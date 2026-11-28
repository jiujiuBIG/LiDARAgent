"""Standalone PyCrown-style individual tree detection and crown delineation."""

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from math import hypot
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from affine import Affine
import geopandas as gpd
import laspy
from numba import jit
import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree
from shapely.geometry import Point, Polygon
from skimage.filters import threshold_otsu

if __package__:
    from .point_instances import (
        MIN_SEPARATED_CROWN_PIXELS, label_crown_and_stem_instances, label_traced_instances,
        las_height_above_ground, recover_occluded_instances, select_separated_crown_tops, write_instance_cloud,
    )
    from .spatial_inputs import require_metric_crs, validate_chm, validate_point_cloud_alignment
else:
    from point_instances import (
        MIN_SEPARATED_CROWN_PIXELS, label_crown_and_stem_instances, label_traced_instances,
        las_height_above_ground, recover_occluded_instances, select_separated_crown_tops, write_instance_cloud,
    )
    from spatial_inputs import require_metric_crs, validate_chm, validate_point_cloud_alignment

try:
    from tools.Segmentation.pycrown_pilot import DalponteSettings, crown_features, dalponte_labels
except ImportError:
    from pycrown_pilot import DalponteSettings, crown_features, dalponte_labels


NEIGHBORS_2D = np.ones((3, 3), dtype=bool)
HORIZONTAL_3D_M = 0.25
VERTICAL_3D_M = 0.5
SEED_BOTTOM_M = 1.5
SEED_TOP_M = 6.5


@dataclass(frozen=True)
class Settings:
    smoothing_width_m: float = 0.35
    peak_radius_m: float = 0.75
    min_peak_m: float = 4.0
    min_canopy_m: float = 2.0
    min_final_tree_m: float = 4.0
    seed_ratio: float = 0.0
    mean_ratio: float = 0.0
    crown_radius_m: float = 6.0
    edge_buffer_m: float = 0.0
    south_edge_buffer_m: float | None = None
    max_smoothed_chm_m: float | None = None
    reproduce_upstream_grid: bool = False
    smoothing_filter: str = 'gaussian'
    min_terrain_std_m: float = 1.0
    use_pointcloud_supplement: bool = True
    height_spacing_slope: float = 0.0
    shoulder_dist_m: float = 5.0
    shoulder_min_drop_m: float = 0.8
    shoulder_min_h_diff_m: float = 3.0
    shoulder_min_tall_h_m: float = 27.0
    stem_candidate_radius_m: float = 1.5
    recover_occluded_crowns: bool = True

    def __post_init__(self):
        if self.smoothing_filter not in ('gaussian', 'median'):
            raise ValueError("smoothing_filter must be 'gaussian' or 'median'")
        values = [
            value for value in asdict(self).values()
            if value is not None and not isinstance(value, (bool, str))
        ]
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError('Settings must be finite and nonnegative')
        if min(self.peak_radius_m, self.crown_radius_m) <= 0:
            raise ValueError('Search and crown radii must be positive')
        if self.smoothing_filter == 'median' and self.smoothing_width_m <= 0:
            raise ValueError('Median smoothing width must be positive')
        if max(self.seed_ratio, self.mean_ratio) > 1:
            raise ValueError('Height ratios must be at most one')
        if self.min_peak_m < self.min_canopy_m:
            raise ValueError('Minimum tree peak height must be at least the minimum canopy height')


# The vendored example clips the north, west and east edges, leaving the south edge open.
OFFICIAL = Settings(
    smoothing_width_m=5, peak_radius_m=5, min_peak_m=16,
    min_canopy_m=15, min_final_tree_m=20, seed_ratio=0.70,
    mean_ratio=0.55, crown_radius_m=10, edge_buffer_m=11,
    south_edge_buffer_m=0, max_smoothed_chm_m=60,
    reproduce_upstream_grid=True, smoothing_filter='median',
    min_terrain_std_m=0.0, use_pointcloud_supplement=False,
    height_spacing_slope=0.0, shoulder_dist_m=0.0,
)
DEFAULT_CONFIG = Path(__file__).with_name('pycrown_v1_defaults.json')


@dataclass(frozen=True)
class Grid:
    chm: np.ndarray
    valid: np.ndarray
    dsm: np.ndarray
    dtm: np.ndarray
    transform: Affine
    crs: CRS
    las_path: Path | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)
    ground_reference: str | None = None

    @property
    def pixel_size(self):
        return self.transform.a


@dataclass(frozen=True)
class Segmentation:
    surface: np.ndarray
    labels: np.ndarray
    initial: np.ndarray
    corrected: np.ndarray
    correction_methods: list[str]
    peak_count: int
    seed_count: int


def _discover_las_path(chm_path: Path, las_path: Path | None = None) -> Path | None:
    if las_path is not None:
        resolved = Path(las_path)
        if not resolved.is_file():
            raise ValueError(f'Point cloud file does not exist: {las_path}')
        return resolved
    parent = Path(chm_path).parent
    for name in ('cloud.laz', 'cloud.las', 'POINTS.laz', 'POINTS.las'):
        candidate = parent / name
        if candidate.is_file():
            return candidate
    return None


def read_grids(chm_path, dsm_path, dtm_path, las_path=None):
    if dtm_path is None:
        raise ValueError('An aligned DTM is required when reading terrain grids')
    arrays = []
    reference = None
    for path in (chm_path, dtm_path, dsm_path):
        if path is None:
            continue
        with rasterio.open(path) as source:
            signature = (source.shape, source.transform, source.crs)
            if source.count != 1:
                raise ValueError(f'Expected a single-band raster: {path}')
            if reference is None:
                crs = validate_chm(source, square_pixels=True)
                bounds = source.bounds
                reference = signature
            if signature != reference:
                raise ValueError(f'CHM, DSM and DTM must be aligned on the same grid and CRS: {path}')
            values = source.read(1, masked=True).astype(np.float32)
            physical = values.filled(np.nan) * source.scales[0] + source.offsets[0]
            arrays.append(np.where(np.isfinite(physical), physical, np.nan))
    _, transform, _ = reference
    chm, dtm = arrays[:2]
    dsm = arrays[2] if dsm_path is not None else dtm + chm
    resolved_las = _discover_las_path(Path(chm_path), las_path)
    if resolved_las is not None:
        validate_point_cloud_alignment(resolved_las, crs, bounds)
    return Grid(chm, np.isfinite(chm), dsm, dtm, transform, crs,
                las_path=resolved_las, ground_reference='dtm')


def _grid_reduce(x, y, z, minx, maxy, res, shape, how='max'):
    nrows, ncols = shape
    col = np.clip(np.floor((x - minx) / res).astype(np.int64), 0, ncols - 1)
    row = np.clip(np.floor((maxy - y) / res).astype(np.int64), 0, nrows - 1)
    flat = row * ncols + col
    size = nrows * ncols
    if how == 'max':
        out = np.full(size, -np.inf, dtype=np.float64)
        np.maximum.at(out, flat, z)
    else:
        out = np.full(size, np.inf, dtype=np.float64)
        np.minimum.at(out, flat, z)
    out = out.reshape(nrows, ncols)
    out[~np.isfinite(out)] = np.nan
    return out


def _fill_nearest(arr):
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr
    if mask.all():
        return np.zeros_like(arr)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def build_grid_from_las(las_path: Path, resolution_m: float = 0.5, default_crs: str | None = None) -> Grid:
    """Build an aligned CHM, DSM and DTM Grid directly from a LAS/LAZ point cloud."""
    resolved = Path(las_path)
    if not resolved.is_file():
        raise ValueError(f'Point cloud file does not exist: {las_path}')
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('Resolution must be positive and finite')
    las = laspy.read(resolved)
    parsed_crs = las.header.parse_crs()
    crs = require_metric_crs(parsed_crs if parsed_crs is not None else default_crs, f"点云 {resolved.name}")
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    if not len(x):
        raise ValueError(f'Point cloud is empty: {las_path}')
    dims = set(las.point_format.dimension_names)
    cls = np.asarray(las.classification, dtype=np.int32) if 'classification' in dims else np.zeros(len(x), dtype=np.int32)
    ret_num = np.asarray(las.return_number, dtype=np.int32) if 'return_number' in dims else np.ones(len(x), dtype=np.int32)
    explicit_height = las_height_above_ground(las)

    minx, maxx = float(x.min()), float(x.max())
    miny, maxy = float(y.min()), float(y.max())
    shape = (
        int(np.ceil((maxy - miny) / resolution_m)) + 1,
        int(np.ceil((maxx - minx) / resolution_m)) + 1,
    )
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    del las

    dsm_raw = _grid_reduce(x, y, z, minx, maxy, resolution_m, shape, 'max')
    ground_mask = cls == 2
    del cls
    ground_reference = None
    if np.any(ground_mask):
        dtm_raw = _grid_reduce(x[ground_mask], y[ground_mask], z[ground_mask], minx, maxy, resolution_m, shape, 'min')
        dtm = ndimage.median_filter(_fill_nearest(dtm_raw), size=3, mode='nearest')
        ground_reference = 'classified_ground'
    elif float(np.percentile(z, 5)) >= -0.5 and float(np.percentile(z, 1)) <= 0.5:
        dtm = np.zeros_like(dsm_raw)
    else:
        dtm_raw = _grid_reduce(x, y, z, minx, maxy, resolution_m, shape, 'min')
        dtm = ndimage.median_filter(_fill_nearest(dtm_raw), size=3, mode='nearest')
    del ground_mask

    occ = np.isfinite(dsm_raw)
    dist = ndimage.distance_transform_edt(~occ, sampling=resolution_m)
    dsm = _fill_nearest(dsm_raw)
    dsm[dist > 1.0] = dtm[dist > 1.0]
    chm = np.maximum(0.0, dsm - dtm).astype(np.float32)
    dsm = dsm.astype(np.float32)
    dtm = dtm.astype(np.float32)

    cols = np.floor((x - transform.c) / transform.a).astype(np.int32)
    rows = np.floor((y - transform.f) / transform.e).astype(np.int32)
    nrows, ncols = shape
    in_bounds = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    r_c = np.clip(rows, 0, nrows - 1)
    c_c = np.clip(cols, 0, ncols - 1)
    if ground_reference is not None:
        norm_z = z - dtm[r_c, c_c]
    elif explicit_height is not None:
        norm_z = explicit_height
        ground_reference = 'las_height_above_ground'
    else:
        # Terrain inferred from unclassified Z may support a 2-D CHM, but does not
        # establish a ground reference for the three-dimensional stem search.
        norm_z = np.full_like(z, np.nan)
    cache = {'las_arrays': (x, y, z, ret_num, rows, cols, in_bounds, norm_z),
             'pointcloud_height_reference': ground_reference}
    return Grid(chm, occ, dsm, dtm, transform, crs, las_path=resolved,
                _cache=cache, ground_reference=ground_reference)


def smooth_chm(grid, settings):
    if settings.smoothing_filter == 'gaussian':
        mask = grid.valid & np.isfinite(grid.chm)
        if not np.any(mask):
            return np.zeros_like(grid.chm, dtype=np.float32)
        if settings.smoothing_width_m > 0:
            sigma_px = float(settings.smoothing_width_m) / grid.pixel_size
            weights = ndimage.gaussian_filter(mask.astype(np.float64), sigma_px, mode='constant', cval=0.0)
            numerator = ndimage.gaussian_filter(
                np.where(mask, grid.chm, 0.0).astype(np.float64), sigma_px, mode='constant', cval=0.0
            )
            surface = np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 0)
        else:
            surface = np.where(mask, grid.chm, 0.0).astype(np.float64)
        valid = mask & (surface >= 0.5)
        if settings.max_smoothed_chm_m is not None:
            valid &= surface <= settings.max_smoothed_chm_m
        return np.where(valid, surface, 0.0).astype(np.float32)

    width = max(1, round(settings.smoothing_width_m / grid.pixel_size))
    surface = ndimage.median_filter(np.where(grid.valid, grid.chm, 0.0), size=width)
    valid = grid.valid & (surface >= 0.5)
    if settings.max_smoothed_chm_m is not None:
        valid &= surface <= settings.max_smoothed_chm_m
    return np.where(valid, surface, 0.0)


def detect_treetops(surface, valid, search_radius_px, min_height_m):
    """Return one centre per peak plateau as fractional (row, col) coordinates."""
    offsets = np.arange(-search_radius_px, search_radius_px + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= search_radius_px ** 2
    maximum = ndimage.maximum_filter(np.where(valid, surface, -np.inf), footprint=disk)
    groups, count = ndimage.label(valid & (surface > min_height_m) & (surface == maximum))
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    centres = np.asarray(ndimage.center_of_mass(surface, groups, np.arange(1, count + 1))) + 0.5
    cells = centres.astype(int)
    bad = np.flatnonzero(groups[cells[:, 0], cells[:, 1]] != np.arange(1, count + 1))
    if bad.size:
        regions = ndimage.find_objects(groups)
        for index in bad:
            window = regions[index]
            plateau = np.argwhere(groups[window] == index + 1) + [part.start for part in window]
            candidates = plateau + 0.5
            centres[index] = candidates[np.argmin(np.sum((candidates - centres[index]) ** 2, axis=1))]
    return centres


def detect_gaussian_treetops(
    heights,
    smooth,
    valid,
    pixel_size_m,
    min_distance_m,
    min_height_m,
    height_spacing_slope: float = 0.0,
):
    """Detect local maxima on Gaussian-smoothed CHM with height-adaptive distance suppression."""
    mask = valid & np.isfinite(heights) & np.isfinite(smooth)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)

    r_px = int(math.ceil(min_distance_m / pixel_size_m))
    yy, xx = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    footprint = (xx * pixel_size_m) ** 2 + (yy * pixel_size_m) ** 2 <= min_distance_m ** 2
    local_max = ndimage.maximum_filter(
        np.where(mask, smooth, -np.inf), footprint=footprint, mode='constant', cval=-np.inf
    )
    eligible = mask & (heights >= min_height_m) & (smooth == local_max)
    if not np.any(eligible):
        return np.empty((0, 2), dtype=np.float64)

    components, count = ndimage.label(eligible, structure=NEIGHBORS_2D)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    ids, first_cells = np.unique(components, return_index=True)
    candidates = [(int(cell), float(smooth.flat[cell])) for cell in first_cells[ids != 0]]
    candidates.sort(key=lambda item: (-item[1], item[0]))

    chosen = []
    ncols = heights.shape[1]
    if height_spacing_slope <= 0.0:
        min_d = float(min_distance_m)
        min_d_sq = min_d * min_d
        inv_d = 1.0 / min_d
        buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for cell, _ in candidates:
            row, col = divmod(cell, ncols)
            ym = row * pixel_size_m
            xm = col * pixel_size_m
            by = int(math.floor(ym * inv_d))
            bx = int(math.floor(xm * inv_d))
            too_close = False
            for dby in (-1, 0, 1):
                for dbx in (-1, 0, 1):
                    pts = buckets.get((by + dby, bx + dbx))
                    if pts is None:
                        continue
                    for oy, ox in pts:
                        dy = ym - oy
                        dx = xm - ox
                        if dy * dy + dx * dx < min_d_sq:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if not too_close:
                chosen.append((row, col))
                buckets.setdefault((by, bx), []).append((ym, xm))
    else:
        for cell, h_val in candidates:
            row, col = divmod(cell, ncols)
            req_dist = float(min_distance_m)
            if h_val > 16.0:
                req_dist += float(height_spacing_slope) * (h_val - 16.0)
            req_dist_sq = req_dist ** 2
            if all(
                ((row - orow) * pixel_size_m) ** 2 + ((col - ocol) * pixel_size_m) ** 2 >= req_dist_sq
                for orow, ocol in chosen
            ):
                chosen.append((row, col))
    chosen.sort()
    return np.array([(r + 0.5, c + 0.5) for r, c in chosen], dtype=np.float64).reshape(-1, 2)


def suppress_shoulder_peaks(
    peaks: np.ndarray,
    surface: np.ndarray,
    pixel_size_m: float,
    max_dist_m: float = 5.0,
    min_drop_m: float = 0.8,
    min_h_diff_m: float = 3.0,
    min_tall_h_m: float = 27.0,
) -> np.ndarray:
    """Suppress shallow shoulder/branch sub-peaks sitting on the flank of much taller crowns."""
    if len(peaks) <= 1 or max_dist_m <= 0 or min_drop_m <= 0:
        return peaks
    heights = np.array([float(surface[int(r), int(c)]) for r, c in peaks])
    order = np.argsort(-heights)
    kept = []
    nrows, ncols = surface.shape
    inv_d = 1.0 / float(max_dist_m)
    tall_buckets: dict[tuple[int, int], list[int]] = {}
    for idx in order:
        r, c = peaks[idx]
        h = heights[idx]
        by = int(math.floor(r * pixel_size_m * inv_d))
        bx = int(math.floor(c * pixel_size_m * inv_d))
        is_shoulder = False
        for dby in (-1, 0, 1):
            for dbx in (-1, 0, 1):
                for k in tall_buckets.get((by + dby, bx + dbx), ()):
                    rk, ck = peaks[k]
                    hk = heights[k]
                    if (hk - h) < min_h_diff_m:
                        continue
                    d = math.hypot((r - rk) * pixel_size_m, (c - ck) * pixel_size_m)
                    if d > max_dist_m:
                        continue
                    n_steps = max(5, int(math.ceil(d / (0.25 * pixel_size_m))))
                    rs = np.clip(np.floor(np.linspace(r, rk, n_steps)).astype(int), 0, nrows - 1)
                    cs = np.clip(np.floor(np.linspace(c, ck, n_steps)).astype(int), 0, ncols - 1)
                    drop = h - float(np.min(surface[rs, cs]))
                    if drop < min_drop_m:
                        is_shoulder = True
                        break
                if is_shoulder:
                    break
            if is_shoulder:
                break
        if not is_shoulder:
            kept.append(idx)
            if h >= min_tall_h_m:
                tall_buckets.setdefault((by, bx), []).append(idx)
    kept.sort()
    return peaks[kept]


def _voxel_counts(points: np.ndarray, horizontal_m: float, vertical_m: float, min_height_m: float):
    vegetation = points[np.isfinite(points).all(axis=1) & (points[:, 2] >= min_height_m)]
    if not len(vegetation):
        return np.zeros((0, 0, 0), dtype=np.int32), np.zeros(2)
    origin = np.floor(vegetation[:, :2].min(axis=0) / horizontal_m) * horizontal_m
    columns, rows = np.floor((vegetation[:, :2] - origin) / horizontal_m).astype(int).T
    levels = np.floor(vegetation[:, 2] / vertical_m).astype(int)
    shape = (rows.max() + 1, columns.max() + 1, levels.max() + 1)
    indices = np.ravel_multi_index((rows, columns, levels), shape)
    counts = np.bincount(indices, minlength=int(np.prod(shape))).reshape(shape)
    return counts.astype(np.int32), origin


def _band_tops(active: np.ndarray, gap_cells: int):
    tops = np.zeros_like(active)
    if active.shape[-1] <= gap_cells + 2:
        return tops
    above = np.logical_or.accumulate(active[..., ::-1], axis=-1)[..., ::-1]
    stop = active.shape[-1] - gap_cells - 1
    tops[..., 1:stop] = active[..., 1:stop] & active[..., :stop - 1] & above[..., 2 + gap_cells:]
    for offset in range(1, gap_cells + 1):
        tops[..., 1:stop] &= ~active[..., 1 + offset:stop + offset]
    return tops


def _detect_lower_treetops(
    counts: np.ndarray,
    origin: np.ndarray,
    horizontal_m: float = 0.5,
    vertical_m: float = 0.5,
    relative_support: float = 0.05,
    radius_m: float = 2.0,
    gap_m: float = 1.5,
    min_height_m: float = 10.0,
):
    if not counts.size:
        return []
    support = ndimage.convolve(counts, np.ones((3, 3, 1), dtype=np.int32), mode='constant')
    active = support >= np.maximum(5, relative_support * support.max(axis=2, keepdims=True))
    gap_cells = max(1, int(np.ceil(gap_m / vertical_m)))
    tops = _band_tops(active, gap_cells)
    heights = (np.arange(counts.shape[2]) + 0.5) * vertical_m
    tops &= heights >= min_height_m
    tops[..., 1:] &= counts[..., 1:] + counts[..., :-1] >= 3
    if not np.any(tops):
        return []
    score = np.where(tops, heights + 0.1 * support / max(1, support.max()), -np.inf)
    radius = int(np.ceil(radius_m / horizontal_m))
    layer_cells = int(np.floor(1.5 / vertical_m))
    dy, dx, dz = np.ogrid[-radius:radius + 1, -radius:radius + 1, -layer_cells:layer_cells + 1]
    footprint = np.broadcast_to(
        (dx ** 2 + dy ** 2) * horizontal_m ** 2 <= radius_m ** 2,
        (2 * radius + 1, 2 * radius + 1, dz.size),
    )
    peaks = tops & (score == ndimage.maximum_filter(score, footprint=footprint, mode='constant', cval=-np.inf))
    rows, columns, levels = np.nonzero(peaks)
    order = np.argsort(-score[rows, columns, levels], kind='stable')
    return [
        (
            float(origin[0] + (columns[i] + 0.5) * horizontal_m),
            float(origin[1] + (rows[i] + 0.5) * horizontal_m),
            float(heights[levels[i]]),
        )
        for i in order
    ]


def _stem_support_scores(counts: np.ndarray, vertical_m: float):
    start, stop = int(np.ceil(SEED_BOTTOM_M / vertical_m)), int(np.ceil(SEED_TOP_M / vertical_m))
    low = counts[..., start:stop]
    if not low.size:
        empty = np.zeros(counts.shape[:2], dtype=np.int32)
        return empty, empty.copy()
    narrow = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    around = ndimage.convolve(low, narrow[..., None], mode='constant')
    larger = ndimage.convolve(low, np.ones((5, 5, 1), dtype=np.int32), mode='constant')
    supported = (around >= 2) & (around * 2 >= larger - around)
    run = np.zeros(counts.shape[:2], dtype=np.int32)
    longest = np.zeros_like(run)
    for active in np.moveaxis(supported, -1, 0):
        run = np.where(active, run + 1, 0)
        np.maximum(longest, run, out=longest)
    return longest, around.sum(axis=2)


def _stem_positions(scores: np.ndarray, density: np.ndarray, origin: np.ndarray, horizontal_m: float, min_layers: int):
    hits = scores >= min_layers
    if not hits.any():
        return []
    score = scores + 0.25 * density / max(1, density.max())
    radius = int(np.ceil(0.75 / horizontal_m))
    dy, dx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (dy ** 2 + dx ** 2) * horizontal_m ** 2 <= 0.75 ** 2
    representatives = hits & (score == ndimage.maximum_filter(score, footprint=disk, mode='constant'))
    islands, count = ndimage.label(representatives, structure=NEIGHBORS_2D)
    positions = []
    for component in range(1, count + 1):
        cells = np.argwhere(islands == component)
        best = np.argmax(score[cells[:, 0], cells[:, 1]])
        row, col = cells[best]
        positions.append((float(origin[0] + (col + 0.5) * horizontal_m), float(origin[1] + (row + 0.5) * horizontal_m)))
    return positions


def _verified_stem_roots(counts: np.ndarray, origin: np.ndarray, stems, min_stem_layers: int):
    first = int(SEED_BOTTOM_M / VERTICAL_3D_M)
    stop = int(SEED_TOP_M / VERTICAL_3D_M)
    window = np.pad(counts[..., first:stop], ((2, 2), (2, 2), (0, 0)))
    yy, xx = np.ogrid[-2:3, -2:3]
    central = np.abs(yy) + np.abs(xx) <= 1
    offsets = np.argwhere(central) - 2
    roots = {}
    for stem_id, stem in enumerate(stems):
        col, row = np.floor((np.asarray(stem) - origin) / HORIZONTAL_3D_M).astype(int)
        if not (0 <= row < counts.shape[0] and 0 <= col < counts.shape[1]):
            continue
        region = window[row:row + 5, col:col + 5]
        inside = region[central].sum(axis=0)
        occupied = np.flatnonzero((inside >= 2) & (3 * inside >= region.sum(axis=(0, 1))))
        if not len(occupied):
            continue
        runs = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
        longest = max(runs, key=len)
        if len(longest) < min_stem_layers:
            continue
        level = int(longest[0])
        for dy, dx in offsets[region[central, level] > 0]:
            roots.setdefault(first + level, []).append((row + dy, col + dx, stem_id))
    return roots


@jit(nopython=True, nogil=True, cache=True)
def _trace_owners_jit(
    counts,
    origin_x,
    origin_y,
    stem_xy,
    root_levels,
    root_rows,
    root_cols,
    root_ids,
    shifts_dy,
    shifts_dx,
    shifts_step,
    radius_m,
    axis_weight,
    gap_levels,
    start_level,
):
    ny, nx, nz = counts.shape
    owner = np.full((ny, nx, nz), -1, dtype=np.int32)
    costs = np.full((ny, nx, nz), np.inf, dtype=np.float64)
    n_shifts = len(shifts_dy)
    n_roots = len(root_levels)

    for level in range(start_level, nz):
        max_rise = gap_levels + 1
        if level < max_rise:
            max_rise = level
        for rise in range(1, max_rise + 1):
            prev_level = level - rise
            rise_penalty = 0.5 * (rise - 1)
            axis_factor = axis_weight * rise * VERTICAL_3D_M
            for dy_idx in range(n_shifts):
                dy = shifts_dy[dy_idx]
                dx = shifts_dx[dy_idx]
                movement = shifts_step[dy_idx]
                r_start = max(0, dy)
                r_stop = ny + min(0, dy)
                c_start = max(0, dx)
                c_stop = nx + min(0, dx)
                for r in range(r_start, r_stop):
                    sr = r - dy
                    dest_y = origin_y + (r + 0.5) * HORIZONTAL_3D_M
                    for c in range(c_start, c_stop):
                        if counts[r, c, level] < 2:
                            continue
                        sc = c - dx
                        sid = owner[sr, sc, prev_level]
                        if sid < 0:
                            continue
                        earlier = costs[sr, sc, prev_level]
                        dest_x = origin_x + (c + 0.5) * HORIZONTAL_3D_M
                        dist_stem = np.hypot(dest_x - stem_xy[sid, 0], dest_y - stem_xy[sid, 1])
                        if dist_stem <= radius_m:
                            travel = earlier + movement + rise_penalty + axis_factor * dist_stem
                            if travel < costs[r, c, level]:
                                costs[r, c, level] = travel
                                owner[r, c, level] = sid
        for idx in range(n_roots):
            if root_levels[idx] == level:
                r = root_rows[idx]
                c = root_cols[idx]
                sid = root_ids[idx]
                dest_x = origin_x + (c + 0.5) * HORIZONTAL_3D_M
                dest_y = origin_y + (r + 0.5) * HORIZONTAL_3D_M
                initial = np.hypot(dest_x - stem_xy[sid, 0], dest_y - stem_xy[sid, 1])
                if initial < costs[r, c, level]:
                    costs[r, c, level] = initial
                    owner[r, c, level] = sid
    return owner


def _trace_owners(
    counts: np.ndarray,
    origin: np.ndarray,
    stems,
    radius_m: float = 0.75,
    axis_weight: float = 0.3,
    max_step_m: float = 0.5,
    gap_levels: int = 1,
    min_stem_layers: int = 7,
):
    if not len(stems) or not counts.size:
        return np.full(counts.shape, -1, dtype=np.int32)
    stem_xy = np.asarray(stems, dtype=np.float64)
    roots = _verified_stem_roots(counts, origin, stems, min_stem_layers)
    root_entries = [(lv, r, c, sid) for lv, items in sorted(roots.items()) for (r, c, sid) in items]
    if not root_entries:
        return np.full(counts.shape, -1, dtype=np.int32)
    r_lv = np.array([e[0] for e in root_entries], dtype=np.int32)
    r_row = np.array([e[1] for e in root_entries], dtype=np.int32)
    r_col = np.array([e[2] for e in root_entries], dtype=np.int32)
    r_id = np.array([e[3] for e in root_entries], dtype=np.int32)
    max_cells = int(max_step_m / HORIZONTAL_3D_M)
    ny, nx, _ = counts.shape
    shifts = [
        (dy, dx, hypot(dy, dx) * HORIZONTAL_3D_M)
        for dy in range(-max_cells, max_cells + 1)
        for dx in range(-max_cells, max_cells + 1)
        if hypot(dy, dx) * HORIZONTAL_3D_M <= max_step_m and abs(dx) < nx and abs(dy) < ny
    ]
    s_dy = np.array([s[0] for s in shifts], dtype=np.int32)
    s_dx = np.array([s[1] for s in shifts], dtype=np.int32)
    s_step = np.array([s[2] for s in shifts], dtype=np.float64)
    return _trace_owners_jit(
        counts, float(origin[0]), float(origin[1]), stem_xy,
        r_lv, r_row, r_col, r_id, s_dy, s_dx, s_step,
        float(radius_m), float(axis_weight), int(gap_levels), int(SEED_BOTTOM_M / VERTICAL_3D_M),
    )


def _supported_crown_tops(
    counts: np.ndarray,
    owners: np.ndarray,
    origin: np.ndarray,
    stems,
    min_cap_cells: int = 7,
    stem_ids=None,
    max_stem_offset_m=None,
    stop_cap_below: int = 6,
    centroid_blend: float = 0.85,
    peak_drop_stop: float = 0.68,
):
    kernel = np.ones((3, 3), dtype=np.int32)
    result = {}
    start_level = int(SEED_TOP_M / VERTICAL_3D_M)
    ny, nx, nz = counts.shape
    target_ids = list(range(len(stems)) if stem_ids is None else stem_ids)
    if not target_ids or start_level >= nz or not len(stems):
        return result

    lut = np.zeros(len(stems), dtype=bool)
    lut[target_ids] = True
    sub_owners = owners[..., start_level:]
    occ_mask = sub_owners >= 0
    r_all, c_all, _ = np.nonzero(occ_mask)
    if not len(r_all):
        return result
    s_all = sub_owners[occ_mask]
    keep_s = lut[s_all]
    if not np.any(keep_s):
        return result
    r_all, c_all, s_all = r_all[keep_s], c_all[keep_s], s_all[keep_s]
    order_s = np.argsort(s_all, kind='mergesort')
    s_sorted = s_all[order_s]
    r_sorted = r_all[order_s]
    c_sorted = c_all[order_s]
    uniq_s, split_pos = np.unique(s_sorted, return_index=True)
    ends = np.append(split_pos[1:], len(s_sorted))
    stem_bboxes = {}
    for sid, st, en in zip(uniq_s.tolist(), split_pos.tolist(), ends.tolist()):
        r0 = max(0, int(r_sorted[st:en].min()) - 1)
        r1 = min(ny, int(r_sorted[st:en].max()) + 2)
        c0 = max(0, int(c_sorted[st:en].min()) - 1)
        c1 = min(nx, int(c_sorted[st:en].max()) + 2)
        stem_bboxes[sid] = (r0, r1, c0, c1)

    for stem_id in target_ids:
        bbox = stem_bboxes.get(stem_id)
        if bbox is None:
            continue
        r0, r1, c0, c1 = bbox
        local_owners = owners[r0:r1, c0:c1, start_level:]
        local_counts = counts[r0:r1, c0:c1, start_level:]
        belongs = local_owners == stem_id
        stem_x, stem_y = stems[stem_id]

        level_caps, level_core_caps, level_pts = {}, {}, {}
        cap_cache = {}
        for lv in range(start_level, nz):
            rel = lv - start_level
            mask = belongs[..., rel]
            if not mask.any():
                level_caps[lv] = level_core_caps[lv] = level_pts[lv] = 0
                continue
            cap = ndimage.convolve(mask.astype(np.int32), kernel, mode='constant')
            cap_cache[lv] = cap
            rows, cols = np.nonzero(mask & (cap >= min_cap_cells))
            if max_stem_offset_m is not None and len(rows):
                x_e = origin[0] + (cols + c0 + 0.5) * HORIZONTAL_3D_M
                y_e = origin[1] + (rows + r0 + 0.5) * HORIZONTAL_3D_M
                keep = ((x_e - stem_x) ** 2 + (y_e - stem_y) ** 2) <= max_stem_offset_m ** 2
                rows, cols = rows[keep], cols[keep]
            level_caps[lv] = int(cap[rows, cols].max()) if len(rows) else int(cap[mask].max())
            level_pts[lv] = int(local_counts[..., rel][mask].sum())
            all_r, all_c = np.nonzero(mask)
            x = origin[0] + (all_c + c0 + 0.5) * HORIZONTAL_3D_M
            y = origin[1] + (all_r + r0 + 0.5) * HORIZONTAL_3D_M
            core_r = 0.55 if max_stem_offset_m is None else min(0.55, max_stem_offset_m)
            core = ((x - stem_x) ** 2 + (y - stem_y) ** 2) <= (core_r ** 2)
            level_core_caps[lv] = int(cap[all_r[core], all_c[core]].max()) if core.any() else 0

        eff_stop = min(stop_cap_below, min_cap_cells) if stop_cap_below is not None else min_cap_cells
        bands, cur = [], []
        for lv in range(start_level, nz):
            if level_core_caps[lv] >= eff_stop:
                cur.append(lv)
            elif cur:
                bands.append(cur)
                cur = []
        if cur:
            bands.append(cur)

        valid_bands = [b for b in bands if any(level_core_caps[lv] >= min_cap_cells for lv in b)]
        if not valid_bands:
            fallback = [lv for lv in range(start_level, nz) if level_caps[lv] >= min_cap_cells]
            if not fallback:
                continue
            valid_bands = [fallback]

        while len(valid_bands) >= 2:
            b0, b1 = valid_bands[0], valid_bands[1]
            if (b0[-1] + 0.5) * VERTICAL_3D_M <= 9.5 and len(b0) <= 4 and (b1[0] - b0[-1]) <= 3:
                valid_bands[0] = b0 + b1
                valid_bands.pop(1)
            else:
                break

        chosen_band = valid_bands[0]
        strong_lvs = [lv for lv in chosen_band if level_core_caps.get(lv, 0) >= min_cap_cells]
        if not strong_lvs:
            strong_lvs = [lv for lv in chosen_band if level_caps.get(lv, 0) >= min_cap_cells]

        if max_stem_offset_m is not None:
            eligible_strong = []
            for slv in strong_lvs:
                m_s = belongs[..., slv - start_level]
                c_s = cap_cache[slv]
                r_s, col_s = np.nonzero(m_s & (c_s >= min_cap_cells))
                if len(r_s):
                    xs = origin[0] + (col_s + c0 + 0.5) * HORIZONTAL_3D_M
                    ys = origin[1] + (r_s + r0 + 0.5) * HORIZONTAL_3D_M
                    if np.any((xs - stem_x) ** 2 + (ys - stem_y) ** 2 <= max_stem_offset_m ** 2):
                        eligible_strong.append(slv)
            if not eligible_strong:
                continue
            strong_lvs = eligible_strong

        lv = strong_lvs[-1]

        if peak_drop_stop is not None and len(strong_lvs) >= 10:
            mid_lvs = [slv for slv in strong_lvs if 14.0 <= (slv + 0.5) * VERTICAL_3D_M <= 19.5]
            if mid_lvs:
                base_dx, base_dy = [], []
                for mlv in mid_lvs:
                    rel_m = mlv - start_level
                    m_m = belongs[..., rel_m]
                    r_m, c_m = np.nonzero(m_m)
                    w_m = np.maximum(local_counts[r_m, c_m, rel_m].astype(float), 1.0)
                    base_dx.append(float(np.sum((origin[0] + (c_m + c0 + 0.5) * HORIZONTAL_3D_M - stem_x) * w_m) / np.sum(w_m)))
                    base_dy.append(float(np.sum((origin[1] + (r_m + r0 + 0.5) * HORIZONTAL_3D_M - stem_y) * w_m) / np.sum(w_m)))
                ref_dx, ref_dy = float(np.median(base_dx)), float(np.median(base_dy))
            else:
                ref_dx = ref_dy = 0.0

            running_max_pts = 0
            peak_lv = strong_lvs[0]
            for idx_s, slv in enumerate(strong_lvs):
                pts_here = level_pts[slv]
                if pts_here > running_max_pts:
                    running_max_pts = pts_here
                    peak_lv = slv
                elif running_max_pts >= 1500 and (slv + 0.5) * VERTICAL_3D_M >= 19.5 and slv > peak_lv and idx_s > 0:
                    rel_s = slv - start_level
                    m_s = belongs[..., rel_s]
                    r_s, c_s = np.nonzero(m_s)
                    w_s = np.maximum(local_counts[r_s, c_s, rel_s].astype(float), 1.0)
                    dx_s = float(np.sum((origin[0] + (c_s + c0 + 0.5) * HORIZONTAL_3D_M - stem_x) * w_s) / np.sum(w_s))
                    dy_s = float(np.sum((origin[1] + (r_s + r0 + 0.5) * HORIZONTAL_3D_M - stem_y) * w_s) / np.sum(w_s))
                    drift_reversed = (dx_s * ref_dx + dy_s * ref_dy) < -0.005
                    if pts_here <= running_max_pts * peak_drop_stop and drift_reversed:
                        lv = strong_lvs[idx_s - 1]
                        break

        z = (lv + 0.5) * VERTICAL_3D_M
        top_slice = [l for l in strong_lvs if abs(l - lv) <= 2 and l <= lv]
        all_x, all_y, all_w = [], [], []
        for l_idx in top_slice:
            rel_l = l_idx - start_level
            m_l = belongs[..., rel_l]
            c_l = cap_cache[l_idx]
            r_l, col_l = np.nonzero(m_l & (c_l >= min_cap_cells))
            if not len(r_l):
                continue
            x_l = origin[0] + (col_l + c0 + 0.5) * HORIZONTAL_3D_M
            y_l = origin[1] + (r_l + r0 + 0.5) * HORIZONTAL_3D_M
            if max_stem_offset_m is not None:
                keep = ((x_l - stem_x) ** 2 + (y_l - stem_y) ** 2) <= max_stem_offset_m ** 2
                r_l, col_l, x_l, y_l = r_l[keep], col_l[keep], x_l[keep], y_l[keep]
            if not len(r_l):
                continue
            all_x.append(x_l)
            all_y.append(y_l)
            all_w.append(np.maximum(local_counts[r_l, col_l, rel_l].astype(float), 1.0))

        if not all_w:
            continue
        ax, ay, aw = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_w)
        cx = float(np.sum(ax * aw) / np.sum(aw))
        cy = float(np.sum(ay * aw) / np.sum(aw))
        target_x = centroid_blend * cx + (1.0 - centroid_blend) * stem_x
        target_y = centroid_blend * cy + (1.0 - centroid_blend) * stem_y
        result[stem_id] = (float(target_x), float(target_y), float(z))
    return result


def _merge_treetops_3d(baseline, candidates, distance_m: float = 0.75, vertical_m: float = 1.5):
    result = list(baseline)
    positions = np.array(result, dtype=float).reshape(-1, 3) if result else np.empty((0, 3), dtype=float)
    for candidate in candidates:
        if len(positions):
            delta = positions - candidate
            duplicate = (delta[:, 0] ** 2 + delta[:, 1] ** 2 < distance_m ** 2) & (np.abs(delta[:, 2]) < vertical_m)
            if np.any(duplicate):
                continue
        result.append(candidate)
        positions = np.vstack((positions, candidate)) if len(positions) else np.array([candidate], dtype=float)
    return result


def _load_point_cloud_arrays(grid: Grid):
    if 'las_arrays' in grid._cache:
        return grid._cache['las_arrays']
    if grid.las_path is None or not Path(grid.las_path).is_file():
        grid._cache['las_arrays'] = None
        return None
    las = laspy.read(grid.las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    if not len(x):
        grid._cache['las_arrays'] = None
        return None
    dims = set(las.point_format.dimension_names)
    ret_num = np.asarray(las.return_number, dtype=np.int32) if 'return_number' in dims else np.ones(len(x), dtype=np.int32)
    cols = np.floor((x - grid.transform.c) / grid.transform.a).astype(np.int32)
    rows = np.floor((y - grid.transform.f) / grid.transform.e).astype(np.int32)
    nrows, ncols = grid.dtm.shape
    in_bounds = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    r_c = np.clip(rows, 0, nrows - 1)
    c_c = np.clip(cols, 0, ncols - 1)
    reference = grid.ground_reference
    if reference == 'dtm':
        norm_z = z - grid.dtm[r_c, c_c]
    else:
        norm_z = las_height_above_ground(las)
        if norm_z is None:
            norm_z = np.full_like(z, np.nan)
        else:
            reference = 'las_height_above_ground'
    norm_z = np.where(in_bounds, norm_z, np.nan)
    grid._cache['pointcloud_height_reference'] = reference
    arrays = (x, y, z, ret_num, rows, cols, in_bounds, norm_z)
    grid._cache['las_arrays'] = arrays
    return arrays


def _precompute_3d_stem_evidence(grid: Grid):
    if 'stem_evidence' in grid._cache:
        return grid._cache['stem_evidence']
    arrays = _load_point_cloud_arrays(grid)
    if arrays is None:
        grid._cache['stem_evidence'] = None
        return None
    x, y, _, _, _, _, in_bounds, norm_z = arrays
    valid_pts = in_bounds & np.isfinite(norm_z) & np.isfinite(x) & np.isfinite(y)
    grid._cache['pointcloud_supplement_valid_points'] = int(np.count_nonzero(valid_pts))
    if not np.any(valid_pts):
        grid._cache['stem_evidence'] = None
        return None
    cloud = np.column_stack((x[valid_pts], y[valid_pts], norm_z[valid_pts]))

    layer_grid, layer_origin = _voxel_counts(cloud, 0.5, 0.5, 2.0)
    lower_candidates = _detect_lower_treetops(layer_grid, layer_origin)
    low_grid, low_origin = _voxel_counts(cloud[cloud[:, 2] < SEED_TOP_M], HORIZONTAL_3D_M, VERTICAL_3D_M, SEED_BOTTOM_M)
    run, density = _stem_support_scores(low_grid, VERTICAL_3D_M)
    stems = _stem_positions(run, density, low_origin, HORIZONTAL_3D_M, min_layers=7)
    if not stems:
        grid._cache['stem_evidence'] = None
        return None
    evidence = {
        'cloud': cloud,
        'stems': stems,
        'stem_tree': cKDTree(stems),
        'lower_candidates': sorted(lower_candidates, key=lambda pt: (-pt[2], pt[0], pt[1])),
        'vgrid': None,
        'vorigin': None,
        'owners': None,
        'traced_tops': {},
    }
    grid._cache['stem_evidence'] = evidence
    return evidence


def _resolve_traced_tops(evidence: dict, stem_ids: list[int]) -> dict:
    traced_tops = evidence['traced_tops']
    missing = [sid for sid in stem_ids if sid not in traced_tops]
    if not missing:
        return traced_tops
    if evidence['owners'] is None:
        vgrid, vorigin = _voxel_counts(evidence['cloud'], HORIZONTAL_3D_M, VERTICAL_3D_M, SEED_BOTTOM_M)
        evidence['vgrid'] = vgrid
        evidence['vorigin'] = vorigin
        evidence['owners'] = _trace_owners(
            vgrid, vorigin, evidence['stems'],
            radius_m=0.75, axis_weight=0.3, max_step_m=0.5, gap_levels=1, min_stem_layers=7,
        )
    computed = _supported_crown_tops(
        evidence['vgrid'], evidence['owners'], evidence['vorigin'],
        evidence['stems'], min_cap_cells=7, stem_ids=missing,
    )
    for sid in missing:
        traced_tops[sid] = computed.get(sid)
    return traced_tops


def _filter_inside_margins(peaks: np.ndarray, shape: tuple[int, int], pixel_size: float, settings: Settings):
    if not len(peaks):
        return peaks
    nrows, ncols = shape
    margin = settings.edge_buffer_m / pixel_size
    south = settings.south_edge_buffer_m
    south_margin = margin if south is None else south / pixel_size
    row, col = peaks.T
    inside = (
        (col >= margin) & (col < ncols - margin)
        & (row > margin) & (row <= nrows - south_margin)
    )
    return peaks[inside]


def find_candidates(grid, surface, settings):
    grid._cache.pop('supplement_heights', None)
    if not settings.use_pointcloud_supplement:
        grid._cache.pop('stem_evidence', None)
        grid._cache.pop('pointcloud_supplement_valid_points', None)
    if settings.smoothing_filter == 'gaussian':
        peaks = detect_gaussian_treetops(
            grid.chm, surface, grid.valid, grid.pixel_size,
            settings.peak_radius_m, settings.min_peak_m,
            height_spacing_slope=settings.height_spacing_slope,
        )
        peaks = suppress_shoulder_peaks(
            peaks, surface, grid.pixel_size,
            max_dist_m=settings.shoulder_dist_m,
            min_drop_m=settings.shoulder_min_drop_m,
            min_h_diff_m=settings.shoulder_min_h_diff_m,
            min_tall_h_m=settings.shoulder_min_tall_h_m,
        )
    else:
        radius = max(1, round(settings.peak_radius_m / grid.pixel_size))
        peaks = detect_treetops(surface, grid.valid, radius, settings.min_peak_m)

    raw_peak_count = len(peaks)
    candidates = _filter_inside_margins(peaks, surface.shape, grid.pixel_size, settings)

    if settings.use_pointcloud_supplement:
        evidence = _precompute_3d_stem_evidence(grid)
        if evidence is not None:
            stem_tree = evidence['stem_tree']
            lower_candidates = evidence['lower_candidates']
            rows, cols = (
                candidates.astype(int).T
                if len(candidates) else (np.array([], dtype=int), np.array([], dtype=int))
            )
            xy = world_points(grid.transform, candidates) if len(candidates) else np.empty((0, 2))
            height_grid = grid.chm if settings.smoothing_filter == 'gaussian' else surface
            baseline = [(float(x), float(y), float(height_grid[r, c])) for (x, y), r, c in zip(xy, rows, cols)]
            if len(baseline):
                original = np.asarray(baseline, dtype=float).reshape(-1, 3)
                distances, owners = stem_tree.query(original[:, :2])
                used = set(owners[distances <= 2.0])
            else:
                used = set()
            paired = []
            for candidate in lower_candidates:
                dist, stem = stem_tree.query(candidate[:2])
                if dist <= settings.stem_candidate_radius_m and int(stem) not in used:
                    paired.append((candidate, int(stem)))
                    used.add(int(stem))
            if paired:
                stem_ids = [stem_id for _, stem_id in paired]
                traced_tops = _resolve_traced_tops(evidence, stem_ids)
                additions = [traced_tops.get(stem_id) or fallback for fallback, stem_id in paired]
                merged = _merge_treetops_3d(baseline, additions, distance_m=0.75, vertical_m=1.5)
                added_pts = merged[len(baseline):]
                if added_pts:
                    occupied = set(zip(rows.tolist(), cols.tolist()))
                    nrows, ncols = surface.shape
                    extra = []
                    supplement_heights = grid._cache.setdefault('supplement_heights', {})
                    for x, y, height in added_pts:
                        col_f, row_f = ~grid.transform * (x, y)
                        r, c = int(np.floor(row_f)), int(np.floor(col_f))
                        if (
                            0 <= r < nrows and 0 <= c < ncols
                            and grid.valid[r, c]
                            and height_grid[r, c] > settings.min_canopy_m
                            and (r, c) not in occupied
                        ):
                            extra.append((float(row_f), float(col_f)))
                            occupied.add((r, c))
                            supplement_heights[(r, c)] = float(height)
                    if extra:
                        extra_arr = _filter_inside_margins(
                            np.array(extra, dtype=np.float64).reshape(-1, 2),
                            surface.shape, grid.pixel_size, settings,
                        )
                        if len(extra_arr):
                            candidates = (
                                np.vstack((candidates, extra_arr))
                                if len(candidates) else extra_arr
                            )
                            raw_peak_count += len(extra_arr)
    return candidates, raw_peak_count


def correct_treetops(labels, centres, dtm, dsm, min_terrain_std_m=1.0):
    corrected = centres.copy()
    methods = ['unchanged'] * len(centres)
    for index, window in enumerate(ndimage.find_objects(labels)):
        if window is None:
            continue
        in_crown = labels[window] == index + 1
        terrain = dtm[window][in_crown]
        canopy = dsm[window][in_crown]
        orig_row, orig_col = centres[index].astype(int)
        seed_ground = dtm[orig_row, orig_col]
        seed_dsm = dsm[orig_row, orig_col]
        if not np.isfinite(seed_ground) or not np.isfinite(canopy).any():
            methods[index] = 'missing_elevation'
            continue
        deviation = np.nanstd(terrain)
        if deviation <= 0 or deviation < min_terrain_std_m or seed_ground > np.nanmean(terrain) - deviation:
            continue
        crown_cells = np.argwhere(in_crown) + [part.start for part in window]
        centroid = crown_cells.mean(axis=0)
        dsm_peak = crown_cells[np.nanargmax(canopy)] + 0.5
        mean_distance = np.linalg.norm(crown_cells - centroid, axis=1).mean()
        if np.linalg.norm(dsm_peak - centroid) <= mean_distance:
            cand, cand_method = dsm_peak, 'dsm_max'
        else:
            cand, cand_method = centroid, 'crown_centroid'
        row, col = cand.astype(int)
        if labels[row, col] != index + 1:
            interior = crown_cells + 0.5
            cand = interior[np.argmin(np.sum((interior - centroid) ** 2, axis=1))]
            cand_method = 'crown_centroid_inside'
            row, col = cand.astype(int)
        if min_terrain_std_m > 0 and np.isfinite(seed_dsm) and dsm[row, col] < seed_dsm:
            continue
        corrected[index], methods[index] = cand, cand_method
    return corrected, methods


def grow_crowns(grid, surface, centres, peak_count, settings):
    if settings.smoothing_filter == 'gaussian' and not settings.reproduce_upstream_grid:
        valid_chm = grid.valid & np.isfinite(grid.chm) & (grid.chm >= 0.5)
        if settings.max_smoothed_chm_m is not None:
            valid_chm &= grid.chm <= settings.max_smoothed_chm_m
        growth_surface = np.where(valid_chm, grid.chm, 0.0).astype(np.float32)
    else:
        growth_surface = surface

    markers = np.zeros(growth_surface.shape, dtype=np.int32)
    row, col = (
        centres.astype(int).T
        if len(centres) else (np.array([], dtype=int), np.array([], dtype=int))
    )
    if len(centres):
        markers[row, col] = np.arange(1, len(centres) + 1)
    mask = grid.valid & (growth_surface > settings.min_canopy_m)
    growth = DalponteSettings(settings.seed_ratio, settings.mean_ratio, settings.crown_radius_m)
    labels = dalponte_labels(
        growth_surface, markers, mask, grid.pixel_size, growth,
        upstream_grid=settings.reproduce_upstream_grid,
    )
    corrected, methods = correct_treetops(
        labels, centres, grid.dtm, grid.dsm, min_terrain_std_m=settings.min_terrain_std_m
    )
    if len(centres):
        keep = growth_surface[row, col] >= settings.min_final_tree_m
    else:
        keep = np.zeros(0, dtype=bool)
    ids = np.zeros(len(centres) + 1, dtype=np.int32)
    ids[np.flatnonzero(keep) + 1] = np.arange(1, np.count_nonzero(keep) + 1)
    return Segmentation(
        growth_surface, ids[labels], centres[keep], corrected[keep],
        [method for method, retain in zip(methods, keep) if retain],
        peak_count, len(centres),
    )


def segment(grid, settings):
    grid._cache.pop('separated_crown_count', None)
    surface = smooth_chm(grid, settings)
    peaks, peak_count = find_candidates(grid, surface, settings)
    provisional = grow_crowns(grid, surface, peaks, peak_count, settings)
    evidence = grid._cache.get('stem_evidence')
    if not evidence or not len(provisional.initial):
        return provisional

    heights = grid._cache.get('supplement_heights', {})
    treetops = {
        index: (*xy, heights.get(tuple(seed.astype(int)), float(grid.chm[tuple(seed.astype(int))])))
        for index, (xy, seed) in enumerate(zip(world_points(grid.transform, provisional.initial), provisional.initial), 1)
    }
    stems = evidence['stems']
    traces = _resolve_traced_tops(evidence, list(range(len(stems))))
    tops = select_separated_crown_tops(stems, traces, provisional.labels, grid.chm, grid.transform, treetops)
    occupied = {tuple(seed.astype(int)) for seed in peaks}
    additions = []
    inverse = ~grid.transform
    for x, y, height in tops:
        col, row = inverse * (x, y)
        cell = (int(row), int(col))
        if cell not in occupied and height >= settings.min_final_tree_m and provisional.surface[cell] >= settings.min_final_tree_m:
            additions.append(((row, col), height))
            occupied.add(cell)

    while additions:
        seeds = np.array([seed for seed, _ in additions])
        trial = grow_crowns(grid, surface, np.vstack((peaks, seeds)), peak_count + len(seeds), settings)
        first_id = len(provisional.initial) + 1
        sizes = np.bincount(trial.labels.ravel(), minlength=first_id + len(seeds))
        kept = [item for index, item in enumerate(additions, first_id)
                if sizes[index] >= MIN_SEPARATED_CROWN_PIXELS]
        if len(kept) == len(additions):
            grid._cache.setdefault('supplement_heights', {}).update(
                {tuple(int(value) for value in seed): height for seed, height in additions}
            )
            grid._cache['separated_crown_count'] = len(additions)
            return trial
        additions = kept
    return provisional


def world_points(transform, centres):
    if not len(centres):
        return np.empty((0, 2), dtype=np.float64)
    rows, cols = centres.T
    return np.column_stack(transform * (cols, rows))


TOP_FIELDS = (
    'tree_id', 'x_initial', 'y_initial', 'x_corrected', 'y_corrected',
    'height_initial_m', 'height_corrected_m', 'correction',
)


def _fast_convex_hull_polygon(xy: np.ndarray, fallback_geom):
    if len(xy) < 3:
        return fallback_geom
    try:
        hull = ConvexHull(xy)
        poly = Polygon(xy[hull.vertices])
        if not poly.is_valid:
            poly = poly.make_valid()
        if poly.is_empty or poly.geom_type not in ('Polygon', 'MultiPolygon') or poly.area <= 0:
            return fallback_geom
        return poly
    except (QhullError, ValueError):
        return fallback_geom


def smooth_crowns_and_classify_las(grid: Grid, result: Segmentation, raster_crowns: gpd.GeoDataFrame, out_las_path: Path | None = None, first_return: bool = True):
    """Build Dalponte/PyCrown Otsu first-return convex-hull crown polygons and optional classified LAS."""
    arrays = _load_point_cloud_arrays(grid)
    if arrays is None or not len(result.initial):
        return raster_crowns.copy(), 0

    x, y, z, ret_num, rows, cols, in_bounds, _ = arrays
    valid_idx = np.flatnonzero(in_bounds)
    if not len(valid_idx):
        return raster_crowns.copy(), 0

    crown_ids = result.labels[rows[valid_idx], cols[valid_idx]]
    keep_mask = crown_ids > 0
    in_crown_idx = valid_idx[keep_mask]
    crown_ids = crown_ids[keep_mask]
    if not len(in_crown_idx):
        return raster_crowns.copy(), 0

    order = np.argsort(crown_ids, kind='mergesort')
    sorted_idx = in_crown_idx[order]
    sorted_ids = crown_ids[order]
    unique_ids, split_pos = np.unique(sorted_ids, return_index=True)
    groups = np.split(sorted_idx, split_pos[1:])
    points_by_tree = dict(zip(unique_ids.tolist(), groups))

    raster_geom_by_id = dict(zip(raster_crowns['tree_id'].tolist(), raster_crowns.geometry.tolist()))
    smooth_records = []
    kept_point_indices = []
    kept_point_tree_ids = []

    for tree_id in range(1, len(result.initial) + 1):
        fallback_geom = raster_geom_by_id.get(tree_id)
        pts_idx = points_by_tree.get(tree_id)
        if pts_idx is None or len(pts_idx) < 3:
            if fallback_geom is not None:
                smooth_records.append({'tree_id': tree_id, 'geometry': fallback_geom})
            continue

        tz = z[pts_idx]
        if len(tz) > 1 and (np.max(tz) - np.min(tz)) > 1e-3:
            otsu_th = float(threshold_otsu(tz))
            cand_idx = pts_idx[tz >= otsu_th]
            if first_return:
                fr_idx = cand_idx[ret_num[cand_idx] == 1]
                if len(fr_idx) >= 3:
                    cand_idx = fr_idx
        else:
            cand_idx = pts_idx

        xy = np.column_stack((x[cand_idx], y[cand_idx]))
        hull = _fast_convex_hull_polygon(xy, fallback_geom)
        smooth_records.append({'tree_id': tree_id, 'geometry': hull})
        kept_point_indices.append(cand_idx)
        kept_point_tree_ids.append(np.full(len(cand_idx), tree_id, dtype=np.int32))

    if smooth_records:
        smooth_gdf = gpd.GeoDataFrame(smooth_records, crs=grid.crs)
        smooth_gdf['area_m2'] = smooth_gdf.geometry.area
    else:
        smooth_gdf = gpd.GeoDataFrame(
            {'tree_id': np.array([], dtype=np.int32), 'area_m2': np.array([], dtype=np.float64)},
            geometry=[], crs=grid.crs,
        )

    total_classified_points = int(sum(len(arr) for arr in kept_point_indices))
    if out_las_path is not None and kept_point_indices:
        all_idx = np.concatenate(kept_point_indices)
        all_tree_ids = np.concatenate(kept_point_tree_ids)
        header = laspy.LasHeader(point_format=0, version='1.4')
        header.add_crs(PyprojCRS.from_wkt(grid.crs.to_wkt()), keep_compatibility=False)
        header.offsets = np.floor([np.min(x[all_idx]), np.min(y[all_idx]), np.min(z[all_idx])])
        header.scales = [0.001, 0.001, 0.001]
        header.add_extra_dim(laspy.ExtraBytesParams(name='tree_id', type=np.int32))
        out_las = laspy.LasData(header)
        out_las.x = x[all_idx]
        out_las.y = y[all_idx]
        out_las.z = z[all_idx]
        out_las.intensity = np.clip(all_tree_ids, 0, 65535).astype(np.uint16)
        out_las.tree_id = all_tree_ids
        out_las.write(out_las_path)

    return smooth_gdf, total_classified_points


def result_vectors(grid, result):
    records = []
    initial_xy = world_points(grid.transform, result.initial)
    corrected_xy = world_points(grid.transform, result.corrected)
    for tree_id, (original, corrected, xy, adjusted, method) in enumerate(
        zip(result.initial, result.corrected, initial_xy, corrected_xy, result.correction_methods), 1
    ):
        fields = (
            tree_id, *xy, *adjusted, float(result.surface[tuple(original.astype(int))]),
            float(result.surface[tuple(corrected.astype(int))]), method,
        )
        records.append(dict(zip(TOP_FIELDS, fields)))
    tops = gpd.GeoDataFrame(
        records, columns=TOP_FIELDS,
        geometry=[Point(x, y) for x, y in corrected_xy], crs=grid.crs,
    )
    features = list(crown_features(result.labels, grid.transform))
    if features:
        crowns = gpd.GeoDataFrame.from_features(features, crs=grid.crs)
        crowns.geometry = crowns.geometry.make_valid()
    else:
        crowns = gpd.GeoDataFrame({'tree_id': np.array([], dtype=np.int32)}, geometry=[], crs=grid.crs)
    crowns['area_m2'] = crowns.geometry.area
    return records, tops, crowns


def write_results(folder, grid, result, export_las=True, *, settings=Settings()):
    grid._cache.pop('instance_export', None)
    records, tops, crowns = result_vectors(grid, result)
    with rasterio.open(
        folder / 'crown_labels.tif', 'w', driver='GTiff', count=1, dtype='int32',
        width=result.labels.shape[1], height=result.labels.shape[0],
        crs=grid.crs, transform=grid.transform, nodata=0, compress='deflate',
    ) as dst:
        dst.write(result.labels, 1)
    tops.to_file(folder / 'treetops.gpkg', index=False, driver='GPKG', engine='pyogrio', geometry_type='Point')
    crowns.to_file(folder / 'crowns.gpkg', index=False, driver='GPKG', engine='pyogrio')
    with (folder / 'treetops.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=TOP_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    classified_points = 0
    if grid.las_path is not None and Path(grid.las_path).is_file():
        las_out = (folder / 'trees.las') if export_las else None
        smooth_crowns, classified_points = smooth_crowns_and_classify_las(grid, result, crowns, out_las_path=las_out)
        smooth_crowns.to_file(folder / 'crowns_smooth.gpkg', index=False, driver='GPKG', engine='pyogrio')
        if export_las:
            arrays = _load_point_cloud_arrays(grid)
            if arrays is None:
                raise ValueError('Source point cloud is empty or unavailable for export.')
            x, y, _, _, _, _, in_bounds, norm_z = arrays
            heights = np.where(in_bounds, norm_z, np.nan)
            cloud = np.column_stack((x, y, heights))
            evidence = grid._cache.get('stem_evidence')
            raster_ids = set(crowns['tree_id'])
            supplements = grid._cache.get('supplement_heights', {})
            treetops = {
                record['tree_id']: (
                    record['x_initial'], record['y_initial'],
                    supplements.get(tuple(seed.astype(int)), float(grid.chm[tuple(seed.astype(int))])),
                )
                for record, seed in zip(records, result.initial)
                if settings.use_pointcloud_supplement and record['tree_id'] in raster_ids
            }
            traced = None
            if evidence and treetops:
                traced_tops = _resolve_traced_tops(evidence, list(range(len(evidence['stems']))))
                traced = label_traced_instances(
                    cloud, evidence['owners'], evidence['vorigin'], HORIZONTAL_3D_M, VERTICAL_3D_M,
                    traced_tops, treetops, SEED_BOTTOM_M,
                )
            point_ids = label_crown_and_stem_instances(
                cloud, result.labels, grid.transform, treetops, settings.min_canopy_m,
                stem_positions=(evidence or {}).get('stems', ()), traced_labels=traced,
                stem_top_m=SEED_TOP_M,
            )
            occluded_parents = {}
            if settings.recover_occluded_crowns and evidence and treetops:
                point_ids, occluded_parents = recover_occluded_instances(
                    cloud, point_ids, result.labels, grid.transform, grid.chm,
                    evidence['stems'], traced_tops, treetops,
                )
            grid._cache['instance_export'] = write_instance_cloud(
                grid.las_path, folder / 'tree_instances.las', heights, point_ids,
                method="crown_raster_and_verified_stems",
                summary_path=folder / 'tree_summary.csv' if settings.recover_occluded_crowns else None,
                occluded_parent_ids=occluded_parents,
            )
    return classified_points


def run(chm, dsm, dtm, output, settings, las=None, export_las=True, resolution_m=0.5):
    start = perf_counter()
    if output.exists():
        raise FileExistsError(f'Output path already exists: {output}')
    with rasterio.Env(GDAL_PAM_ENABLED='NO'):
        if chm is not None and dtm is not None:
            grid = read_grids(chm, dsm, dtm, las_path=las)
        elif las is not None and chm is None and dsm is None and dtm is None:
            grid = build_grid_from_las(las, resolution_m=resolution_m)
        else:
            raise ValueError('Provide aligned --chm and --dtm with optional --dsm, or a single --las point cloud')
        loaded = perf_counter()
        result = segment(grid, settings)
        segmented = perf_counter()
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix='.pycrown_v1_') as staging:
            folder = Path(staging)
            classified_points = write_results(
                folder, grid, result, export_las=export_las, settings=settings,
            )
            summary = {
                'settings': asdict(settings), 'crs': grid.crs.to_string(), 'pixel_size_m': grid.pixel_size,
                'peak_count': result.peak_count, 'seed_count': result.seed_count, 'tree_count': len(result.initial),
                'moved_tops': int(np.count_nonzero(np.any(result.corrected != result.initial, axis=1))),
                'crown_area_m2': float(np.count_nonzero(result.labels) * grid.pixel_size ** 2),
                'classified_lidar_points': classified_points,
                'point_instance_export': grid._cache.get('instance_export'),
                'pointcloud_height_reference': grid._cache.get('pointcloud_height_reference'),
                'pointcloud_supplement_used': bool(settings.use_pointcloud_supplement and
                                                   grid._cache.get('pointcloud_supplement_valid_points', 0)),
                'timing_s': {
                    'read': loaded - start, 'segment': segmented - loaded,
                    'total': perf_counter() - start,
                },
            }
            (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
            folder.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('chm', 'dsm', 'dtm'):
        parser.add_argument(f'--{name}', required=False, default=None, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--las', type=Path, default=None, help='Optional LiDAR point cloud (.las/.laz) for 3D supplement and smooth crowns')
    parser.add_argument('--resolution', type=float, default=0.5, help='Grid resolution in metres when building grids directly from --las')
    parser.add_argument('--no-export-las', action='store_true', help='Skip writing classified trees.las when a point cloud is available')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preset', choices=('official',), help='Reproduce the official c06 example settings')
    mode.add_argument('--config', type=Path, help='Use a custom JSON settings file')
    args = parser.parse_args()
    try:
        if args.preset == 'official':
            settings = OFFICIAL
        else:
            config = args.config or DEFAULT_CONFIG
            settings = Settings(**json.loads(config.read_text(encoding='utf-8')))
        result = run(
            args.chm, args.dsm, args.dtm, args.output, settings,
            las=args.las, export_las=not args.no_export_las, resolution_m=args.resolution,
        )
    except (OSError, ValueError, TypeError, rasterio.errors.RasterioError) as exc:
        parser.exit(1, f'PyCrown: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
