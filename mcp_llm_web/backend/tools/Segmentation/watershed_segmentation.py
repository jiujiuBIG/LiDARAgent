"""Marker-controlled watershed tree crown segmentation with 3D stem-to-crown treetop recovery."""

import csv
import json
import math
import zipfile
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    gaussian_filter,
    label as nd_label,
    maximum_filter,
    median_filter,
)
from scipy.spatial import cKDTree
from shapely import get_parts, make_valid
from shapely.geometry import MultiPolygon, shape
from skimage.measure import label as sk_label
from skimage.segmentation import watershed

if __package__:
    from .point_instances import (
        MIN_SEPARATED_CROWN_PIXELS, label_crown_and_stem_instances, label_traced_instances,
        las_height_above_ground, select_separated_crown_tops, write_instance_cloud,
    )
    from .spatial_inputs import require_metric_crs, validate_chm, validate_point_cloud_alignment
else:
    from point_instances import (
        MIN_SEPARATED_CROWN_PIXELS, label_crown_and_stem_instances, label_traced_instances,
        las_height_above_ground, select_separated_crown_tops, write_instance_cloud,
    )
    from spatial_inputs import require_metric_crs, validate_chm, validate_point_cloud_alignment

mcp = FastMCP("WatershedTreeCrownSegmentation")

NEIGHBORS = np.ones((3, 3), dtype=bool)
HORIZONTAL_M = 0.25
VERTICAL_M = 0.5
SEED_BOTTOM_M = 1.5
SEED_TOP_M = 6.5


@dataclass(frozen=True)
class LayerParameters:
    relative_support: float = 0.05
    radius_m: float = 2.0
    gap_m: float = 1.5
    min_height_m: float = 10.0


def detect_treetops(
    heights: np.ndarray,
    valid: np.ndarray,
    resolution: tuple[float, float],
    *,
    sigma_m: float = 0.35,
    min_distance_m: float = 0.75,
    min_height_m: float = 2.0,
) -> np.ndarray:
    """Return int32 treetop markers numbered in row-major order."""
    values = np.asarray(heights)
    mask = np.asarray(valid)
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError("heights and boolean valid mask must be same-size 2D arrays")
    if len(resolution) != 2:
        raise ValueError("resolution must be (x metres, y metres)")
    rx, ry = map(float, resolution)
    params = (rx, ry, float(sigma_m), float(min_distance_m), float(min_height_m))
    if not all(math.isfinite(v) for v in params) or rx <= 0 or ry <= 0 or sigma_m < 0 or min_distance_m <= 0:
        raise ValueError("resolution and separation must be positive, and Gaussian sigma must be nonnegative")
    mask = mask & np.isfinite(values)
    if not np.any(mask):
        return np.zeros(values.shape, dtype=np.int32)

    sigma_px = (float(sigma_m) / ry, float(sigma_m) / rx)
    weights = gaussian_filter(mask.astype(np.float64), sigma_px, mode="constant", cval=0.0)
    numerator = gaussian_filter(np.where(mask, values, 0).astype(np.float64), sigma_px, mode="constant", cval=0.0)
    smooth = np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 0)

    rows = int(math.ceil(min_distance_m / ry))
    cols = int(math.ceil(min_distance_m / rx))
    yy, xx = np.ogrid[-rows:rows + 1, -cols:cols + 1]
    footprint = (xx * rx) ** 2 + (yy * ry) ** 2 <= min_distance_m ** 2
    local_max = maximum_filter(np.where(mask, smooth, -np.inf), footprint=footprint, mode="constant", cval=-np.inf)
    eligible = mask & (values >= min_height_m) & (smooth == local_max)

    peak_levels = np.zeros(values.shape, dtype=np.int32)
    _, levels = np.unique(smooth[eligible], return_inverse=True)
    peak_levels[eligible] = levels + 1
    components = sk_label(peak_levels, connectivity=2)
    ids, first_cells = np.unique(components, return_index=True)
    candidates = [(int(cell), float(smooth.flat[cell])) for cell in first_cells[ids != 0]]
    candidates.sort(key=lambda item: (-item[1], item[0]))

    chosen: list[int] = []
    min_distance_sq = float(min_distance_m) ** 2
    ncols = values.shape[1]
    inv_d = 1.0 / float(min_distance_m)
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for cell, _ in candidates:
        row, col = divmod(cell, ncols)
        ym = row * ry
        xm = col * rx
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
                    if dy * dy + dx * dx < min_distance_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if not too_close:
            chosen.append(cell)
            buckets.setdefault((by, bx), []).append((ym, xm))

    markers = np.zeros(values.shape, dtype=np.int32)
    for marker_id, cell in enumerate(sorted(chosen), start=1):
        markers.flat[cell] = marker_id
    return markers


def voxel_counts(points: np.ndarray, horizontal_m: float, vertical_m: float, min_height_m: float):
    """Bin returns with axes ordered as northing (row), easting (col) and height (level)."""
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


def band_tops(active: np.ndarray, gap_cells: int):
    """Find supported band tops with a low-density gap and another layer above."""
    tops = np.zeros_like(active)
    if active.shape[-1] <= gap_cells + 2:
        return tops
    above = np.logical_or.accumulate(active[..., ::-1], axis=-1)[..., ::-1]
    stop = active.shape[-1] - gap_cells - 1
    tops[..., 1:stop] = active[..., 1:stop] & active[..., :stop - 1] & above[..., 2 + gap_cells:]
    for offset in range(1, gap_cells + 1):
        tops[..., 1:stop] &= ~active[..., 1 + offset:stop + offset]
    return tops


def detect_lower_treetops(counts: np.ndarray, origin: np.ndarray, horizontal_m: float, vertical_m: float, parameters: LayerParameters):
    """Detect peaks of lower bands using spatial support and a local vertical gap."""
    if not counts.size:
        return []
    support = convolve(counts, np.ones((3, 3, 1), dtype=np.int32), mode="constant")
    active = support >= np.maximum(5, parameters.relative_support * support.max(axis=2, keepdims=True))
    gap_cells = max(1, int(np.ceil(parameters.gap_m / vertical_m)))
    tops = band_tops(active, gap_cells)
    heights = (np.arange(counts.shape[2]) + 0.5) * vertical_m
    tops &= heights >= parameters.min_height_m
    tops[..., 1:] &= counts[..., 1:] + counts[..., :-1] >= 3
    if not np.any(tops):
        return []
    score = np.where(tops, heights + 0.1 * support / max(1, support.max()), -np.inf)
    radius = int(np.ceil(parameters.radius_m / horizontal_m))
    layer_cells = int(np.floor(1.5 / vertical_m))
    dy, dx, dz = np.ogrid[-radius:radius + 1, -radius:radius + 1, -layer_cells:layer_cells + 1]
    footprint = np.broadcast_to(
        (dx ** 2 + dy ** 2) * horizontal_m ** 2 <= parameters.radius_m ** 2,
        (2 * radius + 1, 2 * radius + 1, dz.size),
    )
    peaks = tops & (score == maximum_filter(score, footprint=footprint, mode="constant", cval=-np.inf))
    rows, columns, levels = np.nonzero(peaks)
    order = np.argsort(-score[rows, columns, levels], kind="stable")
    return [
        (
            float(origin[0] + (columns[i] + 0.5) * horizontal_m),
            float(origin[1] + (rows[i] + 0.5) * horizontal_m),
            float(heights[levels[i]]),
        )
        for i in order
    ]


def stem_support_scores(counts: np.ndarray, vertical_m: float):
    """Measure vertically continuous and horizontally concentrated low returns."""
    start, stop = int(np.ceil(1.5 / vertical_m)), int(np.ceil(6.5 / vertical_m))
    low = counts[..., start:stop]
    if not low.size:
        empty = np.zeros(counts.shape[:2], dtype=np.int32)
        return empty, empty.copy()
    narrow = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    around = convolve(low, narrow[..., None], mode="constant")
    larger = convolve(low, np.ones((5, 5, 1), dtype=np.int32), mode="constant")
    supported = (around >= 2) & (around * 2 >= larger - around)
    run = np.zeros(counts.shape[:2], dtype=np.int32)
    longest = np.zeros_like(run)
    for active in np.moveaxis(supported, -1, 0):
        run = np.where(active, run + 1, 0)
        np.maximum(longest, run, out=longest)
    return longest, around.sum(axis=2)


def stem_positions(scores: np.ndarray, density: np.ndarray, origin: np.ndarray, horizontal_m: float, min_layers: int):
    """Represent adjacent persistent columns by a dominant XY position."""
    hits = scores >= min_layers
    if not hits.any():
        return []
    score = scores + 0.25 * density / max(1, density.max())
    radius = int(np.ceil(0.75 / horizontal_m))
    dy, dx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (dy ** 2 + dx ** 2) * horizontal_m ** 2 <= 0.75 ** 2
    representatives = hits & (score == maximum_filter(score, footprint=disk, mode="constant"))
    islands, count = nd_label(representatives, structure=NEIGHBORS)
    positions = []
    for component in range(1, count + 1):
        cells = np.argwhere(islands == component)
        best = np.argmax(score[cells[:, 0], cells[:, 1]])
        row, col = cells[best]
        positions.append((float(origin[0] + (col + 0.5) * horizontal_m), float(origin[1] + (row + 0.5) * horizontal_m)))
    return positions


def stem_supported_candidates(existing, candidates, stems, owner_radius: float = 2.0, candidate_radius: float = 1.25):
    """Add at most one lower-layer top per stem position unclaimed by existing tops."""
    if not stems or not candidates:
        return []
    index = cKDTree(stems)
    original = np.asarray(existing, dtype=float).reshape(-1, 3)
    if len(original):
        distances, owners = index.query(original[:, :2])
        used = set(owners[distances <= owner_radius])
    else:
        used = set()
    result = []
    for candidate in sorted(candidates, key=lambda point: (-point[2], point[0], point[1])):
        distance, stem = index.query(candidate[:2])
        if distance <= candidate_radius and int(stem) not in used:
            result.append((candidate, int(stem)))
            used.add(int(stem))
    return result


def verified_stem_roots(counts: np.ndarray, origin: np.ndarray, stems, min_stem_layers: int):
    """Seed only occupied voxels at the bottom of each verified low vertical band."""
    first = int(SEED_BOTTOM_M / VERTICAL_M)
    stop = int(SEED_TOP_M / VERTICAL_M)
    window = np.pad(counts[..., first:stop], ((2, 2), (2, 2), (0, 0)))
    yy, xx = np.ogrid[-2:3, -2:3]
    central = np.abs(yy) + np.abs(xx) <= 1
    offsets = np.argwhere(central) - 2
    roots = {}
    for stem_id, stem in enumerate(stems):
        col, row = np.floor((np.asarray(stem) - origin) / HORIZONTAL_M).astype(int)
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


def trace_owners(
    counts: np.ndarray,
    origin: np.ndarray,
    stems,
    radius_m: float = 0.75,
    axis_weight: float = 0.3,
    max_step_m: float = 0.5,
    gap_levels: int = 1,
    min_stem_layers: int = 7,
):
    """Assign occupied voxels to competing stems by short upward paths."""
    owner = np.full(counts.shape, -1, dtype=np.int32)
    costs = np.full(counts.shape, np.inf)
    if not len(stems) or not counts.size:
        return owner
    stem_xy = np.asarray(stems, dtype=float)
    ny, nx, nz = counts.shape
    xx, yy = np.meshgrid(
        origin[0] + (np.arange(nx) + 0.5) * HORIZONTAL_M,
        origin[1] + (np.arange(ny) + 0.5) * HORIZONTAL_M,
    )
    roots = verified_stem_roots(counts, origin, stems, min_stem_layers)
    occupancy = counts >= 2
    max_cells = int(max_step_m / HORIZONTAL_M)
    shifts = [
        (dy, dx, hypot(dy, dx) * HORIZONTAL_M)
        for dy in range(-max_cells, max_cells + 1)
        for dx in range(-max_cells, max_cells + 1)
        if hypot(dy, dx) * HORIZONTAL_M <= max_step_m and abs(dx) < nx and abs(dy) < ny
    ]
    windows = []
    for dy, dx, step in shifts:
        origin_y = slice(max(0, -dy), ny - max(0, dy))
        origin_x = slice(max(0, -dx), nx - max(0, dx))
        dest_y = slice(max(0, dy), ny - max(0, -dy))
        dest_x = slice(max(0, dx), nx - max(0, -dx))
        windows.append(((origin_y, origin_x), (dest_y, dest_x), step))
    for level in range(int(SEED_BOTTOM_M / VERTICAL_M), nz):
        open_cells = occupancy[..., level]
        level_roots = roots.get(level, ())
        if not open_cells.any() and not level_roots:
            continue
        layer_cost = costs[..., level]
        layer_owner = owner[..., level]
        for rise in range(1, min(gap_levels + 1, level) + 1):
            previous_cost = costs[..., level - rise]
            previous_owner = owner[..., level - rise]
            for source, dest, movement in windows:
                earlier = previous_cost[source]
                valid = np.isfinite(earlier) & open_cells[dest]
                if not valid.any():
                    continue
                ids = previous_owner[source][valid]
                distance_to_stem = np.hypot(xx[dest][valid] - stem_xy[ids, 0], yy[dest][valid] - stem_xy[ids, 1])
                travel = earlier[valid] + movement + 0.5 * (rise - 1) + axis_weight * distance_to_stem * rise * VERTICAL_M
                best = layer_cost[dest]
                take_local = (distance_to_stem <= radius_m) & (travel < best[valid])
                take = valid.copy()
                take[valid] = take_local
                best[take] = travel[take_local]
                layer_owner[dest][take] = ids[take_local]
        for row, col, stem_id in level_roots:
            stem_x, stem_y = stem_xy[stem_id]
            initial = hypot(xx[row, col] - stem_x, yy[row, col] - stem_y)
            if initial < layer_cost[row, col]:
                layer_cost[row, col] = initial
                layer_owner[row, col] = stem_id
    return owner


def supported_crown_tops(
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
    """Pick the locally supported crown top assigned to each stem before canopy jump."""
    kernel = np.ones((3, 3), dtype=np.int32)
    result = {}
    start_level = int(SEED_TOP_M / VERTICAL_M)
    ny, nx, nz = counts.shape
    target_ids = list(range(len(stems)) if stem_ids is None else stem_ids)
    if not target_ids or start_level >= nz or not len(stems):
        return result

    # Precompute exact XY bounding boxes for all target stems in a single pass.
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
    order_s = np.argsort(s_all, kind="mergesort")
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
            cap = convolve(mask.astype(np.int32), kernel, mode="constant")
            cap_cache[lv] = cap
            rows, cols = np.nonzero(mask & (cap >= min_cap_cells))
            if max_stem_offset_m is not None and len(rows):
                x_e = origin[0] + (cols + c0 + 0.5) * HORIZONTAL_M
                y_e = origin[1] + (rows + r0 + 0.5) * HORIZONTAL_M
                keep = ((x_e - stem_x) ** 2 + (y_e - stem_y) ** 2) <= max_stem_offset_m ** 2
                rows, cols = rows[keep], cols[keep]
            level_caps[lv] = int(cap[rows, cols].max()) if len(rows) else int(cap[mask].max())
            level_pts[lv] = int(local_counts[..., rel][mask].sum())
            all_r, all_c = np.nonzero(mask)
            x = origin[0] + (all_c + c0 + 0.5) * HORIZONTAL_M
            y = origin[1] + (all_r + r0 + 0.5) * HORIZONTAL_M
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
            if (b0[-1] + 0.5) * VERTICAL_M <= 9.5 and len(b0) <= 4 and (b1[0] - b0[-1]) <= 3:
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
                    xs = origin[0] + (col_s + c0 + 0.5) * HORIZONTAL_M
                    ys = origin[1] + (r_s + r0 + 0.5) * HORIZONTAL_M
                    if np.any((xs - stem_x) ** 2 + (ys - stem_y) ** 2 <= max_stem_offset_m ** 2):
                        eligible_strong.append(slv)
            if not eligible_strong:
                continue
            strong_lvs = eligible_strong

        lv = strong_lvs[-1]

        if peak_drop_stop is not None and len(strong_lvs) >= 10:
            mid_lvs = [slv for slv in strong_lvs if 14.0 <= (slv + 0.5) * VERTICAL_M <= 19.5]
            if mid_lvs:
                base_dx, base_dy = [], []
                for mlv in mid_lvs:
                    rel_m = mlv - start_level
                    m_m = belongs[..., rel_m]
                    r_m, c_m = np.nonzero(m_m)
                    w_m = np.maximum(local_counts[r_m, c_m, rel_m].astype(float), 1.0)
                    base_dx.append(float(np.sum((origin[0] + (c_m + c0 + 0.5) * HORIZONTAL_M - stem_x) * w_m) / np.sum(w_m)))
                    base_dy.append(float(np.sum((origin[1] + (r_m + r0 + 0.5) * HORIZONTAL_M - stem_y) * w_m) / np.sum(w_m)))
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
                elif running_max_pts >= 1500 and (slv + 0.5) * VERTICAL_M >= 19.5 and slv > peak_lv and idx_s > 0:
                    rel_s = slv - start_level
                    m_s = belongs[..., rel_s]
                    r_s, c_s = np.nonzero(m_s)
                    w_s = np.maximum(local_counts[r_s, c_s, rel_s].astype(float), 1.0)
                    dx_s = float(np.sum((origin[0] + (c_s + c0 + 0.5) * HORIZONTAL_M - stem_x) * w_s) / np.sum(w_s))
                    dy_s = float(np.sum((origin[1] + (r_s + r0 + 0.5) * HORIZONTAL_M - stem_y) * w_s) / np.sum(w_s))
                    drift_reversed = (dx_s * ref_dx + dy_s * ref_dy) < -0.005
                    if pts_here <= running_max_pts * peak_drop_stop and drift_reversed:
                        lv = strong_lvs[idx_s - 1]
                        break

        z = (lv + 0.5) * VERTICAL_M
        top_slice = [l for l in strong_lvs if abs(l - lv) <= 2 and l <= lv]
        all_x, all_y, all_w = [], [], []
        for l_idx in top_slice:
            rel_l = l_idx - start_level
            m_l = belongs[..., rel_l]
            c_l = cap_cache[l_idx]
            r_l, col_l = np.nonzero(m_l & (c_l >= min_cap_cells))
            if not len(r_l):
                continue
            x_l = origin[0] + (col_l + c0 + 0.5) * HORIZONTAL_M
            y_l = origin[1] + (r_l + r0 + 0.5) * HORIZONTAL_M
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


def merge_treetops(baseline, candidates, distance_m: float = 0.75, vertical_m: float = 1.5):
    """Keep nearby markers when their normalized heights separate their layers."""
    result = list(baseline)
    if not result:
        positions = np.empty((0, 3), dtype=float)
    else:
        positions = np.array(result, dtype=float).reshape(-1, 3)
    for candidate in candidates:
        if len(positions):
            delta = positions - candidate
            duplicate = (delta[:, 0] ** 2 + delta[:, 1] ** 2 < distance_m ** 2) & (np.abs(delta[:, 2]) < vertical_m)
            if np.any(duplicate):
                continue
        result.append(candidate)
        positions = np.vstack((positions, candidate)) if len(positions) else np.array([candidate], dtype=float)
    return result


def _grid_reduce(x, y, z, minx, maxy, res, shape, how="max"):
    nrows, ncols = shape
    col = np.clip(((x - minx) / res).astype(np.int64), 0, ncols - 1)
    row = np.clip(((maxy - y) / res).astype(np.int64), 0, nrows - 1)
    flat = row * ncols + col
    size = nrows * ncols
    if how == "max":
        out = np.full(size, -np.inf)
        np.maximum.at(out, flat, z)
    else:
        out = np.full(size, np.inf)
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
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def build_chm_and_normalized_cloud_from_las(source_path: Path, resolution: float = 0.5):
    """Build a 0.5 m CHM, DTM, transform, CRS, and normalized point cloud from LAS/LAZ."""
    las = laspy.read(source_path)
    crs = require_metric_crs(las.header.parse_crs(), f"点云 {source_path.name}")
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)
    if not len(x):
        raise ValueError("输入点云为空。")
    dims = set(las.point_format.dimension_names)
    cls = np.asarray(las.classification, dtype=np.int32) if "classification" in dims else np.zeros(len(x), dtype=np.int32)
    explicit_height = las_height_above_ground(las)

    minx, maxx = float(x.min()), float(x.max())
    miny, maxy = float(y.min()), float(y.max())
    shape = (int(np.ceil((maxy - miny) / resolution)) + 1, int(np.ceil((maxx - minx) / resolution)) + 1)
    transform = from_origin(minx, maxy, resolution, resolution)
    crs_wkt = crs.to_wkt()
    del las

    dsm_raw = _grid_reduce(x, y, z, minx, maxy, resolution, shape, "max")
    ground_mask = cls == 2
    del cls
    has_classified_ground = bool(np.any(ground_mask))
    if has_classified_ground:
        dtm_raw = _grid_reduce(x[ground_mask], y[ground_mask], z[ground_mask], minx, maxy, resolution, shape, "min")
        dtm = median_filter(_fill_nearest(dtm_raw), size=3, mode="nearest")
    elif float(np.percentile(z, 5)) >= -0.5 and float(np.percentile(z, 1)) <= 0.5:
        dtm = np.zeros_like(dsm_raw)
    else:
        dtm_raw = _grid_reduce(x, y, z, minx, maxy, resolution, shape, "min")
        dtm = median_filter(_fill_nearest(dtm_raw), size=3, mode="nearest")
    del ground_mask

    occ = np.isfinite(dsm_raw)
    dist = distance_transform_edt(~occ, sampling=resolution)
    dsm = _fill_nearest(dsm_raw)
    dsm[dist > 1.0] = dtm[dist > 1.0]

    chm = dsm - dtm
    chm[~np.isfinite(chm)] = 0.0
    chm[chm < 0] = 0.0

    cols = np.clip(np.floor((x - minx) / resolution).astype(np.int32), 0, shape[1] - 1)
    rows = np.clip(np.floor((maxy - y) / resolution).astype(np.int32), 0, shape[0] - 1)
    # Elevation heuristics may construct a 2-D CHM. Only measured ground
    # classes or explicit point heights can establish a 3-D height reference.
    if has_classified_ground:
        norm_z = z - dtm[rows, cols]
    elif explicit_height is not None:
        norm_z = explicit_height
    else:
        norm_z = np.full_like(z, np.nan)
    del rows, cols, z
    cloud = np.column_stack((x, y, norm_z))
    del x, y, norm_z
    return chm.astype(np.float64), occ, transform, crs_wkt, (resolution, resolution), cloud


def supplement_treetops_from_cloud(baseline_points, cloud: np.ndarray, instance_evidence: dict | None = None):
    """Supplement 2D CHM treetops and optionally keep 3D trace ownership."""
    if not len(cloud):
        return list(baseline_points), 0
    layer_grid, layer_origin = voxel_counts(cloud, 0.5, 0.5, 2.0)
    lower_candidates = detect_lower_treetops(layer_grid, layer_origin, 0.5, 0.5, LayerParameters())
    low_grid, low_origin = voxel_counts(cloud[cloud[:, 2] < 6.5], HORIZONTAL_M, VERTICAL_M, 1.5)
    run, density = stem_support_scores(low_grid, VERTICAL_M)
    stems = stem_positions(run, density, low_origin, HORIZONTAL_M, min_layers=7)
    paired = stem_supported_candidates(baseline_points, lower_candidates, stems, owner_radius=2.0, candidate_radius=1.25)
    if not stems or (not paired and instance_evidence is None):
        return list(baseline_points), 0
    stem_ids = [stem_id for _, stem_id in paired] if instance_evidence is None else None
    grid, origin = voxel_counts(cloud, HORIZONTAL_M, VERTICAL_M, 1.5)
    owners = trace_owners(grid, origin, stems, radius_m=0.75, axis_weight=0.3, max_step_m=0.5, gap_levels=1, min_stem_layers=7)
    traced_tops = supported_crown_tops(grid, owners, origin, stems, min_cap_cells=7, stem_ids=stem_ids)
    if instance_evidence is not None:
        instance_evidence.update(owners=owners, origin=origin, tops=traced_tops, stems=stems)
    additions = [traced_tops.get(stem_id, fallback) for fallback, stem_id in paired]
    merged = merge_treetops(baseline_points, additions, distance_m=0.75, vertical_m=1.5)
    return merged, len(merged) - len(baseline_points)


def recover_separated_crowns(markers, labels, heights, mask, transform, treetops_by_id, evidence):
    tops = {index: top[:3] for index, top in treetops_by_id.items()}
    candidates = select_separated_crown_tops(evidence["stems"], evidence["tops"], labels, heights, transform, tops)
    occupied = set(zip(*np.nonzero(markers)))
    additions = []
    for top in candidates:
        cell = rasterio.transform.rowcol(transform, *top[:2])
        if cell not in occupied:
            additions.append((cell, top))
            occupied.add(cell)

    surface = np.where(mask, -heights, 0.0)
    first_id = len(treetops_by_id) + 1
    while additions:
        trial_markers = markers.copy()
        for index, (cell, _) in enumerate(additions, first_id):
            trial_markers[cell] = index
        trial_labels = watershed(surface, trial_markers, mask=mask, connectivity=1, watershed_line=False)
        sizes = np.bincount(trial_labels.ravel(), minlength=first_id + len(additions))
        kept = [item for index, item in enumerate(additions, first_id)
                if sizes[index] >= MIN_SEPARATED_CROWN_PIXELS]
        if len(kept) == len(additions):
            treetops_by_id.update({
                index: (*top, "stem_to_crown_3d") for index, (_, top) in enumerate(additions, first_id)
            })
            return trial_markers, trial_labels, len(additions)
        additions = kept
    return markers, labels, 0


def write_crowns_vector(shp_path: Path, geojson_path: Path, labels: np.ndarray, heights: np.ndarray, treetops_by_id: dict, transform, crs, pixel_area: float):
    pos_mask = labels > 0
    pos_labels = labels[pos_mask]
    ids, pixel_counts = np.unique(pos_labels, return_counts=True)
    pixels_per_tree = dict(zip(ids.tolist(), pixel_counts.tolist()))
    max_id = int(ids[-1]) if len(ids) else 0
    max_h_by_id = np.full(max_id + 1, -np.inf, dtype=np.float64)
    if len(pos_labels):
        pos_heights = heights[pos_mask]
        valid_h = np.isfinite(pos_heights)
        if np.any(valid_h):
            np.maximum.at(max_h_by_id, pos_labels[valid_h], pos_heights[valid_h])
    records = []
    for geometry, value in shapes(labels, mask=pos_mask, connectivity=4, transform=transform):
        tree_id = int(value)
        polygon = shape(geometry)
        if not polygon.is_valid:
            parts = [
                piece
                for part in get_parts(make_valid(polygon))
                for piece in get_parts(part)
                if piece.geom_type == "Polygon"
            ]
            if not parts:
                continue
            polygon = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        mh = max_h_by_id[tree_id] if tree_id <= max_id else -np.inf
        max_h = float(mh) if np.isfinite(mh) else 0.0
        top_x, top_y, top_z, source_kind = treetops_by_id.get(tree_id, (polygon.centroid.x, polygon.centroid.y, max_h, "chm"))
        area_m2 = float(pixels_per_tree[tree_id] * pixel_area)
        diam_m = float(2.0 * math.sqrt(area_m2 / math.pi)) if area_m2 > 0 else 0.0
        records.append({
            "tree_id": tree_id,
            "x_top": round(float(top_x), 3),
            "y_top": round(float(top_y), 3),
            "height_m": round(float(max(top_z, max_h)), 2),
            "area_m2": round(area_m2, 2),
            "diam_m": round(diam_m, 2),
            "n_pixels": int(pixels_per_tree[tree_id]),
            "source": source_kind,
            "geometry": polygon,
        })
    records.sort(key=lambda r: r["tree_id"])
    if records:
        crowns = gpd.GeoDataFrame(records, crs=crs)
        crowns.to_file(shp_path, driver="ESRI Shapefile", engine="pyogrio", index=False)
        crowns.to_file(geojson_path, driver="GeoJSON", engine="pyogrio", index=False)
    else:
        empty = gpd.GeoDataFrame(
            {"tree_id": [], "x_top": [], "y_top": [], "height_m": [], "area_m2": [], "diam_m": [], "n_pixels": [], "source": []},
            geometry=[],
            crs=crs,
        )
        empty.to_file(shp_path, driver="ESRI Shapefile", engine="pyogrio", index=False, geometry_type="Polygon")
        empty.to_file(geojson_path, driver="GeoJSON", engine="pyogrio", index=False, geometry_type="Polygon")
    return records


@mcp.tool(name="segment_trees_watershed")
async def segment_trees_watershed(
    file_path: str,
    sigma_m: Annotated[float, Field(ge=0, description="Gaussian smoothing sigma in metres.")] = 0.35,
    min_distance_m: Annotated[float, Field(gt=0, description="Minimum horizontal treetop spacing in metres.")] = 0.75,
    min_canopy_height_m: Annotated[float, Field(ge=0, description="Minimum canopy height in metres.")] = 2.0,
    chm_resolution_m: Annotated[float, Field(gt=0, description="CHM grid resolution in metres when input is LAS/LAZ.")] = 0.5,
    use_pointcloud_supplement: bool = True,
    point_cloud_path: str | None = None,
    output_path: str = "./",
    export_las: bool = True,
) -> dict:
    """Segment individual tree crowns using marker-controlled watershed on CHM with 3D stem-to-crown treetop recovery.

    Accepts either a point cloud (.las/.laz) or a canopy height model GeoTIFF (.tif/.tiff).
    When a point cloud is provided (directly or via point_cloud_path), lower-canopy trees obscured in 2D CHM
    are recovered by verifying 1.5-6.5 m vertical stems and tracing upward with constriction and dome-drift truncation.
    If LAS export is enabled, tree_instances.las preserves source points and appends point heights and
    supported tree IDs. Unassigned points keep ID zero.
    """
    try:
        for val, label_name, allow_zero in (
            (sigma_m, "平滑半径 sigma_m", True),
            (min_distance_m, "最小树顶间距 min_distance_m", False),
            (min_canopy_height_m, "最小冠层高度 min_canopy_height_m", True),
            (chm_resolution_m, "CHM 分辨率 chm_resolution_m", False),
        ):
            if type(val) not in (int, float) or not math.isfinite(val) or (val < 0 if allow_zero else val <= 0):
                raise ValueError(f"{label_name} 必须为有效的正数。")

        source = Path(file_path)
        if not source.is_file():
            raise ValueError(f"输入文件不存在: {file_path}")
        ext = source.suffix.lower()
        if ext not in (".las", ".laz", ".tif", ".tiff"):
            raise ValueError("请选择 LAS/LAZ 点云或 TIF/TIFF 冠层高度图文件。")

        cloud = None
        cloud_path = None
        if ext in (".las", ".laz"):
            cloud_path = source
            heights, valid, transform, crs, resolution, cloud = build_chm_and_normalized_cloud_from_las(
                source, float(chm_resolution_m)
            )
        else:
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(source, "r") as ds:
                    crs = validate_chm(ds)
                    bounds = ds.bounds
                    raw = ds.read(1, masked=True).astype(np.float64)
                    raw = raw * ds.scales[0] + ds.offsets[0]
                    valid = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
                    heights = np.where(valid, raw.data, 0.0)
                    transform = ds.transform
                    resolution = (abs(float(ds.res[0])), abs(float(ds.res[1])))
            if use_pointcloud_supplement or (export_las and point_cloud_path is not None):
                pc_candidate = Path(point_cloud_path) if point_cloud_path else None
                if pc_candidate is None:
                    for sibling in ("cloud.laz", "cloud.las", f"{source.stem}.laz", f"{source.stem}.las"):
                        cand = source.parent / sibling
                        if cand.is_file():
                            pc_candidate = cand
                            break
                if pc_candidate is not None:
                    validate_point_cloud_alignment(pc_candidate, crs, bounds)
                    _, _, _, _, _, cloud = build_chm_and_normalized_cloud_from_las(pc_candidate, resolution[0])
                    cloud_path = pc_candidate

        mask = valid & (heights >= float(min_canopy_height_m))
        chm_markers = detect_treetops(
            heights,
            valid,
            resolution,
            sigma_m=float(sigma_m),
            min_distance_m=float(min_distance_m),
            min_height_m=float(min_canopy_height_m),
        )
        chm_rows, chm_cols = np.nonzero(chm_markers > 0)
        order = np.argsort(chm_markers[chm_rows, chm_cols])
        chm_rows, chm_cols = chm_rows[order], chm_cols[order]
        baseline_points = [
            (*rasterio.transform.xy(transform, int(r), int(c)), float(heights[int(r), int(c)]))
            for r, c in zip(chm_rows, chm_cols)
        ]

        supplemented_count = 0
        all_points = list(baseline_points)
        instance_evidence = {}
        pointcloud_used = bool(use_pointcloud_supplement and cloud is not None and np.isfinite(cloud[:, 2]).any())
        if pointcloud_used:
            all_points, supplemented_count = supplement_treetops_from_cloud(
                baseline_points, cloud, instance_evidence=instance_evidence,
            )

        nrows, ncols = heights.shape
        markers = np.zeros((nrows, ncols), dtype=np.int32)
        treetops_by_id = {}
        next_id = 1
        for idx_pt, (x, y, z) in enumerate(all_points):
            r, c = rasterio.transform.rowcol(transform, x, y)
            if 0 <= r < nrows and 0 <= c < ncols and mask[r, c] and markers[r, c] == 0:
                markers[r, c] = next_id
                source_kind = "chm" if idx_pt < len(baseline_points) else "stem_to_crown_3d"
                treetops_by_id[next_id] = (float(x), float(y), float(z), source_kind)
                next_id += 1

        seeds = markers > 0
        surface = np.where(mask, -heights, 0.0)
        labels = watershed(surface, markers, mask=mask, connectivity=1, watershed_line=False) if seeds.any() else markers.copy()
        separated_count = 0
        if instance_evidence and treetops_by_id:
            markers, labels, separated_count = recover_separated_crowns(
                markers, labels, heights, mask, transform, treetops_by_id, instance_evidence,
            )

        out_dir = Path(output_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_file = out_dir / f"{source.stem}_watershed_crowns.zip"
        if zip_file.exists():
            raise FileExistsError(f"output already exists: {zip_file}")

        pixel_area = float(resolution[0] * resolution[1])
        int_profile = {
            "driver": "GTiff",
            "dtype": "int32",
            "count": 1,
            "nodata": 0,
            "width": ncols,
            "height": nrows,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
        }
        chm_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": ncols,
            "height": nrows,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
        }

        with TemporaryDirectory(prefix=".watershed_", dir=out_dir) as tmp:
            staging = Path(tmp)
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(staging / "chm.tif", "w", **chm_profile) as dst:
                    dst.write(heights.astype(np.float32), 1)
                with rasterio.open(staging / "markers.tif", "w", **int_profile) as dst:
                    dst.write(markers, 1)
                with rasterio.open(staging / "crowns.tif", "w", **int_profile) as dst:
                    dst.write(labels, 1)
                records = write_crowns_vector(
                    staging / "tree_crowns.shp",
                    staging / "tree_crowns.geojson",
                    labels,
                    heights,
                    treetops_by_id,
                    transform,
                    crs,
                    pixel_area,
                )

            csv_rows = [
                {
                    "tree_id": r["tree_id"],
                    "x_top": r["x_top"],
                    "y_top": r["y_top"],
                    "height_m": r["height_m"],
                    "area_m2": r["area_m2"],
                    "diam_m": r["diam_m"],
                    "n_pixels": r["n_pixels"],
                    "source": r["source"],
                }
                for r in records
            ]
            with (staging / "treetops.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["tree_id", "x_top", "y_top", "height_m", "area_m2", "diam_m", "n_pixels", "source"],
                )
                writer.writeheader()
                writer.writerows(csv_rows)

            tree_heights = [r["height_m"] for r in records]
            crown_areas = [r["area_m2"] for r in records]
            chm_seed_count = sum(1 for r in records if r["source"] == "chm")
            supplement_seed_count = sum(1 for r in records if r["source"] == "stem_to_crown_3d")
            instance_export = None
            if export_las and cloud_path is not None:
                treetops = {
                    record["tree_id"]: treetops_by_id[record["tree_id"]][:3]
                    for record in records if pointcloud_used
                }
                traced = None
                if instance_evidence and treetops:
                    traced = label_traced_instances(
                        cloud, instance_evidence["owners"], instance_evidence["origin"],
                        HORIZONTAL_M, VERTICAL_M, instance_evidence["tops"], treetops,
                    )
                point_ids = label_crown_and_stem_instances(
                    cloud, labels, transform, treetops, float(min_canopy_height_m),
                    stem_positions=(instance_evidence or {}).get("stems", ()), traced_labels=traced,
                    stem_top_m=SEED_TOP_M,
                )
                instance_export = write_instance_cloud(
                    cloud_path, staging / "tree_instances.las", cloud[:, 2], point_ids,
                    method="crown_raster_and_verified_stems",
                )

            summary_data = {
                "algorithm": "marker-controlled watershed + 3D stem-to-crown v9",
                "input_type": "point_cloud" if ext in (".las", ".laz") else "chm_raster",
                "pointcloud_supplement_used": pointcloud_used,
                "point_instance_export": instance_export,
                "tree_count": len(records),
                "chm_seed_count": chm_seed_count,
                "stem_supplemented_count": supplement_seed_count,
                "separated_crown_count": separated_count,
                "sigma_m": float(sigma_m),
                "min_distance_m": float(min_distance_m),
                "min_canopy_height_m": float(min_canopy_height_m),
                "chm_resolution_m": float(resolution[0]),
                "height_min_m": round(min(tree_heights, default=0.0), 2),
                "height_max_m": round(max(tree_heights, default=0.0), 2),
                "height_mean_m": round(float(np.mean(tree_heights)) if tree_heights else 0.0, 2),
                "crown_area_mean_m2": round(float(np.mean(crown_areas)) if crown_areas else 0.0, 2),
                "crown_area_max_m2": round(max(crown_areas, default=0.0), 2),
                "canopy_pixels": int(np.count_nonzero(mask)),
                "labelled_pixels": int(np.count_nonzero(labels)),
            }
            (staging / "run.json").write_text(json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            written_files = []
            for item in sorted(staging.iterdir()):
                if not item.is_file():
                    continue
                target = out_dir / item.name
                if not target.exists():
                    item.replace(target)
                    written_files.append(target)
                else:
                    written_files.append(item)

            with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in written_files:
                    zf.write(item, arcname=item.name)

        mode_desc = (
            f"（含 2D CHM 冠顶 {chm_seed_count} 棵 + 3D 树干冠顶追踪补检 {supplement_seed_count} 棵）"
            if summary_data["pointcloud_supplement_used"]
            else f"（纯 2D CHM 冠顶 {chm_seed_count} 棵）"
        )
        message = (
            f"分水岭单木分割完成，共分割出 {len(records)} 棵单木{mode_desc}。"
            f"平均树高 {summary_data['height_mean_m']:.2f} m（最高 {summary_data['height_max_m']:.2f} m），"
            f"平均冠幅面积 {summary_data['crown_area_mean_m2']:.2f} m²。"
        )
        return {
            "status": "success",
            "output_path": str(zip_file),
            "message": message,
            "data": summary_data,
        }
    except Exception as exc:
        return {"status": "error", "output_path": None, "message": str(exc), "data": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")
