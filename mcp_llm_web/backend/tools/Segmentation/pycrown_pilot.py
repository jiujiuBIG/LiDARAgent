"""Fixed-marker wrappers around the two vendored PyCrown Numba kernels."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from rasterio.features import shapes


KERNEL_SOURCE = Path(__file__).resolve().parent / 'pycrown' / 'pycrown'


@dataclass(frozen=True)
class DalponteSettings:
    seed_ratio: float = 0.70
    mean_ratio: float = 0.55
    max_radius_m: float = 10.0
    kernel: str = 'circular'


@lru_cache(maxsize=2)
def get_kernel(name):
    import numba

    if not hasattr(numba, 'float_'):
        numba.float_ = numba.float64
    stems = {'circular': '_crown_dalponteCIRC', 'classic': '_crown_dalponte'}
    function_name = stems[name]
    module_name = f'{function_name}_numba'
    spec = spec_from_file_location(module_name, KERNEL_SOURCE / f'{module_name}.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def dalponte_labels(heights, markers, mask, pixel_size_m, settings, *, upstream_grid=False):
    """Grow from fixed markers and preserve their IDs in the result.

    The radius is rounded to a whole number of pixels and is exclusive, as in
    the upstream kernels. The classic kernel uses a square distance bound.
    """
    if heights.ndim != 2 or heights.shape != markers.shape or heights.shape != mask.shape:
        raise ValueError('CHM, markers and mask must use the same two-dimensional grid')
    if not np.issubdtype(markers.dtype, np.signedinteger) or np.any(markers < 0):
        raise ValueError('Marker IDs must be nonnegative signed integers')
    if mask.dtype != np.bool_ or not np.isfinite(heights[mask]).all():
        raise ValueError('Mask must be boolean and only admit finite heights')
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError('Pixel size must be positive and finite')
    if not (0 <= settings.seed_ratio <= 1 and 0 <= settings.mean_ratio <= 1):
        raise ValueError('Height ratios must be in [0, 1]')
    if not np.isfinite(settings.max_radius_m) or settings.max_radius_m <= 0:
        raise ValueError('Maximum radius must be positive and finite')

    seed_rows, seed_cols = np.nonzero(markers)
    seed_ids = markers[seed_rows, seed_cols]
    if not mask[seed_rows, seed_cols].all() or len(np.unique(seed_ids)) != len(seed_ids):
        raise ValueError('Each positive marker ID must appear once and be inside the mask')
    radius_pixels = round(settings.max_radius_m / pixel_size_m)
    if not seed_ids.size or radius_pixels < 2:
        return markers.copy()

    # The upstream example grows in detection order, and its kernel skips the
    # outer cells. Standalone runs pad the grid to allow valid border growth.
    surface = np.where(mask, heights, -np.inf).astype(np.float32)
    offset = 0 if upstream_grid else 1
    if upstream_grid:
        order = np.argsort(seed_ids)
        seed_rows, seed_cols, seed_ids = seed_rows[order], seed_cols[order], seed_ids[order]
    else:
        surface = np.pad(surface, offset, constant_values=-np.inf)
    seed_xy = np.stack((seed_cols + offset, seed_rows + offset)).astype(np.int32)
    labels = get_kernel(settings.kernel)(
        surface, seed_xy, settings.seed_ratio, settings.mean_ratio, -np.inf, float(radius_pixels)
    )
    ids_by_label = np.r_[np.zeros(1, dtype=markers.dtype), seed_ids]
    inner = labels[1:-1, 1:-1] if offset else labels
    return ids_by_label[inner]


def crown_features(labels, transform):
    """Polygonize the label grid with its IDs and interior rings."""
    for geometry, value in shapes(labels, mask=labels > 0, transform=transform, connectivity=4):
        yield {'type': 'Feature', 'geometry': geometry, 'properties': {'tree_id': int(value)}}
