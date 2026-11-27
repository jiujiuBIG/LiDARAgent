from math import isclose, isfinite
from pathlib import Path

import laspy
from rasterio.crs import CRS


def require_metric_crs(crs, label: str) -> CRS:
    if crs is None:
        raise ValueError(f"{label} 缺少 CRS。请确认真实坐标系并写入文件。")
    project_crs = CRS.from_user_input(crs)
    if not project_crs.is_projected or not isclose(project_crs.linear_units_factor[1], 1.0):
        raise ValueError(f"{label} 的 CRS 必须是以米为单位的投影坐标系。")
    return project_crs


def validate_chm(source, *, square_pixels: bool = False) -> CRS:
    if source.count != 1:
        raise ValueError("请提供单波段 CHM GeoTIFF。")
    crs = require_metric_crs(source.crs, "CHM")
    transform = source.transform
    if (
        not all(isfinite(value) for value in transform[:6])
        or transform.b != 0
        or transform.d != 0
        or transform.a <= 0
        or transform.e >= 0
    ):
        raise ValueError("CHM 栅格必须向北且无旋转。")
    if square_pixels and not isclose(transform.a, -transform.e, rel_tol=1e-5):
        raise ValueError("PyCrown 的 CHM 栅格必须使用正方形像元。")
    return crs


def validate_point_cloud_alignment(path: Path, chm_crs: CRS, chm_bounds) -> None:
    if path.suffix.lower() not in (".las", ".laz") or not path.is_file():
        raise ValueError(f"补充点云不可用，请选择 LAS 或 LAZ：{path.name}")
    with laspy.open(path) as source:
        header = source.header
        point_cloud_crs = require_metric_crs(header.parse_crs(), f"点云 {path.name}")
        if point_cloud_crs != chm_crs:
            raise ValueError("补充点云与 CHM 的 CRS 不一致。请先将两者转换到同一米制投影。")
        if not header.point_count:
            raise ValueError(f"补充点云为空：{path.name}")
        min_x, min_y = header.mins[:2]
        max_x, max_y = header.maxs[:2]
        left, bottom, right, top = chm_bounds
        if max_x < left or min_x > right or max_y < bottom or min_y > top:
            raise ValueError("补充点云与 CHM 的空间范围不重叠。")
