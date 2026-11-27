from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from scipy.interpolate import griddata

from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud
from tools.Segmentation.spatial_inputs import require_metric_crs


@dataclass(frozen=True)
class RasterSource:
    path: Path
    points: np.ndarray
    ground_mask: np.ndarray
    crs: CRS
    transform: Affine
    shape: tuple[int, int]


def load_raster_source(file_path: str, resolution: float, input_crs: str | None) -> RasterSource:
    if not isfinite(resolution) or resolution <= 0:
        raise ValueError("网格分辨率必须为有限的正数。")
    path = Path(file_path)
    header_crs = None
    if path.suffix.lower() in (".las", ".laz"):
        with laspy.open(path) as reader:
            header_crs = reader.header.parse_crs()
    crs = require_metric_crs(header_crs if header_crs is not None else input_crs, f"点云 {path.name}")
    if header_crs is not None and input_crs is not None and CRS.from_user_input(input_crs) != crs:
        raise ValueError("指定的 CRS 与点云文件中的 CRS 不一致。请使用点云原有坐标系。")

    frame = clean_point_cloud(read_point_cloud(str(path)))
    points = frame[["x", "y", "z"]].to_numpy()
    finite = np.isfinite(points).all(axis=1)
    frame = frame.loc[finite]
    points = points[finite]
    if not len(points):
        raise ValueError("点云中没有有效的 XYZ 点。")
    ground_mask = (
        frame["classification"].to_numpy() == 2
        if "classification" in frame else np.zeros(len(points), dtype=bool)
    )
    west, south = points[:, :2].min(axis=0)
    east, north = points[:, :2].max(axis=0)
    transform = from_origin(float(west), float(north), resolution, resolution)
    shape = (floor((north - south) / resolution) + 1, floor((east - west) / resolution) + 1)
    return RasterSource(path, points, ground_mask, crs, transform, shape)


def surface_grid(source: RasterSource) -> np.ndarray:
    points, transform = source.points, source.transform
    cols = np.floor((points[:, 0] - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((points[:, 1] - transform.f) / transform.e).astype(np.int64)
    grid = np.full(source.shape, -np.inf, dtype=np.float64)
    np.maximum.at(grid, (rows, cols), points[:, 2])
    grid[~np.isfinite(grid)] = np.nan
    return grid.astype(np.float32)


def terrain_grid(source: RasterSource, ground: np.ndarray) -> np.ndarray:
    if len(ground) < 3:
        raise ValueError("地面点不足，生成 DTM 至少需要三个非共线地面点。")
    height, width = source.shape
    transform = source.transform
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    return griddata(ground[:, :2], ground[:, 2], (x[None, :], y[:, None]), method="linear").astype(np.float32)


def save_raster_product(source: RasterSource, values: np.ndarray, output_path: str, product: str) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"{product} 没有可用像元。请检查地面覆盖范围和网格分辨率。")
    folder = Path(output_path).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source.path.stem}_{product}.tif"
    height, width = source.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=source.crs, transform=source.transform,
        nodata=np.nan, compress="deflate", BIGTIFF="IF_SAFER",
    ) as target:
        target.write(values, 1)
    data = {
        "resolution": source.transform.a,
        "crs": source.crs.to_string(),
        "width": width,
        "height": height,
        "height_min": round(float(finite.min()), 2),
        "height_max": round(float(finite.max()), 2),
        "height_mean": round(float(finite.mean(dtype=np.float64)), 2),
    }
    return {
        "status": "success",
        "output_path": str(path),
        "message": (
            f"已生成 {product} GeoTIFF，分辨率 {data['resolution']:g} 米，坐标系 {data['crs']}。"
            f"有效高程范围为 {data['height_min']:.2f}–{data['height_max']:.2f} 米。"
        ),
        "data": data,
    }
