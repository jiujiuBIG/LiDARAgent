"""PyCrown Dalponte circular region-growing individual tree segmentation MCP tool."""

import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import numpy as np
import rasterio
from mcp.server.fastmcp import FastMCP
from pydantic import Field

if __package__:
    from .spatial_inputs import validate_chm, validate_point_cloud_alignment
else:
    from spatial_inputs import validate_chm, validate_point_cloud_alignment

try:
    from tools.Segmentation.pycrown_v1 import (
        Grid,
        Settings,
        build_grid_from_las,
        read_grids,
        segment,
        write_results,
    )
except ImportError:
    from pycrown_v1 import (
        Grid,
        Settings,
        build_grid_from_las,
        read_grids,
        segment,
        write_results,
    )

mcp = FastMCP("PyCrownTreeCrownSegmentation")


def _find_sibling_raster(chm_path: Path, target_kind: str) -> Path | None:
    """Locate an aligned sibling DSM or DTM GeoTIFF next to a CHM raster."""
    stem_lower = chm_path.stem.lower()
    candidates = []
    if "chm" in stem_lower:
        idx = stem_lower.index("chm")
        prefix = chm_path.stem[:idx]
        suffix = chm_path.stem[idx + 3 :]
        for replacement in (target_kind.lower(), target_kind.upper()):
            candidates.append(chm_path.with_name(f"{prefix}{replacement}{suffix}{chm_path.suffix}"))
    for name in (f"{target_kind}.tif", f"{target_kind.upper()}.tif", f"{target_kind}_0p50m.tif"):
        candidates.append(chm_path.parent / name)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_grid_from_raster(
    chm_path: Path,
    dsm_path: str | None,
    dtm_path: str | None,
    las_path: Path | None,
) -> Grid:
    if dsm_path is not None or dtm_path is not None:
        if not dtm_path:
            raise ValueError("提供 DSM 时也需要同一网格的 DTM。")
        dsm_cand, dtm_cand = Path(dsm_path) if dsm_path else None, Path(dtm_path)
        for candidate in (dsm_cand, dtm_cand):
            if candidate is not None and not candidate.is_file():
                raise ValueError(f"辅助栅格不可用：{candidate.name}")
    else:
        dsm_cand = _find_sibling_raster(chm_path, "dsm")
        dtm_cand = _find_sibling_raster(chm_path, "dtm")
    if dtm_cand is not None:
        return read_grids(chm_path, dsm_cand, dtm_cand, las_path=las_path)

    with rasterio.open(chm_path) as src:
        crs = validate_chm(src, square_pixels=True)
        transform = src.transform
        bounds = src.bounds
        values = src.read(1, masked=True).astype(np.float32)
        physical = values.filled(np.nan) * src.scales[0] + src.offsets[0]
        chm = np.where(np.isfinite(physical), physical, np.nan)

    dtm = np.full_like(chm, np.nan, dtype=np.float32)
    dsm = np.where(np.isfinite(chm), chm, 0.0).astype(np.float32)
    resolved_las = las_path
    if resolved_las is None:
        for sibling in ("cloud.laz", "cloud.las", f"{chm_path.stem}.laz", f"{chm_path.stem}.las"):
            cand = chm_path.parent / sibling
            if cand.is_file():
                resolved_las = cand
                break
    if resolved_las is not None:
        validate_point_cloud_alignment(resolved_las, crs, bounds)
    return Grid(chm, np.isfinite(chm), dsm, dtm, transform, crs, las_path=resolved_las)


@mcp.tool(name="segment_trees_pycrown")
async def segment_trees_pycrown(
    file_path: str,
    sigma_m: Annotated[float, Field(ge=0, description="Gaussian smoothing sigma in metres.")] = 0.35,
    min_distance_m: Annotated[float, Field(gt=0, description="Minimum horizontal treetop spacing in metres.")] = 0.75,
    min_height_m: Annotated[float, Field(ge=0, description="Minimum treetop height in metres.")] = 4.0,
    min_canopy_height_m: Annotated[float, Field(ge=0, description="Minimum canopy height in metres.")] = 2.0,
    crown_radius_m: Annotated[float, Field(gt=0, description="Maximum crown growth radius in metres.")] = 6.0,
    chm_resolution_m: Annotated[float, Field(gt=0, description="CHM grid resolution in metres when input is LAS/LAZ.")] = 0.5,
    use_pointcloud_supplement: bool = True,
    export_las: bool = True,
    dsm_path: str | None = None,
    dtm_path: str | None = None,
    point_cloud_path: str | None = None,
    output_path: str = "./",
) -> dict:
    """Segment individual tree crowns using PyCrown (Dalponte circular region growing + steep-slope correction).

    Accepts a point cloud or a CHM GeoTIFF with optional aligned DTM, DSM and point cloud.
    Three-dimensional recovery uses a DTM, classified ground on direct LAS input, or an explicit LAS height-above-ground field.
    Uses normalized Gaussian peak detection with shoulder-branch suppression, optional 3D
    stem-to-crown understory recovery, Dalponte circular crown delineation, steep-slope treetop correction, and
    first-return Otsu convex-hull crown smoothing with classified LAS export.
    With point-cloud input and LAS export enabled, tree_instances.las preserves source points and appends
    the height above ground and supported tree IDs. Unassigned points keep ID zero.
    """
    try:
        for val, label_name, allow_zero in (
            (sigma_m, "平滑半径 sigma_m", True),
            (min_distance_m, "最小树顶间距 min_distance_m", False),
            (min_height_m, "最小树顶高度 min_height_m", True),
            (min_canopy_height_m, "最小冠层高度 min_canopy_height_m", True),
            (crown_radius_m, "最大树冠半径 crown_radius_m", False),
            (chm_resolution_m, "CHM 分辨率 chm_resolution_m", False),
        ):
            if type(val) not in (int, float) or not math.isfinite(val) or (val < 0 if allow_zero else val <= 0):
                raise ValueError(f"{label_name} 必须为有效的正数。")
        if min_height_m < min_canopy_height_m:
            raise ValueError("最小树顶高度 min_height_m 不能小于最小冠层高度 min_canopy_height_m。")

        source = Path(file_path)
        if not source.is_file():
            raise ValueError(f"输入文件不存在: {file_path}")
        ext = source.suffix.lower()
        if ext not in (".las", ".laz", ".tif", ".tiff"):
            raise ValueError("请选择 LAS/LAZ 点云或 TIF/TIFF 冠层高度图文件。")

        las_override = Path(point_cloud_path) if point_cloud_path else None
        if las_override is not None and not las_override.is_file():
            raise ValueError(f"指定的点云文件不存在: {point_cloud_path}")

        settings = Settings(
            smoothing_width_m=float(sigma_m),
            peak_radius_m=float(min_distance_m),
            min_peak_m=float(min_height_m),
            min_canopy_m=float(min_canopy_height_m),
            min_final_tree_m=float(min_height_m),
            seed_ratio=0.0,
            mean_ratio=0.0,
            crown_radius_m=float(crown_radius_m),
            use_pointcloud_supplement=bool(use_pointcloud_supplement),
        )

        out_dir = Path(output_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_file = out_dir / f"{source.stem}_pycrown_crowns.zip"
        if zip_file.exists():
            raise FileExistsError(f"output already exists: {zip_file}")

        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            if ext in (".las", ".laz"):
                grid = build_grid_from_las(source, resolution_m=float(chm_resolution_m))
            else:
                grid = _load_grid_from_raster(source, dsm_path, dtm_path, las_override)

            result = segment(grid, settings)

            with TemporaryDirectory(prefix=".pycrown_mcp_", dir=out_dir) as tmp:
                staging = Path(tmp)
                classified_points = write_results(
                    staging, grid, result, export_las=bool(export_las),
                    settings=settings,
                )
                heights = (
                    result.surface[result.corrected.astype(int)[:, 0], result.corrected.astype(int)[:, 1]].tolist()
                    if len(result.corrected)
                    else []
                )
                labels = result.labels
                ids, counts = np.unique(labels[labels > 0], return_counts=True)
                pixel_area = float(grid.pixel_size ** 2)
                crown_areas = (counts * pixel_area).tolist()
                moved_tops = int(np.count_nonzero(np.any(result.corrected != result.initial, axis=1)))
                valid_3d_points = grid._cache.get("pointcloud_supplement_valid_points", 0)
                pointcloud_used = bool(settings.use_pointcloud_supplement and valid_3d_points)

                summary_data = {
                    "algorithm": "PyCrown (Gaussian + 3D stem supplement + Dalponte circular + steep-slope correction)",
                    "input_type": "point_cloud" if ext in (".las", ".laz") else "chm_raster",
                    "pointcloud_supplement_used": pointcloud_used,
                    "pointcloud_height_reference": grid._cache.get("pointcloud_height_reference"),
                    "pointcloud_supplement_valid_points": valid_3d_points if settings.use_pointcloud_supplement else 0,
                    "smooth_crowns_exported": bool(grid.las_path is not None and (staging / "crowns_smooth.gpkg").is_file()),
                    "classified_lidar_points": int(classified_points),
                    "point_instance_export": grid._cache.get("instance_export"),
                    "separated_crown_count": grid._cache.get("separated_crown_count", 0),
                    "tree_count": int(len(result.initial)),
                    "peak_count": int(result.peak_count),
                    "seed_count": int(result.seed_count),
                    "moved_tops": moved_tops,
                    "sigma_m": float(sigma_m),
                    "min_distance_m": float(min_distance_m),
                    "min_height_m": float(min_height_m),
                    "min_canopy_height_m": float(min_canopy_height_m),
                    "crown_radius_m": float(crown_radius_m),
                    "chm_resolution_m": float(grid.pixel_size),
                    "height_min_m": round(min(heights, default=0.0), 2),
                    "height_max_m": round(max(heights, default=0.0), 2),
                    "height_mean_m": round(float(np.mean(heights)) if heights else 0.0, 2),
                    "crown_area_mean_m2": round(float(np.mean(crown_areas)) if crown_areas else 0.0, 2),
                    "crown_area_max_m2": round(max(crown_areas, default=0.0), 2),
                    "crown_area_total_m2": round(float(np.count_nonzero(labels) * pixel_area), 2),
                    "settings": asdict(settings),
                }
                (staging / "summary.json").write_text(
                    json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

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
            "（已使用有地面基准的点云进行 3D 树干补检）"
            if pointcloud_used
            else "（树顶分割使用 2D 栅格）"
        )
        if settings.use_pointcloud_supplement and grid.las_path is not None and not pointcloud_used:
            mode_desc += "。点云缺少可用的离地高程，已跳过 3D 补检"
        message = (
            f"PyCrown 单木分割完成，共分割出 {summary_data['tree_count']} 棵单木{mode_desc}。"
            f"平均树高 {summary_data['height_mean_m']:.2f} m（最高 {summary_data['height_max_m']:.2f} m），"
            f"平均冠幅面积 {summary_data['crown_area_mean_m2']:.2f} m²，陡坡校正 {moved_tops} 棵。"
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
