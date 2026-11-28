"""Deterministic pre-flight inspection and multi-tool router for tree segmentation.

Inspects `.las`/`.laz` headers in < 0.2 s (without loading point arrays or touching
the remote GPU) and dispatches to the best-fit segmentation tool:
- `segment_trees_segmentanytree`: primary 3D deep-learning engine (>= 15 pts/m²)
- `segment_trees_treelearn`: high-purity 3D alternative on clear-trunk temperate plots
- `segment_trees_pycrown`: sparse ALS (< 15 pts/m²) or CHM GeoTIFF inputs
- `segment_trees_watershed`: ultra-fast local CPU preview
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import laspy
from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("AutoTreeSegmentationRouter")

DENSITY_THRESHOLD_PTS_M2 = 15.0
LARGE_CLOUD_CONFIRM_POINTS = 100_000_000
VALID_PREFERENCES = ("auto", "segmentanytree", "treelearn", "high_purity", "pycrown", "fast", "watershed")


def inspect_input_file(file_path: Path) -> dict[str, Any]:
    """Read lightweight metadata from a LAS/LAZ header or GeoTIFF path in < 0.2 s."""
    if not file_path.is_file():
        raise ValueError(f"输入文件不存在: {file_path}")
    ext = file_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return {
            "file_path": str(file_path),
            "input_type": "chm_raster",
            "point_count": None,
            "extent_x_m": None,
            "extent_y_m": None,
            "extent_z_m": None,
            "bbox_area_m2": None,
            "bbox_density_pts_m2": None,
        }
    if ext not in (".las", ".laz"):
        raise ValueError(f"不支持的文件格式: {ext}（仅支持 .las/.laz 点云或 .tif/.tiff 冠层高度图）")

    with laspy.open(file_path) as reader:
        hdr = reader.header
        point_count = int(hdr.point_count)
        if point_count <= 0:
            raise ValueError(f"输入点云为空: {file_path}")
        extent_x = float(max(0.0, hdr.maxs[0] - hdr.mins[0]))
        extent_y = float(max(0.0, hdr.maxs[1] - hdr.mins[1]))
        extent_z = float(max(0.0, hdr.maxs[2] - hdr.mins[2]))
        area_m2 = float(max(extent_x, 0.1) * max(extent_y, 0.1))
        density = float(point_count / area_m2)

    return {
        "file_path": str(file_path),
        "input_type": "point_cloud",
        "point_count": point_count,
        "extent_x_m": round(extent_x, 2),
        "extent_y_m": round(extent_y, 2),
        "extent_z_m": round(extent_z, 2),
        "bbox_area_m2": round(area_m2, 2),
        "bbox_density_pts_m2": round(density, 2),
    }


def choose_segmentation_route(
    profile: dict[str, Any],
    *,
    preference: str = "auto",
    include_small_trees: bool = False,
    allow_large_cloud: bool = False,
) -> dict[str, Any]:
    """Select the target MCP segmentation tool and rationale from file metadata."""
    pref = (preference or "auto").strip().lower()
    if pref not in VALID_PREFERENCES:
        raise ValueError(f"不支持的 preference={preference!r}，可选: {', '.join(VALID_PREFERENCES)}")

    if profile["input_type"] == "chm_raster":
        if pref in ("fast", "watershed"):
            return {
                "action": "run",
                "selected_tool": "segment_trees_watershed",
                "reason": "输入为 CHM 栅格且选择快速模式，路由至 Watershed 分水岭分割。",
                "tool_kwargs": {},
            }
        return {
            "action": "run",
            "selected_tool": "segment_trees_pycrown",
            "reason": "输入为 CHM 栅格，自动路由至 PyCrown 单木冠层分割。",
            "tool_kwargs": {},
        }

    point_count = int(profile["point_count"] or 0)
    density = float(profile["bbox_density_pts_m2"] or 0.0)

    large_cloud = point_count > LARGE_CLOUD_CONFIRM_POINTS
    if large_cloud and not allow_large_cloud:
        return {
            "action": "needs_confirmation",
            "selected_tool": None,
            "reason": (
                f"检测到超大规模点云（{point_count:,} 点，密度 {density:.1f} 点/m²），"
                "直接分割可能触及内存上限。"
                "请确认使用 0.05 m 体素去重进行三维分割，或先分块再提交。"
            ),
            "tool_kwargs": {"voxel_dedup_m": 0.05},
        }

    dedup_kwargs = {"voxel_dedup_m": 0.05} if large_cloud else {}
    if large_cloud and (pref in ("fast", "watershed", "pycrown") or (pref == "auto" and density < DENSITY_THRESHOLD_PTS_M2)):
        raise ValueError("此算法不支持超大点云体素去重。请先分块，或选择支持去重的三维算法。")

    if pref in ("fast", "watershed"):
        return {
            "action": "run",
            "selected_tool": "segment_trees_watershed",
            "reason": f"按快速预览偏好路由至 Watershed（点云密度 {density:.1f} 点/m²，{point_count:,} 点）。",
            "tool_kwargs": {},
        }

    if pref == "pycrown":
        return {
            "action": "run",
            "selected_tool": "segment_trees_pycrown",
            "reason": f"按指定偏好路由至 PyCrown（点云密度 {density:.1f} 点/m²，{point_count:,} 点）。",
            "tool_kwargs": {},
        }

    if pref in ("treelearn", "high_purity"):
        weight_variant = "small" if include_small_trees else "default"
        return {
            "action": "run",
            "selected_tool": "segment_trees_treelearn",
            "reason": (
                f"按高纯度/TreeLearn 偏好路由至 TreeLearn（{weight_variant} 权重，"
                f"点云密度 {density:.1f} 点/m²，{point_count:,} 点）。"
            ),
            "tool_kwargs": {"weight_variant": weight_variant, **dedup_kwargs},
        }

    if pref == "segmentanytree":
        return {
            "action": "run",
            "selected_tool": "segment_trees_segmentanytree",
            "reason": f"按指定偏好路由至 SegmentAnyTree（点云密度 {density:.1f} 点/m²，{point_count:,} 点）。",
            "tool_kwargs": dedup_kwargs,
        }

    # Default automatic routing (preference == "auto")
    if density >= DENSITY_THRESHOLD_PTS_M2:
        return {
            "action": "run",
            "selected_tool": "segment_trees_segmentanytree",
            "reason": (
                f"检测到点云密度 {density:.1f} 点/m²（>= {DENSITY_THRESHOLD_PTS_M2:.0f} 点/m²，"
                f"{point_count:,} 点），自动选用 SegmentAnyTree 执行 3D 深度学习单木分割。"
            ),
            "tool_kwargs": dedup_kwargs,
        }

    return {
        "action": "run",
        "selected_tool": "segment_trees_pycrown",
        "reason": (
            f"检测到点云密度偏低（{density:.1f} 点/m² < {DENSITY_THRESHOLD_PTS_M2:.0f} 点/m²，"
            f"{point_count:,} 点），自动路由至更适合稀疏 ALS 的 PyCrown 单木分割。"
        ),
        "tool_kwargs": {},
    }


async def _dispatch_tool(
    selected_tool: str,
    file_path: str,
    output_path: str,
    tool_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Lazily import and execute the selected MCP segmentation tool."""
    if selected_tool == "segment_trees_segmentanytree":
        try:
            from tools.Segmentation.segmentanytree_segmentation import segment_trees_segmentanytree
        except ImportError:
            from segmentanytree_segmentation import segment_trees_segmentanytree
        return await segment_trees_segmentanytree(
            file_path=file_path, output_path=output_path, **tool_kwargs
        )

    if selected_tool == "segment_trees_treelearn":
        try:
            from tools.Segmentation.treelearn_segmentation import segment_trees_treelearn
        except ImportError:
            from treelearn_segmentation import segment_trees_treelearn
        return await segment_trees_treelearn(
            file_path=file_path, output_path=output_path, **tool_kwargs
        )

    if selected_tool == "segment_trees_pycrown":
        try:
            from tools.Segmentation.pycrown_segmentation import segment_trees_pycrown
        except ImportError:
            from pycrown_segmentation import segment_trees_pycrown
        return await segment_trees_pycrown(
            file_path=file_path, output_path=output_path, **tool_kwargs
        )

    if selected_tool == "segment_trees_watershed":
        try:
            from tools.Segmentation.watershed_segmentation import segment_trees_watershed
        except ImportError:
            from watershed_segmentation import segment_trees_watershed
        return await segment_trees_watershed(
            file_path=file_path, output_path=output_path, **tool_kwargs
        )

    raise ValueError(f"未知目标分割工具: {selected_tool}")


@mcp.tool(name="auto_segment_trees")
async def auto_segment_trees(
    file_path: Annotated[
        str,
        Field(description="输入森林点云（.las/.laz）或冠层高度图（.tif/.tiff）路径。"),
    ],
    output_path: Annotated[
        str,
        Field(description="输出目录路径。"),
    ] = "./results",
    preference: Annotated[
        str,
        Field(
            description=(
                "路由偏好：'auto'（默认，按点云密度自动选择 SegmentAnyTree 或 PyCrown）、"
                "'high_purity'/'treelearn'（树干清晰的温带林优先走 TreeLearn）、"
                "'fast'/'watershed'（本地 CPU 极速分水岭预览）、"
                "'segmentanytree' 或 'pycrown'。"
            )
        ),
    ] = "auto",
    include_small_trees: Annotated[
        bool,
        Field(description="当路由至 TreeLearn 时，是否启用含小树权重（small）以增强 <10 m 低矮树检出。"),
    ] = False,
    allow_large_cloud: Annotated[
        bool,
        Field(description="若点云超过 1 亿点，设为 true 表示用户已确认自动启用 0.05 m 体素去重并继续执行。"),
    ] = False,
) -> dict[str, Any]:
    """单木分割智能路由入口：0.2 秒预检文件头元数据并自动调度最合适的分割工具。

    当用户未明确指定具体算法时优先调用本工具：
    - 密度 >= 15 点/m² 的 3D 点云 -> 自动路由至 `segment_trees_segmentanytree`
    - 密度 < 15 点/m² 的稀疏 ALS 或 CHM 栅格 -> 自动路由至 `segment_trees_pycrown`
    - 需要树干高纯度边界（`preference='high_purity'`）-> 路由至 `segment_trees_treelearn`
    - 需要本地极速预览（`preference='fast'`）-> 路由至 `segment_trees_watershed`
    - 超过 1 亿点的超大规模点云 -> 先返回风险提示请用户确认，避免远端内存溢出
    """
    try:
        source = Path(file_path).resolve()
        profile = inspect_input_file(source)
        route = choose_segmentation_route(
            profile,
            preference=preference,
            include_small_trees=bool(include_small_trees),
            allow_large_cloud=bool(allow_large_cloud),
        )

        if route["action"] == "needs_confirmation":
            return {
                "status": "needs_confirmation",
                "output_path": None,
                "message": route["reason"],
                "data": {"profile": profile, "routing": route},
            }

        res = await _dispatch_tool(
            route["selected_tool"],
            str(source),
            output_path,
            route.get("tool_kwargs", {}),
        )
        if isinstance(res, dict):
            res_data = res.get("data")
            if isinstance(res_data, dict):
                res_data["routing"] = {
                    "selected_tool": route["selected_tool"],
                    "reason": route["reason"],
                    "profile": profile,
                }
            if res.get("status") == "success" and res.get("message"):
                res["message"] = f"[自动路由：{route['reason']}]\n{res['message']}"
        return res
    except Exception as exc:
        return {
            "status": "error",
            "output_path": None,
            "message": f"单木分割自动路由失败: {exc}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
