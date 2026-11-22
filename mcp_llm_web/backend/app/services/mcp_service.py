import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from sys import float_info
from typing import Optional, Dict, Any, List
from uuid import uuid4
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.core.config import settings

NORMALS_TOOL = "estimate_point_cloud_normals"
WATERSHED_TOOL = "segment_trees_watershed"
PYCROWN_TOOL = "segment_trees_pycrown"

# TOOL_TRIGGERS = {
#     "read_point_cloud": ["读取点云", "加载点云", "打开点云", "read point cloud", "load point cloud", "open point cloud"],
#     "normalize_point_cloud": ["归一化", "归一", "normalization"],
#     "extract_ground_points": ["地面点", "ground", "地面点滤波"],
#     "statistical_filter": ["统计滤波", "statistical filter"],
#     "radius_filter": ["半径滤波", "半径过滤"],
#     "clip_by_bounds": ["裁剪", "clip"],
#     "convert_point_cloud_format": ["转换", "convert", "格式转换"],
#     "voxel_downsample": ["下采样", "降采样", "downsample"],
#     "generate_dsm": ["DSM", "数字表面模型"],
#     "generate_dtm": ["DTM", "数字地形模型"],
#     "generate_chm": ["CHM", "冠层高度模型"],
#     "extract_forest_dbh": ["胸径", "DBH", "树干直径", "提取胸径", "继续提取胸径"],
#     "extract_tree_height": ["树高", "提取树高", "继续提取树高", "继续提取高度"],
#     "extract_tree_crown": ["冠幅","树冠","提取树冠","继续提取冠幅","继续提取树冠",],
#     "extract_gap_fraction": ["冠层间隙分数", "GapFraction", "gap fraction", "继续提取冠层间隙分数","冠层孔隙率",'冠层孔隙度'],
#     "extract_top_rugosity": ["表面粗糙度","TopRugosity","粗糙度","冠层粗糙度","继续提取粗糙度","冠层表面粗糙度","冠层表面粗糙度"],
#     "extract_rumple_index": ["凹凸指数", "RumpleIndex", "rumple", "继续提取凹凸指数","冠层褶皱度"],
#     "extract_gini_coeff": ["Gini系数", "GiniCoefficient", "gini", "继续提取Gini系数","冠层高度Gini系数","Gini指数"],
#     # "extract_individual_trees": ["单木分割", "树木分割", "individual tree", "树分割"],
#     # "compute_3d_fill_index": ["填充指数", "Filling", "填充", "继续提取填充指数",'空间填充','空间填充度','空间填充指数'],
#     "extract_lai": ["叶面积指数", "LAI", "叶面积", "继续提取叶面积"],
#     "extract_top_rugosity": [
#         "表面粗糙度",
#         "TopRugosity",
#         "粗糙度",
#         "冠层粗糙度",
#         "继续提取粗糙度",
#         "冠层表面粗糙度",
#     ],
#     "extract_rumple_index": ["凹凸指数", "RumpleIndex", "rumple", "继续提取凹凸指数","冠层表面褶皱度","冠层褶皱度"],
#     "extract_fhd": ["FHD", "叶面积高度分布", "叶层高度多样性", "继续提取FHD"],
#     "compute_3d_fill_index": ["填充指数", "Filling", "填充", "继续提取填充指数",'空间填充','空间填充度','空间填充指数'],
# }


class MCPService:
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.all_tools: Dict[str, Dict[str, Any]] = {}

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        if not is_python:
            raise ValueError("服务器脚本必须是 Python (.py) 文件")

        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env={
                **os.environ,
                "PYTHONPATH": str(settings.BASE_DIR),
            },
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        response = await session.list_tools()
        for tool in response.tools:
            self.all_tools[tool.name] = {
                "session": session,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            print(f"✅ 已加载工具: {tool.name}")

        self.sessions.append(session)

    async def connect_all_tools(self):
        base_dir = settings.BASE_DIR
        
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/1.TreeHight.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/2.TreeCrown.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/3.DHB.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/4.GapFraction.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/5.TopRugosity.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/6.RumpleIndex.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/7.GiniCoeff.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/8.LAI.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/9.FHD.py")
        await self.connect_to_server(f"{base_dir}/tools/ForestParams/10.Filling.py")

        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/normalization.py")
        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/ground.py")
        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/filter.py")
        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/clip.py")
        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/convert.py")
        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/downsample.py")
        await self.connect_to_server(f"{base_dir}/tools/Product/dsm.py")
        await self.connect_to_server(f"{base_dir}/tools/Product/dtm.py")
        await self.connect_to_server(f"{base_dir}/tools/Product/chm.py")
        await self.connect_to_server(f"{base_dir}/tools/Preprocessing/estimate_normals.py")
        await self.connect_to_server(f"{base_dir}/tools/Segmentation/watershed_segmentation.py")
        await self.connect_to_server(f"{base_dir}/tools/Segmentation/pycrown_segmentation.py")
        print(f"📦 共加载 {len(self.all_tools)} 个工具")

    def detect_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        text_lower = text.lower()
        for tool_name, triggers in TOOL_TRIGGERS.items():
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    return {"tool_name": tool_name, "trigger": trigger}
        return None

    def get_tool_catalog(self) -> List[Dict[str, Any]]:
        """Return MCP tool metadata for LLM-based tool selection."""
        tools = []
        for name, info in self.all_tools.items():
            tools.append(
                {
                    "name": name,
                    "description": info.get("description") or "",
                    "inputSchema": info.get("inputSchema") or {},
                }
            )
        return tools

    async def call_tool(
        self,
        tool_name: str,
        query: str,
        file_context: list = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> dict:
        print()
        if tool_name not in self.all_tools:
            return {
                "status": "error",
                "output_path": None,
                "message": f"未知工具: {tool_name}",
                "data": None,
            }

        tool_info = self.all_tools[tool_name]
        session = tool_info["session"]
        input_schema = tool_info.get("inputSchema", {})

        try:
            if tool_name == NORMALS_TOOL:
                args = self.parse_normal_args(file_context, arguments)
            elif tool_name == WATERSHED_TOOL:
                args = self.parse_watershed_args(file_context, arguments)
            elif tool_name == PYCROWN_TOOL:
                args = self.parse_pycrown_args(file_context, arguments)
            else:
                args = self.parse_tool_args(tool_name, query, input_schema, file_context)
        except ValueError as exc:
            return {"status": "error", "output_path": None, "message": str(exc), "data": None}

        try:
            result = await session.call_tool(tool_name, args)
            raw_result = result.content[0].text if result.content else "{}"

            if tool_name in (NORMALS_TOOL, WATERSHED_TOOL, PYCROWN_TOOL):
                if getattr(result, "isError", False):
                    raise ValueError(raw_result)
                parsed = json.loads(raw_result)
                if not isinstance(parsed, dict) or parsed.get("status") not in ("success", "error"):
                    raise ValueError("工具返回了无法解析的结果。")
                return parsed

            try:
                parsed = json.loads(raw_result)
                return parsed
            except json.JSONDecodeError:
                return {
                    "status": "success",
                    "output_path": None,
                    "message": raw_result,
                    "data": None,
                }
        except Exception as e:
            return {
                "status": "error",
                "output_path": None,
                "message": f"工具调用失败: {str(e)}",
                "data": None,
            }

    def extract_output_path(self, result: dict) -> Optional[str]:
        return result.get("output_path") if isinstance(result, dict) else None

    def extract_message(self, result: dict) -> str:
        if isinstance(result, dict):
            return result.get("message", "") or ""
        return str(result) if result else ""

    def parse_normal_args(self, file_context: Optional[list], arguments: Optional[dict]) -> dict:
        if len(file_context or []) != 1:
            raise ValueError("请为法向量估计选择一份 LAS 或 LAZ 文件。")
        file_path = file_context[0].get("file_path") or ""
        if Path(file_path).suffix.lower() not in (".las", ".laz") or not os.path.isfile(file_path):
            raise ValueError("选中的 LAS 或 LAZ 文件不可用，请重新上传或选择。")

        arguments = arguments if isinstance(arguments, dict) else {}
        radius = arguments.get("radius")
        if radius is None:
            raise ValueError(
                "请确认搜索半径，使用点云 XYZ 的坐标单位。"
                "如果坐标单位是米，可以从 0.25 米试起。"
                "邻居上限可参考混合点云 32、平整地面 128、移动扫描树干 64、细枝与树冠试跑 16。"
                "请回复所选半径和邻居数，例如「确认坐标单位是米，用 0.25 米半径、"
                "最多 32 个邻居计算法向量」。所选参数用于整份点云。"
            )
        if type(radius) not in (int, float) or not 0 < radius <= float_info.max:
            raise ValueError("请提供大于 0 的搜索半径，并使用点云坐标的单位。")
        max_nn = arguments.get("max_nn", 32)
        if type(max_nn) is not int or max_nn < 12:
            raise ValueError("最大邻居数必须是不小于 12 的整数。")
        orientation = arguments.get("orientation", "none")
        if orientation not in ("none", "up", "stem_outward"):
            raise ValueError("请选择基础法向量、朝上（+Z）或树干朝外。")
        tool_arguments = {
            "file_path": file_path,
            "radius": radius,
            "max_nn": max_nn,
            "orientation": orientation,
            "output_path": str(settings.RESULTS_DIR / uuid4().hex),
        }
        if orientation == "stem_outward":
            label_schema = arguments.get("label_schema")
            if label_schema not in ("forinstance", "nibio_mls"):
                raise ValueError("树干朝外需要标签约定。请说明使用 FORinstance 还是 NIBIO_MLS 的标签。")
            tool_arguments["label_schema"] = label_schema
        return tool_arguments

    def parse_watershed_args(self, file_context: Optional[list], arguments: Optional[dict]) -> dict:
        valid_files = [
            f for f in (file_context or [])
            if Path(f.get("file_path") or "").suffix.lower() in (".las", ".laz", ".tif", ".tiff")
            and os.path.isfile(f.get("file_path") or "")
        ]
        if not valid_files:
            raise ValueError("请为分水岭单木分割选择一份 LAS/LAZ 点云或 TIF/TIFF 冠层高度图文件。")

        point_clouds = [f for f in valid_files if Path(f["file_path"]).suffix.lower() in (".las", ".laz")]
        rasters = [f for f in valid_files if Path(f["file_path"]).suffix.lower() in (".tif", ".tiff")]

        primary_file = (rasters[0] if rasters else point_clouds[0])["file_path"]
        point_cloud_path = point_clouds[0]["file_path"] if (rasters and point_clouds) else None

        arguments = arguments if isinstance(arguments, dict) else {}
        sigma_m = arguments.get("sigma_m", 0.35)
        min_distance_m = arguments.get("min_distance_m", 0.75)
        min_canopy_height_m = arguments.get("min_canopy_height_m", 2.0)
        chm_resolution_m = arguments.get("chm_resolution_m", 0.5)
        use_pointcloud_supplement = arguments.get("use_pointcloud_supplement", True)

        if type(sigma_m) not in (int, float) or not 0 <= sigma_m <= float_info.max:
            raise ValueError("平滑半径 sigma_m 必须是不小于 0 的数字（米）。")
        if type(min_distance_m) not in (int, float) or not 0 < min_distance_m <= float_info.max:
            raise ValueError("最小树顶间距 min_distance_m 必须是大于 0 的数字（米）。")
        if type(min_canopy_height_m) not in (int, float) or not 0 <= min_canopy_height_m <= float_info.max:
            raise ValueError("最小冠层高度 min_canopy_height_m 必须是不小于 0 的数字（米）。")
        if type(chm_resolution_m) not in (int, float) or not 0 < chm_resolution_m <= float_info.max:
            raise ValueError("CHM 分辨率 chm_resolution_m 必须是大于 0 的数字（米）。")
        if not isinstance(use_pointcloud_supplement, bool):
            raise ValueError("use_pointcloud_supplement 必须为布尔值。")

        tool_arguments = {
            "file_path": primary_file,
            "sigma_m": float(sigma_m),
            "min_distance_m": float(min_distance_m),
            "min_canopy_height_m": float(min_canopy_height_m),
            "chm_resolution_m": float(chm_resolution_m),
            "use_pointcloud_supplement": use_pointcloud_supplement,
            "output_path": str(settings.RESULTS_DIR / uuid4().hex),
        }
        if point_cloud_path:
            tool_arguments["point_cloud_path"] = point_cloud_path
        return tool_arguments

    def parse_pycrown_args(self, file_context: Optional[list], arguments: Optional[dict]) -> dict:
        valid_files = [
            f for f in (file_context or [])
            if Path(f.get("file_path") or "").suffix.lower() in (".las", ".laz", ".tif", ".tiff")
            and os.path.isfile(f.get("file_path") or "")
        ]
        if not valid_files:
            raise ValueError("请为 PyCrown 单木分割选择一份 LAS/LAZ 点云或 CHM/DSM/DTM GeoTIFF 文件。")

        point_clouds = [f for f in valid_files if Path(f["file_path"]).suffix.lower() in (".las", ".laz")]
        rasters = [f for f in valid_files if Path(f["file_path"]).suffix.lower() in (".tif", ".tiff")]

        chm_file = None
        dsm_file = None
        dtm_file = None
        for r in rasters:
            name_lower = Path(r["file_path"]).name.lower()
            if "dsm" in name_lower and dsm_file is None:
                dsm_file = r["file_path"]
            elif "dtm" in name_lower and dtm_file is None:
                dtm_file = r["file_path"]
            elif chm_file is None:
                chm_file = r["file_path"]
        if chm_file is None and rasters:
            chm_file = rasters[0]["file_path"]

        primary_file = chm_file or point_clouds[0]["file_path"]
        point_cloud_path = point_clouds[0]["file_path"] if (chm_file and point_clouds) else None

        arguments = arguments if isinstance(arguments, dict) else {}
        sigma_m = arguments.get("sigma_m", 0.35)
        min_distance_m = arguments.get("min_distance_m", 0.75)
        min_height_m = arguments.get("min_height_m", 4.0)
        min_canopy_height_m = arguments.get("min_canopy_height_m", 2.0)
        crown_radius_m = arguments.get("crown_radius_m", 6.0)
        chm_resolution_m = arguments.get("chm_resolution_m", 0.5)
        use_pointcloud_supplement = arguments.get("use_pointcloud_supplement", True)
        export_las = arguments.get("export_las", True)

        if type(sigma_m) not in (int, float) or not 0 <= sigma_m <= float_info.max:
            raise ValueError("平滑半径 sigma_m 必须是不小于 0 的数字（米）。")
        if type(min_distance_m) not in (int, float) or not 0 < min_distance_m <= float_info.max:
            raise ValueError("最小树顶间距 min_distance_m 必须是大于 0 的数字（米）。")
        if type(min_height_m) not in (int, float) or not 0 <= min_height_m <= float_info.max:
            raise ValueError("最小树顶高度 min_height_m 必须是不小于 0 的数字（米）。")
        if type(min_canopy_height_m) not in (int, float) or not 0 <= min_canopy_height_m <= float_info.max:
            raise ValueError("最小冠层高度 min_canopy_height_m 必须是不小于 0 的数字（米）。")
        if min_height_m < min_canopy_height_m:
            raise ValueError("最小树顶高度 min_height_m 不能小于最小冠层高度 min_canopy_height_m。")
        if type(crown_radius_m) not in (int, float) or not 0 < crown_radius_m <= float_info.max:
            raise ValueError("最大树冠半径 crown_radius_m 必须是大于 0 的数字（米）。")
        if type(chm_resolution_m) not in (int, float) or not 0 < chm_resolution_m <= float_info.max:
            raise ValueError("CHM 分辨率 chm_resolution_m 必须是大于 0 的数字（米）。")
        if not isinstance(use_pointcloud_supplement, bool):
            raise ValueError("use_pointcloud_supplement 必须为布尔值。")
        if not isinstance(export_las, bool):
            raise ValueError("export_las 必须为布尔值。")

        tool_arguments = {
            "file_path": primary_file,
            "sigma_m": float(sigma_m),
            "min_distance_m": float(min_distance_m),
            "min_height_m": float(min_height_m),
            "min_canopy_height_m": float(min_canopy_height_m),
            "crown_radius_m": float(crown_radius_m),
            "chm_resolution_m": float(chm_resolution_m),
            "use_pointcloud_supplement": use_pointcloud_supplement,
            "export_las": export_las,
            "output_path": str(settings.RESULTS_DIR / uuid4().hex),
        }
        if dsm_file:
            tool_arguments["dsm_path"] = dsm_file
        if dtm_file:
            tool_arguments["dtm_path"] = dtm_file
        if point_cloud_path:
            tool_arguments["point_cloud_path"] = point_cloud_path
        return tool_arguments

    def parse_tool_args(
        self, tool_name: str, query: str, input_schema: Dict, file_context: list = None
    ) -> Dict:
        args = {}

        if file_context:
            for f in file_context:
                if f.get("file_path") and os.path.exists(f["file_path"]):
                    args["file_path"] = f["file_path"]
                    break

        if not args.get("file_path"):
            file_patterns = [
                r"([^\s]+\.(ply|las|laz|tif))",
                r"文件[^\s]*([^\s]+)",
                r"路径[^\s]*([^\s]+)",
            ]

            for pattern in file_patterns:
                match = re.search(pattern, query)
                if match:
                    file_path = match.group(1) if "(" in pattern else match.group(1)
                    if file_path and os.path.exists(file_path):
                        args["file_path"] = file_path
                        break

        if tool_name == "extract_forest_dbh" and "file_path" in args:
            args.setdefault("output_path", "./")

        elif tool_name == "extract_tree_height" and "file_path" in args:
            args.setdefault("output_path", "./")

        elif tool_name == "extract_tree_crown" and "file_path" in args:
            args.setdefault("output_path", "./")

        # elif tool_name == "compute_diversity_index" and "file_path" in args:
        #     args.setdefault("output_path", "./")

        elif (
            tool_name
            in ["extract_filling_index", "extract_lai", "extract_top_rugosity", "extract_rumple_index", "extract_fhd"]
            and "file_path" in args
        ):
            args.setdefault("output_path", "./")

        elif (
            tool_name in ["generate_dsm", "generate_dtm", "generate_chm"]
            and "file_path" in args
        ):
            args.setdefault("output_path", "./")

        elif tool_name == "extract_individual_trees" and "file_path" in args:
            args.setdefault("output_path", "./")

        elif (
            tool_name
            in [
                "normalize_point_cloud",
                "extract_ground_points",
                "statistical_filter",
                "radius_filter",
                "voxel_downsample",
            ]
            and "file_path" in args
        ):
            args.setdefault("output_path", "./")

        return args

    async def cleanup(self):
        await self.exit_stack.aclose()


mcp_service = MCPService()
