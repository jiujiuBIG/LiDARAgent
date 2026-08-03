import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.core.config import settings

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
        # await self.connect_to_server(f"{base_dir}/tools/Segmentation/Treeseg_base.py")
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
        self, tool_name: str, query: str, file_context: list = None
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

        args = self.parse_tool_args(tool_name, query, input_schema, file_context)

        try:
            result = await session.call_tool(tool_name, args)
            raw_result = result.content[0].text if result.content else "{}"

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
