import json
import re
from typing import Dict, Any, Optional, List, AsyncIterator
from app.services.llm_service import llm_service
from pydantic import BaseModel
from app.services.mcp_service import (
    NORMALS_TOOL,
    PYCROWN_TOOL,
    RASTER_PRODUCT_TOOLS,
    SEGMENTATION_TOOLS,
    WATERSHED_TOOL,
    mcp_service,
)


class AgentThought(BaseModel):
    step: str
    thought: str
    tool_calls: List[str] = []
    progress: int


class VisualizationData(BaseModel):
    type: str
    title: str
    data: Dict[str, Any]


SYSTEM_PROMPT = """你是一个专业的林业生态分析助手，专注于森林参数计算、点云数据处理和生态分析。

## 重要提示
- 每个森林参数工具（胸径、树高、冠幅、多样性）是**独立的**，不会相互影响
- 用户问"继续提取冠幅"时，应该直接调用 **extract_tree_crown** 工具，不是其他工具
- 用户问"继续提取胸径"时，应该直接调用 **extract_forest_dbh** 工具
- 不要混淆不同工具的结果，每个工具返回什么就分析什么

## 可用MCP工具
当用户需要使用工具时，系统会自动调用以下MCP工具，请根据返回结果进行分析：

### 数据预处理工具
- **normalize_point_cloud** - 点云归一化，将点云高程转换为相对于地面的高度
- **extract_ground_points** - 地面点提取，从点云中分离地面点和非地面点
- **statistical_filter** - 统计滤波，基于点邻域统计特征去除噪声点
- **radius_filter** - 半径滤波，基于点的近邻数量去除稀疏点
- **voxel_downsample** - 体素下采样，降低点云密度以提高处理效率
- **clip_by_bounds** - 边界裁剪，根据指定范围裁剪点云数据
- **convert_point_cloud_format** - 格式转换，支持多种点云格式转换

### 森林参数提取工具（每个工具独立运行）
- **extract_forest_dbh** - 需使用保留树干且带米制离地高度字段的单木点云，测量离地 1.2–1.4 米截面的胸径
- **extract_tree_height** - 提取单木 Z 值统计和 P95−P05 垂直跨度。仅在结果提供离地树高指标时汇报树高
- **extract_tree_crown** - 冠幅提取，计算树冠投影面积和尺寸（长轴和短轴），单位：米
- **compute_diversity_index** - 多样性指数计算，分析森林物种多样性
- **extract_filling_index** - 填充指数，计算林冠填充度（canopy filling index）
- **extract_lai** - 叶面积指数（LAI），计算单位地面面积上的叶面积
- **extract_gap_fraction** - 林冠空隙率，计算冠层空隙比例
- **extract_gini_coeff** - 基尼系数，衡量冠层内点的空间分布均匀程度

### 遥感产品生成与单木分割工具
- **generate_dsm** - 数字表面模型（DSM），包含所有地物顶面的高程模型
- **generate_dtm** - 数字地形模型（DTM），仅包含地形表面的高程模型
- **generate_chm** - 冠层高度模型（CHM），树冠顶部相对于地面的高度
- **segment_trees_watershed** - 分水岭单木树冠分割，支持 CHM 与三维点云树干-冠顶追踪补检，输出单木树冠栅格、矢量及树顶表
- **segment_trees_pycrown** - PyCrown 单木树冠分割（Dalponte 圆形区域生长 + 3D 树干回溯补点 + 陡坡地形树顶校正 + 首回波点云凸包平滑与分类导出）

## 回复要求
请用简洁专业的中文回复，包括：
1. 数据概况
2. 参数统计（注意单位：树高、冠幅、胸径都是米）
3. 专业建议

重要：回复中提及工具时，请直接使用上述MCP工具名称（如 extract_ground_points），不要提及 LASTools、lasground 等外部工具名称。
"""


def parse_tool_result_for_visualization(
    tool_name: str, result: dict
) -> List[Dict[str, Any]]:
    visualizations = []

    if not isinstance(result, dict):
        result = {"message": str(result), "data": None}

    output_path = result.get("output_path")
    tool_data = result.get("data") or {}
    message = result.get("message", "")

    if tool_name in ["extract_forest_dbh", "extract_tree_height", "extract_tree_crown"]:
        visualizations.append(
            {
                "type": "statistic",
                "title": "🌲 森林参数统计",
                "data": {
                    **tool_data,
                    "output_file": output_path,
                    "description": message[:200],
                },
            }
        )

    elif tool_name == "compute_diversity_index":
        visualizations.append(
            {
                "type": "table",
                "title": "📊 多样性指数",
                "data": {
                    **tool_data,
                    "output_file": output_path,
                    "description": message[:200],
                },
            }
        )

    elif tool_name in ["generate_dsm", "generate_dtm", "generate_chm"]:
        visualizations.append(
            {
                "type": "image",
                "title": "🏔️ 生成结果",
                "data": {
                    "description": message[:300],
                    "output_file": output_path,
                    **tool_data,
                },
            }
        )

    elif tool_name == "normalize_point_cloud":
        visualizations.append(
            {
                "type": "result",
                "title": "📁 归一化结果",
                "data": {
                    "description": message[:300],
                    "output_file": output_path,
                    **tool_data,
                },
            }
        )

    elif tool_name == "extract_ground_points":
        visualizations.append(
            {
                "type": "result",
                "title": "📁 地面点提取结果",
                "data": {
                    "description": message[:300],
                    "output_file": output_path,
                    **tool_data,
                },
            }
        )

    elif tool_name == WATERSHED_TOOL:
        visualizations.append(
            {
                "type": "statistic",
                "title": "🌲 分水岭单木分割结果",
                "data": {
                    **tool_data,
                    "output_file": output_path,
                    "description": message[:300],
                },
            }
        )

    elif tool_name == PYCROWN_TOOL:
        visualizations.append(
            {
                "type": "statistic",
                "title": "🌲 PyCrown 单木分割结果",
                "data": {
                    **tool_data,
                    "output_file": output_path,
                    "description": message[:300],
                },
            }
        )

    return visualizations


class AgentService:
    async def process_message(
        self,
        message: str,
        history: List[Dict[str, str]],
        file_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        thoughts = []
        visualizations = []
        used_files = []
        tool_results = []

        print(f"DEBUG agent_service: file_context = {file_context}")
        if file_context:
            used_files = [f.get("id", "") for f in file_context]
        tool_call_info = await llm_service.select_tool(
            message=message,
            history=history,
            file_context=file_context,
            tools=mcp_service.get_tool_catalog(),
        )

        if tool_call_info.get("tool_name"):
            tool_name = tool_call_info["tool_name"]
            trigger = tool_call_info.get("reason", "LLM selected this MCP tool")

            thoughts.append(
                {
                    "step": "detecting",
                    "thought": f"🔧 检测到工具调用意图: {tool_name} (触发词: {trigger})",
                    "tool_calls": [tool_name],
                    "progress": 20,
                }
            )

            if tool_name == NORMALS_TOOL or tool_name in SEGMENTATION_TOOLS | RASTER_PRODUCT_TOOLS:
                tool_result = await mcp_service.call_tool(
                    tool_name, message, file_context, arguments=tool_call_info.get("arguments")
                )
            else:
                tool_result = await mcp_service.call_tool(tool_name, message, file_context)
            normal_error = tool_name == NORMALS_TOOL and tool_result.get("status") == "error"
            watershed_error = tool_name == WATERSHED_TOOL and tool_result.get("status") == "error"
            pycrown_error = tool_name == PYCROWN_TOOL and tool_result.get("status") == "error"
            tool_pending = tool_result.get("status") in {"error", "needs_confirmation"}
            actual_arguments = {}
            if tool_name == NORMALS_TOOL and not normal_error:
                normal_data = tool_result["data"]
                actual_arguments = {
                    key: normal_data[key]
                    for key in ("radius", "max_nn", "orientation", "label_schema", "tile_size")
                    if key in normal_data
                }
            elif tool_name == WATERSHED_TOOL and not watershed_error:
                ws_data = tool_result.get("data") or {}
                actual_arguments = {
                    key: ws_data[key]
                    for key in ("sigma_m", "min_distance_m", "min_canopy_height_m", "chm_resolution_m", "pointcloud_supplement_used")
                    if key in ws_data
                }
            elif tool_name == PYCROWN_TOOL and not pycrown_error:
                pc_data = tool_result.get("data") or {}
                actual_arguments = {
                    key: pc_data[key]
                    for key in ("sigma_m", "min_distance_m", "min_height_m", "min_canopy_height_m", "crown_radius_m", "chm_resolution_m", "pointcloud_supplement_used")
                    if key in pc_data
                }
            tool_results.append(
                {
                    "tool": tool_name,
                    "arguments": actual_arguments,
                    "result": tool_result,
                }
            )

            thoughts.append(
                {
                    "step": "executing",
                    "thought": (
                        "法向量估计未完成" if normal_error
                        else ("分水岭分割未完成" if watershed_error
                              else ("PyCrown 分割未完成" if pycrown_error
                                    else (f"工具 {tool_name} 未完成" if tool_pending else f"✅ 工具 {tool_name} 执行完成")))
                    ),
                    "tool_calls": [tool_name],
                    "progress": 60,
                }
            )

            visualizations = parse_tool_result_for_visualization(tool_name, tool_result)

            message_for_llm = mcp_service.extract_message(tool_result)
            tool_data = (
                tool_result.get("data", {}) if isinstance(tool_result, dict) else {}
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {
                    "role": "system",
                    "content": f"用户意图: {message}\n\n工具调用结果:\n{message_for_llm}\n\n统计数据:\n{json.dumps(tool_data, ensure_ascii=False)}",
                },
            ]

            if tool_pending:
                response = {"content": message_for_llm}
            else:
                response = await llm_service.chat_complete(messages=messages)

            thoughts.append(
                {
                    "step": "completed",
                    "thought": message_for_llm if tool_pending else "分析完成",
                    "tool_calls": [tool_name],
                    "progress": 100,
                }
            )

        else:
            thoughts.append(
                {
                    "step": "understanding",
                    "thought": "正在理解用户需求...",
                    "progress": 30,
                }
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": message},
            ]

            response = await llm_service.chat_complete(messages=messages)

            thoughts.append(
                {
                    "step": "completed",
                    "thought": "已生成回答",
                    "progress": 100,
                }
            )

        output_path = (
            mcp_service.extract_output_path(tool_results[0]["result"])
            if tool_results
            else None
        )
        tool_data = (
            tool_results[0]["result"].get("data", {})
            if (tool_results and isinstance(tool_results[0]["result"], dict))
            else None
        )

        return {
            "content": response["content"],
            "thoughts": thoughts,
            "visualizations": visualizations,
            "used_files": used_files,
            "tool_results": tool_results,
            "output_path": output_path,
            "tool_data": tool_data,
        }


agent_service = AgentService()
