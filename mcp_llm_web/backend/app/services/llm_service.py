from openai import AsyncOpenAI
from app.core.config import settings
from app.services.mcp_service import (
    AUTO_SEGMENTATION_TOOL, DEEP_SEGMENTATION_TOOLS, NORMALS_TOOL, PYCROWN_TOOL,
    RASTER_PRODUCT_TOOLS, SEGMENTATION_TOOLS, WATERSHED_TOOL,
)
from typing import AsyncIterator, Dict, Any, Optional, List
import json
import re


class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL

    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.7,
        }

        response = await self.client.chat.completions.create(**kwargs)

        if stream:
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield {"type": "content", "content": chunk.choices[0].delta.content}
        else:
            yield {
                "type": "complete",
                "content": response.choices[0].message.content,
            }

    async def _chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        message = response.choices[0].message
        return {
            "content": message.content if message.content else "",
            "role": "assistant",
        }

    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        return await self._chat_complete(messages, temperature=0.7)

    async def select_tool(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        file_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not tools:
            return {"tool_name": None, "reason": "No MCP tools are loaded."}

        tool_names = {tool["name"] for tool in tools}
        tool_catalog = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
            for tool in tools
        ]
        files = [
            {
                "filename": f.get("filename") or f.get("original_name"),
                "file_path": f.get("file_path"),
                "file_type": f.get("file_type"),
                "status": f.get("status"),
            }
            for f in (file_context or [])
        ]

        normals_instructions = ""
        if NORMALS_TOOL in tool_names:
            normals_instructions = (
                "For a clear request to estimate point-cloud normals, select "
                "estimate_point_cloud_normals even when the file or radius is missing; "
                "the backend will ask for missing inputs. For that tool only, include an "
                "arguments object with the numerical radius in the input XYZ coordinate units "
                "and max_nn as an integer ONLY when explicitly specified by the user. "
                "Include tile_size (XY tile width in the input coordinate units) only when the user explicitly gives a tile width. "
                "If the user has NOT explicitly stated a search radius, you MUST omit 'radius' "
                "from arguments (return {} if no parameters were specified) and NEVER invent a "
                "placeholder or default radius such as 1.0 or 0.25, even though radius is listed "
                "as required in inputSchema. Omit orientation by default, including "
                "when the user only mentions labelled data or a type of surface. Set orientation "
                "to 'up' only for an explicit upward orientation request, or 'stem_outward' only "
                "for an explicit request to orient stem normals outward. For 'stem_outward', "
                "include label_schema 'forinstance' or 'nibio_mls' only when the user has supplied "
                "that label convention for the current file. If the convention is unknown, omit it "
                "so the backend can ask. "
                "When the assistant has asked the user to confirm a search radius, wait for the "
                "user to state the desired radius before including it in arguments. "
                "Omit parameters the user has not established. Convert units only when the "
                "point cloud coordinate units are known. Never return input or output paths."
            )
        watershed_instructions = ""
        if WATERSHED_TOOL in tool_names:
            watershed_instructions = (
                " For a request to segment individual trees or tree crowns with watershed "
                "(分水岭单木分割 / 树冠分割 / watershed segmentation), select segment_trees_watershed "
                "and include an arguments object with any explicitly stated numerical parameters: "
                "sigma_m (Gaussian smoothing sigma in metres, default 0.35), min_distance_m (minimum treetop "
                "spacing in metres, default 0.75), min_canopy_height_m (minimum canopy height in metres, default 2.0), "
                "chm_resolution_m (CHM resolution in metres, default 0.5), use_pointcloud_supplement (boolean, default true), "
                "and recover_occluded_crowns (boolean, default true; set to false when the user explicitly disables "
                "occluded lower-crown recovery/splitting)."
            )
        pycrown_instructions = ""
        if PYCROWN_TOOL in tool_names:
            pycrown_instructions = (
                " For a request to segment individual trees or tree crowns with PyCrown / Dalponte "
                "(PyCrown 单木分割 / Dalponte 树冠分割 / 陡坡树顶校正 / 首回波凸包树冠), select segment_trees_pycrown "
                "and include an arguments object with any explicitly stated numerical parameters: "
                "sigma_m (Gaussian smoothing sigma in metres, default 0.35), min_distance_m (minimum treetop "
                "spacing in metres, default 0.75), min_height_m (minimum treetop height in metres, default 4.0), "
                "min_canopy_height_m (minimum canopy height in metres, default 2.0), crown_radius_m (maximum crown "
                "growth radius in metres, default 6.0), chm_resolution_m (CHM resolution in metres, default 0.5), "
                "use_pointcloud_supplement (boolean, default true), export_las (boolean, default true), "
                "and recover_occluded_crowns (boolean, default true; set to false when the user explicitly disables "
                "occluded lower-crown recovery/splitting)."
            )
        segmentation_instructions = ""
        if AUTO_SEGMENTATION_TOOL in tool_names:
            segmentation_instructions += (
                " For individual-tree segmentation without a named algorithm, select auto_segment_trees. "
                "Use preference 'fast' for a quick preview or 'high_purity' for that explicit preference. "
                "Set include_small_trees only on request. Set allow_large_cloud to true only when "
                "the most recent user message explicitly confirms 0.05 m voxel deduplication for the selected input."
            )
        if DEEP_SEGMENTATION_TOOLS & tool_names:
            segmentation_instructions += (
                " For explicit TreeLearn or SegmentAnyTree requests, select the corresponding tool and "
                "include arguments explicitly stated by the user and listed in its inputSchema."
            )
        router_messages = [
            {
                "role": "system",
                "content": (
                    "You are a router for MCP tools in a LiDAR forestry analysis system. "
                    "Choose exactly one tool only when the user clearly wants to run an available MCP tool on data. "
                    "If the user is asking a general/conceptual question (such as asking for definitions, concepts, principles, or differences between terms like CHM and DSM), or there is not enough information to execute a tool, ALWAYS choose null — never select a data-processing tool just because it shares a keyword. "
                    'Return JSON only with "tool_name" and "reason", plus "arguments" when instructed. '
                    + normals_instructions
                    + watershed_instructions
                    + pycrown_instructions
                    + segmentation_instructions
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_message": message,
                        "recent_history": (history or [])[-6:],
                        "available_files": files,
                        "available_tools": tool_catalog,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            if "chat_complete" in self.__dict__:
                response = await self.chat_complete(router_messages)
            else:
                response = await self._chat_complete(router_messages, temperature=0.0)
            content = response.get("content", "").strip()
            if "</think>" in content:
                content = content.rsplit("</think>", 1)[-1].strip()
            match = re.search(r"\{.*\}", content, flags=re.S)
            data = json.loads(match.group(0) if match else content)
            tool_name = data.get("tool_name")

            if tool_name in ("", "null"):
                tool_name = None
            if tool_name is not None and tool_name not in tool_names:
                return {
                    "tool_name": None,
                    "reason": f"LLM selected unknown tool: {tool_name}",
                }

            selection = {
                "tool_name": tool_name,
                "reason": data.get("reason", ""),
            }
            if tool_name == NORMALS_TOOL or tool_name in SEGMENTATION_TOOLS | RASTER_PRODUCT_TOOLS:
                arguments = data.get("arguments")
                if isinstance(arguments, dict):
                    arguments = {
                        k: v for k, v in arguments.items()
                        if k not in ("file_path", "output_path", "point_cloud_path", "dsm_path", "dtm_path")
                    }
                else:
                    arguments = {}
                selection["arguments"] = arguments
            return selection
        except Exception as e:
            return {
                "tool_name": None,
                "reason": f"Tool selection failed: {str(e)}",
            }


llm_service = LLMService()
