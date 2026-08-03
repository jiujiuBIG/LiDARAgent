from openai import AsyncOpenAI
from app.core.config import settings
from typing import AsyncIterator, Dict, Any, Optional, List
import json
import re


class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key="", base_url=settings.LLM_BASE_URL)
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

    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=False,
        )

        message = response.choices[0].message
        return {
            "content": message.content if message.content else "",
            "role": "assistant",
        }

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

        router_messages = [
            {
                "role": "system",
                "content": (
                    "You are a router for MCP tools in a LiDAR forestry analysis system. "
                    "Choose exactly one tool only when the user clearly wants to run an available MCP tool. "
                    "If the user is asking a general question, or there is not enough information to execute a tool, choose null. "
                    'Return JSON only, with this shape: {"tool_name": string|null, "reason": string}.'
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
            response = await self.chat_complete(router_messages)
            content = response.get("content", "").strip()
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

            return {
                "tool_name": tool_name,
                "reason": data.get("reason", ""),
            }
        except Exception as e:
            return {
                "tool_name": None,
                "reason": f"Tool selection failed: {str(e)}",
            }


llm_service = LLMService()
