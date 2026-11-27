import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.core.config import settings
from app.models.database import MessageModel, db_service
from app.services.agent_service import agent_service
from app.services.mcp_service import (
    FOREST_PARAMETER_TOOLS,
    NORMALS_TOOL,
    PYCROWN_TOOL,
    RASTER_PRODUCT_TOOLS,
    WATERSHED_TOOL,
)

chat_router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    file_ids: List[str] = []
    result_message_id: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    message: str
    thoughts: List[Dict]
    visualizations: List[Dict]
    used_files: List[str]
    output_path: Optional[str] = None
    tool_data: Optional[Dict] = None
    download_url: Optional[str] = None
    reusable_result: Optional[Dict] = None


class Message(BaseModel):
    role: str
    content: str
    timestamp: str = None


def extract_title(message: str) -> str:
    return message[:30] + "..." if len(message) > 30 else message


def normal_result_file(
    message: Optional[MessageModel], tool_filter: Optional[str] = None
) -> Optional[Path]:
    result = downloadable_tool_result(message, tool_filter)
    return result[0] if result is not None else None


def downloadable_tool_result(
    message: Optional[MessageModel], tool_filter: Optional[str] = None
) -> Optional[tuple[Path, Dict]]:
    if message is None or message.role != "assistant" or not message.tool_results:
        return None
    try:
        tool_results = json.loads(message.tool_results)
    except json.JSONDecodeError:
        return None
    if not isinstance(tool_results, list):
        return None

    results_dir = settings.RESULTS_DIR.resolve()
    allowed_tools = {
        NORMALS_TOOL: (".las",),
        WATERSHED_TOOL: (".zip", ".tif", ".geojson", ".shp"),
        PYCROWN_TOOL: (".zip", ".tif", ".gpkg", ".las"),
        **{name: (".csv",) for name in FOREST_PARAMETER_TOOLS},
        **{name: (".tif",) for name in RASTER_PRODUCT_TOOLS},
    }
    for entry in tool_results:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool")
        if tool_name not in allowed_tools or (tool_filter is not None and tool_name != tool_filter):
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        status = result.get("status")
        if status != "success" and not (tool_name in FOREST_PARAMETER_TOOLS and status == "warning"):
            continue
        output_path = result.get("output_path")
        if not isinstance(output_path, str) or not output_path:
            continue
        try:
            path = Path(output_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_relative_to(results_dir) and path.suffix.lower() in allowed_tools[tool_name] and path.is_file():
            return path, result
    return None


def sibling_result_file(output_path: Path, filename: str) -> Optional[Path]:
    candidate = output_path.with_name(filename)
    try:
        if candidate.resolve(strict=True) == candidate and candidate.is_file():
            return candidate
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def reusable_result_file(message: Optional[MessageModel]) -> Optional[tuple[Path, str]]:
    normal_file = normal_result_file(message, NORMALS_TOOL)
    if normal_file is not None:
        return normal_file, "point_cloud"

    legacy_crown = None
    for tool in (PYCROWN_TOOL, WATERSHED_TOOL):
        result = downloadable_tool_result(message, tool)
        if result is None:
            continue
        path, tool_result = result
        if tool == PYCROWN_TOOL and path.suffix.lower() == ".las":
            legacy_crown = path
        if path.suffix.lower() != ".zip":
            continue

        data = tool_result.get("data")
        records_instance_export = isinstance(data, dict) and "point_instance_export" in data
        if records_instance_export and data["point_instance_export"] is None:
            continue
        instance = sibling_result_file(path, "tree_instances.las")
        if instance is not None:
            return instance, "tree_instances"
        if tool == PYCROWN_TOOL and not records_instance_export:
            legacy_crown = sibling_result_file(path, "trees.las")

    return (legacy_crown, "crown_points") if legacy_crown is not None else None


def reusable_result_info(message: MessageModel) -> Optional[Dict]:
    result = reusable_result_file(message)
    if result is None:
        return None
    path, kind = result
    return {"message_id": message.id, "filename": path.name, "kind": kind}


def normal_download_url(request: Request, message: MessageModel) -> Optional[str]:
    if normal_result_file(message) is None:
        return None
    return request.url_for(
        "download_normal_result",
        session_id=quote(message.session_id, safe=""),
        message_id=message.id,
    ).path


@chat_router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest, http_request: Request):
    session_id = request.session_id or str(uuid.uuid4())
    file_context = []
    if request.result_message_id is not None:
        if not request.session_id or request.file_ids:
            raise HTTPException(status_code=400, detail="请选择当前会话的一份工具成果，或使用上传文件。")
        previous_message = await db_service.get_message(session_id, request.result_message_id)
        selected = reusable_result_file(previous_message)
        if selected is None:
            raise HTTPException(status_code=404, detail="所选工具成果不存在或无法用于续算。")
        path, kind = selected
        file_context.append({
            "id": f"result:{request.result_message_id}",
            "filename": path.name,
            "file_type": path.suffix[1:].lower(),
            "file_path": str(path),
            "forest_input_kind": kind,
        })

    session = await db_service.get_session(session_id)
    if not session:
        await db_service.create_session(session_id, extract_title(request.message))

    for file_id in request.file_ids:
        file_info = await db_service.get_file(file_id)
        if file_info:
            file_context.append(
                {
                    "id": file_info.id,
                    "filename": file_info.filename,
                    "file_type": file_info.file_type,
                    "file_path": file_info.file_path,
                }
            )
        else:
            print(f"WARNING: file_id {file_id} not found in database")

    print(f"DEBUG: request.file_ids = {request.file_ids}")
    print(f"DEBUG: file_context = {file_context}")

    history = await db_service.get_messages(session_id)
    history_dicts = [
        {"role": msg.role, "content": msg.content} for msg in history[-20:]
    ]

    await db_service.add_message(session_id, "user", request.message)

    result = await agent_service.process_message(
        message=request.message,
        history=history_dicts,
        file_context=file_context if file_context else None,
    )

    tool_results_json = (
        json.dumps(result.get("tool_results", []), ensure_ascii=False)
        if result.get("tool_results")
        else None
    )
    assistant_message = await db_service.add_message(
        session_id,
        "assistant",
        result["content"],
        tool_results=tool_results_json,
    )

    await db_service.update_session_title(session_id, extract_title(request.message))

    downloadable_tool_used = any(
        entry.get("tool") in (
            {NORMALS_TOOL, WATERSHED_TOOL, PYCROWN_TOOL} | FOREST_PARAMETER_TOOLS | RASTER_PRODUCT_TOOLS
        )
        for entry in result.get("tool_results", [])
    )
    return ChatResponse(
        session_id=session_id,
        message=result["content"],
        thoughts=result.get("thoughts", []),
        visualizations=result.get("visualizations", []),
        used_files=[entry["id"] for entry in file_context],
        output_path=result.get("output_path") if downloadable_tool_used else None,
        tool_data=result.get("tool_data") if downloadable_tool_used else None,
        download_url=normal_download_url(http_request, assistant_message),
        reusable_result=reusable_result_info(assistant_message),
    )


@chat_router.get("/sessions")
async def list_sessions():
    sessions = await db_service.list_sessions()
    result = []
    for s in sessions:
        messages = await db_service.get_messages(s.id)
        result.append(
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": len(messages),
            }
        )
    return {"sessions": result}


@chat_router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    session = await db_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = await db_service.get_messages(session_id)
    return {
        "id": session.id,
        "title": session.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "download_url": normal_download_url(request, m),
                "reusable_result": reusable_result_info(m),
            }
            for m in messages
        ],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@chat_router.get("/sessions/{session_id}/messages/{message_id}/download")
async def download_normal_result(session_id: str, message_id: int):
    message = await db_service.get_message(session_id, message_id)
    path = normal_result_file(message)
    if path is None:
        raise HTTPException(status_code=404, detail="结果文件不存在或已不可用。")
    filename = path.name
    # Uploads keep a UUID4 storage prefix before the original filename.
    prefix, _, original_name = filename.partition("_")
    try:
        upload_id = uuid.UUID(prefix)
    except ValueError:
        pass
    else:
        if original_name and upload_id.version == 4 and str(upload_id) == prefix:
            filename = original_name
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@chat_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await db_service.delete_session(session_id)
    return {"success": True}
