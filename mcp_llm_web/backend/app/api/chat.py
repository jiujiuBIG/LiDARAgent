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
from app.services.mcp_service import NORMALS_TOOL

chat_router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    file_ids: List[str] = []


class ChatResponse(BaseModel):
    session_id: str
    message: str
    thoughts: List[Dict]
    visualizations: List[Dict]
    used_files: List[str]
    output_path: Optional[str] = None
    tool_data: Optional[Dict] = None
    download_url: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str
    timestamp: str = None


def extract_title(message: str) -> str:
    return message[:30] + "..." if len(message) > 30 else message


def normal_result_file(message: Optional[MessageModel]) -> Optional[Path]:
    if message is None or message.role != "assistant" or not message.tool_results:
        return None
    try:
        tool_results = json.loads(message.tool_results)
    except json.JSONDecodeError:
        return None
    if not isinstance(tool_results, list):
        return None

    results_dir = settings.RESULTS_DIR.resolve()
    for entry in tool_results:
        if not isinstance(entry, dict) or entry.get("tool") != NORMALS_TOOL:
            continue
        result = entry.get("result")
        if not isinstance(result, dict) or result.get("status") != "success":
            continue
        output_path = result.get("output_path")
        if not isinstance(output_path, str) or not output_path:
            continue
        try:
            path = Path(output_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_relative_to(results_dir) and path.suffix.lower() == ".las" and path.is_file():
            return path
    return None


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

    session = await db_service.get_session(session_id)
    if not session:
        await db_service.create_session(session_id, extract_title(request.message))

    file_context = []
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

    normal_tool_used = any(entry.get("tool") == NORMALS_TOOL for entry in result.get("tool_results", []))
    return ChatResponse(
        session_id=session_id,
        message=result["content"],
        thoughts=result.get("thoughts", []),
        visualizations=result.get("visualizations", []),
        used_files=request.file_ids,
        output_path=result.get("output_path") if normal_tool_used else None,
        tool_data=result.get("tool_data") if normal_tool_used else None,
        download_url=normal_download_url(http_request, assistant_message),
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
        raise HTTPException(status_code=404, detail="法向量结果文件不存在或已不可用。")
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
