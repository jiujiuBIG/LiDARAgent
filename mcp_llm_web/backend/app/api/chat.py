import uuid
import json
from datetime import datetime
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.database import db_service
from app.services.agent_service import agent_service

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


class Message(BaseModel):
    role: str
    content: str
    timestamp: str = None


def extract_title(message: str) -> str:
    return message[:30] + "..." if len(message) > 30 else message


@chat_router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
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
    await db_service.add_message(
        session_id,
        "assistant",
        result["content"],
        tool_results=tool_results_json,
    )

    await db_service.update_session_title(session_id, extract_title(request.message))

    return ChatResponse(
        session_id=session_id,
        message=result["content"],
        thoughts=result.get("thoughts", []),
        visualizations=result.get("visualizations", []),
        used_files=request.file_ids,
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
async def get_session(session_id: str):
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
            }
            for m in messages
        ],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@chat_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await db_service.delete_session(session_id)
    return {"success": True}
