from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FileStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Message(BaseModel):
    id: Optional[str] = None
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


class ChatSession(BaseModel):
    id: Optional[str] = None
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    file_ids: List[str] = []


class UploadedFile(BaseModel):
    id: str
    filename: str
    original_name: str
    file_path: str
    file_type: str
    file_size: int
    status: FileStatus = FileStatus.UPLOADED
    metadata: Optional[Dict[str, Any]] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None


class ToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class Task(BaseModel):
    id: str
    session_id: str
    type: str
    status: TaskStatus = TaskStatus.PENDING
    tool_calls: List[ToolCall] = []
    input_data: Dict[str, Any] = {}
    output_data: Optional[Dict[str, Any]] = None
    progress: int = 0
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class VisualizationData(BaseModel):
    type: str
    data: Dict[str, Any]
    title: Optional[str] = None
    description: Optional[str] = None


class AgentThought(BaseModel):
    step: str
    thought: str
    tool_calls: List[str] = []
    progress: Optional[int] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    file_ids: List[str] = []


class ChatResponse(BaseModel):
    session_id: str
    message: str
    thoughts: List[AgentThought] = []
    visualizations: List[VisualizationData] = []
    used_files: List[str] = []


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    file_size: int
    metadata: Dict[str, Any]
