from typing import List, Optional

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    id: int
    time: str
    step: str
    status: str
    detail: str = ""
    scope: str = "system"


class ChatMessage(BaseModel):
    role: str
    content: str
    steps: List[dict] = Field(default_factory=list)
    citations: List[dict] = Field(default_factory=list)
    eval_score: Optional[float] = None
    retry_count: int = 0
    self_healing_enabled: bool = False


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    session_id: str = Field(
        default="default",
        max_length=64,
        pattern=r"^[a-zA-Z0-9\-_]+$",
        description="Session identifier (alphanumeric, hyphens, underscores)",
    )
    enable_self_healing: bool = False


class QueryResponse(BaseModel):
    answer: str
    steps: List[dict] = Field(default_factory=list)
    citations: List[dict] = Field(default_factory=list)
    messages: List[ChatMessage] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
    eval_score: Optional[float] = None
    retry_count: int = 0
    self_healing_enabled: bool = False


class DocumentEntry(BaseModel):
    id: str
    filename: str
    status: str = "pending"
    vector_count: int = 0
    topic: Optional[str] = None
    created_at: Optional[str] = None
class UploadResponse(BaseModel):
    status: str
    uploaded_files: List[str] = Field(default_factory=list)
    documents_loaded: int = 0
    chunks_created: int = 0
    vector_count: int = 0
    message: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    documents: List[DocumentEntry] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)


class LogResponse(BaseModel):
    logs: List[LogEntry] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: List[DocumentEntry] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    status: str
    message: str
    documents: List[DocumentEntry] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
