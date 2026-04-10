from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionType(str, Enum):
    RESPOND = "respond"
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    READ_FILE = "read_file"
    CREATE_FOLDER = "create_folder"
    ANALYZE_IMAGE = "analyze_image"
    SEARCH = "search"
    AUTO = "auto"


@dataclass(slots=True)
class Message:
    role: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ChatSession:
    id: str
    title: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass(slots=True)
class ActionLogEntry:
    message: str
    success: bool = True
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_console_line(self) -> str:
        status = "[OK]" if self.success else "[ERROR]"
        return f"{status} {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    content: str
    logs: list[ActionLogEntry] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentAction:
    action_type: ActionType
    reason: str
    response_text: str = ""
    target_path: str | None = None
    content: str | None = None
    image_path: str | None = None
    search_query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisionRequest:
    image_path: str | None = None
    image_base64: str | None = None
    prompt: str = "Describe this image."


@dataclass(slots=True)
class VisionResult:
    summary: str
    details: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass(slots=True)
class RetrievalChunk:
    path: str
    snippet: str
    score: float

    def to_prompt_block(self) -> str:
        return f"File: {self.path}\nScore: {self.score:.3f}\n{self.snippet}"


@dataclass(slots=True)
class AssistantResponse:
    text: str
    logs: list[ActionLogEntry] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
