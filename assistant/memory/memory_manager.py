from __future__ import annotations

import json
import uuid
from pathlib import Path

from assistant.config.settings import DEFAULT_DATA_DIR, SettingsManager
from assistant.models import ChatSession, Message, utc_now_iso


class MemoryManager:
    def __init__(
        self,
        settings_manager: SettingsManager,
        history_path: Path | str | None = None,
    ) -> None:
        self.settings_manager = settings_manager
        self.history_path = Path(history_path or DEFAULT_DATA_DIR / "history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_session_id: str | None = None
        self._sessions: dict[str, ChatSession] = {}
        self._load()

    def add_message(self, role: str, content: str) -> None:
        session = self.get_current_session()
        session.messages.append(Message(role=role, content=content))
        if role == "user" and self._should_autorename(session):
            session.title = self._generate_title(content)
        session.updated_at = utc_now_iso()
        self._shrink_if_needed(session)
        self._save()

    def get_context(self) -> list[Message]:
        settings = self.settings_manager.get_settings()
        tail = self.get_current_session().messages[-settings.memory_length :]
        token_budget = settings.memory_max_tokens
        collected: list[Message] = []
        current_tokens = 0

        for item in reversed(tail):
            estimated = self._estimate_tokens(item.content)
            if current_tokens + estimated > token_budget:
                break
            collected.append(item)
            current_tokens += estimated

        return list(reversed(collected))

    def summarize_context(self) -> str:
        history = self.get_current_session().messages
        if not history:
            return ""

        summary_source = history[:-8] if len(history) > 8 else history
        summary_lines: list[str] = []
        for message in summary_source:
            role = "User" if message.role == "user" else "Assistant"
            summary_lines.append(f"{role}: {message.content[:240]}")
        summary = "\n".join(summary_lines)
        return f"Conversation summary:\n{summary}"

    def get_all_messages(self, *, include_system: bool = False) -> list[Message]:
        messages = self.get_current_session().messages
        if include_system:
            return list(messages)
        return [message for message in messages if message.role != "system"]

    def clear(self) -> None:
        session = self.get_current_session()
        session.messages = []
        session.updated_at = utc_now_iso()
        self._save()

    def get_stats(self) -> dict[str, int]:
        visible_messages = self.get_all_messages()
        return {
            "stored_messages": len(visible_messages),
            "stored_tokens_estimate": sum(self._estimate_tokens(item.content) for item in visible_messages),
            "context_messages": len(self.get_context()),
            "session_count": len(self._sessions),
        }

    def list_sessions(self) -> list[ChatSession]:
        return sorted(
            self._sessions.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def create_session(self, title: str | None = None) -> ChatSession:
        session_number = len(self._sessions) + 1
        session = ChatSession(
            id=str(uuid.uuid4()),
            title=title or f"Новый чат {session_number}",
        )
        self._sessions[session.id] = session
        self._active_session_id = session.id
        self._save()
        return session

    def rename_session(self, session_id: str, new_title: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        cleaned = new_title.strip()
        if not cleaned:
            return False
        session.title = cleaned[:80]
        session.updated_at = utc_now_iso()
        self._save()
        return True

    def delete_session(self, session_id: str) -> str:
        if session_id not in self._sessions:
            return self.get_current_session().id

        del self._sessions[session_id]
        if not self._sessions:
            session = self.create_session()
            return session.id

        if self._active_session_id == session_id:
            self._active_session_id = self.list_sessions()[0].id
        self._save()
        return self.get_current_session().id

    def set_active_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        self._active_session_id = session_id
        self._save()
        return True

    def get_active_session_id(self) -> str:
        return self.get_current_session().id

    def get_current_session(self) -> ChatSession:
        if not self._sessions:
            return self.create_session()
        if self._active_session_id not in self._sessions:
            self._active_session_id = self.list_sessions()[0].id
        return self._sessions[self._active_session_id]

    def _shrink_if_needed(self, session: ChatSession) -> None:
        settings = self.settings_manager.get_settings()
        if len(session.messages) <= settings.memory_length * 2:
            return

        summary = self.summarize_context()
        recent_tail = session.messages[-settings.memory_length :]
        session.messages = [Message(role="system", content=summary), *recent_tail]

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _load(self) -> None:
        if not self.history_path.exists():
            self.create_session()
            return

        payload = json.loads(self.history_path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            session = ChatSession(
                id=str(uuid.uuid4()),
                title="Новый чат 1",
                messages=[Message(**item) for item in payload],
            )
            self._sessions = {session.id: session}
            self._active_session_id = session.id
            self._save()
            return

        sessions_payload = payload.get("sessions", [])
        for item in sessions_payload:
            messages = [Message(**message) for message in item.get("messages", [])]
            session = ChatSession(
                id=item["id"],
                title=item["title"],
                created_at=item.get("created_at", utc_now_iso()),
                updated_at=item.get("updated_at", utc_now_iso()),
                messages=messages,
            )
            self._sessions[session.id] = session

        if not self._sessions:
            self.create_session()
            return

        active_session_id = payload.get("active_session_id")
        self._active_session_id = active_session_id if active_session_id in self._sessions else self.list_sessions()[0].id

    def _save(self) -> None:
        payload = {
            "active_session_id": self._active_session_id,
            "sessions": [session.to_dict() for session in self.list_sessions()],
        }
        self.history_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _should_autorename(self, session: ChatSession) -> bool:
        default_prefix = "Новый чат"
        return session.title.startswith(default_prefix) and len([m for m in session.messages if m.role == "user"]) <= 1

    def _generate_title(self, content: str) -> str:
        title = " ".join(content.split())
        return (title[:50] or "Новый чат").strip()
