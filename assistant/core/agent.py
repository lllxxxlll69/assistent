from __future__ import annotations

import re

from assistant.models import ActionType, AgentAction


class Agent:
    def decide(self, user_input: str) -> AgentAction:
        text = user_input.strip()
        lowered = text.lower()

        if self._is_auto_mode(lowered):
            return AgentAction(
                action_type=ActionType.AUTO,
                reason="User requested autonomous execution.",
                response_text=text,
            )

        if image_path := self._extract_image_path(text):
            return AgentAction(
                action_type=ActionType.ANALYZE_IMAGE,
                reason="Detected image analysis intent.",
                image_path=image_path,
                response_text=self._strip_image_path(text, image_path) or "Что изображено на картинке?",
            )

        if match := re.match(r"^(?:create|создай)\s+folder\s+(.+)$", text, flags=re.IGNORECASE):
            return AgentAction(
                action_type=ActionType.CREATE_FOLDER,
                reason="Detected folder creation intent.",
                target_path=match.group(1).strip(),
            )

        if match := re.match(r"^(?:create|создай)\s+file\s+(.+)$", text, flags=re.IGNORECASE):
            target_path, inline_content = self._split_path_and_content(match.group(1).strip())
            return AgentAction(
                action_type=ActionType.CREATE_FILE,
                reason="Detected file creation intent.",
                target_path=target_path,
                content=inline_content,
            )

        if match := re.match(r"^(?:read|прочитай)\s+file\s+(.+)$", text, flags=re.IGNORECASE):
            return AgentAction(
                action_type=ActionType.READ_FILE,
                reason="Detected file read intent.",
                target_path=match.group(1).strip(),
            )

        if match := re.match(r"^(?:edit|измени)\s+file\s+(.+)$", text, flags=re.IGNORECASE):
            target_path, inline_content = self._split_path_and_content(match.group(1).strip())
            return AgentAction(
                action_type=ActionType.EDIT_FILE,
                reason="Detected file edit intent.",
                target_path=target_path,
                content=inline_content,
            )

        if self._needs_search(lowered):
            return AgentAction(
                action_type=ActionType.SEARCH,
                reason="Detected request that benefits from local retrieval.",
                search_query=text,
                response_text=text,
            )

        return AgentAction(
            action_type=ActionType.RESPOND,
            reason="Default conversational response.",
            response_text=text,
        )

    def _is_auto_mode(self, text: str) -> bool:
        markers = (
            "auto mode",
            "do everything",
            "сделай все",
            "сделай всё",
            "создай проект",
            "create rest api",
            "create a project",
        )
        return any(marker in text for marker in markers)

    def _needs_search(self, text: str) -> bool:
        markers = ("search", "найди", "find in project", "поиск", "where is", "где находится")
        return any(marker in text for marker in markers)

    def _extract_image_path(self, text: str) -> str | None:
        match = re.search(
            r"([A-Za-z]:\\[^\s]+\.(?:png|jpg|jpeg|webp)|\S+\.(?:png|jpg|jpeg|webp))",
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _strip_image_path(self, text: str, image_path: str) -> str:
        cleaned = text.replace(image_path, "", 1)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _split_path_and_content(self, value: str) -> tuple[str, str | None]:
        marker = " content:"
        lowered = value.lower()
        if marker not in lowered:
            return value.strip(), None
        split_at = lowered.index(marker)
        return value[:split_at].strip(), value[split_at + len(marker) :].strip()
