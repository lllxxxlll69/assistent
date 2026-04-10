from __future__ import annotations

from pathlib import Path

from assistant.config.settings import Settings
from assistant.models import ActionLogEntry, ToolResult


class FileTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.workspace_root = Path(settings.workspace_root).resolve(strict=False)

    def create_file(self, path: str, content: str) -> ToolResult:
        target, error = self._resolve_workspace_path(path)
        if error:
            return self._error_result(error)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(
            content=f"File created: {target}",
            logs=[ActionLogEntry(message=f"Created file {target}")],
        )

    def edit_file(self, path: str, new_content: str) -> ToolResult:
        target, error = self._resolve_workspace_path(path)
        if error:
            return self._error_result(error)
        if not target.exists():
            return self._error_result(f"File does not exist: {target}")
        if not target.is_file():
            return self._error_result(f"Path is not a file: {target}")

        target.write_text(new_content, encoding="utf-8")
        return ToolResult(
            content=f"File updated: {target}",
            logs=[ActionLogEntry(message=f"Updated file {target}")],
        )

    def read_file(self, path: str) -> ToolResult:
        target, error = self._resolve_workspace_path(path)
        if error:
            return self._error_result(error)
        if not target.exists():
            return self._error_result(f"File does not exist: {target}")
        if not target.is_file():
            return self._error_result(f"Path is not a file: {target}")

        content = target.read_text(encoding="utf-8")
        return ToolResult(
            content=content,
            logs=[ActionLogEntry(message=f"Read file {target}")],
            structured_data={"path": str(target)},
        )

    def create_folder(self, path: str) -> ToolResult:
        target, error = self._resolve_workspace_path(path)
        if error:
            return self._error_result(error)

        target.mkdir(parents=True, exist_ok=True)
        return ToolResult(
            content=f"Folder created: {target}",
            logs=[ActionLogEntry(message=f"Created folder {target}")],
        )

    def _resolve_workspace_path(self, path: str) -> tuple[Path | None, str | None]:
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else self.workspace_root / raw_path
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            return None, f"Access denied outside workspace root: {resolved}"
        return resolved, None

    def _error_result(self, message: str) -> ToolResult:
        return ToolResult(
            content=message,
            logs=[ActionLogEntry(message=message, success=False)],
        )
