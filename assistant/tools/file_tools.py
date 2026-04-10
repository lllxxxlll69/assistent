from __future__ import annotations

from pathlib import Path

from assistant.models import ActionLogEntry, ToolResult


class FileTools:
    def create_file(self, path: str, content: str) -> ToolResult:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(
            content=f"File created: {target}",
            logs=[ActionLogEntry(message=f"Created file {target}")],
        )

    def edit_file(self, path: str, new_content: str) -> ToolResult:
        target = Path(path)
        if not target.exists():
            return ToolResult(
                content=f"File does not exist: {target}",
                logs=[ActionLogEntry(message=f"Failed to update missing file {target}", success=False)],
            )
        target.write_text(new_content, encoding="utf-8")
        return ToolResult(
            content=f"File updated: {target}",
            logs=[ActionLogEntry(message=f"Updated file {target}")],
        )

    def read_file(self, path: str) -> ToolResult:
        target = Path(path)
        if not target.exists():
            return ToolResult(
                content=f"File does not exist: {target}",
                logs=[ActionLogEntry(message=f"Failed to read missing file {target}", success=False)],
            )
        content = target.read_text(encoding="utf-8")
        return ToolResult(
            content=content,
            logs=[ActionLogEntry(message=f"Read file {target}")],
            structured_data={"path": str(target)},
        )

    def create_folder(self, path: str) -> ToolResult:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        return ToolResult(
            content=f"Folder created: {target}",
            logs=[ActionLogEntry(message=f"Created folder {target}")],
        )
