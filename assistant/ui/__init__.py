from __future__ import annotations

try:
    from .window import AssistantWindow, main
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in headless test envs
    if exc.name != "PySide6":
        raise

    missing_dependency = exc
    AssistantWindow = None

    def main() -> None:
        raise ModuleNotFoundError("PySide6 is required to launch the desktop UI.") from missing_dependency

__all__ = ["AssistantWindow", "main"]
