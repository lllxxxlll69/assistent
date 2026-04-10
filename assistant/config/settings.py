from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_DATA_DIR = Path(os.getenv("ASSISTANT_DATA_DIR", ".assistant_data"))
DEFAULT_SETTINGS_PATH = DEFAULT_DATA_DIR / "settings.json"


@dataclass(slots=True)
class Settings:
    model: str = "qwen2.5:3b"
    vision_model: str = "qwen3-vl:4b"
    api_url: str = "http://127.0.0.1:11434/api/chat"
    temperature: float = 0.2
    max_tokens: int = 1200
    context_size: int = 8192
    memory_length: int = 20
    memory_max_tokens: int = 6000
    max_search_results: int = 5
    search_chunk_size: int = 900
    search_root: str = "."
    system_prompt_language: str = "ru"
    request_timeout: int = 180
    stream: bool = True
    show_logs: bool = True


class SettingsManager:
    def __init__(self, settings_path: Path | str = DEFAULT_SETTINGS_PATH) -> None:
        self.settings_path = Path(settings_path)
        self._lock = Lock()

    def ensure_storage(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    def get_settings(self) -> Settings:
        self.ensure_storage()
        if not self.settings_path.exists():
            settings = Settings()
            self._write(settings)
            return settings

        with self._lock:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        return Settings(**payload)

    def update_settings(self, **updates: Any) -> Settings:
        settings = self.get_settings()
        merged = asdict(settings)
        merged.update({key: value for key, value in updates.items() if value is not None})
        updated = Settings(**merged)
        self._write(updated)
        return updated

    def reset_settings(self) -> Settings:
        settings = Settings()
        self._write(settings)
        return settings

    def _write(self, settings: Settings) -> None:
        self.ensure_storage()
        with self._lock:
            self.settings_path.write_text(
                json.dumps(asdict(settings), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
