from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_DATA_DIR = Path(os.getenv("ASSISTANT_DATA_DIR", ".assistant_data"))
DEFAULT_SETTINGS_PATH = DEFAULT_DATA_DIR / "settings.json"


@dataclass(slots=True)
class Settings:
    model: str = "qwen2.5-coder:7b"
    vision_model: str = "qwen3-vl:4b"
    api_url: str = os.getenv("ASSISTANT_API_URL", "http://127.0.0.1:11434/api/chat")
    temperature: float = 0.2
    localscript_temperature: float = 0.0
    max_tokens: int = 1200
    context_size: int = 8192
    memory_length: int = 20
    memory_max_tokens: int = 6000
    max_search_results: int = 5
    search_chunk_size: int = 900
    max_search_file_size_kb: int = 512
    workspace_root: str = "."
    search_root: str = "."
    system_prompt_language: str = "ru"
    request_timeout: int = 180
    stream: bool = True
    show_logs: bool = True
    batch_size: int = 1
    assistant_profile: str = "localscript"
    agent_self_check_attempts: int = 3
    localscript_context_size: int = 4096
    localscript_num_predict: int = 256
    localscript_auto_validate: bool = True
    localscript_repair_attempts: int = 2
    localscript_candidate_count: int = 3
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    api_allowed_origins: str = "http://127.0.0.1,http://localhost"
    api_token: str = ""
    max_request_bytes: int = 1_048_576
    max_image_size_mb: int = 12


class SettingsManager:
    def __init__(self, settings_path: Path | str = DEFAULT_SETTINGS_PATH) -> None:
        self.settings_path = Path(settings_path)
        self._lock = Lock()
        self._field_names = {item.name for item in fields(Settings)}

    def ensure_storage(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    def get_settings(self) -> Settings:
        self.ensure_storage()
        if not self.settings_path.exists():
            settings = self._apply_env_overrides(Settings())
            self._write(settings)
            return settings

        try:
            with self._lock:
                payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            corrupt_backup = self.settings_path.with_suffix(".corrupt.json")
            try:
                if self.settings_path.exists():
                    self.settings_path.replace(corrupt_backup)
            except OSError:
                pass
            settings = self._apply_env_overrides(Settings())
            self._write(settings)
            return settings

        filtered_payload = {key: value for key, value in payload.items() if key in self._field_names}
        settings = Settings(**filtered_payload)
        return self._apply_env_overrides(settings)

    def update_settings(self, **updates: Any) -> Settings:
        settings = self.get_settings()
        merged = asdict(settings)
        merged.update({key: value for key, value in updates.items() if value is not None})
        updated = Settings(**{key: value for key, value in merged.items() if key in self._field_names})
        self._write(updated)
        return updated

    def reset_settings(self) -> Settings:
        settings = self._apply_env_overrides(Settings())
        self._write(settings)
        return settings

    def _write(self, settings: Settings) -> None:
        self.ensure_storage()
        with self._lock:
            _write_atomic_text(
                self.settings_path,
                json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            )

    def _apply_env_overrides(self, settings: Settings) -> Settings:
        overrides: dict[str, Any] = {}
        env_map: dict[str, tuple[str, type[Any]]] = {
            "ASSISTANT_MODEL": ("model", str),
            "ASSISTANT_VISION_MODEL": ("vision_model", str),
            "ASSISTANT_API_URL": ("api_url", str),
            "ASSISTANT_TEMPERATURE": ("temperature", float),
            "ASSISTANT_LOCALSCRIPT_TEMPERATURE": ("localscript_temperature", float),
            "ASSISTANT_MAX_TOKENS": ("max_tokens", int),
            "ASSISTANT_CONTEXT_SIZE": ("context_size", int),
            "ASSISTANT_MEMORY_LENGTH": ("memory_length", int),
            "ASSISTANT_MEMORY_MAX_TOKENS": ("memory_max_tokens", int),
            "ASSISTANT_WORKSPACE_ROOT": ("workspace_root", str),
            "ASSISTANT_SEARCH_ROOT": ("search_root", str),
            "ASSISTANT_MAX_SEARCH_FILE_SIZE_KB": ("max_search_file_size_kb", int),
            "ASSISTANT_REQUEST_TIMEOUT": ("request_timeout", int),
            "ASSISTANT_API_HOST": ("api_host", str),
            "ASSISTANT_API_PORT": ("api_port", int),
            "ASSISTANT_API_ALLOWED_ORIGINS": ("api_allowed_origins", str),
            "ASSISTANT_API_TOKEN": ("api_token", str),
            "ASSISTANT_MAX_REQUEST_BYTES": ("max_request_bytes", int),
            "ASSISTANT_MAX_IMAGE_SIZE_MB": ("max_image_size_mb", int),
            "ASSISTANT_BATCH_SIZE": ("batch_size", int),
            "ASSISTANT_PROFILE": ("assistant_profile", str),
            "ASSISTANT_AGENT_SELF_CHECK_ATTEMPTS": ("agent_self_check_attempts", int),
            "ASSISTANT_LOCALSCRIPT_CONTEXT_SIZE": ("localscript_context_size", int),
            "ASSISTANT_LOCALSCRIPT_NUM_PREDICT": ("localscript_num_predict", int),
            "ASSISTANT_LOCALSCRIPT_REPAIR_ATTEMPTS": ("localscript_repair_attempts", int),
            "ASSISTANT_LOCALSCRIPT_CANDIDATE_COUNT": ("localscript_candidate_count", int),
        }

        for env_name, (field_name, caster) in env_map.items():
            raw_value = os.getenv(env_name)
            if raw_value is None or raw_value == "":
                continue
            overrides[field_name] = caster(raw_value)

        bool_map = {
            "ASSISTANT_STREAM": "stream",
            "ASSISTANT_SHOW_LOGS": "show_logs",
            "ASSISTANT_LOCALSCRIPT_AUTO_VALIDATE": "localscript_auto_validate",
        }
        for env_name, field_name in bool_map.items():
            raw_value = os.getenv(env_name)
            if raw_value is None or raw_value == "":
                continue
            overrides[field_name] = raw_value.strip().lower() in {"1", "true", "yes", "on"}

        if not overrides:
            return settings

        merged = asdict(settings)
        merged.update(overrides)
        return Settings(**merged)


def _write_atomic_text(path: Path, content: str, *, retries: int = 8, delay_seconds: float = 0.03) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(retries):
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(content, encoding="utf-8")
            temp_path.replace(path)
            return
        except OSError as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            time.sleep(delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
