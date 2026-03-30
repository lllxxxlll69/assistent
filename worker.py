import json
import logging
import re
import requests
from PySide6.QtCore import QThread, Signal

from config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    MAX_AGENT_STEPS,
    MAX_HISTORY_MESSAGES,
    SYSTEM_PROMPT,
)
from storage import get_recent_messages
from tools import ToolManager

logger = logging.getLogger(__name__)


class AssistantWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    status = Signal(str)
    stream_chunk = Signal(str)

    _http_session = None

    def __init__(
        self,
        session_id,
        user_text,
        model_name=DEFAULT_MODEL,
        ollama_url=DEFAULT_OLLAMA_URL,
        speak_enabled=False,
    ):
        super().__init__()
        self.session_id = session_id
        self.user_text = user_text
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.speak_enabled = speak_enabled
        self.tool_manager = ToolManager(status_callback=self.status.emit)

    @classmethod
    def get_http_session(cls):
        if cls._http_session is None:
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json; charset=utf-8"})
            cls._http_session = session
        return cls._http_session

    def run(self):
        try:
            route = self.route_user_text(self.user_text)

            if route["mode"] == "direct_tool":
                result = self.tool_manager.execute(
                    route["tool"]["name"],
                    route["tool"]["arguments"],
                )
                self.finished_ok.emit(result)
                return

            self.status.emit("Анализирую запрос...")
            answer = self.ask_model_with_tools(self.user_text)
            self.finished_ok.emit(answer)

        except Exception as exc:
            logger.exception("Worker failed")
            self.failed.emit(str(exc))

    def normalize_text(self, text: str) -> str:
        text = text.lower().replace("ё", "е").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def cleanup_text(self, text: str) -> str:
        text = text.strip().strip('"').strip("'")
        text = re.sub(r"\s+", " ", text).strip()

        garbage_prefixes = [
            "сайт ",
            "сайта ",
            "сайте ",
            "страницу ",
            "страница ",
            "приложение ",
            "приложения ",
            "программу ",
            "программа ",
            "файл ",
            "файла ",
            "документ ",
            "документа ",
            "папку ",
            "папка ",
            "каталог ",
        ]

        lowered = text.lower().replace("ё", "е").strip()

        changed = True
        while changed:
            changed = False
            for prefix in garbage_prefixes:
                if lowered.startswith(prefix):
                    text = text[len(prefix):].strip()
                    lowered = text.lower().replace("ё", "е").strip()
                    changed = True

        return text

    def starts_with_open_prefix(self, text: str) -> bool:
        normalized = self.normalize_text(text)
        prefixes = (
            "открой файл ",
            "открой сайт ",
            "открой приложение ",
            "открой программу ",
            "открой папку ",
        )
        return normalized.startswith(prefixes)

    def looks_like_open_intent(self, text: str) -> bool:
        normalized = self.normalize_text(text)
        patterns = [
            r"^(?:пожалуйста\s+)?открой\s+.+$",
            r"^(?:пожалуйста\s+)?запусти\s+.+$",
            r"^(?:пожалуйста\s+)?зайди\s+на\s+.+$",
            r"^(?:пожалуйста\s+)?перейди\s+на\s+.+$",
        ]
        return any(re.match(p, normalized, flags=re.IGNORECASE) for p in patterns)

    def route_user_text(self, text: str) -> dict:
        direct_tool = self.infer_direct_tool_from_user_text(text)

        if direct_tool and self.starts_with_open_prefix(text):
            return {"mode": "direct_tool", "tool": direct_tool}

        if direct_tool and self.looks_like_open_intent(text):
            return {"mode": "direct_tool", "tool": direct_tool}

        return {"mode": "llm"}

    def infer_direct_tool_from_user_text(self, text):
        raw = self.cleanup_text(text)
        lowered = self.normalize_text(raw)

        site_aliases = {
            "ютуб": "youtube.com",
            "ютьюб": "youtube.com",
            "youtube": "youtube.com",
            "гугл": "google.com",
            "google": "google.com",
            "википедия": "wikipedia.org",
            "вики": "wikipedia.org",
            "wiki": "wikipedia.org",
            "гитхаб": "github.com",
            "гитхуб": "github.com",
            "github": "github.com",
            "вк": "vk.com",
            "vk": "vk.com",
            "openai": "openai.com",
            "чат gpt": "chatgpt.com",
            "chatgpt": "chatgpt.com",
            "телеграм": "web.telegram.org",
            "телеграм веб": "web.telegram.org",
            "telegram": "web.telegram.org",
            "дискорд": "discord.com/app",
            "discord": "discord.com/app",
            "яндекс": "yandex.ru",
            "яндекс музыка": "music.yandex.ru",
            "яндекс мьюзик": "music.yandex.ru",
            "yandex music": "music.yandex.ru",
        }

        incomplete = {
            "открой",
            "открой сайт",
            "открой приложение",
            "открой программу",
            "открой файл",
            "открой папку",
            "запусти",
            "найди",
            "поиск",
        }

        if lowered in incomplete:
            return None

        site_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай|зайди\s+на|перейди\s+на)\s+(?:сайт|сайта|сайте|страницу|page)?\s*(.+)$",
        ]
        app_patterns = [
            r"^(?:пожалуйста\s+)?(?:запусти|запускай)\s+(?:приложение|программу)?\s*(.+)$",
            r"^(?:пожалуйста\s+)?(?:открой|открывай)\s+(?:приложение|программу)\s+(.+)$",
        ]
        file_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай)\s+(?:файл|документ)\s+(.+)$",
        ]
        folder_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай)\s+(?:папку|папка|каталог)\s+(.+)$",
        ]
        search_patterns = [
            r"^(?:пожалуйста\s+)?(?:найди|поищи|загугли|поиск)\s+(.+)$",
        ]

        for pattern in site_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                target = self.cleanup_text(raw[match.start(1):match.end(1)].strip())
                target = site_aliases.get(self.normalize_text(target), target)
                if target:
                    return {"name": "open_site", "arguments": {"target": target}}

        for pattern in app_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                app_name = self.cleanup_text(raw[match.start(1):match.end(1)].strip())
                if self.normalize_text(app_name) not in incomplete and app_name:
                    return {"name": "open_app", "arguments": {"app_name": app_name}}

        for pattern in file_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                file_name = self.cleanup_text(raw[match.start(1):match.end(1)].strip())
                if file_name:
                    return {"name": "open_file", "arguments": {"file_name": file_name}}

        for pattern in folder_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                folder_name = self.cleanup_text(raw[match.start(1):match.end(1)].strip())
                if folder_name:
                    return {"name": "open_folder", "arguments": {"folder_name": folder_name}}

        for pattern in search_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                query = self.cleanup_text(raw[match.start(1):match.end(1)].strip())
                if query:
                    return {"name": "open_search_in_browser", "arguments": {"query": query}}

        if lowered in site_aliases:
            return {"name": "open_site", "arguments": {"target": site_aliases[lowered]}}

        if re.fullmatch(r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Zа-яА-Я]{2,}(?:/.*)?", raw):
            return {"name": "open_site", "arguments": {"target": raw}}

        if re.match(r"^(?:открой|открывай)\s+(.+)$", lowered):
            maybe_target = self.cleanup_text(
                re.sub(r"^(?:открой|открывай)\s+", "", raw, flags=re.IGNORECASE).strip()
            )
            maybe_norm = self.normalize_text(maybe_target)

            if maybe_norm in site_aliases:
                return {"name": "open_site", "arguments": {"target": site_aliases[maybe_norm]}}

            if re.search(r"\.[a-zA-Z0-9]{1,8}$", maybe_target):
                return {"name": "open_file", "arguments": {"file_name": maybe_target}}

            return {"name": "open_app", "arguments": {"app_name": maybe_target}}

        return None

    def build_messages(self, user_text):
        recent_history = get_recent_messages(self.session_id, limit=MAX_HISTORY_MESSAGES)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def stream_model_text(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        response = self.get_http_session().post(
            self.ollama_url,
            json=payload,
            stream=True,
            timeout=180,
        )
        response.raise_for_status()

        full_text = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            message = data.get("message", {})
            content = message.get("content", "")

            if content:
                full_text += content
                self.stream_chunk.emit(content)

            if data.get("done", False):
                break

        return full_text.strip()

    def ask_model_once_nonstream(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        response = self.get_http_session().post(
            self.ollama_url,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    def ask_model_with_tools(self, user_text):
        messages = self.build_messages(user_text)

        for step in range(MAX_AGENT_STEPS):
            self.status.emit(f"Думаю... шаг {step + 1}/{MAX_AGENT_STEPS}")

            answer = self.ask_model_once_nonstream(messages)
            tool_call = self.extract_tool_call(answer)

            if not tool_call:
                return self.stream_model_text(messages)

            valid, error_message = self.tool_manager.validate_tool_call(tool_call)
            if not valid:
                messages.append({"role": "assistant", "content": answer})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Ты вернул некорректный tool_call: "
                            f"{error_message}. "
                            "Ответь пользователю обычным текстом "
                            "или верни корректный JSON."
                        ),
                    }
                )
                continue

            tool_name = tool_call["name"]
            self.status.emit(f"Вызываю инструмент: {tool_name}")
            result = self.tool_manager.execute(tool_name, tool_call["arguments"])

            if tool_name in {
                "open_site",
                "open_app",
                "open_file",
                "open_folder",
                "open_search_in_browser",
            }:
                return result

            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Результат инструмента {tool_name}:\n"
                        f"{result}\n\n"
                        "Используй эти данные и ответь пользователю по-русски. "
                        "Если информации достаточно, просто ответь."
                    ),
                }
            )

        return "Не удалось завершить обработку запроса за разумное число шагов."

    def extract_tool_call(self, answer):
        candidates = []

        raw = answer.strip()
        if raw:
            candidates.append(raw)

        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        candidates.extend(fenced_blocks)
        candidates.extend(self.extract_json_objects(answer))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(parsed, dict):
                continue

            tool_call = parsed.get("tool_call")
            if not isinstance(tool_call, dict):
                continue

            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if isinstance(name, str) and isinstance(arguments, dict):
                return {"name": name, "arguments": arguments}

        return None

    def extract_json_objects(self, text):
        objects = []
        start = None
        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if char == "\\" and in_string:
                escape = not escape
                continue

            if char == '"' and not escape:
                in_string = not in_string

            if not in_string:
                if char == "{":
                    if depth == 0:
                        start = index
                    depth += 1
                elif char == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start is not None:
                            objects.append(text[start:index + 1])

            if char != "\\":
                escape = False

        return objects