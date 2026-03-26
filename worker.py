import json
import logging
import re
import threading

import requests
from PySide6.QtCore import QThread, Signal

from config import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, MAX_AGENT_STEPS, SYSTEM_PROMPT
from storage import get_recent_messages
from tools import ToolManager

logger = logging.getLogger(__name__)


class AssistantWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    status = Signal(str)

    _tts_engine = None
    _tts_lock = threading.Lock()
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
            direct_tool = self.infer_direct_tool_from_user_text(self.user_text)
            if direct_tool:
                logger.info("Direct tool inferred: %s", direct_tool)
                result = self.tool_manager.execute(direct_tool["name"], direct_tool["arguments"])
                if self.speak_enabled:
                    self.speak(result)
                self.finished_ok.emit(result)
                return

            self.status.emit("Анализирую запрос...")
            answer = self.ask_model_with_tools(self.user_text)

            if self.speak_enabled:
                self.speak(answer)

            self.finished_ok.emit(answer)
        except Exception as exc:
            logger.exception("Worker failed")
            self.failed.emit(str(exc))

    def normalize_text(self, text):
        text = text.lower().replace("ё", "е").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def cleanup_text(self, text):
        return re.sub(r"\s+", " ", text.strip().strip('"').strip("'")).strip()

    def infer_direct_tool_from_user_text(self, text):
        raw = self.cleanup_text(text)
        lowered = self.normalize_text(raw)

        site_aliases = {
            "ютуб": "youtube.com",
            "youtube": "youtube.com",
            "гугл": "google.com",
            "google": "google.com",
            "википедия": "wikipedia.org",
            "wiki": "wikipedia.org",
            "гитхаб": "github.com",
            "github": "github.com",
            "вк": "vk.com",
            "vk": "vk.com",
            "openai": "openai.com",
            "чат gpt": "chatgpt.com",
            "chatgpt": "chatgpt.com",
            "телеграм веб": "web.telegram.org",
            "discord": "discord.com/app",
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
            r"^(?:пожалуйста\s+)?(?:открой|открывай|зайди\s+на|перейди\s+на)\s+(?:сайт|страницу|page)\s+(.+)$",
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
                target = raw[match.start(1):match.end(1)].strip()
                target = site_aliases.get(self.normalize_text(target), target)
                if target:
                    return {"name": "open_site", "arguments": {"target": target}}

        for pattern in app_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                app_name = raw[match.start(1):match.end(1)].strip()
                if self.normalize_text(app_name) not in incomplete and app_name:
                    return {"name": "open_app", "arguments": {"app_name": app_name}}

        for pattern in file_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                file_name = raw[match.start(1):match.end(1)].strip()
                if file_name:
                    return {"name": "open_file", "arguments": {"file_name": file_name}}

        for pattern in folder_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                folder_name = raw[match.start(1):match.end(1)].strip()
                if folder_name:
                    return {"name": "open_folder", "arguments": {"folder_name": folder_name}}

        for pattern in search_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                query = raw[match.start(1):match.end(1)].strip()
                if query:
                    return {"name": "open_search_in_browser", "arguments": {"query": query}}

        if lowered in site_aliases:
            return {"name": "open_site", "arguments": {"target": site_aliases[lowered]}}

        if re.fullmatch(r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Zа-яА-Я]{2,}(?:/.*)?", raw):
            return {"name": "open_site", "arguments": {"target": raw}}

        if re.match(r"^(?:открой|открывай)\s+(.+)$", lowered):
            maybe_target = re.sub(r"^(?:открой|открывай)\s+", "", raw, flags=re.IGNORECASE).strip()
            maybe_norm = self.normalize_text(maybe_target)
            if maybe_norm in site_aliases:
                return {"name": "open_site", "arguments": {"target": site_aliases[maybe_norm]}}

        return None

    def build_messages(self, user_text):
        recent_history = get_recent_messages(self.session_id, limit=100)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def ask_model_once(self, messages):
        payload = {
            "model": self.model_name,
            "stream": False,
            "keep_alive": "15m",
            "messages": messages,
            "options": {"temperature": 0.0},
        }

        response = self.get_http_session().post(
            self.ollama_url,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()

        data = response.json()
        content = data["message"]["content"].strip()
        logger.info("Model answer: %s", content[:1000])
        return content

    def ask_model_with_tools(self, user_text):
        messages = self.build_messages(user_text)

        for step in range(MAX_AGENT_STEPS):
            self.status.emit(f"Думаю... шаг {step + 1}/{MAX_AGENT_STEPS}")
            answer = self.ask_model_once(messages)
            tool_call = self.extract_tool_call(answer)

            if not tool_call:
                return answer

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
                        "Если информации недостаточно, можешь вызвать ещё один инструмент. "
                        "Если данных уже достаточно, не делай tool_call, а просто ответь."
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

    def speak(self, text):
        try:
            import pyttsx3

            cls = self.__class__
            with cls._tts_lock:
                if cls._tts_engine is None:
                    cls._tts_engine = pyttsx3.init()
                cls._tts_engine.say(text)
                cls._tts_engine.runAndWait()
        except Exception:
            logger.exception("TTS failed")