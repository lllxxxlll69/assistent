
import difflib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import requests
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from voice_listener import VoiceListener, get_input_devices, test_microphone_levels

APP_NAME = "Local PC Assistant"
DB_PATH = "assistant_history.db"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:4b"
AUTO_START_VOICE = True
HTTP_TIMEOUT = 25
MAX_AGENT_STEPS = 4
APP_CACHE_TTL_SECONDS = 300
FILESYSTEM_CACHE_TTL_SECONDS = 300
MAX_WEB_RESULTS = 5
MAX_WEB_PAGE_CHARS = 5000

SYSTEM_PROMPT = """Ты локальный ассистент для ПК. Отвечай по-русски.

Ты можешь вызывать только эти инструменты:
- open_site(target): открыть сайт в браузере
- open_app(app_name): открыть приложение
- open_file(file_name): открыть файл
- open_folder(folder_name): открыть папку
- open_search_in_browser(query): открыть поиск в браузере
- web_search(query): найти информацию в интернете и вернуть текстовые результаты
- fetch_url(url): прочитать страницу сайта и вернуть текст

Правила:
1. Для фактических вопросов, новостей, справки, инструкций и поиска информации используй web_search.
2. Если одного поиска мало, сначала вызови web_search, потом fetch_url по самому полезному URL.
3. open_search_in_browser используй только когда пользователь явно просит открыть поиск в браузере.
4. Команды открытия по ПК используй только когда пользователь явно просит открыть или запустить что-то.
5. Если данных уже достаточно, отвечай обычным текстом, без tool_call.
6. Если вызываешь инструмент, отвечай строго JSON-объектом без пояснений и markdown.
7. Формат tool_call строго такой:
{"tool_call": {"name": "web_search", "arguments": {"query": "погода в Хельсинки"}}}
"""


def ensure_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    conn.commit()

    row = conn.execute("SELECT id FROM chat_sessions LIMIT 1").fetchone()
    if not row:
        session_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO chat_sessions(id, title, created_at) VALUES (?, ?, ?)",
            (session_id, "Основной чат", datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()

    conn.close()


def get_setting(key, default=""):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else default


def set_setting(key, value):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO app_settings(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def list_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title FROM chat_sessions ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return rows


def create_session(title=None):
    session_id = str(uuid.uuid4())
    if not title:
        title = f"Сценарий {datetime.now().strftime('%H:%M:%S')}"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_sessions(id, title, created_at) VALUES (?, ?, ?)",
        (session_id, title, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return session_id


def rename_session_if_needed(session_id, first_user_text):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT title FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if row and row[0].startswith("Сценарий "):
        title = first_user_text.strip()[:40] or row[0]
        conn.execute("UPDATE chat_sessions SET title = ? WHERE id = ?", (title, session_id))
        conn.commit()
    conn.close()


def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def load_session_history(session_id, limit=1000):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return rows


def get_recent_messages(session_id, limit=120):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


class ChatInput(QTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
                event.accept()
        else:
            super().keyPressEvent(event)


class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        ollama_model=DEFAULT_MODEL,
        ollama_url=DEFAULT_OLLAMA_URL,
        speak=False,
        input_device=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.resize(560, 300)

        self.model_edit = QLineEdit(ollama_model)
        self.url_edit = QLineEdit(ollama_url)
        self.speak_checkbox = QCheckBox("Озвучивать ответы")
        self.speak_checkbox.setChecked(speak)

        self.input_device_combo = QComboBox()
        self.input_device_combo.addItem("Система по умолчанию", None)
        for idx, name in get_input_devices():
            self.input_device_combo.addItem(f"[{idx}] {name}", idx)

        self._set_combo_value(self.input_device_combo, input_device)

        form = QFormLayout()
        form.addRow("Ollama model:", self.model_edit)
        form.addRow("Ollama URL:", self.url_edit)
        form.addRow("Микрофон:", self.input_device_combo)
        form.addRow("", self.speak_checkbox)

        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(self.accept)

        test_button = QPushButton("Проверить микрофон")
        test_button.clicked.connect(self.test_microphone)

        buttons = QHBoxLayout()
        buttons.addWidget(test_button)
        buttons.addWidget(save_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def _set_combo_value(self, combo, value):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def values(self):
        return (
            self.model_edit.text().strip(),
            self.url_edit.text().strip(),
            self.speak_checkbox.isChecked(),
            self.input_device_combo.currentData(),
        )

    def test_microphone(self):
        device = self.input_device_combo.currentData()
        try:
            mean_level, max_level = test_microphone_levels(device=device, seconds=3)

            if max_level < 0.01:
                QMessageBox.warning(
                    self,
                    "Проверка микрофона",
                    f"Сигнал почти не слышен.\n\n"
                    f"Средний уровень: {mean_level:.5f}\n"
                    f"Максимальный уровень: {max_level:.5f}\n\n"
                    f"Попробуй выбрать другой микрофон или сказать что-нибудь громче."
                )
            else:
                QMessageBox.information(
                    self,
                    "Проверка микрофона",
                    f"Микрофон работает.\n\n"
                    f"Средний уровень: {mean_level:.5f}\n"
                    f"Максимальный уровень: {max_level:.5f}"
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Проверка микрофона",
                f"Не удалось проверить микрофон:\n{exc}"
            )


class AssistantWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    status = Signal(str)

    _global_app_cache = None
    _global_filesystem_cache = None
    _app_cache_built_at = 0.0
    _filesystem_cache_built_at = 0.0
    _tts_engine = None
    _tts_lock = threading.Lock()
    _http_session = None

    def __init__(self, session_id, user_text, model_name, ollama_url, speak_enabled):
        super().__init__()
        self.session_id = session_id
        self.user_text = user_text
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.speak_enabled = speak_enabled

        self.term_variants_map = {
            "блокнот": ["notepad", "windows notepad"],
            "notepad": ["блокнот", "windows notepad"],
            "калькулятор": ["calculator", "calc"],
            "calculator": ["калькулятор", "calc"],
            "calc": ["калькулятор", "calculator"],
            "проводник": ["explorer", "file explorer", "windows explorer"],
            "explorer": ["проводник", "file explorer", "windows explorer"],
            "file explorer": ["проводник", "explorer"],
            "терминал": ["terminal", "windows terminal", "wt", "cmd", "powershell"],
            "terminal": ["терминал", "windows terminal", "wt", "cmd", "powershell"],
            "командная строка": ["cmd", "command prompt"],
            "cmd": ["командная строка", "command prompt"],
            "пауэршелл": ["powershell", "power shell"],
            "powershell": ["пауэршелл", "power shell"],
            "паинт": ["paint", "mspaint"],
            "paint": ["паинт", "mspaint"],
            "браузер": ["browser", "chrome", "edge", "firefox", "opera", "brave"],
            "browser": ["браузер", "chrome", "edge", "firefox", "opera", "brave"],
            "хром": ["chrome", "google chrome"],
            "chrome": ["хром", "google chrome"],
            "эдж": ["edge", "microsoft edge"],
            "edge": ["эдж", "microsoft edge"],
            "фаерфокс": ["firefox"],
            "firefox": ["фаерфокс"],
            "опера": ["opera"],
            "opera": ["опера"],
            "телеграм": ["telegram"],
            "telegram": ["телеграм"],
            "дискорд": ["discord"],
            "discord": ["дискорд"],
            "стим": ["steam"],
            "steam": ["стим"],
            "вс код": ["vs code", "vscode", "visual studio code", "code"],
            "vs code": ["вс код", "vscode", "visual studio code", "code"],
            "vscode": ["вс код", "vs code", "visual studio code", "code"],
            "visual studio code": ["вс код", "vs code", "vscode", "code"],
            "ворд": ["word", "microsoft word"],
            "word": ["ворд", "microsoft word"],
            "эксель": ["excel", "microsoft excel"],
            "excel": ["эксель", "microsoft excel"],
            "пауэрпоинт": ["powerpoint", "microsoft powerpoint"],
            "powerpoint": ["пауэрпоинт", "microsoft powerpoint"],
            "фотошоп": ["photoshop", "adobe photoshop"],
            "photoshop": ["фотошоп", "adobe photoshop"],
            "рабочий стол": ["desktop"],
            "desktop": ["рабочий стол"],
            "документы": ["documents"],
            "documents": ["документы"],
            "загрузки": ["downloads"],
            "downloads": ["загрузки"],
            "изображения": ["pictures", "photos"],
            "pictures": ["изображения", "photos"],
            "photos": ["изображения", "pictures"],
            "музыка": ["music"],
            "music": ["музыка"],
            "видео": ["videos"],
            "videos": ["видео"],
        }

        self.site_aliases = {
            "ютуб": "youtube.com",
            "youtube": "youtube.com",
            "гугл": "google.com",
            "google": "google.com",
            "википедия": "wikipedia.org",
            "wiki": "wikipedia.org",
            "вконтакте": "vk.com",
            "вк": "vk.com",
            "vk": "vk.com",
            "github": "github.com",
            "гитхаб": "github.com",
            "openai": "openai.com",
            "чат gpt": "chatgpt.com",
            "chatgpt": "chatgpt.com",
            "телеграм": "web.telegram.org",
            "telegram": "web.telegram.org",
            "дискорд": "discord.com/app",
            "discord": "discord.com/app",
        }

    @classmethod
    def get_http_session(cls):
        if cls._http_session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
                    )
                }
            )
            cls._http_session = session
        return cls._http_session

    def run(self):
        try:
            direct_tool = self.infer_tool_from_user_text(self.user_text)
            if direct_tool:
                result = self.execute_tool(direct_tool)
                self.finished_ok.emit(result)
                return

            self.status.emit("Анализирую запрос...")
            answer = self.ask_model_with_tools(self.user_text)
            if self.speak_enabled:
                self.speak(answer)
            self.finished_ok.emit(answer)
        except Exception as exc:
            self.failed.emit(str(exc))

    def build_messages(self, user_text):
        recent_history = get_recent_messages(self.session_id, limit=120)
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
            "options": {"temperature": 0.15},
        }

        response = self.get_http_session().post(
            self.ollama_url,
            json=payload,
            timeout=180,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    def ask_model_with_tools(self, user_text):
        messages = self.build_messages(user_text)

        for step in range(MAX_AGENT_STEPS):
            self.status.emit(f"Думаю... шаг {step + 1}/{MAX_AGENT_STEPS}")
            answer = self.ask_model_once(messages)
            tool_call = self.extract_tool_call(answer)

            if not tool_call:
                return answer

            tool_name = tool_call.get("name", "")
            self.status.emit(f"Вызываю инструмент: {tool_name}")
            tool_result = self.execute_tool(tool_call)

            if tool_name in {"open_site", "open_app", "open_file", "open_folder", "open_search_in_browser"}:
                return tool_result

            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Результат инструмента {tool_name}:\n{tool_result}\n\n"
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

        fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", answer, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(fenced_blocks)

        for candidate in self.extract_json_objects(answer):
            candidates.append(candidate)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            tool_call = parsed.get("tool_call") if isinstance(parsed, dict) else None
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

    def infer_tool_from_user_text(self, text):
        raw = self.cleanup_target_text(text)
        lowered = self.normalize_match_text(raw)

        site_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай|зайди\s+на|перейди\s+на)\s+(?:мне\s+)?(?:сайт|site|веб.?сайт|страницу|page)\s+(.+)$",
            r"^(?:please\s+)?(?:open|go\s+to)\s+(?:site|website|page)\s+(.+)$",
        ]
        app_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай|запусти|запускай)\s+(?:мне\s+)?(?:приложение|программу|app|application|program)\s+(.+)$",
            r"^(?:please\s+)?(?:open|run|launch|start)\s+(?:app|application|program)\s+(.+)$",
            r"^(?:пожалуйста\s+)?(?:запусти|запускай)\s+(.+)$",
        ]
        file_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай)\s+(?:мне\s+)?(?:файл|file|документ|document)\s+(.+)$",
            r"^(?:please\s+)?(?:open)\s+(?:file|document)\s+(.+)$",
        ]
        folder_patterns = [
            r"^(?:пожалуйста\s+)?(?:открой|открывай)\s+(?:мне\s+)?(?:папку|папка|folder|directory|каталог)\s+(.+)$",
            r"^(?:please\s+)?(?:open)\s+(?:folder|directory)\s+(.+)$",
        ]
        browser_search_patterns = [
            r"^(?:пожалуйста\s+)?(?:найди|поищи|загугли|найди\s+в\s+интернете|поиск)\s+(.+)$",
            r"^(?:please\s+)?(?:search|google|find)\s+(.+)$",
        ]

        for pattern in browser_search_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                query = raw[match.start(1):match.end(1)].strip()
                if query:
                    return {"name": "open_search_in_browser", "arguments": {"query": query}}

        for pattern in site_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                target = raw[match.start(1):match.end(1)].strip()
                if target:
                    return {"name": "open_site", "arguments": {"target": target}}

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

        for pattern in app_patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if match:
                app_name = raw[match.start(1):match.end(1)].strip()
                if app_name:
                    return {"name": "open_app", "arguments": {"app_name": app_name}}

        if self.looks_like_url_or_domain(raw):
            return {"name": "open_site", "arguments": {"target": raw}}

        if lowered in self.site_aliases:
            return {"name": "open_site", "arguments": {"target": lowered}}

        return None

    def normalize_match_text(self, text):
        text = text.lower().replace("ё", "е").strip()
        text = re.sub(r"[\"'“”«»!?;,]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def cleanup_target_text(self, text):
        text = text.strip().strip('"').strip("'")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def cleanup_fs_query(self, text):
        text = self.cleanup_target_text(text)
        text = re.sub(
            r"^(?:файл|file|документ|document|папка|папку|folder|directory|каталог)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    def looks_like_url_or_domain(self, text):
        raw = text.strip()
        if raw.startswith(("http://", "https://")):
            return True

        if " " in raw:
            return False

        return bool(
            re.fullmatch(
                r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Zа-яА-Я]{2,}(?:/.*)?",
                raw,
                flags=re.IGNORECASE,
            )
        )

    def normalize_url(self, text):
        url = text.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    def expand_path(self, path):
        path = path.strip().strip('"').strip("'")
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        return os.path.abspath(path)

    def try_direct_path(self, target):
        path = self.expand_path(target)
        if os.path.exists(path):
            return path
        return None

    def build_language_variants(self, text):
        base = self.cleanup_target_text(text)
        normalized = self.normalize_match_text(base)

        variants = []

        def add(item):
            item = self.cleanup_target_text(item)
            if item and item not in variants:
                variants.append(item)

        add(base)
        add(normalized)

        generated = set(variants)
        changed = True

        while changed:
            changed = False
            current = list(generated)

            for item in current:
                norm_item = self.normalize_match_text(item)
                for src, alts in self.term_variants_map.items():
                    pattern = rf"\b{re.escape(src)}\b"
                    if re.search(pattern, norm_item):
                        for alt in alts:
                            replaced = re.sub(pattern, alt, norm_item)
                            replaced = self.cleanup_target_text(replaced)
                            if replaced and replaced not in generated:
                                generated.add(replaced)
                                changed = True

        for item in list(generated):
            add(item)

        return variants[:25]

    def score_name_match(self, query, candidate_name):
        q = self.normalize_match_text(query)
        c = self.normalize_match_text(candidate_name)
        c_stem = self.normalize_match_text(os.path.splitext(candidate_name)[0])

        if not q or not c:
            return 0

        if q == c or q == c_stem:
            return 1000

        if c.startswith(q) or c_stem.startswith(q):
            return 900

        if q in c or q in c_stem:
            return 800

        q_tokens = q.split()
        if q_tokens and all(token in c for token in q_tokens):
            return 700

        ratio = difflib.SequenceMatcher(None, q, c).ratio()
        ratio_stem = difflib.SequenceMatcher(None, q, c_stem).ratio()
        return int(max(ratio, ratio_stem) * 500)

    def extract_real_url_from_search_link(self, href):
        href = unescape(href).strip()

        if href.startswith("//"):
            href = "https:" + href

        if href.startswith("/l/?"):
            href = "https://duckduckgo.com" + href

        parsed = urlparse(href)
        query = parse_qs(parsed.query)

        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])

        return href

    def is_good_public_url(self, url):
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False

            host = (parsed.netloc or "").lower()
            blocked = [
                "google.",
                "duckduckgo.com",
                "bing.com",
                "yandex.",
                "search.yahoo.",
            ]
            return not any(item in host for item in blocked)
        except Exception:
            return False

    def search_best_site_url(self, query):
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.get_http_session().get(search_url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        html = response.text

        matches = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="(.*?)"',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not matches:
            matches = re.findall(
                r'<a[^>]+href="(https?://[^"]+)"',
                html,
                flags=re.IGNORECASE | re.DOTALL,
            )

        for href in matches:
            real_url = self.extract_real_url_from_search_link(href)
            if self.is_good_public_url(real_url):
                return real_url

        return None

    def get_app_search_roots(self):
        roots = [
            os.path.join(os.environ.get("PROGRAMDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps"),
        ]
        unique = []
        for root in roots:
            root = root.strip()
            if root and os.path.isdir(root) and root not in unique:
                unique.append(root)
        return unique

    def get_filesystem_search_roots(self):
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, "Desktop"),
            os.path.join(home, "Documents"),
            os.path.join(home, "Downloads"),
            os.path.join(home, "Pictures"),
            os.path.join(home, "Music"),
            os.path.join(home, "Videos"),
            os.path.join(home, "OneDrive"),
        ]
        unique = []
        for path in candidates:
            if os.path.isdir(path) and path not in unique:
                unique.append(path)
        return unique

    def build_app_cache(self):
        now = time.monotonic()
        if (
            AssistantWorker._global_app_cache is not None
            and now - AssistantWorker._app_cache_built_at < APP_CACHE_TTL_SECONDS
        ):
            return AssistantWorker._global_app_cache

        self.status.emit("Сканирую приложения...")
        entries = []
        roots = self.get_app_search_roots()
        seen_paths = set()
        scanned = 0
        max_scanned = 50000

        for root in roots:
            for base, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d.lower() not in {"windows", "__pycache__"}]

                for file_name in files:
                    scanned += 1
                    if scanned > max_scanned:
                        break

                    lower = file_name.lower()
                    if not lower.endswith((".lnk", ".exe", ".url", ".bat", ".cmd")):
                        continue

                    full_path = os.path.join(base, file_name)
                    if full_path in seen_paths:
                        continue
                    seen_paths.add(full_path)

                    name = os.path.splitext(file_name)[0]
                    entries.append(
                        {
                            "name": name,
                            "norm": self.normalize_match_text(name),
                            "path": full_path,
                        }
                    )

                if scanned > max_scanned:
                    break
            if scanned > max_scanned:
                break

        AssistantWorker._global_app_cache = entries
        AssistantWorker._app_cache_built_at = now
        return AssistantWorker._global_app_cache

    def build_filesystem_cache(self):
        now = time.monotonic()
        if (
            AssistantWorker._global_filesystem_cache is not None
            and now - AssistantWorker._filesystem_cache_built_at < FILESYSTEM_CACHE_TTL_SECONDS
        ):
            return AssistantWorker._global_filesystem_cache

        self.status.emit("Сканирую файлы и папки...")
        entries = []
        roots = self.get_filesystem_search_roots()
        scanned = 0
        max_scanned = 80000

        for root in roots:
            for base, dirs, files in os.walk(root):
                dirs[:] = [
                    d for d in dirs
                    if d.lower() not in {"appdata", "__pycache__", ".git", "node_modules", "venv", ".venv"}
                ]

                for dir_name in dirs:
                    scanned += 1
                    if scanned > max_scanned:
                        break
                    full_path = os.path.join(base, dir_name)
                    entries.append(
                        {
                            "name": dir_name,
                            "path": full_path,
                            "kind": "folder",
                        }
                    )

                if scanned > max_scanned:
                    break

                for file_name in files:
                    scanned += 1
                    if scanned > max_scanned:
                        break
                    full_path = os.path.join(base, file_name)
                    entries.append(
                        {
                            "name": file_name,
                            "path": full_path,
                            "kind": "file",
                        }
                    )

                if scanned > max_scanned:
                    break

            if scanned > max_scanned:
                break

        AssistantWorker._global_filesystem_cache = entries
        AssistantWorker._filesystem_cache_built_at = now
        return AssistantWorker._global_filesystem_cache

    def find_best_installed_app(self, query_variants):
        app_cache = self.build_app_cache()
        candidates = []

        for query in query_variants:
            q = self.cleanup_target_text(query)
            if not q:
                continue

            which_match = shutil.which(q)
            if which_match:
                candidates.append((1200, q, which_match))

            if not q.lower().endswith(".exe"):
                which_match_exe = shutil.which(q + ".exe")
                if which_match_exe:
                    candidates.append((1180, q + ".exe", which_match_exe))

        for query in query_variants:
            for item in app_cache:
                score = self.score_name_match(query, item["name"])
                if score >= 520:
                    candidates.append((score, item["name"], item["path"]))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        return best[1], best[2]

    def find_best_filesystem_match(self, query_variants, prefer_kind=None):
        fs_cache = self.build_filesystem_cache()
        candidates = []

        for query in query_variants:
            cleaned_query = self.cleanup_fs_query(query)
            if not cleaned_query:
                continue

            for item in fs_cache:
                score = self.score_name_match(cleaned_query, item["name"])
                if prefer_kind and item["kind"] == prefer_kind:
                    score += 120
                if score >= 520:
                    candidates.append((score, item["name"], item["path"], item["kind"]))

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        return best[1], best[2], best[3]

    def open_path(self, path):
        if not os.path.exists(path):
            return f"Не найдено: {path}"

        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

        if os.path.isdir(path):
            return f"Открыл папку: {path}"
        return f"Открыл файл: {path}"

    def resolve_site_target(self, target):
        target = self.cleanup_target_text(target)
        normalized = self.normalize_match_text(target)
        if not target:
            return "Не получил адрес или описание сайта."

        if normalized in self.site_aliases:
            target = self.site_aliases[normalized]

        if self.looks_like_url_or_domain(target):
            url = self.normalize_url(target)
            webbrowser.open(url)
            return f"Открыл сайт: {url}"

        best_url = self.search_best_site_url(target)
        if best_url:
            webbrowser.open(best_url)
            return f"Открыл сайт по запросу: {target}"

        fallback_url = f"https://www.google.com/search?q={quote_plus(target)}"
        webbrowser.open(fallback_url)
        return f"Не нашёл точный сайт. Открыл поиск по запросу: {target}"

    def resolve_app_target(self, app_name):
        app_name = self.cleanup_target_text(app_name)
        if not app_name:
            return "Не получил название приложения."

        variants = self.build_language_variants(app_name)

        found_name, found_path = self.find_best_installed_app(variants)
        if not found_path:
            return f"Приложение не найдено: {app_name}"

        try:
            if sys.platform.startswith("win"):
                os.startfile(found_path)
            else:
                subprocess.Popen([found_path])
            return f"Открыл приложение: {found_name}"
        except Exception as exc:
            return f"Нашёл приложение '{found_name}', но не смог открыть: {exc}"

    def resolve_file_target(self, file_name):
        file_name = self.cleanup_target_text(file_name)
        if not file_name:
            return "Не получил название файла."

        direct_path = self.try_direct_path(file_name)
        if direct_path and os.path.isfile(direct_path):
            return self.open_path(direct_path)

        variants = self.build_language_variants(file_name)
        found_name, found_path, found_kind = self.find_best_filesystem_match(variants, prefer_kind="file")
        if found_path and found_kind == "file":
            return self.open_path(found_path)

        return f"Файл не найден: {file_name}"

    def resolve_folder_target(self, folder_name):
        folder_name = self.cleanup_target_text(folder_name)
        if not folder_name:
            return "Не получил название папки."

        direct_path = self.try_direct_path(folder_name)
        if direct_path and os.path.isdir(direct_path):
            return self.open_path(direct_path)

        variants = self.build_language_variants(folder_name)
        found_name, found_path, found_kind = self.find_best_filesystem_match(variants, prefer_kind="folder")
        if found_path and found_kind == "folder":
            return self.open_path(found_path)

        return f"Папка не найдена: {folder_name}"

    def search_web_info(self, query):
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.get_http_session().get(search_url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        html = response.text

        links = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="(.*?)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results = []
        for href, title_html in links:
            url = self.extract_real_url_from_search_link(href)
            if not self.is_good_public_url(url):
                continue

            clean_title = self.strip_html(title_html)
            snippet = self.extract_nearby_snippet(html, href)
            results.append(
                {
                    "title": clean_title[:180],
                    "url": url,
                    "snippet": snippet[:280],
                }
            )
            if len(results) >= MAX_WEB_RESULTS:
                break

        if not results:
            return "Поиск не дал результатов."

        lines = [f"Результаты поиска по запросу: {query}"]
        for idx, item in enumerate(results, start=1):
            lines.append(f"{idx}. {item['title']}")
            lines.append(f"URL: {item['url']}")
            if item["snippet"]:
                lines.append(f"Описание: {item['snippet']}")
            lines.append("")

        return "\n".join(lines).strip()

    def fetch_url_text(self, url):
        url = self.normalize_url(url)
        response = self.get_http_session().get(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return f"Не удалось прочитать страницу как HTML. Content-Type: {content_type}"

        html = response.text
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = self.strip_html(title_match.group(1)) if title_match else url

        body = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        body = re.sub(r"(?is)<style.*?>.*?</style>", " ", body)
        body = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", body)
        body = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", body)
        body = re.sub(r"(?is)<[^>]+>", " ", body)
        body = unescape(body)
        body = re.sub(r"\s+", " ", body).strip()

        if len(body) > MAX_WEB_PAGE_CHARS:
            body = body[:MAX_WEB_PAGE_CHARS] + "..."

        return f"Заголовок страницы: {title}\nURL: {response.url}\nТекст страницы:\n{body}"

    def strip_html(self, text):
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_nearby_snippet(self, html, href):
        pattern = re.escape(href)
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if not match:
            return ""

        start = max(0, match.end())
        window = html[start:start + 1200]
        snippet_match = re.search(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>',
            window,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if snippet_match:
            snippet_html = snippet_match.group(1) or snippet_match.group(2) or ""
            return self.strip_html(snippet_html)

        window_text = self.strip_html(window)
        return window_text[:240]

    def execute_tool(self, tool_call):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if name == "open_site":
            target = arguments.get("target", "").strip()
            text = self.resolve_site_target(target)

        elif name == "open_app":
            app_name = arguments.get("app_name", "").strip()
            text = self.resolve_app_target(app_name)

        elif name == "open_file":
            file_name = arguments.get("file_name", "").strip()
            text = self.resolve_file_target(file_name)

        elif name == "open_folder":
            folder_name = arguments.get("folder_name", "").strip()
            text = self.resolve_folder_target(folder_name)

        elif name == "open_search_in_browser":
            query = arguments.get("query", "").strip()
            if not query:
                return "Не получил поисковый запрос."
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            webbrowser.open(url)
            text = f"Открыл поиск по запросу: {query}"

        elif name == "web_search":
            query = arguments.get("query", "").strip()
            if not query:
                return "Не получил поисковый запрос."
            text = self.search_web_info(query)

        elif name == "fetch_url":
            url = arguments.get("url", "").strip()
            if not url:
                return "Не получил URL страницы."
            text = self.fetch_url_text(url)

        else:
            text = "Этот инструмент не разрешён в текущей версии приложения."

        if self.speak_enabled and name in {"open_site", "open_app", "open_file", "open_folder", "open_search_in_browser"}:
            self.speak(text)
        return text

    def speak(self, text):
        try:
            import pyttsx3
            with AssistantWorker._tts_lock:
                if AssistantWorker._tts_engine is None:
                    AssistantWorker._tts_engine = pyttsx3.init()
                AssistantWorker._tts_engine.say(text)
                AssistantWorker._tts_engine.runAndWait()
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_db()

        self.model_name = get_setting("model_name", DEFAULT_MODEL)
        self.ollama_url = get_setting("ollama_url", DEFAULT_OLLAMA_URL)
        self.speak_enabled = get_setting("speak_enabled", "0") == "1"

        raw_input_device = get_setting("input_device", "")
        self.input_device = int(raw_input_device) if raw_input_device not in ("", "None") else None

        self.worker = None
        self.voice_listener = None
        self.current_session_id = None

        self.setWindowTitle(APP_NAME)
        self.resize(1040, 760)
        self.build_ui()
        self.build_menu()
        self.reload_sessions(select_last=False)

        self.voice_listener = self._make_voice_listener()
        if AUTO_START_VOICE:
            self.voice_listener.start()

    def build_ui(self):
        self.chat_selector = QComboBox()
        self.chat_selector.currentIndexChanged.connect(self.on_session_changed)

        self.new_chat_button = QPushButton("Новый сценарий")
        self.new_chat_button.clicked.connect(self.create_new_chat)

        self.voice_button = QPushButton("Голос: вкл" if AUTO_START_VOICE else "Голос: выкл")
        self.voice_button.clicked.connect(self.toggle_voice)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Чат:"))
        top_row.addWidget(self.chat_selector)
        top_row.addWidget(self.new_chat_button)
        top_row.addWidget(self.voice_button)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setPlaceholderText("Здесь будет история диалога...")

        self.input = ChatInput()
        self.input.setPlaceholderText(
            "Напиши запрос. Например: открой сайт openai, запусти блокнот, открой файл report.pdf, открой папку Downloads, какая сегодня погода в Хельсинки"
        )
        self.input.setFixedHeight(110)
        self.input.send_requested.connect(self.send_message)

        self.status_label = QLabel("Готов")
        self.status_label.setAlignment(Qt.AlignLeft)

        send_button = QPushButton("Отправить")
        send_button.clicked.connect(self.send_message)

        clear_button = QPushButton("Очистить текущий чат")
        clear_button.clicked.connect(self.clear_chat)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(send_button)
        buttons_row.addWidget(clear_button)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.chat)
        layout.addWidget(QLabel("Ввод:"))
        layout.addWidget(self.input)
        layout.addLayout(buttons_row)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def build_menu(self):
        menu = self.menuBar()
        settings_menu = menu.addMenu("Настройки")

        settings_action = QAction("Параметры", self)
        settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(settings_action)

    def _make_voice_listener(self):
        listener = VoiceListener(input_device=self.input_device)
        listener.heard_command.connect(self.on_voice_command)
        listener.heard_text.connect(self.on_voice_text)
        listener.status.connect(self.status_label.setText)
        listener.failed.connect(self.on_voice_error)
        return listener

    def reload_sessions(self, select_last=True):
        sessions = list_sessions()
        self.chat_selector.blockSignals(True)
        self.chat_selector.clear()
        for session_id, title in sessions:
            self.chat_selector.addItem(title, session_id)
        self.chat_selector.blockSignals(False)

        if sessions:
            if self.current_session_id:
                index = self.chat_selector.findData(self.current_session_id)
                if index >= 0:
                    self.chat_selector.setCurrentIndex(index)
                elif select_last:
                    self.chat_selector.setCurrentIndex(len(sessions) - 1)
                else:
                    self.chat_selector.setCurrentIndex(0)
            else:
                self.chat_selector.setCurrentIndex(len(sessions) - 1 if select_last else 0)

            self.current_session_id = self.chat_selector.currentData()
            self.load_history()

    def create_new_chat(self):
        self.current_session_id = create_session()
        self.reload_sessions(select_last=True)
        self.input.setFocus()
        self.status_label.setText("Создан новый сценарий")

    def on_session_changed(self):
        session_id = self.chat_selector.currentData()
        if session_id:
            self.current_session_id = session_id
            self.load_history()

    def append_chat(self, role, text):
        if not self.current_session_id:
            self.current_session_id = create_session()

        time_str = datetime.now().strftime("%H:%M:%S")
        prefix = "Вы" if role == "user" else "Ассистент"
        self.chat.append(f"<b>[{time_str}] {prefix}:</b><br>{text}<br>")
        scrollbar = self.chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        save_message(self.current_session_id, role, text)

    def load_history(self):
        if not self.current_session_id:
            self.chat.clear()
            return

        rows = load_session_history(self.current_session_id, limit=1000)
        self.chat.clear()
        for role, content, created_at in rows:
            prefix = "Вы" if role == "user" else "Ассистент"
            time_str = created_at.split("T")[-1]
            self.chat.append(f"<b>[{time_str}] {prefix}:</b><br>{content}<br>")
        scrollbar = self.chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_chat(self):
        if not self.current_session_id:
            return
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM messages WHERE session_id = ?", (self.current_session_id,))
        conn.commit()
        conn.close()
        self.chat.clear()
        self.status_label.setText("Текущий чат очищен")

    def open_settings(self):
        dlg = SettingsDialog(
            self,
            self.model_name,
            self.ollama_url,
            self.speak_enabled,
            self.input_device,
        )
        if dlg.exec():
            self.model_name, self.ollama_url, self.speak_enabled, self.input_device = dlg.values()

            set_setting("model_name", self.model_name)
            set_setting("ollama_url", self.ollama_url)
            set_setting("speak_enabled", "1" if self.speak_enabled else "0")
            set_setting("input_device", "" if self.input_device is None else str(self.input_device))

            if self.voice_listener and self.voice_listener.isRunning():
                self.voice_listener.stop()
                self.voice_listener.wait(2000)

            self.voice_listener = self._make_voice_listener()
            self.voice_listener.start()
            self.voice_button.setText("Голос: вкл")
            self.status_label.setText("Настройки сохранены и голос перезапущен")

    def send_message(self):
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Подожди, предыдущая команда ещё выполняется")
            return

        if not self.current_session_id:
            self.create_new_chat()

        text = self.input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, APP_NAME, "Введите запрос.")
            return

        self.input.clear()
        self.append_chat("user", text)
        rename_session_if_needed(self.current_session_id, text)
        self.reload_sessions(select_last=False)

        index = self.chat_selector.findData(self.current_session_id)
        if index >= 0:
            self.chat_selector.setCurrentIndex(index)

        self.status_label.setText("Выполняю...")

        self.worker = AssistantWorker(
            self.current_session_id,
            text,
            self.model_name,
            self.ollama_url,
            self.speak_enabled,
        )
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.on_answer)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_answer(self, text):
        self.append_chat("assistant", text)
        self.status_label.setText("Готов")
        self.worker = None

    def on_error(self, error_text):
        self.append_chat("assistant", f"Ошибка: {error_text}")
        self.status_label.setText("Ошибка")
        self.worker = None

    def toggle_voice(self):
        if self.voice_listener and self.voice_listener.isRunning():
            self.voice_button.setEnabled(False)
            self.voice_button.setText("Голос: выключается...")
            self.voice_listener.stop()
            self.voice_listener.wait(2000)
            self.voice_listener = self._make_voice_listener()
            self.voice_button.setEnabled(True)
            self.voice_button.setText("Голос: выкл")
            self.status_label.setText("Голосовой режим выключен")
            return

        self.voice_listener = self._make_voice_listener()
        self.voice_listener.start()
        self.voice_button.setText("Голос: вкл")

    def on_voice_command(self, text):
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Команда услышана, но ассистент ещё занят")
            return
        self.input.setPlainText(text)
        self.send_message()

    def on_voice_text(self, text):
        self.status_label.setText(f"Слышу: {text}")

    def on_voice_error(self, error_text):
        self.voice_button.setText("Голос: выкл")
        self.status_label.setText(f"Ошибка голоса: {error_text}")
        QMessageBox.warning(self, APP_NAME, f"Ошибка голосового режима:\n{error_text}")

    def closeEvent(self, event):
        if self.voice_listener and self.voice_listener.isRunning():
            self.voice_listener.stop()
            self.voice_listener.wait(1500)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

