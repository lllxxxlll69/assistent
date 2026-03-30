import json
import os
import queue
import re
import sys
import threading

import numpy as np
import requests
import sounddevice as sd
import torch
import vosk
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

SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 48000
SPEAKER = "baya"
MODEL_PATH = "model"


class VoiceAssistant(QThread):
    recognized_text = Signal(str)
    assistant_text = Signal(str)
    status = Signal(str)
    error = Signal(str)

    _tts_model = None
    _tts_lock = threading.Lock()
    _speak_lock = threading.Lock()

    def __init__(
        self,
        session_id,
        ollama_url=DEFAULT_OLLAMA_URL,
        model_name=DEFAULT_MODEL,
        parent=None,
    ):
        super().__init__(parent)
        self.session_id = session_id
        self.ollama_url = ollama_url
        self.model_name = model_name

        self._running = False
        self.audio_queue = queue.Queue()
        self.http = requests.Session()
        self.http.headers.update({"Content-Type": "application/json; charset=utf-8"})

        self.vosk_model = None
        self.rec = None
        self.tool_manager = ToolManager(status_callback=self.status.emit)

    def stop(self):
        self._running = False
        try:
            sd.stop()
        except Exception:
            pass

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        if self._running:
            self.audio_queue.put(bytes(indata))

    @classmethod
    def ensure_tts_loaded(cls):
        with cls._tts_lock:
            if cls._tts_model is None:
                device = torch.device("cpu")
                tts_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v4_ru",
                    trust_repo=True,
                )

                try:
                    if hasattr(tts_model, "to"):
                        tts_model.to(device)
                except Exception:
                    pass

                try:
                    if hasattr(tts_model, "eval"):
                        tts_model.eval()
                except Exception:
                    pass

                cls._tts_model = tts_model

        return cls._tts_model

    def init_vosk(self):
        self.status.emit("Загружаю распознавание...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Папка модели Vosk '{MODEL_PATH}' не найдена. "
                f"Положи туда локальную модель Vosk."
            )
        self.vosk_model = vosk.Model(MODEL_PATH)
        self.rec = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)

    def warmup_ollama(self):
        self.status.emit("Проверяю локальную модель...")
        payload = {
            "model": self.model_name,
            "messages": [],
            "stream": False,
            "keep_alive": "30m",
        }
        response = self.http.post(self.ollama_url, json=payload, timeout=30)
        response.raise_for_status()

    def safe_status(self, text: str):
        try:
            self.status.emit(text)
        except Exception:
            pass

    def speak(self, text):
        """
        Нефатальная озвучка:
        - если TTS или sounddevice падает, голосовой режим НЕ выключается
        - просто показываем статус и продолжаем слушать дальше
        """
        clean_text = re.sub(r"[*_#`]", "", text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            return

        if len(clean_text) > 600:
            clean_text = clean_text[:600]

        with self.__class__._speak_lock:
            try:
                tts_model = self.ensure_tts_loaded()

                parts = re.split(r"([,.!?\n])", clean_text)
                chunks = []
                current_chunk = ""

                for part in parts:
                    if part in ",.!?\n":
                        current_chunk += part
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = ""
                    else:
                        current_chunk += part

                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                final_audio_blocks = []

                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    if len(chunk) > 220:
                        subchunks = [chunk[i:i + 220] for i in range(0, len(chunk), 220)]
                    else:
                        subchunks = [chunk]

                    for subchunk in subchunks:
                        subchunk = subchunk.strip()
                        if not subchunk:
                            continue

                        audio = tts_model.apply_tts(
                            text=subchunk,
                            speaker=SPEAKER,
                            sample_rate=TTS_SAMPLE_RATE,
                        )

                        if hasattr(audio, "cpu"):
                            audio = audio.cpu()

                        if hasattr(audio, "numpy"):
                            audio = audio.numpy()

                        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

                        if audio.size == 0:
                            continue

                        if not np.isfinite(audio).all():
                            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

                        short_silence = np.zeros(int(TTS_SAMPLE_RATE * 0.08), dtype=np.float32)
                        final_audio_blocks.append(audio)
                        final_audio_blocks.append(short_silence)

                if not final_audio_blocks:
                    self.safe_status("Озвучка пропущена: пустой аудиобуфер")
                    return

                tail_silence = np.zeros(int(TTS_SAMPLE_RATE * 0.25), dtype=np.float32)
                final_audio_blocks.append(tail_silence)

                full_audio = np.concatenate(final_audio_blocks).astype(np.float32, copy=False)

                if full_audio.ndim != 1:
                    full_audio = full_audio.reshape(-1)

                if full_audio.size == 0:
                    self.safe_status("Озвучка пропущена: пустой итоговый буфер")
                    return

                try:
                    sd.stop()
                except Exception:
                    pass

                # sounddevice стабильнее принимает 2D массив (frames, channels)
                playback_audio = full_audio.reshape(-1, 1)

                sd.play(playback_audio, samplerate=TTS_SAMPLE_RATE, blocking=True)

            except Exception as e:
                self.safe_status(f"Озвучка недоступна: {type(e).__name__}: {e}")

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

    def route_user_text(self, text: str) -> dict:
        direct_tool = self.infer_direct_tool_from_user_text(text)

        if direct_tool and self.starts_with_open_prefix(text):
            return {"mode": "direct_tool", "tool": direct_tool}

        if direct_tool and self.looks_like_open_intent(text):
            return {"mode": "direct_tool", "tool": direct_tool}

        return {"mode": "llm"}

    def build_messages(self, user_text):
        recent_history = get_recent_messages(self.session_id, limit=MAX_HISTORY_MESSAGES)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def ask_model_once(self, messages):
        payload = {
            "model": self.model_name,
            "stream": False,
            "keep_alive": "30m",
            "messages": messages,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        response = self.http.post(
            self.ollama_url,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

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

    def ask_model_with_tools(self, user_text):
        messages = self.build_messages(user_text)

        for _step in range(MAX_AGENT_STEPS):
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
                        "Если информации уже достаточно, не делай tool_call, а просто ответь."
                    ),
                }
            )

        return "Не удалось завершить обработку голосового запроса."

    def process_user_text(self, user_text: str) -> str:
        route = self.route_user_text(user_text)

        if route["mode"] == "direct_tool":
            return self.tool_manager.execute(
                route["tool"]["name"],
                route["tool"]["arguments"],
            )

        return self.ask_model_with_tools(user_text)

    def handle_text(self, user_text):
        user_text = user_text.strip()
        if not user_text:
            return

        self.recognized_text.emit(user_text)

        lowered = self.normalize_text(user_text)
        if any(word in lowered for word in ["выход", "стоп", "пока"]):
            answer = "До связи"
            self.assistant_text.emit(answer)
            self.speak(answer)
            self.stop()
            return

        self.status.emit("Обрабатываю голосовой запрос...")

        try:
            answer = self.process_user_text(user_text)
        except requests.exceptions.ConnectionError:
            self.error.emit("Не удалось подключиться к Ollama. Проверь, что ollama serve запущен.")
            return
        except Exception as e:
            self.error.emit(f"Ошибка голосового режима: {e}")
            return

        self.assistant_text.emit(answer)

        spoken_answer = answer
        if len(spoken_answer) > 300:
            spoken_answer = spoken_answer[:300]

        self.speak(spoken_answer)

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        if self.rec:
            self.rec.Reset()

    def run(self):
        try:
            self._running = True
            self.status.emit("Загружаю голос...")
            self.ensure_tts_loaded()
            self.init_vosk()
            self.warmup_ollama()
            self.status.emit("Голосовой режим включён")

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback,
            ):
                while self._running:
                    data = self.audio_queue.get()
                    if self.rec.AcceptWaveform(data):
                        result = json.loads(self.rec.Result())
                        user_text = result.get("text", "").strip()

                        if user_text:
                            self.handle_text(user_text)

            self.status.emit("Голосовой режим выключен")

        except Exception as e:
            self.error.emit(str(e))