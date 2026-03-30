import json
import os
import queue
import re
import sys
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import vosk
except Exception:
    vosk = None


SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.20
COMMAND_WAIT_SECONDS = 3.0
DUPLICATE_COMMAND_BLOCK_SECONDS = 4.0
MIN_COMMAND_LENGTH = 3
MIN_AUDIO_LEVEL = 0.012

WAKE_WORDS = ("ассистент", "асистент", "ассистен", "асистен")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "model")

# Для снижения ложных срабатываний лучше оставить False.
# Если потом захочешь прямые команды без wake word, поставь True.
ALLOW_DIRECT_COMMANDS_WITHOUT_WAKE_WORD = False

DIRECT_COMMAND_PATTERNS = [
    r"^(?:пожалуйста\s+)?(?:открой|открывай|зайди\s+на|перейди\s+на)\s+.+$",
    r"^(?:пожалуйста\s+)?(?:запусти|запускай)\s+.+$",
    r"^(?:пожалуйста\s+)?(?:найди|поищи|загугли|найди\s+в\s+интернете|поиск)\s+.+$",
]

INCOMPLETE_COMMANDS = {
    "открой",
    "открывай",
    "открой сайт",
    "открой приложение",
    "открой программу",
    "открой файл",
    "открой папку",
    "запусти",
    "запускай",
    "найди",
    "поищи",
    "загугли",
    "поиск",
}


def get_input_devices():
    if sd is None:
        return []

    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append((idx, dev["name"]))
    return result


def get_device_default_samplerate(device_index):
    if sd is None:
        raise RuntimeError("Не установлен sounddevice. Установи: pip install sounddevice")

    if device_index is None:
        info = sd.query_devices(kind="input")
        return int(info["default_samplerate"])

    info = sd.query_devices(device_index)
    return int(info["default_samplerate"])


def test_microphone_levels(device=None, seconds=3):
    if sd is None:
        raise RuntimeError("Не установлен sounddevice. Установи: pip install sounddevice")

    sample_rate = get_device_default_samplerate(device)
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio = audio.flatten()

    mean_level = float(np.mean(np.abs(audio))) if len(audio) else 0.0
    max_level = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return mean_level, max_level


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    duration = len(audio) / float(orig_sr)
    new_length = max(1, int(duration * target_sr))
    old_times = np.linspace(0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0, duration, num=new_length, endpoint=False)
    resampled = np.interp(new_times, old_times, audio)
    return resampled.astype(np.float32)


def float_audio_to_pcm16_bytes(audio):
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767).astype(np.int16)
    return int16_audio.tobytes()


class VoiceListener(QThread):
    heard_command = Signal(str)
    status = Signal(str)
    failed = Signal(str)
    heard_text = Signal(str)

    def __init__(self, input_device=None):
        super().__init__()
        self.input_device = input_device
        self.running = False
        self.stream = None
        self.audio_queue = queue.Queue()

        self.vosk_model = None
        self.recognizer = None
        self.device_sample_rate = None

        self.awaiting_command = False
        self.awaiting_until = 0.0
        self.last_emitted_command = ""
        self.last_emit_ts = 0.0
        self.last_partial_text = ""
        self.last_partial_ts = 0.0

    def preload_model(self):
        if vosk is None:
            raise RuntimeError("Не установлен vosk. Установи: pip install vosk")

        if self.vosk_model is not None:
            return

        if not os.path.exists(VOSK_MODEL_PATH):
            raise RuntimeError(
                f"Папка модели Vosk не найдена: {VOSK_MODEL_PATH}\n"
                "Скачай русскую модель и распакуй её рядом с проектом."
            )

        self.status.emit("Загружаю модель Vosk...")
        try:
            self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            self.recognizer.SetWords(False)
            self.recognizer.SetPartialWords(False)
        except Exception as exc:
            raise RuntimeError(f"Не удалось загрузить модель Vosk: {exc}") from exc

    def normalize_recognized_text(self, text):
        text = text.lower().replace("ё", "е").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def cleanup_command_text(self, text):
        text = self.normalize_recognized_text(text)
        text = text.strip(" ,.!?-:;")
        text = re.sub(
            r"^(?:пожалуйста|слушай|ну|а ну|сможешь|можешь|будь добр|будь добра)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip(" ,.!?-:;")
        return text

    def is_complete_command(self, text):
        cleaned = self.cleanup_command_text(text)
        if not cleaned or len(cleaned) < MIN_COMMAND_LENGTH:
            return False

        if cleaned in INCOMPLETE_COMMANDS:
            return False

        parts = cleaned.split()
        if len(parts) < 2:
            return False

        return True

    def find_wake_word_match(self, text):
        normalized = self.normalize_recognized_text(text)

        for match in re.finditer(r"\b[\w-]+\b", normalized):
            token = match.group(0)
            if token in WAKE_WORDS or token.startswith(("ассист", "асист")):
                return match
        return None

    def extract_command_after_wake_word(self, text):
        normalized = self.normalize_recognized_text(text)
        match = self.find_wake_word_match(normalized)
        if not match:
            return None
        return self.cleanup_command_text(normalized[match.end():])

    def looks_like_direct_command(self, text):
        cleaned = self.cleanup_command_text(text)
        if not self.is_complete_command(cleaned):
            return False

        for pattern in DIRECT_COMMAND_PATTERNS:
            if re.match(pattern, cleaned, flags=re.IGNORECASE):
                return True
        return False

    def emit_command_once(self, command):
        command = self.cleanup_command_text(command)
        if not self.is_complete_command(command):
            return False

        now = time.monotonic()
        if (
            command == self.last_emitted_command
            and now - self.last_emit_ts < DUPLICATE_COMMAND_BLOCK_SECONDS
        ):
            return False

        self.last_emitted_command = command
        self.last_emit_ts = now
        self.awaiting_command = False
        self.awaiting_until = 0.0
        self.last_partial_text = ""

        self.status.emit(f"Распознано: {command}")
        self.heard_command.emit(command)
        return True

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        if self.running:
            self.audio_queue.put(indata.copy())

    def stop(self):
        self.running = False
        self.awaiting_command = False
        self.awaiting_until = 0.0
        self.last_partial_text = ""

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception:
            pass

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def handle_partial_text(self, text):
        normalized = self.normalize_recognized_text(text)
        if not normalized:
            return

        now = time.monotonic()
        if normalized != self.last_partial_text or now - self.last_partial_ts > 0.8:
            self.last_partial_text = normalized
            self.last_partial_ts = now
            self.heard_text.emit(normalized)

    def handle_final_text(self, text):
        normalized = self.normalize_recognized_text(text)
        if not normalized:
            return

        self.heard_text.emit(normalized)
        now = time.monotonic()

        command_after_wake = self.extract_command_after_wake_word(normalized)
        if command_after_wake is not None:
            if command_after_wake:
                self.emit_command_once(command_after_wake)
            else:
                self.awaiting_command = True
                self.awaiting_until = now + COMMAND_WAIT_SECONDS
                self.status.emit("Ключевое слово услышано, жду команду...")
            return

        if self.awaiting_command:
            if now > self.awaiting_until:
                self.awaiting_command = False
                self.awaiting_until = 0.0
                self.status.emit("Время ожидания команды истекло")
                return

            cleaned = self.cleanup_command_text(normalized)
            if self.is_complete_command(cleaned):
                self.emit_command_once(cleaned)
            return

        if ALLOW_DIRECT_COMMANDS_WITHOUT_WAKE_WORD and self.looks_like_direct_command(normalized):
            self.emit_command_once(normalized)

    def run(self):
        try:
            if sd is None:
                raise RuntimeError("Не установлен sounddevice. Установи: pip install sounddevice")

            self.running = True
            self.preload_model()
            self.device_sample_rate = get_device_default_samplerate(self.input_device)

            self.status.emit(
                f"Голосовой режим включён. Частота устройства: {self.device_sample_rate} Hz. "
                f"Жду фразу '{WAKE_WORDS[0]}'..."
            )

            self.stream = sd.InputStream(
                samplerate=self.device_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=max(1, int(self.device_sample_rate * CHUNK_SECONDS)),
                callback=self.audio_callback,
                device=self.input_device,
            )
            self.stream.start()

            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.15)
                except queue.Empty:
                    if self.awaiting_command and time.monotonic() > self.awaiting_until:
                        self.awaiting_command = False
                        self.awaiting_until = 0.0
                        self.status.emit("Команда после ключевого слова не распознана")
                    continue

                audio = chunk.flatten()
                mean_level = float(np.mean(np.abs(audio))) if len(audio) else 0.0
                if mean_level < MIN_AUDIO_LEVEL:
                    continue

                audio_16k = resample_audio(audio, self.device_sample_rate, SAMPLE_RATE)
                pcm_bytes = float_audio_to_pcm16_bytes(audio_16k)

                if self.recognizer.AcceptWaveform(pcm_bytes):
                    try:
                        result = json.loads(self.recognizer.Result())
                    except Exception:
                        result = {}
                    text = (result.get("text") or "").strip()
                    if text:
                        self.handle_final_text(text)
                    self.last_partial_text = ""
                else:
                    try:
                        partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
                    except Exception:
                        partial = ""

                    if partial:
                        self.handle_partial_text(partial)

        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.stop()


if __name__ == "__main__":
    print("Проверка voice_listener.py")
    print(f"VOSK_MODEL_PATH = {VOSK_MODEL_PATH}")
    print("Доступные микрофоны:")
    for idx, name in get_input_devices():
        print(f"[{idx}] {name}")

    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Ошибка: папка модели не найдена: {VOSK_MODEL_PATH}")
        sys.exit(1)

    if sd is None:
        print("Ошибка: не установлен sounddevice")
        sys.exit(1)

    if vosk is None:
        print("Ошибка: не установлен vosk")
        sys.exit(1)

    print("Файл загружается без синтаксических ошибок.")