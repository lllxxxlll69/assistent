import os
import queue
import json
import sys
import re
import requests
import torch
import numpy as np
import sounddevice as sd
import vosk

# === 1. НАСТРОЙКИ ===
MODEL_PATH = "model"  # Путь к папке Vosk
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:4b"

SAMPLE_RATE = 16000
SPEAKER = "baya"

SYSTEM_PROMPT = "Ты — нейроассистент Нэкс. Отвечай кратко, 1-2 предложения. Ты женского пола."

# === 2. ИНИЦИАЛИЗАЦИЯ ПАМЯТИ ДИАЛОГА ===
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# === 3. ИНИЦИАЛИЗАЦИЯ ГОЛОСА (Silero TTS) ===
device = torch.device("cpu")
print("--- Загрузка синтезатора речи... ---")
tts_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="v4_ru",
    trust_repo=True
)
tts_model.to(device)

# === 4. ИНИЦИАЛИЗАЦИЯ СЛУХА (Vosk) ===
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Папка модели '{MODEL_PATH}' не найдена!")
    sys.exit(1)

vosk_model = vosk.Model(MODEL_PATH)
rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
audio_queue = queue.Queue()

http = requests.Session()
http.headers.update({"Content-Type": "application/json; charset=utf-8"})


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


def speak(text):
    clean_text = re.sub(r"[*_#]", "", text)
    print(f"Nex говорит: {clean_text}")

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

    try:
        for chunk in chunks:
            if not chunk.strip():
                continue

            audio = tts_model.apply_tts(
                text=chunk,
                speaker=SPEAKER,
                sample_rate=48000
            ).numpy()

            short_silence = np.zeros(int(48000 * 0.1), dtype=np.float32)
            final_audio_blocks.append(audio)
            final_audio_blocks.append(short_silence)

        if final_audio_blocks:
            tail_silence = np.zeros(int(48000 * 0.5), dtype=np.float32)
            final_audio_blocks.append(tail_silence)

            full_audio = np.concatenate(final_audio_blocks)
            sd.play(full_audio, 48000)
            sd.wait()

    except Exception as e:
        print(f"Ошибка синтеза: {e}")


def get_ollama_answer(user_text):
    global messages

    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.6,
            "num_ctx": 4096,
        }
    }

    try:
        response = http.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        answer = data["message"]["content"].strip()
        messages.append({"role": "assistant", "content": answer})

        # Чтобы история не разрасталась слишком сильно
        if len(messages) > 20:
            messages = [messages[0]] + messages[-19:]

        return answer

    except requests.exceptions.ConnectionError:
        return "Не удалось подключиться к Ollama. Проверь, запущен ли локальный сервер."
    except Exception as e:
        return f"Произошла ошибка связи с локальной моделью: {e}"


def warmup_ollama():
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [],
        "stream": False,
        "keep_alive": "30m",
    }

    try:
        http.post(OLLAMA_URL, json=payload, timeout=30)
        print(f"--- Модель {OLLAMA_MODEL} прогрета ---")
    except Exception as e:
        print(f"--- Не удалось прогреть модель: {e} ---")


def main():
    print("\n--- Nex готов к работе! Говорите... ---")
    warmup_ollama()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            data = audio_queue.get()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                user_text = result.get("text", "").strip()

                if user_text:
                    print(f"\nВы сказали: {user_text}")

                    if any(word in user_text for word in ["выход", "стоп", "пока"]):
                        speak("До связи")
                        break

                    print("Nex думает...")
                    answer = get_ollama_answer(user_text)

                    speak(answer)

                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break

                    rec.Reset()
                    print("\n--- Снова слушаю... ---")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма остановлена.")