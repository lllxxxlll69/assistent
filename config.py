import os

APP_NAME = "Local PC Assistant"
DB_PATH = "assistant_history.db"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:4b"

# Голос полностью отключён
AUTO_START_VOICE = False

MAX_AGENT_STEPS = 4
MAX_HISTORY_MESSAGES = 14

HTTP_TIMEOUT = 25
MAX_WEB_RESULTS = 5
MAX_WEB_PAGE_CHARS = 5000

APP_CACHE_TTL_SECONDS = 300
FILESYSTEM_CACHE_TTL_SECONDS = 300

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "assistant.log")

SYSTEM_PROMPT = """Ты локальный ассистент для ПК. Отвечай по-русски.

Разрешённые инструменты:
- open_site(target)
- open_app(app_name)
- open_file(file_name)
- open_folder(folder_name)
- open_search_in_browser(query)
- web_search(query)
- fetch_url(url)

Правила:
1. Если пользователь явно просит открыть файл, сайт, приложение, программу, папку или выполнить поиск — верни только JSON tool_call.
2. Если пользователь не просит ничего открывать, отвечай обычным текстом.
3. Не выдумывай, что что-то уже открылось, если tool_call не был вызван.
4. Если нужен инструмент, отвечай строго JSON без markdown и без пояснений.
5. Если инструмент не нужен, отвечай коротко и по делу.
6. Формат строго такой:
{"tool_call":{"name":"open_site","arguments":{"target":"youtube.com"}}}
"""