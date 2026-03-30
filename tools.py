import logging
import os
import re
import subprocess
import webbrowser
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback or (lambda _text: None)

        self.allowed_tools = {
            "open_site": {"required": ["target"]},
            "open_app": {"required": ["app_name"]},
            "open_file": {"required": ["file_name"]},
            "open_folder": {"required": ["folder_name"]},
            "open_search_in_browser": {"required": ["query"]},
            "web_search": {"required": ["query"]},
            "fetch_url": {"required": ["url"]},
        }

        self.search_roots = self._build_search_roots()

    def _build_search_roots(self):
        home = Path.home()
        roots = [
            home / "Desktop",
            home / "Documents",
            home / "Downloads",
            home / "Pictures",
            home / "Music",
            home / "Videos",
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        ]

        result = []
        for root in roots:
            try:
                if root and str(root) and root.exists():
                    result.append(root)
            except Exception:
                pass
        return result

    def validate_tool_call(self, tool_call):
        if not isinstance(tool_call, dict):
            return False, "tool_call должен быть dict"

        name = tool_call.get("name")
        arguments = tool_call.get("arguments")

        if name not in self.allowed_tools:
            return False, f"неизвестный инструмент: {name}"

        if not isinstance(arguments, dict):
            return False, "arguments должен быть dict"

        required = self.allowed_tools[name]["required"]
        for field in required:
            value = arguments.get(field)
            if not isinstance(value, str) or not value.strip():
                return False, f"отсутствует обязательный аргумент: {field}"

        return True, ""

    def execute(self, tool_name, arguments):
        logger.info("Tool call: %s | args=%s", tool_name, arguments)

        method = getattr(self, tool_name, None)
        if method is None:
            result = f"Инструмент не реализован: {tool_name}"
            logger.info("Tool result: %s", result)
            return result

        try:
            result = method(**arguments)
            logger.info("Tool result: %s", result)
            return result
        except Exception as e:
            result = f"Ошибка выполнения инструмента {tool_name}: {e}"
            logger.exception(result)
            return result

    def normalize_text(self, value: str) -> str:
        value = value.lower().replace("ё", "е").strip()
        value = re.sub(r"\s+", " ", value)
        return value

    def normalize_app_name(self, app_name: str) -> str:
        value = self.normalize_text(app_name)

        aliases = {
            "пэйнт": "mspaint",
            "паинт": "mspaint",
            "paint": "mspaint",
            "mspaint": "mspaint",

            "блокнот": "notepad",
            "нотпад": "notepad",
            "notepad": "notepad",

            "калькулятор": "calc",
            "calculator": "calc",
            "calc": "calc",

            "проводник": "explorer",
            "эксплорер": "explorer",
            "explorer": "explorer",

            "яндекс музыка": "yandex music",
            "яндекс мьюзик": "yandex music",
            "yandex music": "yandex music",

            "телеграм": "telegram",
            "telegram": "telegram",

            "дискорд": "discord",
            "discord": "discord",

            "стим": "steam",
            "steam": "steam",
        }

        return aliases.get(value, value)

    def normalize_site_target(self, target: str) -> str:
        value = self.normalize_text(target)
        aliases = {
            "ютуб": "youtube.com",
            "ютьюб": "youtube.com",
            "youtube": "youtube.com",
            "гугл": "google.com",
            "google": "google.com",
            "яндекс": "yandex.ru",
            "яндекс музыка": "music.yandex.ru",
            "яндекс мьюзик": "music.yandex.ru",
            "yandex music": "music.yandex.ru",
            "гитхаб": "github.com",
            "гитхуб": "github.com",
            "github": "github.com",
            "википедия": "wikipedia.org",
            "вики": "wikipedia.org",
            "телеграм": "web.telegram.org",
            "telegram": "web.telegram.org",
            "дискорд": "discord.com/app",
            "discord": "discord.com/app",
        }
        return aliases.get(value, target.strip())

    def _score_match(self, query: str, candidate: str) -> int:
        q = self.normalize_text(query)
        c = self.normalize_text(candidate)

        if q == c:
            return 1000
        if c.startswith(q):
            return 900
        if q in c:
            return 800

        ratio = SequenceMatcher(None, q, c).ratio()
        return int(ratio * 100)

    def _iter_files(self, extensions=None, max_depth=4):
        for root in self.search_roots:
            root_parts_len = len(root.parts)

            try:
                for path in root.rglob("*"):
                    try:
                        if len(path.parts) - root_parts_len > max_depth:
                            continue

                        if not path.exists():
                            continue

                        if extensions is not None:
                            if not path.is_file():
                                continue
                            if path.suffix.lower() not in extensions:
                                continue

                        yield path
                    except Exception:
                        continue
            except Exception:
                continue

    def _find_best_file(self, file_name: str):
        query = self.normalize_text(file_name)
        best_path = None
        best_score = -1

        for path in self._iter_files(extensions=None, max_depth=4):
            try:
                if not path.is_file():
                    continue
                score = self._score_match(query, path.name)
                if score > best_score:
                    best_score = score
                    best_path = path
            except Exception:
                continue

        return best_path, best_score

    def _find_best_folder(self, folder_name: str):
        query = self.normalize_text(folder_name)
        best_path = None
        best_score = -1

        for root in self.search_roots:
            try:
                for path in root.rglob("*"):
                    try:
                        if not path.is_dir():
                            continue
                        score = self._score_match(query, path.name)
                        if score > best_score:
                            best_score = score
                            best_path = path
                    except Exception:
                        continue
            except Exception:
                continue

        return best_path, best_score

    def _find_best_app(self, app_name: str):
        query = self.normalize_app_name(app_name)
        candidates = []

        shortcut_exts = {".lnk", ".url", ".exe"}
        for path in self._iter_files(extensions=shortcut_exts, max_depth=5):
            try:
                candidates.append(path)
            except Exception:
                continue

        best_path = None
        best_score = -1

        for path in candidates:
            try:
                stem_score = self._score_match(query, path.stem)
                name_score = self._score_match(query, path.name)
                score = max(stem_score, name_score)

                if score > best_score:
                    best_score = score
                    best_path = path
            except Exception:
                continue

        return best_path, best_score

    def open_site(self, target: str):
        target = self.normalize_site_target(target)

        if not re.match(r"^https?://", target, flags=re.IGNORECASE):
            if re.fullmatch(r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Zа-яА-Я]{2,}(?:/.*)?", target):
                target = f"https://{target}"
            else:
                webbrowser.open(f"https://www.google.com/search?q={target}")
                return f"Не нашёл точный сайт. Открыл поиск по запросу: {target}"

        webbrowser.open(target)
        return f"Открыл сайт: {target}"

    def open_search_in_browser(self, query: str):
        query = query.strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"Открыл поиск по запросу: {query}"

    def open_app(self, app_name: str):
        app_name = self.normalize_app_name(app_name)

        system_apps = {
            "mspaint": "mspaint",
            "notepad": "notepad",
            "calc": "calc",
            "explorer": "explorer",
            "cmd": "cmd",
            "powershell": "powershell",
        }

        if app_name in system_apps:
            try:
                subprocess.Popen(system_apps[app_name], shell=True)
                return f"Открыл приложение: {app_name}"
            except Exception as e:
                return f"Не удалось открыть приложение {app_name}: {e}"

        best_path, best_score = self._find_best_app(app_name)
        if best_path and best_score >= 70:
            try:
                os.startfile(str(best_path))
                return f"Открыл приложение: {best_path.stem}"
            except Exception as e:
                return f"Не удалось открыть приложение {best_path.stem}: {e}"

        return f"Приложение не найдено: {app_name}"

    def open_file(self, file_name: str):
        best_path, best_score = self._find_best_file(file_name)

        if best_path and best_score >= 70:
            try:
                os.startfile(str(best_path))
                return f"Открыл файл: {best_path.name}"
            except Exception as e:
                return f"Не удалось открыть файл {best_path.name}: {e}"

        return f"Файл не найден: {file_name}"

    def open_folder(self, folder_name: str):
        best_path, best_score = self._find_best_folder(folder_name)

        if best_path and best_score >= 70:
            try:
                os.startfile(str(best_path))
                return f"Открыл папку: {best_path.name}"
            except Exception as e:
                return f"Не удалось открыть папку {best_path.name}: {e}"

        return f"Папка не найдена: {folder_name}"

    def web_search(self, query: str):
        return self.open_search_in_browser(query)

    def fetch_url(self, url: str):
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = f"https://{url}"
        webbrowser.open(url)
        return f"Открыл URL: {url}"
