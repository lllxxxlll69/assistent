import difflib
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

from config import (
    APP_CACHE_TTL_SECONDS,
    FILESYSTEM_CACHE_TTL_SECONDS,
    HTTP_TIMEOUT,
    MAX_WEB_PAGE_CHARS,
    MAX_WEB_RESULTS,
)

logger = logging.getLogger(__name__)


class ToolManager:
    _app_cache = None
    _app_cache_built_at = 0.0
    _filesystem_cache = None
    _filesystem_cache_built_at = 0.0
    _http_session = None

    def __init__(self, status_callback=None):
        self.status_callback = status_callback
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
        self.tools = {
            "open_site": self._tool_open_site,
            "open_app": self._tool_open_app,
            "open_file": self._tool_open_file,
            "open_folder": self._tool_open_folder,
            "open_search_in_browser": self._tool_open_search,
            "web_search": self._tool_web_search,
            "fetch_url": self._tool_fetch_url,
        }

    @classmethod
    def get_http_session(cls):
        if cls._http_session is None:
            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
                )
            })
            cls._http_session = session
        return cls._http_session

    def emit_status(self, text):
        if self.status_callback:
            self.status_callback(text)

    def execute(self, tool_name, arguments):
        logger.info("Tool call: %s | args=%s", tool_name, arguments)
        handler = self.tools.get(tool_name)
        if not handler:
            return "Этот инструмент не разрешён."
        try:
            result = handler(arguments)
            logger.info("Tool result: %s", result[:500])
            return result
        except Exception as exc:
            logger.exception("Tool execution failed: %s", tool_name)
            return f"Ошибка инструмента {tool_name}: {exc}"

    def validate_tool_call(self, tool_call):
        if not isinstance(tool_call, dict):
            return False, "tool_call должен быть объектом"
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if not isinstance(name, str) or not name:
            return False, "Не указано имя инструмента"
        if name not in self.tools:
            return False, f"Инструмент {name} не разрешён"
        if not isinstance(arguments, dict):
            return False, "arguments должен быть объектом"
        return True, ""

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
        text = re.sub(r"^(?:файл|file|документ|document|папка|папку|folder|directory|каталог)\s+", "", text, flags=re.IGNORECASE)
        return text.strip()

    def looks_like_url_or_domain(self, text):
        raw = text.strip()
        if raw.startswith(("http://", "https://")):
            return True
        if " " in raw:
            return False
        return bool(re.fullmatch(r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Zа-яА-Я]{2,}(?:/.*)?", raw, flags=re.IGNORECASE))

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
                    pattern = rf"{re.escape(src)}"
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
            blocked = ["google.", "duckduckgo.com", "bing.com", "yandex.", "search.yahoo."]
            return not any(item in host for item in blocked)
        except Exception:
            return False

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
        snippet_match = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>', window, flags=re.IGNORECASE | re.DOTALL)
        if snippet_match:
            snippet_html = snippet_match.group(1) or snippet_match.group(2) or ""
            return self.strip_html(snippet_html)
        return self.strip_html(window)[:240]

    def search_best_site_url(self, query):
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.get_http_session().get(search_url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        html = response.text
        matches = re.findall(r'<a[^>]+class="result__a"[^>]+href="(.*?)"', html, flags=re.IGNORECASE | re.DOTALL)
        for href in matches:
            real_url = self.extract_real_url_from_search_link(href)
            if self.is_good_public_url(real_url):
                return real_url
        return None

    def search_web_info(self, query):
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.get_http_session().get(search_url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        html = response.text
        links = re.findall(r'<a[^>]+class="result__a"[^>]+href="(.*?)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL)
        results = []
        for href, title_html in links:
            url = self.extract_real_url_from_search_link(href)
            if not self.is_good_public_url(url):
                continue
            clean_title = self.strip_html(title_html)
            snippet = self.extract_nearby_snippet(html, href)
            results.append({"title": clean_title[:180], "url": url, "snippet": snippet[:280]})
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

    def get_app_search_roots(self):
        roots = [
            os.path.join(os.environ.get("PROGRAMDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps"),
        ]
        return [root for root in roots if root.strip() and os.path.isdir(root)]

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
        return [path for path in candidates if os.path.isdir(path)]

    def build_app_cache(self):
        now = time.monotonic()
        if self._app_cache is not None and now - self._app_cache_built_at < APP_CACHE_TTL_SECONDS:
            return self._app_cache
        self.emit_status("Сканирую приложения...")
        entries = []
        seen_paths = set()
        scanned = 0
        max_scanned = 50000
        for root in self.get_app_search_roots():
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
                    entries.append({"name": os.path.splitext(file_name)[0], "path": full_path})
                if scanned > max_scanned:
                    break
            if scanned > max_scanned:
                break
        ToolManager._app_cache = entries
        ToolManager._app_cache_built_at = now
        return ToolManager._app_cache

    def build_filesystem_cache(self):
        now = time.monotonic()
        if self._filesystem_cache is not None and now - self._filesystem_cache_built_at < FILESYSTEM_CACHE_TTL_SECONDS:
            return self._filesystem_cache
        self.emit_status("Сканирую файлы и папки...")
        entries = []
        scanned = 0
        max_scanned = 80000
        for root in self.get_filesystem_search_roots():
            for base, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d.lower() not in {"appdata", "__pycache__", ".git", "node_modules", "venv", ".venv"}]
                for dir_name in dirs:
                    scanned += 1
                    if scanned > max_scanned:
                        break
                    entries.append({"name": dir_name, "path": os.path.join(base, dir_name), "kind": "folder"})
                if scanned > max_scanned:
                    break
                for file_name in files:
                    scanned += 1
                    if scanned > max_scanned:
                        break
                    entries.append({"name": file_name, "path": os.path.join(base, file_name), "kind": "file"})
                if scanned > max_scanned:
                    break
            if scanned > max_scanned:
                break
        ToolManager._filesystem_cache = entries
        ToolManager._filesystem_cache_built_at = now
        return ToolManager._filesystem_cache

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
            if not q.lower().endswith('.exe'):
                which_match_exe = shutil.which(q + '.exe')
                if which_match_exe:
                    candidates.append((1180, q + '.exe', which_match_exe))
        for query in query_variants:
            for item in app_cache:
                score = self.score_name_match(query, item['name'])
                if score >= 520:
                    candidates.append((score, item['name'], item['path']))
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
                score = self.score_name_match(cleaned_query, item['name'])
                if prefer_kind and item['kind'] == prefer_kind:
                    score += 120
                if score >= 520:
                    candidates.append((score, item['name'], item['path'], item['kind']))
        if not candidates:
            return None, None, None
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        return best[1], best[2], best[3]

    def open_path(self, path):
        if not os.path.exists(path):
            return f"Не найдено: {path}"
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
        return f"Открыл папку: {path}" if os.path.isdir(path) else f"Открыл файл: {path}"

    def resolve_site_target(self, target):
        target = self.cleanup_target_text(target)
        normalized = self.normalize_match_text(target)
        if not target:
            return 'Не получил адрес или описание сайта.'
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
            return 'Не получил название приложения.'
        variants = self.build_language_variants(app_name)
        found_name, found_path = self.find_best_installed_app(variants)
        if not found_path:
            return f"Приложение не найдено: {app_name}"
        try:
            if sys.platform.startswith('win'):
                os.startfile(found_path)
            else:
                subprocess.Popen([found_path])
            return f"Открыл приложение: {found_name}"
        except Exception as exc:
            return f"Нашёл приложение '{found_name}', но не смог открыть: {exc}"

    def resolve_file_target(self, file_name):
        file_name = self.cleanup_target_text(file_name)
        if not file_name:
            return 'Не получил название файла.'
        direct_path = self.try_direct_path(file_name)
        if direct_path and os.path.isfile(direct_path):
            return self.open_path(direct_path)
        variants = self.build_language_variants(file_name)
        found_name, found_path, found_kind = self.find_best_filesystem_match(variants, prefer_kind='file')
        if found_path and found_kind == 'file':
            return self.open_path(found_path)
        return f"Файл не найден: {file_name}"

    def resolve_folder_target(self, folder_name):
        folder_name = self.cleanup_target_text(folder_name)
        if not folder_name:
            return 'Не получил название папки.'
        direct_path = self.try_direct_path(folder_name)
        if direct_path and os.path.isdir(direct_path):
            return self.open_path(direct_path)
        variants = self.build_language_variants(folder_name)
        found_name, found_path, found_kind = self.find_best_filesystem_match(variants, prefer_kind='folder')
        if found_path and found_kind == 'folder':
            return self.open_path(found_path)
        return f"Папка не найдена: {folder_name}"

    def _tool_open_site(self, arguments):
        return self.resolve_site_target(arguments.get('target', '').strip())

    def _tool_open_app(self, arguments):
        return self.resolve_app_target(arguments.get('app_name', '').strip())

    def _tool_open_file(self, arguments):
        return self.resolve_file_target(arguments.get('file_name', '').strip())

    def _tool_open_folder(self, arguments):
        return self.resolve_folder_target(arguments.get('folder_name', '').strip())

    def _tool_open_search(self, arguments):
        query = arguments.get('query', '').strip()
        if not query:
            return 'Не получил поисковый запрос.'
        url = f"https://www.google.com/search?q={quote_plus(query)}"
        webbrowser.open(url)
        return f"Открыл поиск по запросу: {query}"

    def _tool_web_search(self, arguments):
        query = arguments.get('query', '').strip()
        if not query:
            return 'Не получил поисковый запрос.'
        return self.search_web_info(query)

    def _tool_fetch_url(self, arguments):
        url = arguments.get('url', '').strip()
        if not url:
            return 'Не получил URL страницы.'
        return self.fetch_url_text(url)
