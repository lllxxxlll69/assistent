<p align="center">
  <img src="assets/vomatix.png" width="150" alt="VOMATIX logo" />
</p>

<h1 align="center">VOMATIX CODE</h1>

<p align="center">
  Локальный AI-ассистент для генерации <b>Lua / LocalScript</b> в защищённом контуре
</p>

<p align="center">
  <a href="docs/CONTEST_EVIDENCE.md"><img src="https://img.shields.io/badge/contest-evidence-0f172a?style=for-the-badge" alt="Contest evidence"></a>
  <img src="https://img.shields.io/badge/local--only-Ollama-0ea5e9?style=for-the-badge" alt="Local only via Ollama">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PySide6-GUI-1f2937?style=for-the-badge" alt="PySide6 GUI">
  <img src="https://img.shields.io/badge/OpenAPI-judged%20API-10b981?style=for-the-badge" alt="OpenAPI judged API">
</p>

<p align="center">
  <b>GUI для интерактивной работы</b> · <b>HTTP API для judged-сценария</b> · <b>LLM-only LocalScript pipeline</b> · <b>eval / self-check / reproducibility</b>
</p>

---

## Обзор

**VOMATIX CODE** — это локальный AI-ассистент, который работает через **Ollama** и помогает генерировать, валидировать и дорабатывать **Lua / LocalScript** без отправки данных во внешние AI API.

Проект ориентирован на два основных сценария:

- **Judged LocalScript path** через `POST /generate`
- **Desktop GUI** для итеративной локальной работы

Ключевая идея репозитория: не просто “чат с моделью”, а инженерный контур с:

- отдельным LocalScript pipeline;
- валидацией и ranking кандидатов;
- repair loop;
- eval / benchmark контуром;
- self-check для judged-окружения;
- локальными инструментами для файлов, поиска и vision.

---

## Что умеет проект

| Возможность | Что есть в репозитории |
|---|---|
| Генерация LocalScript/Lua | LLM-only pipeline с validation, ranking и repair |
| Judged API | `POST /generate`, `GET /health`, OpenAPI-контракт |
| GUI | Оконное приложение на PySide6 |
| Агентный режим | Отдельный режим работы с проектом и файлами |
| Проверка качества | smoke benchmark, full eval, self-check |
| Локальность | локальный Ollama runtime, без внешних AI API |
| Воспроизводимость | Docker, pinned Python dependencies, contest evidence docs |

---

## Быстрый старт

### 1. Поднять Ollama

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:4b
```

### 2. Установить зависимости

```bash
python -m pip install -r requirements.txt
```

### 3. Запустить GUI

```bash
python main.py
```

### 4. Запустить judged API

```bash
python -m assistant.api
```

---

## Рекомендуемый runtime

### Текущий tested/default contour

```text
model = qwen2.5-coder:7b
vision_model = qwen3-vl:4b
num_ctx = 4096
num_predict = 256
batch = 1
num_gpu = -1
keep_alive = 2h
parallel = 1
OLLAMA_NUM_PARALLEL = 1
```

Важно:

- это **tested tag**, а не digest-pinned model artifact;
- если нужен более строгий pinning, его надо фиксировать на стороне локального Ollama-контура;
- `luac` используется только если доступен в `PATH`.

---

## Архитектура judged path

```mermaid
flowchart LR
    A["POST /generate"] --> B["HTTP API"]
    B --> C["Orchestrator"]
    C --> D["LocalScriptService"]
    D --> E["LLM candidate generation"]
    E --> F["Validator"]
    F --> G["Ranking"]
    G --> H["Focused repair loop"]
    H --> I["Final select"]
    I --> J["Response: code"]
```

### Что происходит внутри

1. API принимает `prompt`
2. `orchestrator.generate_localscript_response(...)` запускает judged pipeline
3. `LocalScriptService`:
   - отсеивает не-Lua запросы;
   - при необходимости фиксирует assumptions для non-interactive режима;
   - генерирует кандидатов через LLM;
   - валидирует кандидатов;
   - ранжирует результаты;
   - запускает focused repair loop;
   - выбирает финальный ответ
4. API возвращает только поле `code`

Ключевые точки входа:

- [assistant/api/server.py](assistant/api/server.py)
- [assistant/core/orchestrator.py](assistant/core/orchestrator.py)
- [assistant/localscript/service.py](assistant/localscript/service.py)
- [assistant/localscript/validator.py](assistant/localscript/validator.py)

---

## Режимы работы

### 1. LocalScript / Lua

Основной judged-контур.  
Используется для генерации LocalScript/Lua-кода через LLM-only pipeline с validation и repair.

### 2. Chat-bot

Обычный режим общения с локальной моделью внутри GUI.

### 3. Agent mode

Режим для работы с проектом:

- чтение и изменение файлов;
- локальный retrieval;
- работа с контекстом выбранной папки;
- многошаговая помощь по проекту.

---

## Структура проекта

```text
assistant/
  api/
    openapi.yaml
    server.py
  config/
    settings.py
  core/
    agent.py
    orchestrator.py
  llm/
    client.py
    prompts.py
  localscript/
    benchmark.py
    eval_cases.py
    evaluator.py
    knowledge.py
    self_check.py
    service.py
    validator.py
  memory/
    memory_manager.py
  project_agent/
    service.py
  tools/
    file_tools.py
    search_tools.py
    vision_tools.py
  ui/
    window.py
artifacts/
assets/
docs/
tests/
main.py
Dockerfile
docker-compose.yml
requirements.txt
```

---

## Проверка API

### Health

```bash
curl http://127.0.0.1:8080/health
```

### Generate

```bash
curl -X POST http://127.0.0.1:8080/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Из полученного списка email получи последний. {\\\"wf\\\":{\\\"vars\\\":{\\\"emails\\\":[\\\"a\\\",\\\"b\\\"]}}}\"}"
```

---

## Benchmark и eval

### Smoke benchmark

Быстрый короткий прогон:

```bash
python -m assistant.localscript.benchmark --suite smoke
```

### Full eval

Расширенный eval с машинно-читаемым отчётом:

```bash
python -m assistant.localscript.benchmark --suite full --json-out artifacts/localscript_eval_report.json
```

Отчёт содержит:

- `pass_rate`
- breakdown по категориям
- распределение стратегий
- распределение `luac_status`
- причины падений
- метрики repair / assumptions

---

## Self-check

### Быстрая проверка judged-контура

```bash
python -m assistant.localscript.self_check
```

### Расширенная проверка с full eval

```bash
python -m assistant.localscript.self_check --full-eval
```

Self-check проверяет:

- локальность endpoint;
- judged runtime settings;
- наличие `luac`;
- smoke/full eval path;
- reproducibility-related checks.

---

## Тесты

### Deterministic tests

```bash
python -m unittest discover -s tests -v
```

### Optional live tests against real Ollama

```bash
set ASSISTANT_RUN_LIVE_OLLAMA_TESTS=1
python -m unittest discover -s tests -v
```

---

## Docker / compose

```bash
docker compose up --build
```

Поднимается:

- `ollama/ollama:0.12.4`
- загрузка `qwen2.5-coder:7b`
- API-контур на `:8080`

Базовый образ приложения:

- `python:3.11-slim-bookworm`

---

## Ограничения

- Проект не доказывает семантическую корректность всех возможных LocalScript-задач.
- `luac`-проверка включается только если `luac` доступен в `PATH`.
- Лёгкая модель ограничивает качество на сложных нестандартных кейсах.
- Надёжность достигается не только LLM, но и validator/ranking/repair контуром.
- Модель в Ollama не pinned по digest внутри репозитория.

---

## Для жюри и технической защиты

Отдельный краткий документ по judged pipeline, eval methodology и reproducibility:

- [docs/CONTEST_EVIDENCE.md](docs/CONTEST_EVIDENCE.md)

---

## Ключевые файлы

- [assistant/localscript/service.py](assistant/localscript/service.py)
- [assistant/localscript/validator.py](assistant/localscript/validator.py)
- [assistant/localscript/evaluator.py](assistant/localscript/evaluator.py)
- [assistant/localscript/self_check.py](assistant/localscript/self_check.py)
- [assistant/api/server.py](assistant/api/server.py)
- [assistant/api/openapi.yaml](assistant/api/openapi.yaml)
- [assistant/core/orchestrator.py](assistant/core/orchestrator.py)
- [assistant/project_agent/service.py](assistant/project_agent/service.py)
