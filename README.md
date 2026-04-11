<p align="center">
  <img src="assets/vomatix.png" width="140"/>
</p>

# Локальный AI-агент VOMATIX CODE для LocalScript

Локальный AI-агент для генерации Lua/LocalScript-кода в защищённом контуре.  
Проект ориентирован на judged-сценарий `POST /generate` и на локальную интерактивную работу через desktop GUI.

## Что реально есть в репозитории

- локальный runtime через Ollama, без внешних AI API в runtime
- отдельный LocalScript pipeline: rules-first prompts, валидация, ranking, repair loop
- HTTP API по контракту `POST /generate`
- desktop GUI для итеративной работы
- eval/benchmark контур с расширенным набором задач и JSON-отчётами
- self-check для judged-контура и воспроизводимости
- локальные file/search/vision tools

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
main.py
Dockerfile
docker-compose.yml
requirements.txt
docs/
  CONTEST_EVIDENCE.md
```

## Judged LocalScript pipeline

Целевой judged path:

1. `POST /generate` принимает `prompt`
2. API вызывает `orchestrator.generate_localscript_response(...)`
3. Orchestrator запускает `LocalScriptService`
4. `LocalScriptService` выполняет:
   - mode guard для не-Lua запросов
   - optional assumptions для non-interactive режима
   - LLM candidate generation
   - validation
   - ranking
   - focused repair loop
   - final select
5. API возвращает только поле `code`

Кодовые точки:
- [assistant/api/server.py](assistant/api/server.py)
- [assistant/core/orchestrator.py](assistant/core/orchestrator.py)
- [assistant/localscript/service.py](assistant/localscript/service.py)
- [assistant/localscript/validator.py](assistant/localscript/validator.py)

## Рекомендуемая модель

Текущая рекомендуемая модель для judged LocalScript path:

```bash
ollama pull qwen2.5-coder:7b
```

Текущий tested/default contour:
- `model = qwen2.5-coder:7b`
- `num_ctx = 4096`
- `num_predict = 256`
- `batch = 1`
- `parallel = 1` через `OLLAMA_NUM_PARALLEL=1`

Важно:
- это tested tag, а не digest-pinned artifact модели;
- если нужен более строгий pinning, его надо фиксировать на стороне окружения Ollama отдельно.

## Локальный запуск

1. Поднимите Ollama.
2. Скачайте модель:

```bash
ollama pull qwen2.5-coder:7b
```

3. Установите зависимости:

```bash
python -m pip install -r requirements.txt
```

4. Запустите GUI:

```bash
python main.py
```

5. Запустите judged API:

```bash
python -m assistant.api
```

## Проверка API

```bash
curl http://127.0.0.1:8080/health
```

```bash
curl -X POST http://127.0.0.1:8080/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Из полученного списка email получи последний. {\\\"wf\\\":{\\\"vars\\\":{\\\"emails\\\":[\\\"a\\\",\\\"b\\\"]}}}\"}"
```

## Docker / compose

```bash
docker compose up --build
```

Что поднимается:
- `ollama/ollama:0.12.4`
- загрузка `qwen2.5-coder:7b`
- API-контур на `:8080`

Базовый образ приложения:
- `python:3.11-slim-bookworm`

## Smoke benchmark

Быстрый запуск небольшого smoke-suite:

```bash
python -m assistant.localscript.benchmark --suite smoke
```

## Полный eval

Расширенный eval-набор с машинно-читаемым отчётом:

```bash
python -m assistant.localscript.benchmark --suite full --json-out artifacts/localscript_eval_report.json
```

Отчёт содержит:
- `pass_rate`
- breakdown по категориям
- распределение стратегий
- распределение `luac_status`
- причины падений
- метрики repair/assumptions

## Self-check

Быстрая проверка judged-контура:

```bash
python -m assistant.localscript.self_check
```

Расширенная проверка с полным eval:

```bash
python -m assistant.localscript.self_check --full-eval
```

## Тесты

Обычные deterministic tests:

```bash
python -m unittest discover -s tests -v
```

Опциональные live-тесты против реального Ollama:

```bash
set ASSISTANT_RUN_LIVE_OLLAMA_TESTS=1
python -m unittest discover -s tests -v
```

## Ограничения

- Проект не доказывает семантическую корректность всех возможных LocalScript-задач.
- `luac`-проверка включается только если `luac` доступен в `PATH`.
- Лёгкая модель ограничивает качество на сложных нестандартных кейсах.
- Часть надёжности достигается доменными правилами, валидацией и эвристиками, а не только LLM.
- Модель в Ollama не pinned по digest внутри репозитория.

## Для жюри

Отдельный краткий документ по judged pipeline, eval methodology и reproducibility:

- [docs/CONTEST_EVIDENCE.md](docs/CONTEST_EVIDENCE.md)
