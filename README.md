<p align="center">
  <img src="assets/vomatix.png" width="140"/>
</p>

<h1>
  Локальный AI-агент<br>
  <span style="color:#4FC3F7;">VOMATIX CODE</span><br>
  для LocalScript
</h1>

Локальный AI-агент для генерации Lua/LocalScript-кода в защищённом контуре.

Решение рассчитано на приватную инфраструктуру:
- локальная open-source модель через Ollama
- отсутствие внешних AI-вендоров в runtime
- генерация с учётом правил LocalScript/LowCode
- автоматическая валидация и одна итерация автоисправления
- оконный интерфейс для итеративной работы
- HTTP API по контракту `POST /generate`

## Что делает проект

Агент принимает задачу на русском или английском языке, использует лёгкую локальную модель, генерирует LocalScript-совместимый Lua-код, валидирует результат и при необходимости делает одну автоматическую попытку исправления.

Решение адаптировано под LocalScript-домен:
- использует `wf.vars` и `wf.initVariables`
- запрещает JsonPath
- поддерживает `_utils.array.new()` и `_utils.array.markAsArray(arr)`
- включает локальные доменные примеры из публичной выборки
- поддерживает итеративное уточнение через desktop UI

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
    knowledge.py
    service.py
    templates.py
    validator.py
  memory/
    memory_manager.py
  tools/
    file_tools.py
    search_tools.py
    vision_tools.py
  ui/
    window.py
main.py
Dockerfile
docker-compose.yml
```

## Рекомендуемая модель

Точная команда для демо-контура:

```bash

ollama pull qwen2.5:3b
```

Фиксированные параметры генерации для LocalScript-режима:
- `num_ctx=4096`
- `num_predict=256`
- `batch=1`
- `parallel=1` на уровне деплоя через `OLLAMA_NUM_PARALLEL=1`

Эти значения зафиксированы в LocalScript pipeline и в Docker-конфигурации.

## Локальный запуск

1. Запустите Ollama.
2. Скачайте модель:

```bash

ollama pull qwen2.5:3b
```

3. Запустите оконное приложение:

```bash

python main.py
```

4. Запустите HTTP API:

```bash

python -m assistant.api
```

Проверка здоровья API:

```bash

curl http://127.0.0.1:8080/health
```

Пример генерации кода:

```bash

curl -X POST http://127.0.0.1:8080/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Из полученного списка email получи последний. {\\\"wf\\\":{\\\"vars\\\":{\\\"emails\\\":[\\\"a\\\",\\\"b\\\"]}}}\"}"
```

## Однострочный запуск через Docker

```bash

docker compose up --build
```

`docker-compose.yml` поднимает:
- Ollama
- загрузку модели через `ollama pull qwen2.5:3b`
- LocalScript API на порту `8080`

## Публичный benchmark

Локальная smoke-проверка на публичных задачах:

```bash

python -m assistant.localscript.benchmark
```

## Self-check

Проверка judging-контура, фиксированных параметров и публичного benchmark:

```bash

python -m assistant.localscript.self_check
```

## Тесты

Запуск автоматических проверок:

```bash

python -m unittest discover -s tests -v
```

## Контракт API

Контракт включён в репозиторий:

```text
assistant/api/openapi.yaml
```

Сервер реализует:
- `POST /generate`
- `GET /health`
- `GET /openapi.yaml`

## Важно

- Runtime использует только локальный Ollama.
- OpenAI, Anthropic и другие внешние AI API для генерации не используются.
- Оконный интерфейс сохранён для итеративной доработки результата.
- Основной judging-интерфейс для проверки решения: HTTP API.
