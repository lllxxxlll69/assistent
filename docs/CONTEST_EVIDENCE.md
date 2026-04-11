# Contest Evidence Notes

## 1. Что именно является judged path

Основной judged-интерфейс:
- `POST /generate`

Реализация:
- `assistant/api/server.py`
- `assistant/core/orchestrator.py`
- `assistant/localscript/service.py`

Judged mode запускается без интерактивных уточнений. Если вход неоднозначен, сервис:
- не уходит в бесконечные вопросы;
- фиксирует безопасные допущения во внутреннем trace;
- пытается выбрать минимально рискованную интерпретацию.

## 2. Что реально делает LocalScript pipeline

Pipeline состоит из следующих этапов:

1. mode guard для не-Lua запросов
2. clarification/assumption decision
3. LLM candidate generation
4. validation
5. ranking
6. focused repair loop
7. final quality gate

Файлы:
- `assistant/localscript/service.py`
- `assistant/localscript/knowledge.py`
- `assistant/localscript/validator.py`

## 3. Что именно проверяет validator

Validator не является полноценным интерпретатором Lua, но проверяет:
- отсутствие markdown fences
- запрет JsonPath
- корректное использование `wf.vars` / `wf.initVariables`
- обнаружение hardcoded sample values
- правила `_utils.array.new()` и `_utils.array.markAsArray(...)`
- JSON payload с `lua{...}lua`
- пустые контейнеры вместо кода
- placeholders
- базовую структурную целостность
- `luac -p`, если `luac` доступен в `PATH`

Важно:
- если `luac` недоступен, это не скрывается, а отражается в `luac_status`
- validator остаётся эвристическим и не доказывает семантику для всех задач

## 4. Eval methodology

В репозитории есть два уровня оценки:

- smoke suite: быстрый набор критичных кейсов
- full eval suite: расширенный набор категорий

Файлы:
- `assistant/localscript/eval_cases.py`
- `assistant/localscript/evaluator.py`
- `assistant/localscript/benchmark.py`

Eval-отчёт содержит:
- total / passed / pass_rate
- category breakdown
- strategy distribution
- luac status distribution
- failure reasons
- repair / assumptions metrics

## 5. Reproducibility и local-only

В runtime проект использует:
- локальный Ollama endpoint
- локальные JSON settings/history
- локальные file/search tools

В tracked-коде нет вызовов OpenAI/Anthropic SDK.

Дополнительно:
- `requirements.txt` pinned по exact versions
- `docker-compose.yml` не использует `latest`
- `self_check` отдельно проверяет локальность endpoint и smoke path

Ограничения reproducibility:
- Docker images pinned по tag, но не по digest
- Ollama model tag указан, но digest модели не зафиксирован в репозитории
- отсутствие `luac` на машине понижает строгость синтаксической проверки

## 6. Что нельзя честно утверждать

Нельзя утверждать, что:
- проект гарантирует высокий pass rate на неизвестной закрытой выборке;
- все сгенерированные ответы semantically correct;
- reproduсibility абсолютна до байта;
- GPU-only / no CPU offload автоматически доказаны только кодом репозитория.

Эти ограничения нужно проговаривать на защите явно.
