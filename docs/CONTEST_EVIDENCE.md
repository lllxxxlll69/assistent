# Contest Evidence

## 1. Judged path

The judged interface is:

- `POST /generate`

Main code path:

- `assistant/api/server.py`
- `assistant/core/orchestrator.py`
- `assistant/localscript/service.py`

In judged mode the system does not ask follow-up questions. It either:

- applies minimal explicit assumptions
- generates candidates
- validates and repairs them
- returns final code
- or fails fast if the runtime contour is invalid

## 2. Runtime enforcement

The project now contains a strict runtime guard for judged mode.

Implementation:

- `assistant/localscript/runtime.py`
- `assistant/localscript/service.py`

The guard performs:

1. a warm-up request for the dedicated judged model
2. a live probe of Ollama `/api/version`
3. a live probe of Ollama `/api/ps`
4. an optional `nvidia-smi` sample
5. constraint checks on:
   - exact judged model tag
   - one loaded Ollama model only
   - matching context length
   - VRAM budget
   - full GPU ratio
   - digest presence
   - optional exact digest pin

If any required constraint fails, judged generation aborts with an error instead of silently running with CPU offload.

## 3. Fixed contest settings

Expected judged contour:

- model tag: `qwen2.5-coder:7b`
- `num_ctx=4096`
- `num_predict=256`
- `batch=1`
- `parallel=1`
- `num_gpu=-1`

Compose contour:

- `docker-compose.yml`
- `.env.contest.example`

Important compose limits:

- `OLLAMA_NUM_PARALLEL=1`
- `OLLAMA_MAX_LOADED_MODELS=1`
- `ASSISTANT_LOCALSCRIPT_RUNTIME_GUARD=true`
- `ASSISTANT_LOCALSCRIPT_REQUIRE_FULL_GPU=true`
- `ASSISTANT_LOCALSCRIPT_MAX_VRAM_BYTES=8000000000`

## 4. Local-only evidence

Generation path uses:

- local Ollama HTTP API
- local Python code
- local validation
- local knowledge examples
- local file and search tools

Tracked code does not depend on OpenAI, Anthropic, or another external AI vendor for generation.

## 5. Agentic iteration evidence

The system is not a single-shot answer.

Agentic behavior is implemented in:

- `assistant/localscript/service.py`

Behavior:

- clarification question in interactive mode
- assumption capture in judged mode
- multi-candidate generation
- ranking
- repair loop over invalid or weak candidates

## 6. Validation evidence

Validation lives in:

- `assistant/localscript/validator.py`

Checks include:

- no markdown fences
- no JsonPath
- correct `wf.vars` / `wf.initVariables` usage
- placeholder and template rejection
- workflow-shape heuristics
- JSON payload wrapper rules
- array helper rules
- optional `luac -p`

The validator is heuristic, but the project is explicit about that and does not present it as a full Lua interpreter.

## 7. Eval methodology and public eval integrity

Public eval suite:

- `assistant/localscript/eval_cases.py`
- `assistant/localscript/evaluator.py`

Public knowledge examples:

- `assistant/localscript/knowledge.py`

The repo now includes an exact-overlap audit between public eval prompts and public knowledge prompts. The audit is surfaced in:

- `python -m assistant.localscript.self_check`
- `python -m assistant.localscript.evaluator ...`

This prevents silent drift back to identical example/eval prompts.

## 8. Reproducibility workflow

Recommended verification flow:

1. `ollama pull qwen2.5-coder:7b`
2. configure contest env from `.env.contest.example`
3. `docker compose up --build`
4. `python -m assistant.localscript.self_check`
5. `python -m assistant.localscript.evaluator --suite full --json-out artifacts/localscript_eval_report.json`

## 9. Honest limits

The project does not claim:

- guaranteed correctness on every unseen LocalScript task
- semantic proof of all generated code
- byte-for-byte reproducibility unless a digest is explicitly pinned

What it does claim, and now enforces in code, is:

- local-only generation
- reproducible judged parameters
- runtime rejection of partial-offload judged runs
- visible evidence for validation, eval, and overlap checks
