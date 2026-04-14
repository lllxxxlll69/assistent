# VOMATIX CODE

Local-only assistant for generating, refining, and validating Lua / LocalScript with Ollama.

## Contest contour

Judged path:

- endpoint: `POST /generate`
- exact model tag: `qwen2.5-coder:7b`
- fixed Ollama parameters:
  - `num_ctx=4096`
  - `num_predict=256`
  - `batch=1`
  - `parallel=1`
  - `num_gpu=-1`
- hard limits enforced by code in judged mode:
  - full GPU load required (`no CPU offload`)
  - peak model VRAM budget must stay within `8_000_000_000` bytes
  - runtime must expose a digest
  - only one Ollama model may stay loaded in the contest contour

The judged contour is validated at runtime through `assistant/localscript/runtime.py` and `assistant/localscript/self_check.py`.
LocalScript-profile defaults now enforce the runtime guard in judged mode even outside docker-compose.


## Quick start

Requirements:

- Python `3.11`
- local Ollama
- NVIDIA GPU for the judged contour

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Pull the judged model:

```bash
ollama pull qwen2.5-coder:7b
```

Optional vision model for the desktop GUI:

```bash
ollama pull qwen3-vl:4b
```

Run the desktop app:

```bash
python main.py
```

Run the judged API:

```bash
python -m assistant.api
```

## Contest docker compose

The repo ships a compose contour aligned with the judging requirements:

```bash
docker compose up --build
```

Important compose properties:

- `OLLAMA_NUM_PARALLEL=1`
- `OLLAMA_MAX_LOADED_MODELS=1`
- `ASSISTANT_LOCALSCRIPT_MODEL=qwen2.5-coder:7b`
- `ASSISTANT_LOCALSCRIPT_RUNTIME_GUARD=true`
- `ASSISTANT_LOCALSCRIPT_REQUIRE_FULL_GPU=true`
- `ASSISTANT_LOCALSCRIPT_FULL_GPU_RATIO=0.98`
- `ASSISTANT_LOCALSCRIPT_MAX_VRAM_BYTES=8000000000`
- local syntax gate available through `luac` or bundled `luaparser`

Reference env file:

- [.env.contest.example](.env.contest.example)

If you want to hard-pin the pulled artifact, set `ASSISTANT_LOCALSCRIPT_EXPECTED_DIGEST` after the first pull. The self-check will then verify the digest exactly.

## API

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Generate:

```bash
curl -X POST http://127.0.0.1:8080/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Return orderId from workflow context. {\\\"wf\\\":{\\\"vars\\\":{\\\"orderId\\\":\\\"123\\\"}}}\"}"
```

The response contains:

- `code`
- `clarification_question`
- `metrics`

Notable metrics include:

- `selected_strategy`
- `execution_status`
- `sandbox_status`
- `repair_attempts_used`

If judged runtime constraints are violated, the API returns `503`.

Iterative refinement is available through the same endpoint by passing `context_messages` and `allow_clarification=true`.
Example:

```bash
curl -X POST http://127.0.0.1:8080/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Refine the previous result for JSON payload output.\",\"allow_clarification\":true,\"context_messages\":[{\"role\":\"assistant\",\"content\":\"return wf.vars.orderId\"},{\"role\":\"user\",\"content\":\"Return JSON payload with orderId only.\"}]}"
```

## Self-check

Run the contest self-check:

```bash
python -m assistant.localscript.self_check
```

Extended version with full eval:

```bash
python -m assistant.localscript.self_check --full-eval
```

The self-check verifies:

- fixed judged parameters
- model tag consistency between generic and judged profiles
- local runtime endpoint
- compose limits for `parallel=1` and one loaded model
- judged runtime guard enabled
- optional sandboxed Lua execution availability
- full GPU requirement enabled
- live Ollama runtime probe via `/api/version` and `/api/ps`
- local syntax gate availability (`luac` or `luaparser`)
- deterministic execution probes for supported LocalScript task families
- hidden-task sandbox probes for supported LocalScript task families
- model digest presence
- VRAM budget
- knowledge/eval exact-overlap audit
- semantic overlap audit between public guidance cards and eval prompts
- smoke eval path

## Eval

Smoke eval:

```bash
python -m assistant.localscript.evaluator --suite smoke --json-out artifacts/localscript_eval_report.json
```

Full eval:

```bash
python -m assistant.localscript.evaluator --suite full --json-out artifacts/localscript_eval_report.json
```

Compatibility wrapper:

```bash
python -m assistant.localscript.benchmark --suite full --json-out artifacts/localscript_eval_report.json
```

The report now includes:

- generation timestamp
- model and judged settings
- strategy distribution
- `luac` status distribution
- syntax-engine distribution
- execution-probe distribution
- sandbox distribution
- runtime-info presence
- exact prompt-overlap audit between public knowledge examples and public eval cases
- semantic overlap audit between public guidance cards and eval prompts

## Live E2E

Reproducible live contour:

```bash
python -m assistant.localscript.e2e --suite smoke --json-out artifacts/localscript_e2e_report.json
```

Extended live contour:

```bash
python -m assistant.localscript.e2e --suite full --json-out artifacts/localscript_e2e_report.json
```

## Local knowledge base

If the system uses retrieval or examples, they are local:

- `assistant/localscript/knowledge.py`
- `assistant/localscript/eval_cases.py`
- `assistant/tools/search_tools.py`

There is no external retrieval service in the judged contour.

## Key files

- `assistant/api/server.py`
- `assistant/core/orchestrator.py`
- `assistant/localscript/service.py`
- `assistant/localscript/runtime.py`
- `assistant/localscript/semantic_checks.py`
- `assistant/localscript/lua_sandbox.py`
- `assistant/localscript/syntax_gate.py`
- `assistant/localscript/validator.py`
- `assistant/localscript/evaluator.py`
- `assistant/localscript/e2e.py`
- `assistant/localscript/self_check.py`
- `docs/CONTEST_EVIDENCE.md`

## Tests

Deterministic test suite:

```bash
python -m unittest discover -s tests -v
```

Optional live Ollama tests:

```bash
set ASSISTANT_RUN_LIVE_OLLAMA_TESTS=1
python -m unittest discover -s tests -v
```
