from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from assistant.app import build_backend
from assistant.localscript.benchmark import run_public_benchmark


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    severity: str = "error"


async def run_self_check() -> dict[str, object]:
    backend = build_backend()
    settings = backend.settings_manager.get_settings()
    checks: list[CheckResult] = [
        CheckResult(
            name="localscript_profile",
            ok=settings.assistant_profile.lower() == "localscript",
            detail=f"assistant_profile={settings.assistant_profile}",
        ),
        CheckResult(
            name="fixed_context_size",
            ok=settings.localscript_context_size == 4096,
            detail=f"localscript_context_size={settings.localscript_context_size}",
        ),
        CheckResult(
            name="fixed_num_predict",
            ok=settings.localscript_num_predict == 256,
            detail=f"localscript_num_predict={settings.localscript_num_predict}",
        ),
        CheckResult(
            name="fixed_batch_size",
            ok=settings.batch_size == 1,
            detail=f"batch_size={settings.batch_size}",
        ),
        CheckResult(
            name="local_runtime_endpoint",
            ok=settings.api_url.startswith(("http://127.0.0.1", "http://localhost")),
            detail=f"api_url={settings.api_url}",
        ),
        CheckResult(
            name="openapi_contract",
            ok=Path("assistant/api/openapi.yaml").exists(),
            detail="assistant/api/openapi.yaml",
        ),
        CheckResult(
            name="docker_compose",
            ok=Path("docker-compose.yml").exists(),
            detail="docker-compose.yml",
        ),
        CheckResult(
            name="luac_available",
            ok=shutil.which("luac") is not None,
            detail=f"luac={'found' if shutil.which('luac') else 'missing'}",
            severity="warning",
        ),
    ]

    try:
        benchmark_results = await run_public_benchmark()
        passed = sum(1 for item in benchmark_results if item["ok"])
        checks.append(
            CheckResult(
                name="public_benchmark",
                ok=passed == len(benchmark_results),
                detail=f"passed={passed}/{len(benchmark_results)}",
            )
        )
    except Exception as exc:  # pragma: no cover - requires runtime model
        benchmark_results = []
        checks.append(CheckResult(name="public_benchmark", ok=False, detail=str(exc)))

    ok = all(check.ok or check.severity == "warning" for check in checks)
    return {
        "ok": ok,
        "checks": [asdict(check) for check in checks],
        "settings": {
            "model": settings.model,
            "api_url": settings.api_url,
            "localscript_context_size": settings.localscript_context_size,
            "localscript_num_predict": settings.localscript_num_predict,
            "batch_size": settings.batch_size,
            "assistant_profile": settings.assistant_profile,
        },
        "benchmark_results": benchmark_results,
    }


def main() -> None:
    report = asyncio.run(run_self_check())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
