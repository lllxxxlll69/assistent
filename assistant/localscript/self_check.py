from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from assistant.app import build_backend
from assistant.config.settings import SettingsManager
from assistant.localscript.evaluator import run_eval_suite


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def _status(ok: bool) -> str:
    return "passed" if ok else "failed"


async def run_self_check(*, run_full_eval: bool = False) -> dict[str, object]:
    settings_manager = SettingsManager()
    backend = build_backend(settings_manager=settings_manager)
    settings = backend.settings_manager.get_settings()

    requirements_lines = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    docker_compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerfile_text = Path("Dockerfile").read_text(encoding="utf-8")
    checks: list[CheckResult] = [
        CheckResult(
            name="localscript_profile",
            status=_status(settings.assistant_profile.lower() == "localscript"),
            detail=f"assistant_profile={settings.assistant_profile}",
        ),
        CheckResult(
            name="fixed_context_size",
            status=_status(settings.localscript_context_size == 4096),
            detail=f"localscript_context_size={settings.localscript_context_size}",
        ),
        CheckResult(
            name="fixed_num_predict",
            status=_status(settings.localscript_num_predict == 256),
            detail=f"localscript_num_predict={settings.localscript_num_predict}",
        ),
        CheckResult(
            name="fixed_batch_size",
            status=_status(settings.batch_size == 1),
            detail=f"batch_size={settings.batch_size}",
        ),
        CheckResult(
            name="local_runtime_endpoint",
            status=_status(settings.api_url.startswith(("http://127.0.0.1", "http://localhost"))),
            detail=f"api_url={settings.api_url}",
        ),
        CheckResult(
            name="default_api_host_is_local",
            status=_status(settings.api_host in {"127.0.0.1", "localhost"}),
            detail=f"api_host={settings.api_host}",
        ),
        CheckResult(
            name="openapi_contract",
            status=_status(Path("assistant/api/openapi.yaml").exists()),
            detail="assistant/api/openapi.yaml",
        ),
        CheckResult(
            name="docker_compose",
            status=_status(Path("docker-compose.yml").exists()),
            detail="docker-compose.yml",
        ),
        CheckResult(
            name="pinned_python_dependencies",
            status=_status(all("==" in line for line in requirements_lines)),
            detail="requirements.txt uses exact pins" if requirements_lines else "requirements.txt is empty",
        ),
        CheckResult(
            name="docker_images_not_latest",
            status=_status(":latest" not in docker_compose_text and ":latest" not in dockerfile_text),
            detail="Checked Dockerfile and docker-compose.yml for ':latest'.",
        ),
    ]

    luac_path = shutil.which("luac")
    checks.append(
        CheckResult(
            name="luac_available",
            status="passed" if luac_path else "skipped_with_reason",
            detail=f"luac={luac_path}" if luac_path else "luac binary is not available in PATH.",
            required=False,
        )
    )

    try:
        smoke_report = await run_eval_suite(smoke_only=True)
        smoke_ok = smoke_report["cases_passed"] == smoke_report["cases_total"]
        checks.append(
            CheckResult(
                name="smoke_eval",
                status=_status(smoke_ok),
                detail=f"passed={smoke_report['cases_passed']}/{smoke_report['cases_total']}",
            )
        )
    except Exception as exc:
        smoke_report = {"error": str(exc)}
        checks.append(CheckResult(name="smoke_eval", status="failed", detail=str(exc)))

    if run_full_eval:
        try:
            full_report = await run_eval_suite(smoke_only=False)
            full_ok = full_report["pass_rate"] >= 80.0
            checks.append(
                CheckResult(
                    name="full_eval",
                    status=_status(full_ok),
                    detail=f"pass_rate={full_report['pass_rate']}",
                    required=False,
                )
            )
        except Exception as exc:
            full_report = {"error": str(exc)}
            checks.append(CheckResult(name="full_eval", status="failed", detail=str(exc), required=False))
    else:
        full_report = {"status": "skipped_with_reason", "detail": "Run with --full-eval to include the extended suite."}
        checks.append(
            CheckResult(
                name="full_eval",
                status="skipped_with_reason",
                detail="Run with --full-eval to include the extended suite.",
                required=False,
            )
        )

    ok = all(check.status == "passed" for check in checks if check.required)
    return {
        "ok": ok,
        "checks": [asdict(check) for check in checks],
        "settings": {
            "model": settings.model,
            "vision_model": settings.vision_model,
            "api_url": settings.api_url,
            "api_host": settings.api_host,
            "localscript_context_size": settings.localscript_context_size,
            "localscript_num_predict": settings.localscript_num_predict,
            "batch_size": settings.batch_size,
            "assistant_profile": settings.assistant_profile,
        },
        "smoke_report": smoke_report,
        "full_report": full_report,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contest self-check for the LocalScript judged contour.")
    parser.add_argument("--full-eval", action="store_true", help="Also run the extended evaluation suite.")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    report = asyncio.run(run_self_check(run_full_eval=args.full_eval))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
