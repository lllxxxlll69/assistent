from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from assistant.localscript.evaluator import run_eval_suite


async def run_public_benchmark() -> list[dict[str, object]]:
    report = await run_eval_suite(smoke_only=True)
    return list(report["results"])


async def run_full_eval_report(*, json_out: str | Path | None = None) -> dict[str, object]:
    return await run_eval_suite(smoke_only=False, json_out=json_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LocalScript smoke or full benchmark.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    report = asyncio.run(
        run_eval_suite(
            smoke_only=args.suite == "smoke",
            json_out=args.json_out or None,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
