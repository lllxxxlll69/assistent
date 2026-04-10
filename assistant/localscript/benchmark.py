from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from assistant.app import build_backend


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    prompt: str
    expected_substrings: tuple[str, ...]


PUBLIC_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="last_email",
        prompt=(
            "Из полученного списка email получи последний.\n"
            '{"wf":{"vars":{"emails":["user1@example.com","user2@example.com","user3@example.com"]}}}'
        ),
        expected_substrings=("wf.vars.emails", "#wf.vars.emails"),
    ),
    BenchmarkCase(
        name="increment_try_count",
        prompt='Увеличивай значение переменной try_count_n на каждой итерации. {"wf":{"vars":{"try_count_n":3}}}',
        expected_substrings=("wf.vars.try_count_n", "+ 1"),
    ),
    BenchmarkCase(
        name="rest_cleanup",
        prompt="Для полученных данных из предыдущего REST запроса очисти значения переменных ID, ENTITY_ID, CALL.",
        expected_substrings=("wf.vars.RESTbody.result", "ENTITY_ID", "CALL"),
    ),
    BenchmarkCase(
        name="filter_discount_markdown",
        prompt="Отфильтруй элементы из массива, чтобы включить только те, у которых есть значения в Discount или Markdown.",
        expected_substrings=("_utils.array.new()", "wf.vars.parsedCsv", "table.insert"),
    ),
    BenchmarkCase(
        name="unix_time",
        prompt='Конвертируй время в переменной recallTime в unix-формат. {"wf":{"initVariables":{"recallTime":"2023-10-15T15:30:00+00:00"}}}',
        expected_substrings=("wf.initVariables.recallTime", "return"),
    ),
)


async def run_public_benchmark() -> list[dict[str, object]]:
    backend = build_backend()
    results: list[dict[str, object]] = []
    for case in PUBLIC_CASES:
        backend.memory_manager.create_session(f"benchmark:{case.name}")
        response = await backend.orchestrator.generate_localscript_response(
            case.prompt,
            allow_clarification=False,
            persist_memory=False,
            use_memory_context=False,
        )
        checks = {snippet: snippet in response.text for snippet in case.expected_substrings}
        results.append(
            {
                "name": case.name,
                "ok": all(checks.values()),
                "checks": checks,
                "response": response.text,
                "metrics": response.metrics,
            }
        )
    return results


def main() -> None:
    results = asyncio.run(run_public_benchmark())
    summary = {
        "passed": sum(1 for result in results if result["ok"]),
        "total": len(results),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
