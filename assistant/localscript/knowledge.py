from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


LOCALSCRIPT_RULES = """
You are an autonomous LocalScript/Lua code generation agent for a secure LowCode runtime.
Return only executable output. Do not add markdown fences, prose, or extra commentary.

Mandatory rules:
1. Build every answer from the current task and context only. Do not reuse canned templates.
2. Treat the result as code that will really run. Silently simulate execution before returning.
3. Internally check syntax, workflow paths, undefined names, data flow, and requested side effects.
4. If the task is incomplete in interactive mode, ask one concise clarification question instead of guessing.
5. In judged mode, make only minimal explicit assumptions and keep them consistent.
6. Run an internal repair loop in this order: syntax, logic, edge cases. If any step fails, rebuild the code.
7. Generate LocalScript-compatible Lua only.
8. Never use JsonPath such as $.foo or $[0].
9. Use direct workflow access through wf.vars and wf.initVariables.
10. Do not hardcode sample values from the prompt when workflow data is available.
11. When a new array must be created, use _utils.array.new().
12. When an existing table must be marked as an array, use _utils.array.markAsArray(arr).
13. Keep the result minimal, deterministic, and semantically correct.
14. If the user asks for a JSON payload, return JSON whose Lua values are wrapped as lua{...}lua strings.
15. Do not leave placeholders, TODOs, or generic scaffolding in the final code.
""".strip()


@dataclass(slots=True)
class LocalScriptExample:
    title: str
    prompt: str
    expected_code: str
    keywords: tuple[str, ...]


PUBLIC_EXAMPLES: tuple[LocalScriptExample, ...] = (
    LocalScriptExample(
        title="Return last workflow email",
        prompt=(
            'Return the newest email from wf.vars.inboxEmails. '
            '{"wf":{"vars":{"inboxEmails":["alpha@example.com","beta@example.com","gamma@example.com"]}}}'
        ),
        expected_code="return wf.vars.inboxEmails[#wf.vars.inboxEmails]",
        keywords=("email", "emails", "last", "latest", "array", "wf.vars"),
    ),
    LocalScriptExample(
        title="Increment attempts counter",
        prompt=(
            'Increase wf.vars.retryAttempts by one after each execution. '
            '{"wf":{"vars":{"retryAttempts":3}}}'
        ),
        expected_code="return wf.vars.retryAttempts + 1",
        keywords=("increment", "counter", "retry", "attempts", "plus one"),
    ),
    LocalScriptExample(
        title="Keep only key REST fields",
        prompt="Trim wf.vars.RESTbody.result so each entry keeps only ID, ENTITY_ID, and CALL.",
        expected_code=(
            "local result = wf.vars.RESTbody.result\n"
            "for _, filtered_entry in pairs(result) do\n"
            "    for key, _ in pairs(filtered_entry) do\n"
            "        if key ~= \"ID\" and key ~= \"ENTITY_ID\" and key ~= \"CALL\" then\n"
            "            filtered_entry[key] = nil\n"
            "        end\n"
            "    end\n"
            "end\n"
            "return result"
        ),
        keywords=("rest", "restbody", "entity_id", "call", "filter", "cleanup"),
    ),
    LocalScriptExample(
        title="Format workflow date as ISO 8601",
        prompt="Build an ISO 8601 timestamp from workflow DATUM and TIME fields in Lua.",
        expected_code=(
            "local datum = wf.vars.json.IDOC.ZCDF_HEAD.DATUM\n"
            "local time = wf.vars.json.IDOC.ZCDF_HEAD.TIME\n"
            "local function safe_sub(str, start_pos, end_pos)\n"
            "    local value = string.sub(str, start_pos, math.min(end_pos, #str))\n"
            "    return value ~= \"\" and value or \"00\"\n"
            "end\n"
            "local year = safe_sub(datum, 1, 4)\n"
            "local month = safe_sub(datum, 5, 6)\n"
            "local day = safe_sub(datum, 7, 8)\n"
            "local hour = safe_sub(time, 1, 2)\n"
            "local minute = safe_sub(time, 3, 4)\n"
            "local second = safe_sub(time, 5, 6)\n"
            "return string.format('%s-%s-%sT%s:%s:%s.00000Z', year, month, day, hour, minute, second)"
        ),
        keywords=("iso", "8601", "datum", "time", "timestamp", "format"),
    ),
    LocalScriptExample(
        title="Mark package items as arrays",
        prompt="Ensure every package entry in wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES keeps items as arrays.",
        expected_code=(
            "local function ensure_array(value)\n"
            "    if type(value) ~= \"table\" then\n"
            "        return {value}\n"
            "    end\n"
            "    local is_array = true\n"
            "    for key, _ in pairs(value) do\n"
            "        if type(key) ~= \"number\" or math.floor(key) ~= key then\n"
            "            is_array = false\n"
            "            break\n"
            "        end\n"
            "    end\n"
            "    return is_array and value or {value}\n"
            "end\n"
            "for _, obj in ipairs(wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES) do\n"
            "    if type(obj) == \"table\" and obj.items then\n"
            "        obj.items = ensure_array(obj.items)\n"
            "    end\n"
            "end\n"
            "return wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES"
        ),
        keywords=("items", "arrays", "packages", "markasarray", "zcdf_packages"),
    ),
    LocalScriptExample(
        title="Filter parsedCsv discounts and markdowns",
        prompt="Create a new array with parsedCsv rows where Discount or Markdown is present.",
        expected_code=(
            "local result = _utils.array.new()\n"
            "for _, item in ipairs(wf.vars.parsedCsv) do\n"
            "    if (item.Discount ~= \"\" and item.Discount ~= nil) "
            "or (item.Markdown ~= \"\" and item.Markdown ~= nil) then\n"
            "        table.insert(result, item)\n"
            "    end\n"
            "end\n"
            "return result"
        ),
        keywords=("discount", "markdown", "parsedcsv", "filter", "array"),
    ),
    LocalScriptExample(
        title="Return JSON payload with derived value",
        prompt="Return a JSON payload with fields num and squared for the provided value.",
        expected_code=(
            '{"num":"lua{return tonumber(\'5\')}lua",'
            '"squared":"lua{local n = tonumber(\'5\')\nreturn n * n}lua"}'
        ),
        keywords=("json", "payload", "square", "derived", "fields"),
    ),
    LocalScriptExample(
        title="Convert recallTime to unix timestamp",
        prompt=(
            'Convert wf.initVariables.recallTime to a unix timestamp. '
            '{"wf":{"initVariables":{"recallTime":"2024-01-20T09:10:11+03:00"}}}'
        ),
        expected_code=(
            "local y, m, d, h, mi, s = wf.initVariables.recallTime:match(\"^(%d%d%d%d)%-(%d%d)%-(%d%d)T(%d%d):(%d%d):(%d%d)\")\n"
            "return os.time({year = tonumber(y), month = tonumber(m), day = tonumber(d), hour = tonumber(h), min = tonumber(mi), sec = tonumber(s)})"
        ),
        keywords=("recalltime", "unix", "timestamp", "initvariables", "time"),
    ),
)


def normalize_prompt_text(text: str) -> str:
    return " ".join(text.casefold().split())


def find_exact_prompt_overlaps(
    cases: Sequence[tuple[str, str]],
    examples: Sequence[LocalScriptExample] = PUBLIC_EXAMPLES,
) -> list[dict[str, str]]:
    example_index = {normalize_prompt_text(example.prompt): example for example in examples}
    overlaps: list[dict[str, str]] = []
    for case_id, prompt in cases:
        normalized = normalize_prompt_text(prompt)
        matched = example_index.get(normalized)
        if matched is None:
            continue
        overlaps.append(
            {
                "case_id": case_id,
                "example_title": matched.title,
                "normalized_prompt": normalized,
            }
        )
    return overlaps


class LocalScriptKnowledgeBase:
    def __init__(self, examples: tuple[LocalScriptExample, ...] = PUBLIC_EXAMPLES) -> None:
        self.examples = examples

    def render_rules(self) -> str:
        return LOCALSCRIPT_RULES

    def select_examples(self, task: str, limit: int = 3) -> list[LocalScriptExample]:
        tokens = set(self._tokenize(task))
        scored: list[tuple[int, LocalScriptExample]] = []
        for example in self.examples:
            example_tokens = set(self._tokenize(example.prompt))
            example_tokens.update(token.casefold() for token in example.keywords)
            score = len(tokens & example_tokens)
            if score > 0:
                scored.append((score, example))

        if not scored:
            return list(self.examples[:limit])

        scored.sort(key=lambda item: item[0], reverse=True)
        return [example for _, example in scored[:limit]]

    def render_examples(self, task: str, limit: int = 3) -> str:
        selected = self.select_examples(task, limit=limit)
        rendered_blocks: list[str] = []
        for index, example in enumerate(selected, start=1):
            rendered_blocks.append(
                "\n".join(
                    [
                        f"Example {index}: {example.title}",
                        f"User task:\n{example.prompt}",
                        f"Expected output:\n{example.expected_code}",
                    ]
                )
            )
        return "\n\n".join(rendered_blocks)

    def render_generation_guidance(self, task: str, limit: int = 3) -> str:
        selected = self.select_examples(task, limit=limit)
        rendered_blocks: list[str] = []
        for index, example in enumerate(selected, start=1):
            rendered_blocks.append(
                "\n".join(
                    [
                        f"Reference {index}: {example.title}",
                        f"Task family cues: {', '.join(example.keywords[:6])}",
                        "Use this only as task-shape guidance. Do not copy a canned implementation.",
                    ]
                )
            )
        return "\n\n".join(rendered_blocks)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.casefold())
