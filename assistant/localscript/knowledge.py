from __future__ import annotations

import re
from dataclasses import dataclass


LOCALSCRIPT_RULES = """
You are a LocalScript code generation agent for a secure LowCode environment.
Return only executable output, without explanations, markdown fences, or extra commentary.

Mandatory rules:
1. Generate LocalScript-compatible Lua only.
2. Never use JsonPath such as $.foo or $[0].
3. Use direct access through wf.vars for workflow variables.
4. Use wf.initVariables for startup variables.
5. Do not hardcode sample values from the input context when wf.vars or wf.initVariables are available.
6. When a new array must be created, use _utils.array.new().
7. When an existing table must be marked as an array, use _utils.array.markAsArray(arr).
8. Keep the result minimal and runnable.
9. If the user asks for a JSON payload, return JSON whose Lua values are wrapped as lua{...}lua strings.
10. Before returning, self-check the result against the rules and fix obvious mistakes.
""".strip()


@dataclass(slots=True)
class LocalScriptExample:
    title: str
    prompt: str
    expected_code: str
    keywords: tuple[str, ...]


PUBLIC_EXAMPLES: tuple[LocalScriptExample, ...] = (
    LocalScriptExample(
        title="Last email from wf.vars array",
        prompt=(
            "Из полученного списка email получи последний.\n"
            '{"wf":{"vars":{"emails":["user1@example.com","user2@example.com","user3@example.com"]}}}'
        ),
        expected_code="return wf.vars.emails[#wf.vars.emails]",
        keywords=("email", "emails", "последний", "array", "массив"),
    ),
    LocalScriptExample(
        title="Increment retry counter",
        prompt=(
            "Увеличивай значение переменной try_count_n на каждой итерации.\n"
            '{"wf":{"vars":{"try_count_n":3}}}'
        ),
        expected_code="return wf.vars.try_count_n + 1",
        keywords=("try_count_n", "счетчик", "counter", "increment", "итерации"),
    ),
    LocalScriptExample(
        title="Filter object keys after REST call",
        prompt=(
            "Для полученных данных из предыдущего REST запроса очисти значения переменных "
            "ID, ENTITY_ID, CALL."
        ),
        expected_code=(
            "result = wf.vars.RESTbody.result\n"
            "for _, filtered_entry in pairs(result) do\n"
            "    for key, _ in pairs(filtered_entry) do\n"
            "        if key ~= \"ID\" and key ~= \"ENTITY_ID\" and key ~= \"CALL\" then\n"
            "            filtered_entry[key] = nil\n"
            "        end\n"
            "    end\n"
            "end\n"
            "return result"
        ),
        keywords=("rest", "entity_id", "call", "очисти", "filter"),
    ),
    LocalScriptExample(
        title="Convert date and time to ISO 8601",
        prompt=(
            "Преобразуй время из формата YYYYMMDD и HHMMSS в строку ISO 8601 "
            "с использованием Lua."
        ),
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
        keywords=("iso", "8601", "datum", "time", "yyyy", "hhmmss"),
    ),
    LocalScriptExample(
        title="Ensure items are arrays",
        prompt="Сделай так, чтобы все элементы items в ZCDF_PACKAGES всегда были массивами.",
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
        keywords=("items", "arrays", "packages", "zcdf_packages", "массивами"),
    ),
    LocalScriptExample(
        title="Filter discounts or markdowns",
        prompt=(
            "Отфильтруй элементы из массива, чтобы включить только те, "
            "у которых есть значения в Discount или Markdown."
        ),
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
        keywords=("discount", "markdown", "parsedcsv", "filter", "фильтрация"),
    ),
    LocalScriptExample(
        title="Add derived variable",
        prompt="Добавь переменную с квадратом числа.",
        expected_code=(
            '{"num":"lua{return tonumber(\'5\')}lua",'
            '"squared":"lua{local n = tonumber(\'5\')\nreturn n * n}lua"}'
        ),
        keywords=("square", "квадрат", "переменную", "json"),
    ),
    LocalScriptExample(
        title="ISO to unix time",
        prompt="Конвертируй время в переменной recallTime в unix-формат.",
        expected_code=(
            "local iso_time = wf.initVariables.recallTime\n"
            "if not iso_time or not iso_time:match(\"^%d%d%d%d%-%d%d%-%d%dT\") then\n"
            "    return nil\n"
            "end\n"
            "local year = tonumber(iso_time:sub(1, 4))\n"
            "local month = tonumber(iso_time:sub(6, 7))\n"
            "local day = tonumber(iso_time:sub(9, 10))\n"
            "local hour = tonumber(iso_time:sub(12, 13))\n"
            "local minute = tonumber(iso_time:sub(15, 16))\n"
            "local second = tonumber(iso_time:sub(18, 19))\n"
            "local timestamp = os.time({year = year, month = month, day = day, hour = hour, min = minute, sec = second})\n"
            "local sign, tz_hour, tz_minute = iso_time:match(\"([%+%-])(%d%d):(%d%d)$\")\n"
            "if sign and tz_hour and tz_minute then\n"
            "    local shift = tonumber(tz_hour) * 3600 + tonumber(tz_minute) * 60\n"
            "    if sign == \"+\" then\n"
            "        timestamp = timestamp - shift\n"
            "    else\n"
            "        timestamp = timestamp + shift\n"
            "    end\n"
            "end\n"
            "return timestamp"
        ),
        keywords=("recalltime", "unix", "epoch", "initvariables", "time"),
    ),
)


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
            example_tokens.update(token.lower() for token in example.keywords)
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

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.lower())
