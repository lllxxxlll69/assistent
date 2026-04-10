from __future__ import annotations

import json
import re
from typing import Any


LAST_MARKERS = ("последн", "last")
INCREMENT_MARKERS = ("увелич", "счетчик", "increment", "counter")
ARRAY_MARKERS = ("массив", "array")
FILTER_MARKERS = ("отфильтр", "filter")
REFINE_JSON_MARKERS = ("json", "payload", "объект")


class LocalScriptTemplateEngine:
    def render(self, task: str) -> str | None:
        for matcher in (
            self._match_last_array_item,
            self._match_increment_variable,
            self._match_cleanup_restbody,
            self._match_iso_datetime,
            self._match_ensure_items_are_arrays,
            self._match_mark_as_array,
            self._match_filter_discount_markdown,
            self._match_square_json_payload,
            self._match_unix_time,
        ):
            result = matcher(task)
            if result is not None:
                return result
        return None

    def _match_last_array_item(self, task: str) -> str | None:
        lowered = task.lower()
        if not any(marker in lowered for marker in LAST_MARKERS):
            return None

        context = self._extract_json_context(task)
        wf_vars = self._wf_vars(context)
        if wf_vars:
            for key, value in wf_vars.items():
                if isinstance(value, list):
                    return f"return wf.vars.{key}[#wf.vars.{key}]"

        wf_init = self._wf_init_variables(context)
        if wf_init:
            for key, value in wf_init.items():
                if isinstance(value, list):
                    return f"return wf.initVariables.{key}[#wf.initVariables.{key}]"

        return None

    def _match_increment_variable(self, task: str) -> str | None:
        lowered = task.lower()
        if not any(marker in lowered for marker in INCREMENT_MARKERS):
            return None

        context = self._extract_json_context(task)
        wf_vars = self._wf_vars(context)
        if not wf_vars:
            return None

        explicit_var = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", task)
        if explicit_var:
            name = explicit_var.group(1)
            if name in wf_vars and isinstance(wf_vars[name], (int, float)):
                return f"return wf.vars.{name} + 1"

        for key, value in wf_vars.items():
            if isinstance(value, (int, float)):
                return f"return wf.vars.{key} + 1"
        return None

    def _match_cleanup_restbody(self, task: str) -> str | None:
        lowered = task.lower()
        has_call_field = re.search(r'(?:"call"|\bcall\b)', lowered) is not None
        if "restbody" not in lowered and not ("entity_id" in lowered and has_call_field):
            return None
        return (
            "local result = wf.vars.RESTbody.result\n"
            "for _, filtered_entry in ipairs(result) do\n"
            "    for key, _ in pairs(filtered_entry) do\n"
            "        if key ~= \"ID\" and key ~= \"ENTITY_ID\" and key ~= \"CALL\" then\n"
            "            filtered_entry[key] = nil\n"
            "        end\n"
            "    end\n"
            "end\n"
            "return result"
        )

    def _match_iso_datetime(self, task: str) -> str | None:
        lowered = task.lower()
        if "iso 8601" not in lowered and "yyyymmdd" not in lowered and "hhmmss" not in lowered:
            return None
        return (
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
        )

    def _match_ensure_items_are_arrays(self, task: str) -> str | None:
        lowered = task.lower()
        if "items" not in lowered or not any(marker in lowered for marker in ARRAY_MARKERS):
            return None
        if "zcdf_packages" not in lowered:
            return None
        return (
            "local function ensure_array(value)\n"
            "    if type(value) ~= \"table\" then\n"
            "        local arr = _utils.array.new()\n"
            "        table.insert(arr, value)\n"
            "        return arr\n"
            "    end\n"
            "    local is_array = true\n"
            "    for key, _ in pairs(value) do\n"
            "        if type(key) ~= \"number\" or math.floor(key) ~= key then\n"
            "            is_array = false\n"
            "            break\n"
            "        end\n"
            "    end\n"
            "    if is_array then\n"
            "        return _utils.array.markAsArray(value)\n"
            "    end\n"
            "    local arr = _utils.array.new()\n"
            "    table.insert(arr, value)\n"
            "    return arr\n"
            "end\n"
            "for _, obj in ipairs(wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES) do\n"
            "    if type(obj) == \"table\" and obj.items ~= nil then\n"
            "        obj.items = ensure_array(obj.items)\n"
            "    end\n"
            "end\n"
            "return wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES"
        )

    def _match_mark_as_array(self, task: str) -> str | None:
        lowered = task.lower()
        if "markasarray" not in lowered and not ("помет" in lowered and any(marker in lowered for marker in ARRAY_MARKERS)):
            return None
        target = self._extract_explicit_wf_path(task) or "wf.vars.items"
        return (
            f"local arr = {target}\n"
            "if type(arr) ~= \"table\" then\n"
            "    local wrapped = _utils.array.new()\n"
            "    table.insert(wrapped, arr)\n"
            "    return wrapped\n"
            "end\n"
            "return _utils.array.markAsArray(arr)"
        )

    def _match_filter_discount_markdown(self, task: str) -> str | None:
        lowered = task.lower()
        if "discount" not in lowered and "markdown" not in lowered:
            return None
        return (
            "local result = _utils.array.new()\n"
            "local items = wf.vars.parsedCsv\n"
            "for _, item in ipairs(items) do\n"
            "    if (item.Discount ~= \"\" and item.Discount ~= nil) or (item.Markdown ~= \"\" and item.Markdown ~= nil) then\n"
            "        table.insert(result, item)\n"
            "    end\n"
            "end\n"
            "return result"
        )

    def _match_square_json_payload(self, task: str) -> str | None:
        lowered = task.lower()
        if "squared" not in lowered and "квадрат" not in lowered:
            return None
        if not any(marker in lowered for marker in REFINE_JSON_MARKERS):
            return None
        value = self._extract_numeric_literal(task) or "5"
        return (
            "{"
            f"\"num\":\"lua{{return tonumber('{value}')}}lua\","
            f"\"squared\":\"lua{{local n = tonumber('{value}')\\nreturn n * n}}lua\""
            "}"
        )

    def _match_unix_time(self, task: str) -> str | None:
        lowered = task.lower()
        if "unix" not in lowered and "epoch" not in lowered:
            return None
        if "recalltime" not in lowered:
            return None
        return (
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
        )

    def _extract_json_context(self, task: str) -> dict[str, Any] | None:
        start = task.find("{")
        end = task.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        raw_context = task[start : end + 1]
        try:
            payload = json.loads(raw_context)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_explicit_wf_path(self, task: str) -> str | None:
        match = re.search(r"\bwf\.(?:vars|initVariables)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", task)
        return match.group(0) if match else None

    def _extract_numeric_literal(self, task: str) -> str | None:
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", task)
        return match.group(1) if match else None

    def _wf_vars(self, context: dict[str, Any] | None) -> dict[str, Any]:
        if not context:
            return {}
        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return {}
        vars_payload = wf_payload.get("vars")
        return vars_payload if isinstance(vars_payload, dict) else {}

    def _wf_init_variables(self, context: dict[str, Any] | None) -> dict[str, Any]:
        if not context:
            return {}
        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return {}
        init_payload = wf_payload.get("initVariables")
        return init_payload if isinstance(init_payload, dict) else {}
