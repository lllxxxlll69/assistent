from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalCase:
    id: str
    category: str
    prompt: str
    difficulty: str
    notes: str = ""
    smoke: bool = False
    required_substrings: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ("```", "$.", "$[")
    property_checks: tuple[str, ...] = ("validation_passed", "no_markdown", "no_jsonpath")
    context_messages: tuple[tuple[str, str], ...] = ()
    expected_assumptions_min: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


FULL_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="last_email_smoke",
        category="selection_last",
        difficulty="easy",
        smoke=True,
        prompt='Из полученного списка email получи последний. {"wf":{"vars":{"emails":["user1@example.com","user2@example.com","user3@example.com"]}}}',
        required_substrings=("wf.vars.emails", "#wf.vars.emails"),
        property_checks=("validation_passed", "no_markdown", "no_jsonpath", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="last_phone_variant",
        category="selection_last",
        difficulty="easy",
        prompt='Верни последний телефон из массива. {"wf":{"vars":{"phones":["111","222","333"]}}}',
        required_substrings=("wf.vars.phones", "#wf.vars.phones"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="last_token_english",
        category="selection_last",
        difficulty="easy",
        prompt='Get the last token from the workflow list. {"wf":{"vars":{"tokens":["aa","bb","cc"]}}}',
        required_substrings=("wf.vars.tokens", "#wf.vars.tokens"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="last_init_orders",
        category="selection_last",
        difficulty="medium",
        prompt='Верни последний заказ из стартового массива. {"wf":{"initVariables":{"orders":["A-1","A-2","A-3"]}}}',
        required_substrings=("wf.initVariables.orders", "#wf.initVariables.orders"),
        property_checks=("validation_passed", "contains_return", "uses_init_variables"),
    ),
    EvalCase(
        id="last_nested_step_id",
        category="selection_last",
        difficulty="hard",
        prompt='Верни последний stepId из wf.vars.audit.steps. {"wf":{"vars":{"audit":{"steps":["s1","s2","s3"]}}}}',
        required_substrings=("wf.vars.audit.steps", "#wf.vars.audit.steps"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
        notes="Nested workflow array path to reduce prompt memorization.",
    ),
    EvalCase(
        id="increment_try_count_smoke",
        category="increment",
        difficulty="easy",
        smoke=True,
        prompt='Увеличивай значение переменной try_count_n на каждой итерации. {"wf":{"vars":{"try_count_n":3}}}',
        required_substrings=("wf.vars.try_count_n", "+ 1"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="increment_retry_counter",
        category="increment",
        difficulty="easy",
        prompt='Increment retryCounter for every run. {"wf":{"vars":{"retryCounter":7}}}',
        required_substrings=("wf.vars.retryCounter", "+ 1"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="increment_error_count",
        category="increment",
        difficulty="easy",
        prompt='Увеличь errorCount на единицу. {"wf":{"vars":{"errorCount":11}}}',
        required_substrings=("wf.vars.errorCount", "+ 1"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="increment_attempts",
        category="increment",
        difficulty="easy",
        prompt='Increase attempts counter. {"wf":{"vars":{"attempts":1}}}',
        required_substrings=("wf.vars.attempts", "+ 1"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="increment_nested_retry_budget",
        category="increment",
        difficulty="hard",
        prompt='Increment wf.vars.metrics.retryBudget by one. {"wf":{"vars":{"metrics":{"retryBudget":4}}}}',
        required_substrings=("wf.vars.metrics.retryBudget", "+ 1"),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
        notes="Nested numeric path for symbolic increment coverage.",
    ),
    EvalCase(
        id="rest_cleanup_smoke",
        category="rest_cleanup",
        difficulty="medium",
        smoke=True,
        prompt="Для полученных данных из предыдущего REST запроса очисти значения переменных ID, ENTITY_ID, CALL.",
        required_substrings=("wf.vars.RESTbody.result", "ENTITY_ID", "CALL"),
        property_checks=("validation_passed", "rest_cleanup_pattern", "uses_wf_vars"),
    ),
    EvalCase(
        id="rest_cleanup_variant",
        category="rest_cleanup",
        difficulty="medium",
        prompt="Очисти RESTbody result и оставь только поля ID, ENTITY_ID и CALL.",
        required_substrings=("wf.vars.RESTbody.result", "filtered_entry[key] = nil"),
        property_checks=("validation_passed", "rest_cleanup_pattern", "uses_wf_vars"),
    ),
    EvalCase(
        id="rest_cleanup_english",
        category="rest_cleanup",
        difficulty="medium",
        prompt="For the previous REST response keep only ID, ENTITY_ID and CALL fields.",
        required_substrings=("wf.vars.RESTbody.result", "ENTITY_ID", "CALL"),
        property_checks=("validation_passed", "rest_cleanup_pattern", "uses_wf_vars"),
    ),
    EvalCase(
        id="rest_cleanup_call_field",
        category="rest_cleanup",
        difficulty="medium",
        prompt="Удаляй из результата RESTbody все поля, кроме ID, ENTITY_ID, CALL.",
        required_substrings=("wf.vars.RESTbody.result", "filtered_entry[key] = nil"),
        property_checks=("validation_passed", "rest_cleanup_pattern", "uses_wf_vars"),
    ),
    EvalCase(
        id="iso_datetime_smoke",
        category="datetime_iso",
        difficulty="medium",
        prompt="Преобразуй время из формата YYYYMMDD и HHMMSS в строку ISO 8601 с использованием Lua.",
        required_substrings=("string.format", "DATUM", "TIME"),
        property_checks=("validation_passed", "iso_8601_pattern", "uses_wf_vars"),
    ),
    EvalCase(
        id="iso_datetime_variant",
        category="datetime_iso",
        difficulty="medium",
        prompt="Convert YYYYMMDD + HHMMSS from workflow data into ISO 8601 string.",
        required_substrings=("string.format", "DATUM", "TIME"),
        property_checks=("validation_passed", "iso_8601_pattern", "uses_wf_vars"),
    ),
    EvalCase(
        id="ensure_items_are_arrays",
        category="array_helpers",
        difficulty="medium",
        prompt="Сделай так, чтобы все элементы items в ZCDF_PACKAGES всегда были массивами.",
        required_substrings=("_utils.array.new()", "_utils.array.markAsArray", "obj.items"),
        property_checks=("validation_passed", "uses_array_helper_new", "uses_mark_as_array", "uses_wf_vars"),
    ),
    EvalCase(
        id="ensure_items_arrays_variant",
        category="array_helpers",
        difficulty="medium",
        prompt="Нужно, чтобы items в ZCDF_PACKAGES всегда интерпретировались как массив.",
        required_substrings=("_utils.array.new()", "_utils.array.markAsArray", "ZCDF_PACKAGES"),
        property_checks=("validation_passed", "uses_array_helper_new", "uses_mark_as_array", "uses_wf_vars"),
    ),
    EvalCase(
        id="mark_as_array_prompt",
        category="array_helpers",
        difficulty="easy",
        prompt='Пометь wf.vars.items как массив через markAsArray. {"wf":{"vars":{"items":[{"id":1}]}}}',
        required_substrings=("wf.vars.items", "_utils.array.markAsArray"),
        property_checks=("validation_passed", "uses_mark_as_array", "contains_return"),
    ),
    EvalCase(
        id="mark_as_array_explicit_path",
        category="array_helpers",
        difficulty="easy",
        prompt='Use markAsArray for wf.initVariables.packages. {"wf":{"initVariables":{"packages":[{"id":"p1"}]}}}',
        required_substrings=("wf.initVariables.packages", "_utils.array.markAsArray"),
        property_checks=("validation_passed", "uses_mark_as_array"),
    ),
    EvalCase(
        id="filter_discount_markdown_smoke",
        category="array_filter",
        difficulty="medium",
        smoke=True,
        prompt="Отфильтруй элементы из массива, чтобы включить только те, у которых есть значения в Discount или Markdown.",
        required_substrings=("_utils.array.new()", "wf.vars.parsedCsv", "table.insert"),
        property_checks=("validation_passed", "uses_array_helper_new", "uses_wf_vars"),
    ),
    EvalCase(
        id="filter_discount_markdown_variant",
        category="array_filter",
        difficulty="medium",
        prompt="Оставь только строки parsedCsv, где заполнены Discount или Markdown.",
        required_substrings=("_utils.array.new()", "wf.vars.parsedCsv", "Discount"),
        property_checks=("validation_passed", "uses_array_helper_new", "uses_wf_vars"),
    ),
    EvalCase(
        id="filter_discount_markdown_english",
        category="array_filter",
        difficulty="medium",
        prompt="Filter parsedCsv and keep rows where Discount or Markdown is present.",
        required_substrings=("_utils.array.new()", "wf.vars.parsedCsv", "Markdown"),
        property_checks=("validation_passed", "uses_array_helper_new", "uses_wf_vars"),
    ),
    EvalCase(
        id="square_json_payload_smoke",
        category="json_payload",
        difficulty="medium",
        prompt="Добавь переменную с квадратом числа и верни JSON payload с полями num и squared.",
        required_substrings=('"num":"lua{', '"squared":"lua{'),
        property_checks=("validation_passed", "json_payload_wrapped"),
    ),
    EvalCase(
        id="square_json_payload_variant",
        category="json_payload",
        difficulty="medium",
        prompt="Верни JSON объект с num и squared для числа 12.",
        required_substrings=('"num":"lua{', '"squared":"lua{'),
        property_checks=("validation_passed", "json_payload_wrapped"),
    ),
    EvalCase(
        id="selected_json_fields",
        category="json_payload",
        difficulty="medium",
        prompt='Верни JSON payload с полями orderId и customerEmail. {"wf":{"vars":{"orderId":"ORD-7","customerEmail":"a@example.com"}}}',
        required_substrings=('"orderId":"lua{', '"customerEmail":"lua{'),
        property_checks=("validation_passed", "json_payload_wrapped", "uses_wf_vars"),
    ),
    EvalCase(
        id="selected_init_json_fields",
        category="json_payload",
        difficulty="medium",
        prompt='Return JSON payload with fields sourceSystem and recallTime. {"wf":{"initVariables":{"sourceSystem":"sap","recallTime":"2024-01-20T09:10:11+03:00"}}}',
        required_substrings=('"sourceSystem":"lua{', '"recallTime":"lua{'),
        property_checks=("validation_passed", "json_payload_wrapped", "uses_init_variables"),
    ),
    EvalCase(
        id="unix_time_smoke",
        category="datetime_unix",
        difficulty="medium",
        smoke=True,
        prompt='Конвертируй время в переменной recallTime в unix-формат. {"wf":{"initVariables":{"recallTime":"2023-10-15T15:30:00+00:00"}}}',
        required_substrings=("wf.initVariables.recallTime", "os.time"),
        property_checks=("validation_passed", "unix_time_pattern", "uses_init_variables"),
    ),
    EvalCase(
        id="unix_time_variant",
        category="datetime_unix",
        difficulty="medium",
        prompt='Convert recallTime from initVariables to unix timestamp. {"wf":{"initVariables":{"recallTime":"2024-01-20T09:10:11+03:00"}}}',
        required_substrings=("wf.initVariables.recallTime", "os.time"),
        property_checks=("validation_passed", "unix_time_pattern", "uses_init_variables"),
    ),
    EvalCase(
        id="return_order_id",
        category="direct_return",
        difficulty="easy",
        prompt='Верни orderId из workflow контекста. {"wf":{"vars":{"orderId":"123"}}}',
        required_substrings=("return wf.vars.orderId",),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="return_customer_email",
        category="direct_return",
        difficulty="easy",
        prompt='Return customerEmail from workflow context. {"wf":{"vars":{"customerEmail":"test@example.com"}}}',
        required_substrings=("return wf.vars.customerEmail",),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="return_init_user_id",
        category="direct_return",
        difficulty="easy",
        prompt='Верни userId из initVariables. {"wf":{"initVariables":{"userId":"U-9"}}}',
        required_substrings=("return wf.initVariables.userId",),
        property_checks=("validation_passed", "contains_return", "uses_init_variables"),
    ),
    EvalCase(
        id="return_explicit_path",
        category="direct_return",
        difficulty="easy",
        prompt='Верни значение wf.vars.json.order.id как есть. {"wf":{"vars":{"json":{"order":{"id":"O-1"}}}}}',
        required_substrings=("return wf.vars.json.order.id",),
        property_checks=("validation_passed", "contains_return", "uses_wf_vars"),
    ),
    EvalCase(
        id="return_nested_init_path",
        category="direct_return",
        difficulty="hard",
        prompt='Return wf.initVariables.payload.customer.primaryEmail as is. {"wf":{"initVariables":{"payload":{"customer":{"primaryEmail":"a@example.com"}}}}}',
        required_substrings=("return wf.initVariables.payload.customer.primaryEmail",),
        property_checks=("validation_passed", "contains_return", "uses_init_variables"),
    ),
    EvalCase(
        id="judged_assumption_without_context",
        category="assumptions",
        difficulty="hard",
        prompt="Доработай скрипт и верни json payload с полем squared.",
        required_substrings=('"squared":"lua{' ,),
        property_checks=("no_markdown", "no_jsonpath"),
        expected_assumptions_min=1,
        notes="Проверка judged-mode assumptions без интерактивного уточнения.",
    ),
)


def get_eval_cases(*, smoke_only: bool = False) -> list[EvalCase]:
    if not smoke_only:
        return list(FULL_EVAL_CASES)
    return [case for case in FULL_EVAL_CASES if case.smoke]
