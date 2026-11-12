"""Tools the agent can call.

Each tool has two parts:
- a JSON schema (OpenAI function-calling format, which LiteLLM translates for every provider);
- a Python function that returns a JSON string.

Tools never raise: errors are returned as text so the LLM can read them and try again.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import duckdb

from demand_copilot import services
from demand_copilot.db import describe_schema
from demand_copilot.sql_tool import UnsafeQueryError, run_query

MAX_SQL_ROWS = 50


def _get_schema() -> dict:
    return {"schema": describe_schema()}


def _run_sql(sql: str) -> dict:
    try:
        result = run_query(sql, max_rows=MAX_SQL_ROWS)
    except (UnsafeQueryError, duckdb.Error) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "columns": result.columns,
        "rows": result.rows,
        "truncated": result.truncated,
        "notes": result.notes,
    }


def _get_forecast(store_nbr: int, family: str) -> dict:
    result = services.get_forecast(int(store_nbr), family)
    if result is None:
        return {
            "error": f"No forecast for store {store_nbr} and family '{family}'.",
            "valid_families": services.list_families(),
        }
    return result


def _get_forecast_accuracy(family: str | None = None) -> dict:
    if not family:
        return {"summary": services.accuracy_summary(), "by_family": services.accuracy_by_family()}
    result = services.accuracy_for_family(family)
    if result is None:
        return {"error": f"Unknown family '{family}'.", "valid_families": services.list_families()}
    return result


TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "get_schema": _get_schema,
    "run_sql": _run_sql,
    "get_forecast": _get_forecast,
    "get_forecast_accuracy": _get_forecast_accuracy,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "List the database tables, what they contain and their columns. Call it before writing SQL."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run one read-only DuckDB SELECT query over historical sales, stores, holidays, oil prices "
                f"and forecasts. Returns at most {MAX_SQL_ROWS} rows, so aggregate in SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "A single DuckDB SELECT statement."}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Get the 16-day sales forecast (2017-08-16 to 2017-08-31) for one store and product family."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Store number, 1 to 54."},
                    "family": {"type": "string", "description": "Product family, e.g. BEVERAGES."},
                },
                "required": ["store_nbr", "family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_accuracy",
            "description": (
                "How accurate the forecasting model was in the backtest. "
                "RMSLE measures the size of the error "
                "(0 is perfect); bias_pct measures its direction (negative = under-forecasting). "
                "Without a family it returns the overall summary (model vs baseline RMSLE and the RMSLE "
                "for each of the 16 days of the forecast horizon) plus the metrics of every family."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    # Some providers send null for "all families"; strict validators reject it unless allowed.
                    "family": {"type": ["string", "null"], "description": "Optional product family."}
                },
            },
        },
    },
]


def call_tool(name: str, arguments: str | dict) -> str:
    """Run a tool by name with JSON arguments and return its result as a JSON string."""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {sorted(TOOL_FUNCTIONS)}"})
    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Arguments are not valid JSON: {exc}"})
    try:
        result = TOOL_FUNCTIONS[name](**kwargs)
    except TypeError as exc:
        return json.dumps({"error": f"Invalid arguments for {name}: {exc}"})
    return json.dumps(result, default=str)
