"""Safe execution of SQL that comes from users or from an LLM.

Three layers of defense, from cheapest to strongest:
1. Parse the query with DuckDB's own parser and accept a single SELECT statement only.
2. Run it on a read-only connection, so writes fail even if the parser check is bypassed.
3. Disable external access, so the query cannot read arbitrary files from disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import duckdb

from demand_copilot.db import get_connection

DEFAULT_MAX_ROWS = 200


class UnsafeQueryError(ValueError):
    """Raised when a query is not a single read-only SELECT statement."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


def validate_query(sql: str) -> None:
    """Accept exactly one SELECT statement (CTEs with WITH are SELECT statements too)."""
    if not sql or not sql.strip():
        raise UnsafeQueryError("The query is empty.")
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.ParserException as exc:
        raise UnsafeQueryError(f"The query could not be parsed: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeQueryError(f"Expected exactly one statement, got {len(statements)}.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got {statements[0].type.name}.")


def _to_json_value(value):
    """Convert DuckDB values into JSON-friendly Python types."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def run_query(sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> QueryResult:
    """Validate and run a query, returning at most max_rows rows."""
    validate_query(sql)
    with get_connection(read_only=True, allow_external_access=False) as con:
        cursor = con.execute(sql)
        columns = [col[0] for col in cursor.description]
        fetched = cursor.fetchmany(max_rows + 1)

    truncated = len(fetched) > max_rows
    rows = [[_to_json_value(v) for v in row] for row in fetched[:max_rows]]
    notes = [f"Result truncated to the first {max_rows} rows."] if truncated else []
    return QueryResult(columns=columns, rows=rows, row_count=len(rows), truncated=truncated, notes=notes)
