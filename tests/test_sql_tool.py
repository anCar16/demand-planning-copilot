import pytest

from demand_copilot import config
from demand_copilot.sql_tool import UnsafeQueryError, run_query, validate_query


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select family from sales limit 3",
        "WITH t AS (SELECT 1 AS x) SELECT * FROM t",
    ],
)
def test_select_queries_are_accepted(sql):
    validate_query(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DELETE FROM stores",
        "DROP TABLE sales",
        "INSERT INTO stores VALUES (99, 'x', 'x', 'x', 1)",
        "SELECT 1; DROP TABLE sales",
        "COPY stores TO 'stores.csv'",
        "ATTACH 'other.db'",
        "SELEC 1",
    ],
)
def test_unsafe_or_invalid_queries_are_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        validate_query(sql)


needs_db = pytest.mark.skipif(not config.DUCKDB_PATH.exists(), reason="Database not built")


@needs_db
def test_file_access_is_blocked():
    with pytest.raises(Exception, match="(?i)permission"):
        run_query("SELECT * FROM read_csv('pyproject.toml')")


@needs_db
def test_results_are_truncated():
    result = run_query("SELECT * FROM stores", max_rows=5)
    assert result.row_count == 5
    assert result.truncated
