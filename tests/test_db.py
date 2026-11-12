import duckdb
import pytest

from demand_copilot import config
from demand_copilot.db import TABLE_DESCRIPTIONS, describe_schema, get_connection

pytestmark = pytest.mark.skipif(
    not config.DUCKDB_PATH.exists(),
    reason="Database not built; run: uv run python -m demand_copilot.db",
)


def test_all_tables_exist():
    with get_connection() as con:
        existing = {
            row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        }
    assert set(TABLE_DESCRIPTIONS) <= existing


def test_sales_key_is_unique():
    with get_connection() as con:
        dupes = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT date, store_nbr, family FROM sales
                GROUP BY ALL HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
    assert dupes == 0


def test_expected_series_count():
    with get_connection() as con:
        n_series = con.execute("SELECT COUNT(DISTINCT (store_nbr, family)) FROM sales").fetchone()[0]
    assert n_series == 54 * 33


def test_connection_is_read_only():
    with get_connection() as con, pytest.raises(duckdb.Error):
        con.execute("DELETE FROM stores")


def test_schema_mentions_every_table():
    schema = describe_schema()
    for table in TABLE_DESCRIPTIONS:
        assert table in schema


def test_missing_files_are_reported(tmp_path, monkeypatch):
    from demand_copilot import db

    monkeypatch.setattr(db, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(db, "FORECAST_OUTPUTS_DIR", tmp_path / "data" / "raw" / "forecast_outputs")
    monkeypatch.setattr(db, "DUCKDB_PATH", tmp_path / "data" / "processed" / "sales.duckdb")
    with pytest.raises(FileNotFoundError, match="train.csv"):
        db.build_database()
