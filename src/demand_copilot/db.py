"""Build and access the DuckDB database with Corporación Favorita sales.

Usage from the terminal:
    uv run python -m demand_copilot.db
"""

from __future__ import annotations

import duckdb

from demand_copilot.config import DUCKDB_PATH, FORECAST_OUTPUTS_DIR, RAW_DIR

# Target table -> (CSV file, columns with their types).
# Declaring types keeps DuckDB from guessing wrong (e.g. reading dates as text).
TABLES: dict[str, tuple[str, dict[str, str]]] = {
    "sales": (
        "train.csv",
        {
            "id": "BIGINT",
            "date": "DATE",
            "store_nbr": "INTEGER",
            "family": "VARCHAR",
            "sales": "DOUBLE",
            "onpromotion": "INTEGER",
        },
    ),
    "test_sales": (
        "test.csv",
        {
            "id": "BIGINT",
            "date": "DATE",
            "store_nbr": "INTEGER",
            "family": "VARCHAR",
            "onpromotion": "INTEGER",
        },
    ),
    "stores": (
        "stores.csv",
        {
            "store_nbr": "INTEGER",
            "city": "VARCHAR",
            "state": "VARCHAR",
            "type": "VARCHAR",
            "cluster": "INTEGER",
        },
    ),
    "transactions": (
        "transactions.csv",
        {"date": "DATE", "store_nbr": "INTEGER", "transactions": "INTEGER"},
    ),
    "oil": (
        "oil.csv",
        {"date": "DATE", "dcoilwtico": "DOUBLE"},
    ),
    "holidays_events": (
        "holidays_events.csv",
        {
            "date": "DATE",
            "type": "VARCHAR",
            "locale": "VARCHAR",
            "locale_name": "VARCHAR",
            "description": "VARCHAR",
            "transferred": "BOOLEAN",
        },
    ),
}

# Outputs of the upstream forecasting system (batch scoring), loaded as tables.
# In production a nightly job would refresh these; the copilot only reads them.
FORECAST_TABLES_SQL: dict[str, str] = {
    "forecasts": """
        SELECT t.date, t.store_nbr, t.family, t.onpromotion,
               GREATEST(f.sales, 0) AS forecast_sales
        FROM read_csv(?, header = true, columns = {'id': 'BIGINT', 'sales': 'DOUBLE'}) f
        JOIN test_sales t USING (id)
    """,
    "validation_predictions": """
        SELECT CAST(date AS DATE) AS date, CAST(store_nbr AS INTEGER) AS store_nbr,
               CAST(family AS VARCHAR) AS family,
               CAST(day_of_forecast AS INTEGER) AS day_of_forecast,
               sales AS actual_sales, prediction AS predicted_sales, squared_log_error
        FROM read_parquet(?)
    """,
    "error_by_family": "SELECT * FROM read_csv(?, header = true)",
    "error_by_horizon": "SELECT * FROM read_csv(?, header = true)",
}
FORECAST_FILES = {
    "forecasts": "submission_lightgbm.csv",
    "validation_predictions": "lightgbm_validation_predictions.parquet",
    "error_by_family": "lightgbm_error_by_family.csv",
    "error_by_horizon": "lightgbm_error_by_horizon.csv",
}

# Plain-language descriptions. They are passed to the LLM later
# so it understands what each table holds before writing SQL.
TABLE_DESCRIPTIONS = {
    "sales": "Historical daily sales by store and product family (2013-01-01 to 2017-08-15).",
    "test_sales": "Future dates to forecast (2017-08-16 to 2017-08-31), without sales.",
    "stores": "Metadata for the 54 stores: city, state (province), type and cluster.",
    "transactions": "Number of daily transactions per store.",
    "oil": "Daily WTI oil price (Ecuador's economy is oil-dependent).",
    "holidays_events": "National, regional and local holidays and events in Ecuador.",
    "sales_enriched": "View: the sales table joined with each store's metadata.",
    "forecasts": "LightGBM forecast of daily sales for the next 16 days (2017-08-16 to 2017-08-31).",
    "validation_predictions": (
        "Backtest of the last 16 known days (2017-07-31 to 2017-08-15): actual vs predicted "
        "sales per store, family and day of the forecast horizon."
    ),
    "error_by_family": "Backtest error (RMSLE) and total actual vs predicted sales per family.",
    "error_by_horizon": "Backtest error (RMSLE) by day of the 16-day forecast horizon.",
}


KAGGLE_COMPETITION = "store-sales-time-series-forecasting"


def missing_input_files() -> list[str]:
    """Input files that are not on disk yet, relative to the project root."""
    expected = [RAW_DIR / filename for filename, _ in TABLES.values()]
    expected += [FORECAST_OUTPUTS_DIR / filename for filename in FORECAST_FILES.values()]
    return [str(path.relative_to(RAW_DIR.parents[1])) for path in expected if not path.exists()]


def build_database(overwrite: bool = False) -> None:
    """Load the CSV files from data/raw into a DuckDB file."""
    if DUCKDB_PATH.exists() and not overwrite:
        print(f"Database already exists at {DUCKDB_PATH}. Use overwrite=True to rebuild it.")
        return

    missing = missing_input_files()
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n  - "
            + "\n  - ".join(missing)
            + f"\nDownload the Kaggle competition data ({KAGGLE_COMPETITION}) into data/raw/. "
            "See 'Data setup' in README.md."
        )

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUCKDB_PATH.unlink(missing_ok=True)

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        for table, (filename, columns) in TABLES.items():
            csv_path = (RAW_DIR / filename).as_posix()
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_csv(?, header = true, columns = ?)",
                [csv_path, columns],
            )
            _print_count(con, table)

        con.execute("""
            CREATE VIEW sales_enriched AS
            SELECT s.date, s.store_nbr, st.city, st.state, st.type AS store_type,
                   st.cluster, s.family, s.sales, s.onpromotion
            FROM sales s
            JOIN stores st USING (store_nbr)
        """)

        for table, query in FORECAST_TABLES_SQL.items():
            path = (FORECAST_OUTPUTS_DIR / FORECAST_FILES[table]).as_posix()
            con.execute(f"CREATE TABLE {table} AS {query}", [path])
            _print_count(con, table)

    size_mb = DUCKDB_PATH.stat().st_size / 1_048_576
    print(f"Database created at {DUCKDB_PATH} ({size_mb:.1f} MB)")


def _print_count(con: duckdb.DuckDBPyConnection, table: str) -> None:
    rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<24} {rows:>10,} rows")


def get_connection(read_only: bool = True, allow_external_access: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the database. Read-only by default so nothing (including the agent) can modify it.

    With allow_external_access=False, queries cannot read or write files on disk
    (read_csv, COPY, ATTACH...). This is the mode used for LLM-generated SQL.
    """
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError("Database not found. Run: uv run python -m demand_copilot.db")
    config = {} if allow_external_access else {"enable_external_access": False}
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only, config=config)


def describe_schema() -> str:
    """Return the schema as plain text: tables, descriptions and typed columns."""
    lines = []
    with get_connection() as con:
        for table, description in TABLE_DESCRIPTIONS.items():
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            col_text = ", ".join(f"{name} {dtype}" for name, dtype in cols)
            lines.append(f"- {table}: {description}\n  Columns: {col_text}")
    return "\n".join(lines)


if __name__ == "__main__":
    build_database(overwrite=True)
