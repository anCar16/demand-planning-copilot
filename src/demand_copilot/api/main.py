"""REST API over the forecasting system.

Run locally:
    uv run uvicorn demand_copilot.api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import duckdb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from demand_copilot import services
from demand_copilot.db import describe_schema, get_connection
from demand_copilot.sql_tool import DEFAULT_MAX_ROWS, UnsafeQueryError, run_query

app = FastAPI(
    title="Demand Planning Copilot API",
    description="Read-only access to sales history, 16-day forecasts and forecast accuracy.",
    version="0.1.0",
)


# ---------- Schemas ----------


class QueryRequest(BaseModel):
    sql: str = Field(
        ...,
        examples=[
            "SELECT family, SUM(sales) AS total FROM sales GROUP BY family ORDER BY total DESC LIMIT 5"
        ],
    )
    max_rows: int = Field(DEFAULT_MAX_ROWS, ge=1, le=1000)


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    notes: list[str]


class ForecastPoint(BaseModel):
    date: date
    forecast_sales: float
    onpromotion: int


class ForecastResponse(BaseModel):
    store_nbr: int
    family: str
    total_forecast: float
    points: list[ForecastPoint]


class FamilyAccuracy(BaseModel):
    family: str
    rmsle: float
    actual_sales: float
    predicted_sales: float
    bias_pct: float = Field(
        ..., description="(predicted - actual) / actual * 100. Negative means under-forecasting."
    )
    direction: Literal["under", "over", "exact"] = Field(
        ..., description="Whether the model under- or over-forecast, from unrounded values."
    )


class AccuracySummary(BaseModel):
    baseline_rmsle: float
    lightgbm_rmsle: float
    relative_improvement_pct: float
    rmsle_by_horizon_day: dict[int, float]


# ---------- Endpoints ----------


@app.get("/health")
def health() -> dict:
    with get_connection() as con:
        con.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict:
    """Tables, descriptions and columns. This is the context an LLM needs to write SQL."""
    return {"schema": describe_schema()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run a single read-only SELECT. Errors are returned with a clear message so an agent can retry."""
    try:
        result = run_query(request.sql, max_rows=request.max_rows)
    except UnsafeQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except duckdb.Error as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    return QueryResponse(**result.__dict__)


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    store_nbr: int = Query(..., ge=1, examples=[44]),
    family: str = Query(..., examples=["BEVERAGES"]),
) -> ForecastResponse:
    """16-day forecast for one store and product family."""
    result = services.get_forecast(store_nbr, family)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No forecast for store {store_nbr} and family '{family}'."
        )
    return ForecastResponse(**result)


@app.get("/accuracy/families", response_model=list[FamilyAccuracy])
def accuracy_by_family() -> list[FamilyAccuracy]:
    """Backtest accuracy per family, worst first."""
    return [FamilyAccuracy(**row) for row in services.accuracy_by_family()]


@app.get("/accuracy/families/{family}", response_model=FamilyAccuracy)
def accuracy_for_family(family: str) -> FamilyAccuracy:
    result = services.accuracy_for_family(family)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown family '{family}'.")
    return FamilyAccuracy(**result)


@app.get("/accuracy/summary", response_model=AccuracySummary)
def accuracy_summary() -> AccuracySummary:
    return AccuracySummary(**services.accuracy_summary())
