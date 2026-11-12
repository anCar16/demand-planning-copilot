"""Business logic shared by the REST API and the agent tools.

Keeping it here means the API and the LLM agent always return the same numbers.
"""

from __future__ import annotations

import json

from demand_copilot.config import FORECAST_OUTPUTS_DIR
from demand_copilot.db import get_connection


def normalize_family(family: str) -> str:
    return family.strip().upper()


def list_families() -> list[str]:
    with get_connection() as con:
        return [row[0] for row in con.execute("SELECT DISTINCT family FROM sales ORDER BY 1").fetchall()]


def get_forecast(store_nbr: int, family: str) -> dict | None:
    """16-day forecast for one store and family, or None if the pair does not exist."""
    family = normalize_family(family)
    with get_connection() as con:
        rows = con.execute(
            "SELECT date, forecast_sales, onpromotion FROM forecasts "
            "WHERE store_nbr = ? AND family = ? ORDER BY date",
            [store_nbr, family],
        ).fetchall()
    if not rows:
        return None
    points = [{"date": d.isoformat(), "forecast_sales": round(s, 2), "onpromotion": p} for d, s, p in rows]
    return {
        "store_nbr": store_nbr,
        "family": family,
        "total_forecast": round(sum(p["forecast_sales"] for p in points), 2),
        "points": points,
    }


def _family_accuracy(family: str, rmsle: float, actual: float, predicted: float) -> dict:
    bias = (predicted - actual) / actual * 100 if actual else 0.0
    return {
        "family": family,
        "rmsle": round(rmsle, 4),
        "actual_sales": round(actual, 2),
        "predicted_sales": round(predicted, 2),
        "bias_pct": round(bias, 2),
        # Computed from unrounded values: a bias of -0.03% must still read as under-forecasting.
        "direction": "under" if predicted < actual else "over" if predicted > actual else "exact",
    }


def accuracy_by_family() -> list[dict]:
    """Backtest accuracy per family, worst RMSLE first."""
    with get_connection() as con:
        rows = con.execute(
            "SELECT family, rmsle, actual_sales, predicted_sales FROM error_by_family ORDER BY rmsle DESC"
        ).fetchall()
    return [_family_accuracy(*row) for row in rows]


def accuracy_for_family(family: str) -> dict | None:
    family = normalize_family(family)
    with get_connection() as con:
        row = con.execute(
            "SELECT family, rmsle, actual_sales, predicted_sales FROM error_by_family WHERE family = ?",
            [family],
        ).fetchone()
    return _family_accuracy(*row) if row else None


def accuracy_summary() -> dict:
    metrics = json.loads((FORECAST_OUTPUTS_DIR / "lightgbm_validation_metrics.json").read_text())
    with get_connection() as con:
        by_day = dict(
            con.execute("SELECT day_of_forecast, rmsle FROM error_by_horizon ORDER BY 1").fetchall()
        )
    return {
        "baseline_rmsle": metrics["baseline_rmsle"],
        "lightgbm_rmsle": metrics["lightgbm_rmsle"],
        "relative_improvement_pct": metrics["relative_improvement_pct"],
        "rmsle_by_horizon_day": by_day,
    }
