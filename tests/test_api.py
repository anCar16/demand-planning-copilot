import pytest
from fastapi.testclient import TestClient

from demand_copilot import config
from demand_copilot.api.main import app

pytestmark = pytest.mark.skipif(not config.DUCKDB_PATH.exists(), reason="Database not built")

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_schema_lists_forecast_tables():
    schema = client.get("/schema").json()["schema"]
    assert "forecasts" in schema and "validation_predictions" in schema


def test_query_returns_rows():
    response = client.post("/query", json={"sql": "SELECT COUNT(*) AS n FROM stores"})
    assert response.status_code == 200
    assert response.json()["rows"] == [[54]]


def test_query_rejects_writes():
    response = client.post("/query", json={"sql": "DELETE FROM stores"})
    assert response.status_code == 400


def test_query_reports_sql_errors():
    response = client.post("/query", json={"sql": "SELECT missing_column FROM stores"})
    assert response.status_code == 400
    assert "missing_column" in response.json()["detail"]


def test_unknown_forecast_returns_404():
    assert client.get("/forecast", params={"store_nbr": 1, "family": "NOT A FAMILY"}).status_code == 404


def test_accuracy_families_sorted_worst_first():
    families = client.get("/accuracy/families").json()
    rmsles = [f["rmsle"] for f in families]
    assert rmsles == sorted(rmsles, reverse=True)


def test_accuracy_summary():
    summary = client.get("/accuracy/summary").json()
    assert summary["lightgbm_rmsle"] < summary["baseline_rmsle"]


def test_tiny_bias_keeps_its_direction():
    # EGGS has a bias of about -0.03%: rounding alone would hide that it was under-forecast.
    families = client.get("/accuracy/families").json()
    under = [f for f in families if f["direction"] == "under"]
    assert all(f["predicted_sales"] <= f["actual_sales"] for f in under)
    assert all(f["direction"] in {"under", "over", "exact"} for f in families)
