"""Central paths and configuration for the project."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EVALS_DIR = PROJECT_ROOT / "evals"
FORECAST_OUTPUTS_DIR = RAW_DIR / "forecast_outputs"
DUCKDB_PATH = PROCESSED_DIR / "sales.duckdb"

load_dotenv(PROJECT_ROOT / ".env")
