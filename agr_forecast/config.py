"""Paths and constants for the forecasting pipeline."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Optionally point training/API at a CSV path (recommended for fetched live snapshots):
# PowerShell example: `$env:AGRI_DATA_CSV="C:\\path\\to\\commodities_live.csv"`
_explicit = os.environ.get("AGRI_DATA_CSV")
DATA_RAW = (
    Path(_explicit).expanduser().resolve()
    if _explicit
    else PROJECT_ROOT / "data" / "raw" / "commodities.csv"
)

# Default output path written by agr_forecast.live_fetch.fetch_live_snapshot_to_csv(...)
DATA_LIVE = PROJECT_ROOT / "data" / "raw" / "commodities_live.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default train ratio (sequential tail for test / validation)
DEFAULT_TEST_RATIO = 0.2

# Commodities in sample data
COMMODITIES = ("rice", "wheat", "maize")

# Random seed for reproducibility (synthetic jitter / splits that use stochastic components)
SEED = 42
