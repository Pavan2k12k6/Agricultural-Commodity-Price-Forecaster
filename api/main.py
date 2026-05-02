"""FastAPI service: /predict, /train, /data."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agr_forecast.artifacts import load_metadata
from agr_forecast.config import DATA_RAW
from agr_forecast.predict_pipeline import backtest_overlap, forecasts_for_horizons, merged_actual_recent
from agr_forecast.train_pipeline import train_all

app = FastAPI(
    title="Agricultural Commodity Price Forecaster",
    description="SDG 2 — forecasting support for better sell timing (demo API).",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    commodity: str = Field(..., examples=["rice"])
    horizons: list[int] = Field(default_factory=lambda: [7, 15, 30])


class TrainResponse(BaseModel):
    status: str
    commodities: list[str]
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train_models():
    """Retrain ARIMAX + XGBoost on the CSV at data/raw/commodities.csv and persist joblib artifacts."""
    if not Path(DATA_RAW).exists():
        raise HTTPException(
            status_code=400,
            detail="No dataset found. Run scripts/generate_sample_data.py first.",
        )
    meta = train_all(DATA_RAW)
    return TrainResponse(
        status="completed",
        commodities=meta.get("commodities", []),
        message="Models saved under models/. See models/metadata.json for hold-out and TSCV metrics.",
    )


@app.post("/predict")
def predict(req: PredictRequest):
    """Return future paths for requested horizons (xgboost + arima)."""
    if not Path(DATA_RAW).exists():
        raise HTTPException(status_code=400, detail="Dataset missing.")
    try:
        out = forecasts_for_horizons(DATA_RAW, req.commodity.lower(), req.horizons or [7, 15, 30])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    extras = {
        "evaluation_note": "Use /backtest_overlay for a naive actual vs forecast chart payload.",
    }
    return JSONResponse(content={"commodity": req.commodity.lower(), **extras, "horizons": out})


@app.get("/data")
def get_processed_data(commodity: str | None = None, limit: int = 500):
    """Return recent processed rows (feature-engineered prices + weather subset)."""
    if not Path(DATA_RAW).exists():
        raise HTTPException(status_code=400, detail="Dataset missing.")
    limit = max(30, min(limit, 5000))
    md = load_metadata()
    available = md.get("commodities") or ["rice", "wheat", "maize"]
    comm = (commodity or available[0]).lower()
    df = merged_actual_recent(DATA_RAW, comm, last_n_days=limit)
    blob = []
    for _, r in df.iterrows():
        blob.append(
            {
                "date": r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"]),
                "commodity": r["commodity"],
                "price": float(r["price"]),
                "temp_c": float(r["temp_c"]),
                "rainfall_mm": float(r["rainfall_mm"]),
                "humidity_pct": float(r["humidity_pct"]),
            }
        )
    return {"records": blob, "meta": load_metadata().get("evaluation_metrics_holdout_cv")}


@app.get("/backtest_overlay")
def plot_payload(commodity: str = "rice", days: int = 30):
    """JSON series for comparing last-N actuals with model paths (demo visualization)."""
    if not Path(DATA_RAW).exists():
        raise HTTPException(status_code=400, detail="Dataset missing.")
    try:
        return backtest_overlap(DATA_RAW, commodity, holdout_days=min(max(days, 7), 180))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
