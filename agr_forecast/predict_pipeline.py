"""Load trained models and produce multi-horizon price forecasts (+ optional backtest overlays)."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from agr_forecast.artifacts import artifact_path, load_joblib
from agr_forecast.features import supervised_feature_columns
from agr_forecast.train_pipeline import prepare_frame


def _seasonal_future_weather(grp: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Build future exogenous rows by aligning same calendar day-of-month with historical medians (+ tiny noise)."""
    rng = pd.date_range(start + pd.Timedelta(days=1), periods=horizon, freq="D")
    ref = grp.copy()
    rows = []
    for d in rng:
        cand = ref[(ref["date"].dt.month == d.month) & (ref["date"].dt.day == d.day)]
        if len(cand) < 5:
            cand = ref[ref["date"].dt.month == d.month]
        med = cand[["temp_c", "rainfall_mm", "humidity_pct"]].median()
        jitter = pd.Series({"temp_c": np.random.randn() * 0.4, "rainfall_mm": abs(np.random.randn()) * 0.6})
        jitter["humidity_pct"] = np.random.randn() * 0.5
        t = float(med["temp_c"]) + float(jitter["temp_c"])
        r = float(max(0.0, med["rainfall_mm"] + float(jitter["rainfall_mm"])))
        h = float(np.clip(med["humidity_pct"] + float(jitter["humidity_pct"]), 5.0, 99.0))
        rows.append(pd.DataFrame({"date": [pd.Timestamp(d)], "temp_c": [t], "rainfall_mm": [r], "humidity_pct": [h]}))
    return pd.concat(rows, ignore_index=True)


def xgb_recursive_forecast_single(
    history_grp: pd.DataFrame,
    commodity: str,
    horizon: int,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    """Multi-step forecasting with XGB + StandardScaler persisted per commodity."""
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if history_grp.empty:
        raise ValueError(
            f"Not enough history for {commodity!r}: at least 1 row is required for XGBoost recursion."
        )

    scaler = load_joblib("scaler_xgb", commodity)
    model = load_joblib("xgb", commodity)
    feats = supervised_feature_columns()
    start = pd.Timestamp(history_grp["date"].max())
    fx = _seasonal_future_weather(history_grp, start=start, horizon=horizon)

    rng = pd.date_range(start + pd.Timedelta(days=1), periods=horizon, freq="D")
    pr_hist = deque(history_grp.sort_values("date")["price"].astype(float).tolist(), maxlen=400)

    preds: list[float] = []
    for i in range(horizon):
        d = rng[i]
        row_w = fx.iloc[i]
        pr = list(pr_hist)
        lag1 = pr[-1]
        lag7 = pr[-7] if len(pr) >= 7 else lag1
        lag30 = pr[-30] if len(pr) >= 30 else lag1
        tail7 = pr[-7:]
        tail30 = pr[-30:] if len(pr) >= 30 else tail7
        roll7 = float(np.mean(tail7))
        roll30 = float(np.mean(tail30))
        feat_row = pd.DataFrame(
            [
                {
                    "temp_c": float(row_w["temp_c"]),
                    "rainfall_mm": float(row_w["rainfall_mm"]),
                    "humidity_pct": float(row_w["humidity_pct"]),
                    "month": int(d.month),
                    "quarter": int((d.month - 1) // 3 + 1),
                    "dayofyear": int(d.dayofyear),
                    "price_lag1": lag1,
                    "price_lag7": lag7,
                    "price_lag30": lag30,
                    "roll_mean_7": roll7,
                    "roll_mean_30": roll30,
                    "month_sin": float(np.sin(2 * np.pi * int(d.month) / 12.0)),
                    "month_cos": float(np.cos(2 * np.pi * int(d.month) / 12.0)),
                }
            ]
        )
        x = scaler.transform(feat_row[feats].astype(float))
        y_hat = float(model.predict(x)[0])
        preds.append(y_hat)
        pr_hist.append(y_hat)

    return list(rng.values), np.asarray(preds)


def arima_forecast(history_grp: pd.DataFrame, commodity: str, horizon: int) -> tuple[list[pd.Timestamp], np.ndarray]:
    """ARIMAX multi-step forecasts using synthesized future weather rows."""
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if history_grp.empty:
        raise ValueError(
            f"Not enough history for {commodity!r}: at least 1 row is required for ARIMA forecasting."
        )

    bundle = load_joblib("arima", commodity)
    model = bundle["model"]
    ex_cols = bundle["exog_cols"]
    start = pd.Timestamp(history_grp["date"].max())
    wx = _seasonal_future_weather(history_grp, start=start, horizon=horizon)
    rng = pd.date_range(start + pd.Timedelta(days=1), periods=horizon, freq="D")
    extras = pd.DataFrame(
        {
            "month": rng.month,
            "quarter": (rng.month - 1) // 3 + 1,
            "dayofyear": rng.dayofyear,
        }
    )
    exog_future = pd.concat([wx[["temp_c", "rainfall_mm", "humidity_pct"]].reset_index(drop=True), extras], axis=1)
    ex_arr = exog_future[ex_cols].astype(float).values
    fc = model.get_forecast(steps=horizon, exog=ex_arr)
    return list(pd.DatetimeIndex(rng).values), np.asarray(fc.predicted_mean)


def forecasts_for_horizons(
    csv_path: str | Path,
    commodity: str,
    horizons: list[int],
) -> dict[int, dict[str, list]]:
    """Return per-horizon dict with dates + xgb + arima price paths."""
    np.random.seed(42)
    df = prepare_frame(csv_path)
    history = df[df["commodity"] == commodity.lower()].sort_values("date")
    if history.empty:
        raise ValueError(f"No rows found for commodity {commodity!r}")
    commodity = commodity.lower()

    missing: list[str] = []
    for kind in ("xgb", "arima", "scaler_xgb"):
        p = artifact_path(kind, commodity)
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Trained artifacts missing. Run POST /train or python scripts/train_models.py. Missing: "
            + "; ".join(missing)
        )

    results: dict[int, dict[str, object]] = {}
    for h in horizons:
        int_h = int(h)
        dates_x, pred_x = xgb_recursive_forecast_single(history, commodity, int_h)
        _, pred_a = arima_forecast(history, commodity, int_h)
        results[int_h] = {
            "date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates_x],
            "forecast_xgboost": np.asarray(pred_x, dtype=float).tolist(),
            "forecast_arima": np.asarray(pred_a, dtype=float).tolist(),
        }
    return results


def merged_actual_recent(csv_path: str | Path, commodity: str, last_n_days: int = 120) -> pd.DataFrame:
    """Subset of engineered history for dashboards / backtesting plots."""
    df = prepare_frame(csv_path)
    h = df[df["commodity"] == commodity.lower()].sort_values("date").tail(last_n_days)
    return h.reset_index(drop=True)


def backtest_overlap(
    csv_path: str | Path,
    commodity: str,
    holdout_days: int = 30,
) -> dict:
    """For visualization: reconstruct last-window walk-forward-ish comparison on holdout slice."""
    np.random.seed(42)
    df = prepare_frame(csv_path)
    g = df[df["commodity"] == commodity.lower()].sort_values("date")
    if g.empty:
        raise ValueError(f"No rows found for commodity {commodity!r}")
    if len(g) < 3:
        raise ValueError(
            f"Need at least 3 rows for backtest_overlap on {commodity!r}; got {len(g)}."
        )

    # Keep at least one train row and one test row.
    holdout_days = max(1, int(holdout_days))
    max_holdout = max(1, len(g) - 1)
    if len(g) <= holdout_days + 61:
        holdout_days = max(14, len(g) // 5)
    holdout_days = min(holdout_days, max_holdout)

    train_part = g.iloc[:-holdout_days]
    test_part = g.iloc[-holdout_days:]
    if train_part.empty or test_part.empty:
        raise ValueError(
            f"Not enough rows to split backtest for {commodity!r}: train={len(train_part)}, test={len(test_part)}"
        )
    commodity = commodity.lower()
    # Retrain surrogates on train_part only via statsmodels/XGB naive refit omitted for perf;
    # use persisted production models projecting into test horizon for visual comparison overlay.
    _, pred_x_prod = xgb_recursive_forecast_single(train_part.copy(), commodity, holdout_days)
    _, pred_a_prod = arima_forecast(train_part.copy(), commodity, holdout_days)
    return {
        "dates_actual": test_part["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": test_part["price"].astype(float).tolist(),
        "xgboost_approx": pred_x_prod.tolist(),
        "arima_approx": pred_a_prod.tolist(),
        "note": "Overlay uses production models seeded from shortened history for demo clarity.",
    }
