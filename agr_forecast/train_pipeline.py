"""Train ARIMA/SARIMAX and XGBoost per commodity; evaluate on time-based hold-out; save models."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from agr_forecast.artifacts import save_joblib, save_metadata
from agr_forecast.config import DEFAULT_TEST_RATIO, SEED
from agr_forecast.evaluation import rmse_mae, time_series_cv_xgb_scores
from agr_forecast.features import build_supervised_features, rows_with_valid_supervised_targets, supervised_feature_columns
from agr_forecast.load_data import load_commodities_csv, split_by_time
from agr_forecast.preprocess import clip_price_outliers_iqr, fill_missing_ffill_bfill, fit_scaler_on_train

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")

# Exogenous columns for ARIMAX (aligned with engineered calendar columns)
ARIMA_EXOG_COLS = ["temp_c", "rainfall_mm", "humidity_pct", "month", "quarter", "dayofyear"]


def prepare_frame(csv_path: str | Path) -> pd.DataFrame:
    df = load_commodities_csv(Path(csv_path))
    df = fill_missing_ffill_bfill(df)
    df = clip_price_outliers_iqr(df)
    df = build_supervised_features(df)
    return df


def _fit_arima(endog: pd.Series, exog: pd.DataFrame):
    orders = [(1, 1, 1), (1, 0, 1), (0, 1, 0)]
    last_err = None
    for od in orders:
        try:
            m = SARIMAX(
                endog.astype(float),
                exog=exog.astype(float),
                order=od,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return m.fit(disp=False, maxiter=150)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"ARIMAX could not converge: {last_err}")


def train_all(csv_path, test_ratio: float = DEFAULT_TEST_RATIO) -> dict:
    """
    Phase 1 — report metrics using a chronological hold-out slice.
    Phase 2 — refit models on **full history** so API forecasts use freshest signal.
    """
    path = Path(csv_path)
    df_full = prepare_frame(path)
    train_df, test_df = split_by_time(df_full, test_ratio=test_ratio)
    supervised_cols = supervised_feature_columns()
    commodities = sorted(df_full["commodity"].unique())
    metrics: dict[str, dict] = {}

    # --- Metrics (evaluation only) ---
    for comm in commodities:
        tr = train_df[train_df["commodity"] == comm].sort_values("date")
        te = test_df[test_df["commodity"] == comm].sort_values("date")

        # ARIMAX: fitted on train, predict full test horizon with known test exogenous
        try:
            if len(tr) > 120 and len(te) > 5:
                fit_a = _fit_arima(tr["price"], tr[ARIMA_EXOG_COLS])
                exc = te[ARIMA_EXOG_COLS].astype(float).values
                yhat_a = np.asarray(fit_a.get_forecast(steps=len(te), exog=exc).predicted_mean)
                ar_hold = rmse_mae(te["price"].astype(float).values, yhat_a.astype(float))
            else:
                ar_hold = {"rmse": np.nan, "mae": np.nan, "note": "insufficient_series_length"}
        except Exception as e:
            ar_hold = {"rmse": np.nan, "mae": np.nan, "error": str(e)}

        tr_s = rows_with_valid_supervised_targets(tr.copy())
        te_s = rows_with_valid_supervised_targets(te.copy())
        if len(tr_s) > 150 and len(te_s) >= 10:
            scaler_m = fit_scaler_on_train(tr_s, supervised_cols)
            Xm = scaler_m.transform(tr_s[supervised_cols].astype(float))
            ym = tr_s["price"].astype(float).values
            xgb_m = XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=SEED,
                verbosity=0,
            )
            xgb_m.fit(Xm, ym)
            Xte_n = scaler_m.transform(te_s[supervised_cols].astype(float))
            pred_x = xgb_m.predict(Xte_n)
            x_hold = rmse_mae(te_s["price"].values, pred_x)
        else:
            x_hold = {"rmse": np.nan, "mae": np.nan, "note": "small_supervised_window"}

        hist_s = rows_with_valid_supervised_targets(df_full[df_full["commodity"] == comm].copy())
        if len(hist_s) >= 250:
            X_raw = hist_s[supervised_cols].astype(float).values
            y_raw = hist_s["price"].astype(float).values
            cv = time_series_cv_xgb_scores(X_raw, y_raw, n_splits=min(5, max(3, len(hist_s) // 140)))
        else:
            cv = {"cv_rmse_mean": np.nan, "cv_mae_mean": np.nan, "cv_folds": 0, "note": "short_series"}

        metrics[comm] = {"arima_holdout": ar_hold, "xgb_holdout": x_hold, "xgboost_time_series_cv": cv}

    # --- Production persistence (trained on ALL available sequential data) ---
    for comm in commodities:
        grp = df_full[df_full["commodity"] == comm].sort_values("date")

        bundle_a = {
            "model": _fit_arima(grp["price"], grp[ARIMA_EXOG_COLS]),
            "exog_cols": ARIMA_EXOG_COLS,
        }
        save_joblib(bundle_a, "arima", comm)

        sup = rows_with_valid_supervised_targets(grp.copy())
        scaler_fin = StandardScaler()
        scaler_fin.fit(sup[supervised_cols].astype(float))
        Xf = scaler_fin.transform(sup[supervised_cols].astype(float))
        yf = sup["price"].astype(float).values
        xgb_fin = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=SEED,
            verbosity=0,
        )
        xgb_fin.fit(Xf, yf)
        save_joblib(scaler_fin, "scaler_xgb", comm)
        save_joblib(xgb_fin, "xgb", comm)

    meta = {
        "commodities": commodities,
        "supervised_cols": supervised_cols,
        "arima_exog_cols": ARIMA_EXOG_COLS,
        "evaluation_metrics_holdout_cv": metrics,
        "data_path_used": str(path.resolve()),
        "production_note": "Saved XGB artifacts use StandardScaler fit on supervised rows.",
    }
    save_metadata(meta)
    return meta
