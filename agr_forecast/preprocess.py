"""Handle missing values, outliers, scaling helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fill_missing_ffill_bfill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then back-fill numeric gaps within each commodity series."""
    out = df.copy()
    num_cols = ["price", "temp_c", "rainfall_mm", "humidity_pct"]
    for commodity, grp in out.groupby("commodity", sort=False):
        idx = grp.index
        out.loc[idx, num_cols] = grp[num_cols].ffill().bfill()
    return out


def clip_price_outliers_iqr(df: pd.DataFrame, col: str = "price", k: float = 1.5) -> pd.DataFrame:
    """Winsorize price using IQR per commodity (preserves tails less aggressively than dropping)."""
    out = df.copy()
    def _clip(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return s.clip(lo, hi)
    out[col] = out.groupby("commodity", sort=False)["price"].transform(_clip)
    return out


def fit_scaler_on_train(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit StandardScaler only on training rows for supervised model features."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].astype(float))
    return scaler


def scale_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]) -> np.ndarray:
    return scaler.transform(df[feature_cols].astype(float))


def normalize_series_for_arima(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Min-max normalize a positive series for numerical stability."""
    vmin, vmax = float(series.min()), float(series.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (series.astype(float) - vmin) / (vmax - vmin)
    return scaled, vmin, vmax


def invert_min_max(scaled: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return scaled * (vmax - vmin) + vmin
