"""Feature engineering: lags, rolling stats, seasonal encodings."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month.astype(int)
    out["quarter"] = out["date"].dt.quarter.astype(int)
    out["dayofyear"] = out["date"].dt.dayofyear.astype(int)
    return out


def add_lags_and_rollings(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Lags / rolling statistics within each commodity, sorted by date."""
    out = df.sort_values(["commodity", "date"]).reset_index(drop=True)
    feats = []

    def _grp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        p = g[price_col]
        g[f"{price_col}_lag1"] = p.shift(1)
        g[f"{price_col}_lag7"] = p.shift(7)
        g[f"{price_col}_lag30"] = p.shift(30)
        g["roll_mean_7"] = p.shift(1).rolling(7, min_periods=1).mean()
        g["roll_mean_30"] = p.shift(1).rolling(30, min_periods=1).mean()
        return g

    for _, grp in out.groupby("commodity", sort=False):
        feats.append(_grp(grp))
    merged = pd.concat(feats, ignore_index=True)
    return merged.sort_values(["commodity", "date"]).reset_index(drop=True)


def supervised_feature_columns() -> list[str]:
    """Column names used by XGBoost after engineering (excluding target/date/commodity)."""
    base = ["temp_c", "rainfall_mm", "humidity_pct", "month", "quarter", "dayofyear"]
    price_feats = ["price_lag1", "price_lag7", "price_lag30", "roll_mean_7", "roll_mean_30"]
    cyclic = ["month_sin", "month_cos"]
    return base + price_feats + cyclic


def build_supervised_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + lag/rolling features + cyclic month encoding (keep rows sorted)."""
    out = add_calendar_features(df.copy())
    out = add_lags_and_rollings(out)
    out = cyclic_month_encoding(out)
    return out


def rows_with_valid_supervised_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where lags could not be formed (warm-up period per series)."""
    cols = supervised_feature_columns()
    return df.dropna(subset=["price"] + cols).reset_index(drop=True)


def cyclic_month_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth seasonality signals (trees better capture circular calendar effects)."""
    out = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out
