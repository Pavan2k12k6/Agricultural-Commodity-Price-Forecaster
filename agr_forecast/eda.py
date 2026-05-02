"""Exploratory data analysis plots (time series, seasonality, correlation)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agr_forecast.features import build_supervised_features
from agr_forecast.load_data import load_commodities_csv
from agr_forecast.preprocess import clip_price_outliers_iqr, fill_missing_ffill_bfill


def run_eda_plots(csv_path: str | Path, out_dir: str | Path) -> Path:
    """
    Saves PNGs under out_dir:
    - timeseries_by_commodity.png
    - seasonal_monthly_avg.png
    - correlation_numeric.png
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_commodities_csv(csv_path)
    df = fill_missing_ffill_bfill(df)
    df = clip_price_outliers_iqr(df)
    eng = build_supervised_features(df)

    # Time series overlays
    fig, ax = plt.subplots(figsize=(11, 5))
    for c, grp in df.groupby("commodity"):
        ax.plot(grp["date"], grp["price"], label=c, lw=1.2)
    ax.set_title("Commodity market price trends (daily)")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    ts_path = out_dir / "timeseries_by_commodity.png"
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)

    # Seasonality: average price by calendar month per commodity
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["commodity", "month"])["price"].mean().reset_index()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=monthly, x="month", y="price", hue="commodity", marker="o")
    plt.title("Seasonality — monthly average prices")
    plt.xticks(range(1, 13))
    plt.ylabel("Avg price")
    seas_path = out_dir / "seasonal_monthly_avg.png"
    plt.tight_layout()
    plt.savefig(seas_path, dpi=150)
    plt.close()

    # Correlation numeric (engineered supervised columns + price snapshot)
    num_cols = [c for c in eng.columns if c in {"price", "temp_c", "rainfall_mm", "humidity_pct", "roll_mean_7", "price_lag1"}]

    corr = eng[num_cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation heatmap (subset of engineered numeric features)")
    cor_path = out_dir / "correlation_numeric.png"
    plt.tight_layout()
    plt.savefig(cor_path, dpi=150)
    plt.close()

    insights_path = out_dir / "insights.txt"
    vol = df.groupby("commodity")["price"].std().sort_values(ascending=False)
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write("Price volatility ranking (daily std deviation):\n")
        for k, v in vol.items():
            f.write(f"  {k}: {v:.3f}\n")
        ma = df.groupby("commodity")["price"].mean()
        f.write("\nMean price level snapshot:\n")
        for k, v in ma.items():
            f.write(f"  {k}: {v:.3f}\n")

    return out_dir


if __name__ == "__main__":
    from agr_forecast.config import DATA_RAW, PROJECT_ROOT

    run_eda_plots(DATA_RAW, PROJECT_ROOT / "notebooks" / "figures")
