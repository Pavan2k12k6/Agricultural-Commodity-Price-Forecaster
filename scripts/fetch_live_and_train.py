"""
Download near-real weather (Open-Meteo) + daily futures proxies (Yahoo Finance) and train models.

Usage (from repo root):

  python scripts/fetch_live_and_train.py --days 600

Optional: overwrite the default CSV the API/train scripts use unless AGRI_DATA_CSV is set:

  python scripts/fetch_live_and_train.py --overwrite-main

Then start the API normally; DATA_RAW resolves from env or paths in agr_forecast.config.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agr_forecast.config import DATA_LIVE, DATA_RAW
from agr_forecast.live_fetch import DEFAULT_TICKERS, fetch_live_snapshot_to_csv
from agr_forecast.train_pipeline import train_all


def main():
    parser = argparse.ArgumentParser(description="Fetch live-ish data and retrain forecasting models.")
    parser.add_argument("--lat", type=float, default=28.6139, help="Latitude for Open-Meteo point (India default)")
    parser.add_argument("--lon", type=float, default=77.2090, help="Longitude for Open-Meteo point (India default)")
    parser.add_argument("--days", type=int, default=730, help="History window in days (~2 calendar years)")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_LIVE),
        help=f"CSV output path (default: {DATA_LIVE})",
    )
    parser.add_argument(
        "--overwrite-main",
        action="store_true",
        help="Also copy fetched CSV onto data/raw/commodities.csv (still respect AGRI_DATA_CSV if set).",
    )

    parser.add_argument(
        "--wheat-symbol",
        type=str,
        default=DEFAULT_TICKERS["wheat"],
        metavar="SYM",
        help="Yahoo symbol for wheat proxy futures",
    )
    parser.add_argument(
        "--maize-symbol",
        type=str,
        default=DEFAULT_TICKERS["maize"],
        metavar="SYM",
        help="Yahoo symbol for maize (corn futures) proxy",
    )
    parser.add_argument(
        "--rice-symbol",
        type=str,
        default=DEFAULT_TICKERS["rice"],
        metavar="SYM",
        help="Yahoo symbol for rough rice proxy futures",
    )
    parser.add_argument(
        "--train-only-path",
        type=str,
        default=None,
        help="Train models from this CSV path without downloading (shortcut).",
    )
    flags = parser.parse_args()

    if flags.train_only_path:
        csv_path = Path(flags.train_only_path)
        if not csv_path.exists():
            raise SystemExit(f"train-only path not found: {csv_path}")
    else:
        csv_path = fetch_live_snapshot_to_csv(
            flags.out,
            lat=flags.lat,
            lon=flags.lon,
            tickers={
                "wheat": flags.wheat_symbol,
                "maize": flags.maize_symbol,
                "rice": flags.rice_symbol,
            },
            history_days=int(flags.days),
        )
        print(f"Wrote live-ish dataset to {csv_path.resolve()}")

        if flags.overwrite_main:
            main_path = ROOT / "data" / "raw" / "commodities.csv"
            shutil.copyfile(csv_path, main_path)
            print(
                f"Copied dataset to {main_path.resolve()} - DATA_RAW picks this unless AGRI_DATA_CSV is set."
            )

    print(f"Training on: {csv_path.resolve()}")
    meta = train_all(csv_path)
    print("Training complete.")
    commodities = meta.get("commodities", [])
    print("Artifacts saved for commodities:", commodities)
    for comm, blob in meta.get("evaluation_metrics_holdout_cv", {}).items():
        print(f"\n== {comm} ==")
        print("ARIMA holdout:", blob.get("arima_holdout"))
        print("XGB holdout:", blob.get("xgb_holdout"))
        print("XGB TSCV:", blob.get("xgboost_time_series_cv"))


if __name__ == "__main__":
    main()