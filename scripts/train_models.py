"""CLI entry to preprocess + evaluate + persist model artifacts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agr_forecast.config import DATA_RAW
from agr_forecast.train_pipeline import train_all


def main():
    csv = DATA_RAW if len(sys.argv) < 2 else Path(sys.argv[1])
    meta = train_all(csv)
    print("Training complete.")
    commodities = meta.get("commodities", [])
    print("Artifacts saved for commodities:", commodities)
    for c, blob in meta.get("evaluation_metrics_holdout_cv", {}).items():
        print(f"\n== {c} ==")
        print("ARIMA holdout:", blob.get("arima_holdout"))
        print("XGB holdout:", blob.get("xgb_holdout"))
        print("XGB TSCV:", blob.get("xgboost_time_series_cv"))


if __name__ == "__main__":
    main()
