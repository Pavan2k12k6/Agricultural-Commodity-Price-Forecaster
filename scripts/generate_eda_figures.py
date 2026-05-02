"""Run EDA and write figures to notebooks/figures."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agr_forecast.config import DATA_RAW
from agr_forecast.eda import run_eda_plots


def main():
    out = ROOT / "notebooks" / "figures"
    d = run_eda_plots(DATA_RAW, out)
    print(f"EDA figures saved to {d.resolve()}")


if __name__ == "__main__":
    main()
