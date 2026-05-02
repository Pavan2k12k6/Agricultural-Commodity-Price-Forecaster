"""
Generate synthetic daily commodities.csv for demos (CSV format required by agr_forecast).
Run from project root: python scripts/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COMMODITIES = {
    "rice": {"base": 38.5, "amp": 2.8, "lag_weather": 0.035},
    "wheat": {"base": 32.0, "amp": 2.2, "lag_weather": 0.03},
    "maize": {"base": 24.8, "amp": 1.9, "lag_weather": 0.028},
}


def daily_weather(n: int) -> pd.DataFrame:
    """Rough seasonal temperature/rainfall/humidity with noise."""
    phase = RNG.uniform(0, 2 * np.pi, size=1)[0]
    t = np.arange(n)
    temp = (
        25
        + 7 * np.sin(2 * np.pi * (t / 365.25) + phase)
        + RNG.normal(0, 2.6, size=n)
    ).clip(-5.0, 48.0)
    rain_raw = np.exp(
        0.4 * np.sin(2 * np.pi * ((t % 365) / 365) + RNG.normal(0, 0.2, size=n)) + RNG.normal(-0.1, 0.45, size=n)
    ).clip(0.0, 80.0)
    humidity = (55 + 15 * np.sin(2 * np.pi * ((t % 365) / 365) + RNG.normal(0, 0.1, size=n))).clip(
        20.0, 99.5
    ) + RNG.normal(0, 6.0, size=n)
    humidity = np.clip(humidity, 15.0, 99.0)
    return pd.DataFrame({"temp_c": temp, "rainfall_mm": rain_raw, "humidity_pct": humidity})


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "raw" / "commodities.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2020-01-01")
    n = 1400
    dates = pd.date_range(start, periods=n, freq="D")
    wx = daily_weather(n)

    rows = []
    for comm, spec in COMMODITIES.items():
        seasonal = spec["amp"] * np.sin(2 * np.pi * (np.arange(n) / 365.25) + RNG.uniform(0, 2 * np.pi))
        noise = RNG.normal(0, 0.65, size=n)
        drift = 0.0009 * np.arange(n)
        price = spec["base"] + seasonal + drift + noise
        price += spec["lag_weather"] * (wx["temp_c"].values - np.mean(wx["temp_c"].values))
        price -= 0.012 * (wx["rainfall_mm"].values - np.mean(wx["rainfall_mm"].values))
        price = np.clip(price, 5.0, 200.0)

        part = pd.DataFrame(
            {
                "date": dates,
                "commodity": comm,
                "price": price,
                "temp_c": wx["temp_c"].values + RNG.normal(0, 0.15, size=n),
                "rainfall_mm": np.maximum(0.0, wx["rainfall_mm"].values + RNG.normal(0, 2.5, size=n)),
                "humidity_pct": wx["humidity_pct"].values + RNG.normal(0, 2.6, size=n),
            }
        )
        rows.append(part)

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

    mc = RNG.choice(len(df), size=180, replace=False)
    for i in mc:
        col = RNG.choice(["temp_c", "rainfall_mm", "humidity_pct"])
        df.loc[i, col] = np.nan

    spike_idx = RNG.choice(len(df), size=35, replace=False)
    df.loc[spike_idx, "price"] *= RNG.uniform(0.92, 1.12, size=len(spike_idx))

    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
