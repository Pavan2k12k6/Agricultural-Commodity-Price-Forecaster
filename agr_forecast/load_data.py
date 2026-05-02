"""Load raw commodity CSV into a typed DataFrame."""
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("date", "commodity", "price", "temp_c", "rainfall_mm", "humidity_pct")


def load_commodities_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV with required columns and parse dates."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_csv(p)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["commodity"] = df["commodity"].str.lower().astype(str).str.strip()
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)
    return df


def split_by_time(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split per commodity combined into global sorted frames."""
    parts_train, parts_test = [], []
    for _, g in df.groupby("commodity", sort=False):
        g = g.sort_values("date")
        n = len(g)
        cut = max(1, int(n * (1 - test_ratio)))
        parts_train.append(g.iloc[:cut])
        parts_test.append(g.iloc[cut:])
    train_df = pd.concat(parts_train, ignore_index=True).sort_values(["commodity", "date"])
    test_df = pd.concat(parts_test, ignore_index=True).sort_values(["commodity", "date"])
    return train_df, test_df
