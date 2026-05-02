"""
Fetch recent market + weather panels and assemble `commodities.csv`-compatible rows.

**Prices:** Yahoo Finance via `yfinance` — futures proxies for demo training (global markets):

| commodity | Default symbol |
|-----------|----------------|
| wheat | ZW=F |
| maize | ZC=F (corn) |
| rice | ZR=F (rough rice) |

These track international futures, **not local mandi/cash quotes**. Replace tickers/mapping when you ingest Agmark/FCI data.

**Weather:** Open-Meteo (archive ERA5 + short forecast blend; no API key).
Rows repeat the same daily weather across commodities for the chosen geographic point.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_TICKERS = {
    "wheat": "ZW=F",
    "maize": "ZC=F",
    "rice": "ZR=F",
}


def fetch_open_meteo_archive(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Daily historical ERA5 weather for inclusive `[start, end]` (chunks if span is long)."""
    frames: list[pd.DataFrame] = []
    cursor = start
    sess = requests.Session()
    max_chunk = timedelta(days=360)
    while cursor <= end:
        chunk_end = min(cursor + max_chunk - timedelta(days=1), end)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": cursor.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": "UTC",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
        }
        resp = sess.get(ARCHIVE_URL, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json().get("daily") or {}
        if not data.get("time"):
            raise RuntimeError(f"Open-Meteo archive empty for {cursor}..{chunk_end}: {resp.text[:280]}")
        ts = pd.DatetimeIndex(pd.to_datetime(data["time"])).normalize()
        frames.append(
            pd.DataFrame(
                {
                    "date": ts,
                    "tmax": pd.to_numeric(data.get("temperature_2m_max"), errors="coerce"),
                    "tmin": pd.to_numeric(data.get("temperature_2m_min"), errors="coerce"),
                    "precip_mm": pd.to_numeric(data.get("precipitation_sum"), errors="coerce"),
                    "rh_mean": pd.to_numeric(data.get("relative_humidity_2m_mean"), errors="coerce"),
                }
            )
        )
        cursor = chunk_end + timedelta(days=1)
    return pd.concat(frames, axis=0, ignore_index=True).drop_duplicates("date").sort_values("date")


def fetch_open_meteo_forecast_tail(
    lat: float,
    lon: float,
    past_days: int = 14,
    forecast_days: int = 5,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Past few days blended with horizon forecast."""
    sess = session or requests.Session()
    resp = sess.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "UTC",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json().get("daily") or {}
    if not data.get("time"):
        raise RuntimeError(f"Open-Meteo forecast empty: {resp.text[:280]}")
    ts = pd.DatetimeIndex(pd.to_datetime(data["time"])).normalize()
    return pd.DataFrame(
        {
            "date": ts,
            "tmax": pd.to_numeric(data.get("temperature_2m_max"), errors="coerce"),
            "tmin": pd.to_numeric(data.get("temperature_2m_min"), errors="coerce"),
            "precip_mm": pd.to_numeric(data.get("precipitation_sum"), errors="coerce"),
            "rh_mean": pd.to_numeric(data.get("relative_humidity_2m_mean"), errors="coerce"),
        }
    ).drop_duplicates("date")


def compose_weather_panel(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    archive = fetch_open_meteo_archive(lat, lon, start, end)

    cutoff = pd.Timestamp(date.today()) - timedelta(days=5)
    forecast_start = cutoff if cutoff > pd.Timestamp(start) else pd.Timestamp(start)

    tail = archive[archive["date"] < forecast_start]
    sess = requests.Session()
    blended = fetch_open_meteo_forecast_tail(lat, lon, session=sess)
    blended = blended[blended["date"] >= forecast_start]
    blended = blended[blended["date"] <= pd.Timestamp(end)]

    weather = pd.concat([tail, blended], axis=0, ignore_index=True)
    weather["temp_c"] = weather[["tmax", "tmin"]].astype(float).mean(axis=1, skipna=True)
    weather["rainfall_mm"] = weather["precip_mm"].astype(float).fillna(0.0)
    weather["humidity_pct"] = weather["rh_mean"].astype(float)

    out = weather.loc[:, ["date", "temp_c", "rainfall_mm", "humidity_pct"]]
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]

    # Fill trailing humidity gaps
    med = float(out["humidity_pct"].median()) if bool(out["humidity_pct"].notna().any()) else 65.0
    out["humidity_pct"] = out["humidity_pct"].ffill().bfill().fillna(med)

    out["humidity_pct"] = out["humidity_pct"].clip(lower=10.0, upper=99.0)
    out["rainfall_mm"] = out["rainfall_mm"].clip(lower=0.0)
    out = out.dropna(subset=["temp_c"]).reset_index(drop=True)
    return out


def _yf_daily_close(ticker: str, start: date, end: date) -> pd.Series:
    import yfinance as yf

    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    px = yf.download(
        ticker,
        start=pd.Timestamp(start),
        end=end_excl,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if px.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(px.columns, pd.MultiIndex):
        closes = px["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
    else:
        closes = px["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.squeeze(axis=1)
    ix = pd.to_datetime(pd.Index(px.index))
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_convert(None)
    closes.index = ix.normalize()
    return closes.rename(ticker)


def fetch_yahoo_proxy_panel(tickers: dict[str, str], start: date, end: date) -> pd.DataFrame:
    series_list: dict[str, pd.Series] = {}
    for commodity, ticker in tickers.items():
        s = _yf_daily_close(ticker, start=start, end=end)
        s.name = commodity
        series_list[commodity] = s

    merged = pd.DataFrame(series_list).sort_index()
    merged.index = merged.index.rename("date")
    return merged.astype(float)


def assemble_training_frame(
    lat: float,
    lon: float,
    *,
    tickers: dict[str, str] | None = None,
    history_days: int = 730,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Return long-form rows: date, commodity, price, temp_c, rainfall_mm, humidity_pct

    Futures are forward-filled on the weather calendar to bridge weekends/market holidays.
    """
    tickers = tickers or dict(DEFAULT_TICKERS)
    last = end_date or date.today()
    start = last - timedelta(days=max(120, int(history_days)))

    weather = compose_weather_panel(lat, lon, start, last)
    if weather.empty:
        raise RuntimeError("Weather download returned an empty dataframe.")

    closes = fetch_yahoo_proxy_panel(tickers, start, last)
    if closes.empty:
        raise RuntimeError(
            "Could not fetch Yahoo futures data (empty). Check connectivity or ticker mapping."
        )

    aligned = closes.reindex(pd.DatetimeIndex(weather["date"])).ffill(limit=31)
    if aligned.isna().any().any():
        # Allow short warm-up tails
        aligned = aligned.ffill().bfill()
    aligned.insert(0, "date", weather["date"].values)
    long_px = aligned.melt(id_vars="date", var_name="commodity", value_name="price")

    out = long_px.merge(weather, on="date", how="left")
    out = out.dropna(subset=["price", "temp_c", "humidity_pct"])
    out["commodity"] = out["commodity"].astype(str).str.strip().str.lower()

    cols = ["date", "commodity", "price", "temp_c", "rainfall_mm", "humidity_pct"]
    return out.loc[:, cols].sort_values(["commodity", "date"]).reset_index(drop=True)


def fetch_live_snapshot_to_csv(
    output_path: str | Path,
    lat: float,
    lon: float,
    *,
    tickers: dict[str, str] | None = None,
    history_days: int = 730,
    end_date: date | None = None,
) -> Path:
    """Download live-ish panel and persist CSV."""
    frame = assemble_training_frame(
        lat, lon, tickers=tickers, history_days=history_days, end_date=end_date
    )
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outp, index=False)
    return outp