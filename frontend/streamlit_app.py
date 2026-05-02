"""
Streamlit dashboard — select commodity, view forecasts, download CSV.
Run from project root (after training):
  streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from agr_forecast.config import DATA_RAW

st.set_page_config(page_title="Agri Price Forecaster", layout="wide")
st.title("Agricultural Commodity Price Forecaster")
st.caption("SDG 2 — supporting better sell timing with simple ML forecasts.")

if "forecast_cache" not in st.session_state:
    st.session_state["forecast_cache"] = None

use_api = st.sidebar.checkbox("Call FastAPI backend", value=False)
api_base = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

commodity = st.selectbox("Commodity", ["rice", "wheat", "maize"])
horizons = st.multiselect("Forecast horizons (days)", options=[7, 15, 30], default=[7, 15, 30])

if not Path(DATA_RAW).exists():
    st.error("No dataset at data/raw/commodities.csv — run `python scripts/generate_sample_data.py` first.")
    st.stop()

if st.sidebar.button("Train / retrain models"):
    with st.spinner("Training (may take a minute)…"):
        try:
            from agr_forecast.train_pipeline import train_all

            train_all(DATA_RAW)
            st.success("Training completed. Artifacts refreshed under models/.")
        except Exception as e:
            st.error(str(e))

if st.button("Get forecasts"):
    if not horizons:
        st.warning("Select at least one horizon.")
    else:
        with st.spinner("Computing forecasts…"):
            try:
                if use_api:
                    r = requests.post(
                        f"{api_base.rstrip('/')}/predict",
                        json={"commodity": commodity, "horizons": horizons},
                        timeout=180,
                    )
                    r.raise_for_status()
                    resp = r.json()
                    st.session_state["forecast_cache"] = resp.get("horizons", {})
                    st.session_state["commodity_cached"] = resp.get("commodity", commodity)
                else:
                    from agr_forecast.predict_pipeline import forecasts_for_horizons

                    st.session_state["forecast_cache"] = forecasts_for_horizons(
                        DATA_RAW, commodity, horizons
                    )
                    st.session_state["commodity_cached"] = commodity
                st.success("Forecasts ready.")
            except Exception as e:
                st.error(str(e))

hs = st.session_state.get("forecast_cache")
if not hs:
    st.info('Click "Get forecasts" after training models (sidebar: Train / retrain).')
else:
    for h_key, blob in sorted(hs.items(), key=lambda x: int(x[0])):
        df = pd.DataFrame(
            {
                "date": blob["date"],
                "forecast_xgboost": blob["forecast_xgboost"],
                "forecast_arima": blob["forecast_arima"],
            }
        )
        st.subheader(f"Next {h_key} days ({st.session_state.get('commodity_cached', commodity)})")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["date"], df["forecast_xgboost"], label="XGBoost", lw=2)
        ax.plot(df["date"], df["forecast_arima"], label="ARIMA/SARIMAX", lw=2, alpha=0.85)
        ax.legend()
        ax.set_ylabel("Predicted price")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig)

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            label=f"Download CSV ({h_key}-day)",
            data=buf.getvalue(),
            file_name=f"forecast_{st.session_state.get('commodity_cached', commodity)}_{h_key}d.csv",
            mime="text/csv",
        )

    try:
        from agr_forecast.predict_pipeline import backtest_overlap

        comm = st.session_state.get("commodity_cached", commodity)
        st.subheader("Backtest overlay (last ~28 days)")
        bt = backtest_overlap(DATA_RAW, comm, holdout_days=28)
        bdf = pd.DataFrame(
            {
                "date": bt["dates_actual"],
                "actual": bt["actual"],
                "xgboost": bt["xgboost_approx"],
                "arima": bt["arima_approx"],
            }
        )
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(bdf["date"], bdf["actual"], label="Actual (hold-out tail)", lw=2)
        ax2.plot(bdf["date"], bdf["xgboost"], "--", label="XGB approx")
        ax2.plot(bdf["date"], bdf["arima"], "--", label="ARIMA approx")
        ax2.tick_params(axis="x", rotation=35)
        ax2.legend()
        st.pyplot(fig2)
        st.caption(bt.get("note", ""))
    except Exception:
        pass
