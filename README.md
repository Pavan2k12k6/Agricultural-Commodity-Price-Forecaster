# Agricultural Commodity Price Forecaster

An end-to-end ML project to forecast agricultural commodity prices (`rice`, `wheat`, `maize`) using:
- historical price time series,
- weather features (`temp_c`, `rainfall_mm`, `humidity_pct`),
- seasonal and lag-based engineered features.

The project supports SDG 2 (Zero Hunger) by helping farmers and market participants estimate short-term price movements and plan better selling windows.

## Project Overview

- **Data pipeline:** CSV loading, missing-value handling, outlier clipping, feature engineering.
- **Models:** ARIMA (ARIMAX-style with exogenous variables) and XGBoost regressor.
- **Evaluation:** chronological train/test split + RMSE/MAE + time-series CV.
- **Prediction horizons:** 7, 15, and 30 days (custom horizons supported).
- **Interfaces:** FastAPI backend + optional Streamlit dashboard.
- **Extras:** synthetic data generator and live-ish data fetch script (Open-Meteo + Yahoo proxies).

## Folder Structure

```text
Agri/
├─ agr_forecast/
├─ api/
├─ frontend/
├─ scripts/
├─ data/raw/
├─ models/
├─ notebooks/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

## How to Run the Project

### 1) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

### 3) Prepare data

Use synthetic data (recommended first run):

```powershell
python scripts/generate_sample_data.py
```

Optional: fetch near-real data (weather + futures proxies):

```powershell
python scripts/fetch_live_and_train.py --days 730 --overwrite-main
```

### 4) Train models

```powershell
python scripts/train_models.py
```

Artifacts are saved under `models/`.

### 5) (Optional) Generate EDA charts

```powershell
python scripts/generate_eda_figures.py
```

Outputs: `notebooks/figures/`.

### 6) Start backend API

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 7) (Optional) Start Streamlit dashboard

```powershell
streamlit run frontend/streamlit_app.py
```

Open app: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## API Endpoints

- `GET /health` - health check
- `POST /train` - retrain models
- `POST /predict` - forecast future prices
- `GET /data` - return processed data
- `GET /backtest_overlay` - actual vs forecast overlay payload

## Quick Predict Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"commodity\":\"rice\",\"horizons\":[7,15,30]}"
```

## Docker (Optional)

```bash
docker compose up --build
```

- API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit: [http://localhost:8501](http://localhost:8501)
