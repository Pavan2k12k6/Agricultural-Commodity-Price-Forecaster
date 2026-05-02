#!/bin/sh
set -e
cd /app

if [ ! -f data/raw/commodities.csv ]; then
  python scripts/generate_sample_data.py
fi

if ! ls models/xgb_*.joblib >/dev/null 2>&1; then
  echo "Training models on first startup…"
  python scripts/train_models.py
fi

exec "$@"
