# Multi-stage not required for this small service.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Editable install exposes both `agr_forecast` and `api` packages when running from WORKDIR
RUN pip install -e . && chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app

# Default: API on 8000 — override command to run Streamlit instead.
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
