"""Load/save model artifacts (paths + serialization)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def artifact_path(kind: str, commodity: str) -> Path:
    from agr_forecast.config import MODELS_DIR
    commodity = commodity.lower().replace(" ", "_")
    safe = "".join(c for c in commodity if c.isalnum() or c in "_-")
    return MODELS_DIR / f"{kind}_{safe}.joblib"


def metadata_path() -> Path:
    from agr_forecast.config import MODELS_DIR
    return MODELS_DIR / "metadata.json"


def save_metadata(metadata: dict[str, Any]) -> None:
    p = metadata_path()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata() -> dict[str, Any]:
    p = metadata_path()
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, kind: str, commodity: str) -> Path:
    ap = artifact_path(kind, commodity)
    joblib.dump(obj, ap)
    return ap


def load_joblib(kind: str, commodity: str) -> Any:
    return joblib.load(artifact_path(kind, commodity))
