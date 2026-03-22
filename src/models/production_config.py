"""Utilities for loading and normalizing runtime production model config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

PRODUCTION_CONFIG_PATH = Path("config/production_model.json")

_RF_ALIASES = {"rf", "random_forest", "random-forest", "randomforest", "random forest"}
_XGB_ALIASES = {"xgb", "xgboost", "xg-boost"}
_PRODUCTION_ALIASES = {"production", "prod", "default", "auto"}

_MODEL_NAMES_BY_TYPE = {
    "rf": {"RandomForest_v2", "RandomForestModel"},
    "xgb": {"XGBoost_v2", "XGBoostModel"},
}


def normalize_classifier_model_type(model_type: str, *, allow_production: bool = False) -> str:
    """Normalize user/config model type to `rf` or `xgb`."""
    value = str(model_type or "").strip().lower()
    if value in _RF_ALIASES:
        return "rf"
    if value in _XGB_ALIASES:
        return "xgb"
    if allow_production and value in _PRODUCTION_ALIASES:
        return "production"
    allowed = "'rf' or 'xgb'" + (" (or 'production')" if allow_production else "")
    raise ValueError(f"Unsupported model type: {model_type}. Use {allowed}.")


def load_production_model_config(config_path: Path = PRODUCTION_CONFIG_PATH) -> Dict[str, Any]:
    """Load production model config payload from JSON."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Production config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid production config schema in {config_path}: expected JSON object.")

    if isinstance(payload.get("production_model"), dict):
        return payload["production_model"]
    return payload


def resolve_production_classifier_model_type(config_path: Path = PRODUCTION_CONFIG_PATH) -> str:
    """Resolve classifier model type from production config."""
    config = load_production_model_config(config_path=config_path)
    raw = config.get("model_type")
    if raw is None:
        raise ValueError(f"`model_type` missing in production config: {config_path}")
    return normalize_classifier_model_type(str(raw), allow_production=False)


def classifier_model_names_for_type(model_type: str) -> Set[str]:
    """Return known prediction-table model names for a normalized classifier model type."""
    normalized = normalize_classifier_model_type(model_type, allow_production=False)
    names = set(_MODEL_NAMES_BY_TYPE.get(normalized, set()))
    names.add(normalized)
    return names
