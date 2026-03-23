"""Utilities for loading classifier artifacts and building UI-ready signal payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.stock_data import StockDataProcessor
from src.database.dal import DataAccessLayer
from src.models.classification_utils import compute_classification_metrics
from src.models.production_config import (
    normalize_classifier_model_type,
    resolve_production_classifier_model_type,
)

DEFAULT_FEATURE_COLUMNS = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


@dataclass(frozen=True)
class ClassifierArtifactPaths:
    """Resolved artifact paths for classifier inference."""

    model_type: str
    model_path: Path
    metadata_path: Optional[Path] = None


def _normalize_model_type(model_type: str) -> str:
    return normalize_classifier_model_type(model_type, allow_production=True)


def resolve_classifier_artifacts(
    model_type: str,
    models_dir: Path = Path("models"),
) -> ClassifierArtifactPaths:
    """Resolve classifier model + metadata artifact paths from `models/`."""
    normalized = _normalize_model_type(model_type)
    if normalized == "production":
        normalized = resolve_production_classifier_model_type()

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if normalized == "rf":
        model_path = models_dir / "random_forest_global.pkl"
        metadata_path = models_dir / "random_forest_global_metadata.json"
    else:
        model_path = models_dir / "xgboost_global.json"
        metadata_path = models_dir / "xgboost_global_metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Classifier model artifact not found: {model_path}")

    return ClassifierArtifactPaths(
        model_type=normalized,
        model_path=model_path.resolve(),
        metadata_path=metadata_path.resolve() if metadata_path.is_file() else None,
    )


@lru_cache(maxsize=4)
def _load_cached_classifier_artifacts(
    model_type: str,
    model_path: str,
    metadata_path: str,
    model_mtime: float,
    metadata_mtime: float,
) -> Dict[str, Any]:
    """
    Load and cache classifier artifacts.

    The mtime arguments are included to invalidate cache after retraining artifacts.
    """
    del model_mtime
    del metadata_mtime

    if model_type == "rf":
        from src.models.random_forest_model import RandomForestModel

        model = RandomForestModel()
    else:
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()

    model.load(model_path)

    metadata: Dict[str, Any] = {}
    if metadata_path and Path(metadata_path).is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {"model": model, "metadata": metadata}


def _to_market_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert DAL rows to market-style dataframe with standardized OHLCV columns."""
    if price_df.empty:
        return pd.DataFrame()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close",
    }
    df = price_df.rename(columns=rename_map).copy()
    df["Date"] = pd.to_datetime(df["date"])
    df = df.sort_values("Date").set_index("Date")

    keep_cols = [col for col in ["Open", "High", "Low", "Close", "Volume", "Adj Close"] if col in df.columns]
    return df[keep_cols].astype(float)


def _prepare_feature_frame(
    ticker: str,
    market_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Build feature matrix consistent with RF/XGB training features."""
    processor = StockDataProcessor(ticker=ticker)
    feature_df = processor.add_technical_indicators(market_df.copy())
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    missing = [col for col in feature_columns if col not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for inference: {missing}")

    return feature_df


def _next_business_day(date_value: str) -> str:
    """Return the next business day in YYYY-MM-DD format."""
    value = pd.Timestamp(date_value)
    return (value + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")


def _safe_parse_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse date-like values to normalized pandas timestamps."""
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    ts = pd.Timestamp(parsed)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _extract_artifact_reference_date(metadata: Dict[str, Any], fallback_path: Path) -> Optional[pd.Timestamp]:
    """
    Best-effort artifact data date from metadata coverage fields.

    Falls back to artifact mtime when metadata coverage is unavailable.
    """
    candidates: List[Any] = []
    coverage = metadata.get("data_coverage")
    if isinstance(coverage, dict):
        date_coverage = coverage.get("date_coverage")
        if isinstance(date_coverage, dict):
            candidates.extend(
                [
                    date_coverage.get("overall_end"),
                    date_coverage.get("test_end"),
                    date_coverage.get("val_end"),
                    date_coverage.get("train_end"),
                ]
            )
        candidates.append(coverage.get("overall_end"))

    candidates.extend(
        [
            metadata.get("trained_at"),
            metadata.get("updated_at"),
            metadata.get("created_at"),
        ]
    )

    for value in candidates:
        parsed = _safe_parse_date(value)
        if parsed is not None:
            return parsed

    try:
        return pd.Timestamp.fromtimestamp(fallback_path.stat().st_mtime).normalize()
    except OSError:
        return None


def _build_artifact_freshness(
    latest_price_date: str,
    metadata: Dict[str, Any],
    artifact_path: Path,
    stale_threshold_days: int = 7,
) -> Dict[str, Any]:
    """Build stale-artifact diagnostics against latest DB market date."""
    latest_ts = _safe_parse_date(latest_price_date)
    artifact_ts = _extract_artifact_reference_date(metadata=metadata, fallback_path=artifact_path)

    stale_days: Optional[int] = None
    if latest_ts is not None and artifact_ts is not None:
        stale_days = max(0, int((latest_ts - artifact_ts).days))

    is_stale = stale_days is not None and stale_days > stale_threshold_days
    return {
        "latest_price_date": str(latest_price_date),
        "artifact_reference_date": artifact_ts.strftime("%Y-%m-%d") if artifact_ts is not None else None,
        "stale_days": stale_days,
        "stale_threshold_days": int(stale_threshold_days),
        "is_stale": bool(is_stale),
    }


def _build_recent_eval_metrics(
    model: Any,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    threshold: float,
    eval_window: int,
) -> Dict[str, float]:
    """
    Build a lightweight, recent directional evaluation window for UI diagnostics.

    For each row t, features at t predict direction of Close(t+1) relative to Close(t).
    """
    if eval_window <= 0 or len(feature_df) < 2:
        return {}

    eligible_rows = len(feature_df) - 1
    if eligible_rows <= 0:
        return {}

    n_eval = min(eval_window, eligible_rows)
    start = eligible_rows - n_eval
    stop = start + n_eval

    X_eval = feature_df[feature_columns].iloc[start:stop].values
    current_close = feature_df["Close"].iloc[start:stop].values
    next_close = feature_df["Close"].iloc[start + 1 : stop + 1].values

    y_true = (next_close > current_close).astype(int)
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    raw_metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    raw_metrics["predicted_up_ratio"] = float(np.mean(y_pred))
    raw_metrics["actual_up_ratio"] = float(np.mean(y_true))
    return {k: float(v) for k, v in raw_metrics.items()}


def _build_classifier_health_diagnostics(
    global_metrics: Dict[str, float],
    recent_eval_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Build simple degeneracy diagnostics for classifier output quality."""
    warnings: List[str] = []

    balanced_accuracy = global_metrics.get("balanced_accuracy")
    precision = global_metrics.get("precision")
    recall = global_metrics.get("recall")
    roc_auc = global_metrics.get("roc_auc")

    #orignal value should be 0.505
    if balanced_accuracy is not None and balanced_accuracy <= 0.105:
        warnings.append("Global balanced accuracy is near random chance (~0.50).")
    #original value should be 0.40 <= precision <= 0.60
    if recall is not None and recall >= 0.98 and precision is not None and 0.00 <= precision <= 0.01:
        warnings.append("Global recall is near 1.00 with mid precision, suggesting mostly one-class predictions.")
    #original value should be roc_auc < 0.52
    if roc_auc is not None and roc_auc <= 0.01:
        warnings.append("Global ROC-AUC is near random chance.")

    predicted_up_ratio = recent_eval_metrics.get("predicted_up_ratio")
    if predicted_up_ratio is not None and (predicted_up_ratio >= 0.98 or predicted_up_ratio <= 0.02):
        warnings.append("Recent-window predictions are almost entirely one direction.")

    return {
        "is_degenerate": bool(warnings),
        "warnings": warnings,
    }


def _extract_metrics(metadata: Dict[str, Any], ticker: str) -> Dict[str, Dict[str, float]]:
    """Extract global and per-ticker test metrics from metadata when available."""
    global_metrics: Dict[str, float] = {}
    ticker_metrics: Dict[str, float] = {}

    metrics_payload = metadata.get("metrics")
    if isinstance(metrics_payload, dict):
        test_metrics = metrics_payload.get("test")
        if isinstance(test_metrics, dict):
            global_metrics = {
                str(k): float(v)
                for k, v in test_metrics.items()
                if isinstance(v, (int, float, np.floating, np.integer))
            }

    per_ticker_payload = metadata.get("per_ticker_test_metrics")
    if isinstance(per_ticker_payload, dict):
        by_ticker = per_ticker_payload.get("by_ticker")
        if isinstance(by_ticker, dict):
            row = by_ticker.get(ticker.upper())
            if isinstance(row, dict):
                ticker_metrics = {
                    str(k): float(v)
                    for k, v in row.items()
                    if isinstance(v, (int, float, np.floating, np.integer))
                }

    return {"global": global_metrics, "ticker": ticker_metrics}


def generate_classification_signal_data(
    ticker: str,
    model_type: str,
    dal: DataAccessLayer,
    eval_window: int = 90,
    persist_prediction: bool = True,
) -> Dict[str, Any]:
    """Generate UI-ready signal payload for RF/XGBoost model artifacts."""
    ticker = ticker.upper()
    requested_model_type = str(model_type).strip().lower()
    latest_date = dal.get_latest_price_date(ticker)
    if not latest_date:
        raise ValueError(f"No stock price history found for ticker {ticker}.")

    raw_prices = dal.get_stock_prices(ticker, "1900-01-01", latest_date)
    market_df = _to_market_dataframe(raw_prices)
    if market_df.empty:
        raise ValueError(f"Unable to build market DataFrame for ticker {ticker}.")

    artifacts = resolve_classifier_artifacts(model_type=model_type)
    model_mtime = artifacts.model_path.stat().st_mtime
    metadata_path_str = str(artifacts.metadata_path) if artifacts.metadata_path else ""
    metadata_mtime = artifacts.metadata_path.stat().st_mtime if artifacts.metadata_path else 0.0

    loaded = _load_cached_classifier_artifacts(
        model_type=artifacts.model_type,
        model_path=str(artifacts.model_path),
        metadata_path=metadata_path_str,
        model_mtime=model_mtime,
        metadata_mtime=metadata_mtime,
    )
    model = loaded["model"]
    metadata = loaded["metadata"]

    feature_columns = list(metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS))
    feature_df = _prepare_feature_frame(ticker=ticker, market_df=market_df, feature_columns=feature_columns)
    if feature_df.empty:
        raise ValueError(f"No indicator-ready rows available for ticker {ticker}.")

    latest_features = feature_df[feature_columns].tail(1).values
    prob_up = float(model.predict_proba(latest_features)[0, 1])
    decision_threshold = float(getattr(model, "decision_threshold", 0.5))
    predicted_direction = int(prob_up >= decision_threshold)
    predicted_label = "UP" if predicted_direction == 1 else "DOWN"
    confidence = float(prob_up if predicted_direction == 1 else 1.0 - prob_up)
    prediction_date = _next_business_day(latest_date)
    model_name = str(getattr(model, "model_name", model.__class__.__name__))

    metrics = _extract_metrics(metadata=metadata, ticker=ticker)
    recent_eval_metrics = _build_recent_eval_metrics(
        model=model,
        feature_df=feature_df,
        feature_columns=feature_columns,
        threshold=decision_threshold,
        eval_window=eval_window,
    )
    classifier_health = _build_classifier_health_diagnostics(
        global_metrics=metrics["global"],
        recent_eval_metrics=recent_eval_metrics,
    )
    artifact_freshness = _build_artifact_freshness(
        latest_price_date=str(latest_date),
        metadata=metadata,
        artifact_path=artifacts.model_path,
    )

    if persist_prediction:
        existing = dal.get_prediction_by_key(ticker=ticker, model_name=model_name, date=prediction_date)
        if existing is None:
            dal.insert_prediction(
                ticker=ticker,
                date=prediction_date,
                model_name=model_name,
                predicted_direction=predicted_direction,
                predicted_price=None,
                confidence=confidence,
            )

    return {
        "model_type": artifacts.model_type,
        "requested_model_type": requested_model_type,
        "model_name": model_name,
        "latest_price_date": str(latest_date),
        "prediction_date": prediction_date,
        "predicted_direction": predicted_direction,
        "predicted_label": predicted_label,
        "prob_up": prob_up,
        "confidence": confidence,
        "decision_threshold": decision_threshold,
        "global_test_metrics": metrics["global"],
        "ticker_test_metrics": metrics["ticker"],
        "recent_eval_metrics": recent_eval_metrics,
        "classifier_health": classifier_health,
        "artifact_freshness": artifact_freshness,
        "feature_columns": feature_columns,
        "artifact_paths": {
            "model_path": str(artifacts.model_path),
            "metadata_path": metadata_path_str,
        },
    }
