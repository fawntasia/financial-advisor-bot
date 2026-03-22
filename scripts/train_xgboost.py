import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
sys.path.append(PROJECT_ROOT)

from src.models.classification_utils import compute_classification_metrics
from src.models.global_classification_data import build_global_classification_dataset
from src.models.io_utils import ensure_parent_dir, get_library_versions
from src.models.reproducibility import set_global_seeds
from src.models.xgboost_model import XGBoostModel

# ==================== Hard-Coded Calendar Split ====================
# Edit these dates directly when you want a different train/validation/test window.
TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE = "2024-12-31"
VALIDATION_START_DATE = "2025-01-01"
VALIDATION_END_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = None  # Example: "2026-12-31"

# Optional source data clipping window used before split assignment.
DATA_WINDOW_START_DATE = TRAIN_START_DATE
DATA_WINDOW_END_DATE = TEST_END_DATE


def _xgboost_gpu_available() -> bool:
    """Return True when XGBoost can run a tiny fit with GPU tree method."""
    try:
        from xgboost import XGBClassifier

        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int32)
        probe = XGBClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            random_state=0,
            eval_metric="logloss",
            tree_method="gpu_hist",
            n_jobs=-1,
        )
        probe.fit(X, y, verbose=False)
        return True
    except Exception:
        return False


def train_xgboost(
    tune: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    use_gpu: bool = True,
    data_source: str = "db",
    db_path: str = "data/financial_advisor.db",
    enforce_diversity: bool = True,
):
    print("Starting XGBoost global training...")
    set_global_seeds(seed=seed, include_tensorflow=False)

    print(f"Preparing pooled global dataset from source={data_source}...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        split_meta,
        test_tickers,
    ) = _prepare_global_data(
        data_source=data_source,
        db_path=db_path,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    effective_use_gpu = bool(use_gpu)
    if effective_use_gpu and not _xgboost_gpu_available():
        print("GPU requested, but XGBoost GPU backend is unavailable. Falling back to CPU training.")
        effective_use_gpu = False

    model = XGBoostModel(random_state=seed, use_gpu=effective_use_gpu)
    print(f"Training model (Tuning: {tune})...")
    metrics = model.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparameters=tune,
        val_prices=None,
        enforce_prediction_diversity=enforce_diversity,
    )

    model_path = os.path.join(output_dir, "xgboost_global.json")
    model.save(model_path)

    feature_importances = {}
    if hasattr(model.model, "feature_importances_"):
        importances = model.model.feature_importances_
        feature_importances = {k: float(v) for k, v in zip(feature_cols, importances)}
        print("\n=== Feature Importance Analysis ===")
        for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
            print(f"{k:<15}: {v:.4f}")

    model_stem, _ = os.path.splitext(model_path)
    metadata_path = f"{model_stem}_metadata.json"
    manifest_path = f"{model_stem}.manifest.json"
    per_ticker_test = _per_ticker_metrics(model, X_test, y_test, test_tickers)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "scope": "global",
        "seed": seed,
        "model_name": model.get_name(),
        "model_path": model_path,
        "feature_columns": list(feature_cols),
        "hard_coded_split": {
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "validation_start": VALIDATION_START_DATE,
            "validation_end": VALIDATION_END_DATE,
            "test_start": TEST_START_DATE,
            "test_end": TEST_END_DATE,
        },
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "data_coverage": split_meta,
        "use_gpu_requested": bool(use_gpu),
        "use_gpu_effective": bool(effective_use_gpu),
        "decision_threshold": float(model.decision_threshold),
        "metrics": metrics,
        "per_ticker_test_metrics": per_ticker_test,
        "feature_importances": feature_importances,
    }
    ensure_parent_dir(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    manifest = {
        "schema_version": 1,
        "model_kind": "xgboost_classification",
        "model_path": model_path,
        "created_at": datetime.now().isoformat(),
        "scope": "global",
        "seed": seed,
        "feature_columns": list(feature_cols),
        "data_source": {
            "source": data_source,
            "db_path": db_path if data_source == "db" else None,
            "start_date": DATA_WINDOW_START_DATE,
            "end_date": DATA_WINDOW_END_DATE,
            "years": None,
            "ticker": "ALL",
        },
        "data_coverage": split_meta,
        "split_config": {
            "split_strategy": "hard_coded_calendar",
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "validation_start": VALIDATION_START_DATE,
            "validation_end": VALIDATION_END_DATE,
            "test_start": TEST_START_DATE,
            "test_end": TEST_END_DATE,
            "tune_hyperparameters": tune,
            "use_gpu_requested": bool(use_gpu),
            "use_gpu_effective": bool(effective_use_gpu),
            "enforce_prediction_diversity": enforce_diversity,
        },
        "metrics": metrics,
        "library_versions": get_library_versions(),
    }
    ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nModel artifact saved to {model_path}")
    print(f"Run metadata saved to {metadata_path}")
    print(f"Run manifest saved to {manifest_path}")
    print("\nTraining completed successfully.")


def _prepare_global_data(
    data_source: str,
    db_path: str,
):
    dataset = build_global_classification_dataset(
        data_source=data_source,
        db_path=db_path,
        start_date=DATA_WINDOW_START_DATE,
        end_date=DATA_WINDOW_END_DATE,
        years=10,
        explicit_train_start_date=TRAIN_START_DATE,
        explicit_train_end_date=TRAIN_END_DATE,
        explicit_val_start_date=VALIDATION_START_DATE,
        explicit_val_end_date=VALIDATION_END_DATE,
        explicit_test_start_date=TEST_START_DATE,
        explicit_test_end_date=TEST_END_DATE,
    )
    return (
        dataset["X_train"],
        dataset["y_train"],
        dataset["X_val"],
        dataset["y_val"],
        dataset["X_test"],
        dataset["y_test"],
        dataset["feature_cols"],
        dataset["metadata"],
        dataset["test_tickers"],
    )


def _per_ticker_metrics(model, X_test: np.ndarray, y_test: np.ndarray, test_tickers: np.ndarray) -> Dict[str, object]:
    if len(X_test) == 0 or len(test_tickers) == 0:
        return {"num_tickers": 0, "summary": {}, "by_ticker": {}}

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    by_ticker: Dict[str, Dict[str, float]] = {}
    for ticker in sorted(set(test_tickers.tolist())):
        mask = test_tickers == ticker
        ticker_metrics = compute_classification_metrics(y_test[mask], pred[mask], prob[mask])
        by_ticker[ticker] = {k: float(v) for k, v in ticker_metrics.items()}

    summary = {}
    if by_ticker:
        keys = list(next(iter(by_ticker.values())).keys())
        for key in keys:
            values = [m[key] for m in by_ticker.values() if not np.isnan(m[key])]
            summary[key] = float(np.mean(values)) if values else float("nan")

    return {
        "num_tickers": len(by_ticker),
        "summary": summary,
        "by_ticker": by_ticker,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train global XGBoost model across all available tickers.")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save model artifacts (default: project_root/models).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Enable GPU tree method (default).")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="Disable GPU and force CPU training.")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "yfinance"],
        default="db",
        help="Training data source",
    )
    parser.add_argument("--db-path", type=str, default="data/financial_advisor.db", help="SQLite DB path")
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Disable anti-degeneracy checks for validation prediction diversity.",
    )
    args = parser.parse_args()

    train_xgboost(
        tune=not args.no_tune,
        output_dir=args.output_dir,
        seed=args.seed,
        use_gpu=args.gpu,
        data_source=args.data_source,
        db_path=args.db_path,
        enforce_diversity=not args.allow_degenerate,
    )
