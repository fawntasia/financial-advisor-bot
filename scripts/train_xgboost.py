import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classification_utils import compute_classification_metrics
from src.models.global_classification_data import build_global_classification_dataset
from src.models.io_utils import ensure_parent_dir, get_library_versions
from src.models.reproducibility import set_global_seeds
from src.models.xgboost_model import XGBoostModel


def train_xgboost(
    tune: bool = True,
    train_split: float = 0.8,
    val_split: float = 0.1,
    output_dir: str = "models",
    seed: int = 42,
    use_gpu: bool = False,
    data_source: str = "db",
    db_path: str = "data/financial_advisor.db",
    start_date: str | None = None,
    end_date: str | None = None,
    years: int = 10,
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
        train_split=train_split,
        val_split=val_split,
        data_source=data_source,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        years=years,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    model = XGBoostModel(random_state=seed, use_gpu=use_gpu)
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
        "train_split": train_split,
        "validation_split_within_train": val_split,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "data_coverage": split_meta,
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
            "start_date": start_date,
            "end_date": end_date,
            "years": years,
            "ticker": "ALL",
        },
        "data_coverage": split_meta,
        "split_config": {
            "train_split": train_split,
            "val_split": val_split,
            "tune_hyperparameters": tune,
            "use_gpu": use_gpu,
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
    train_split: float,
    val_split: float,
    data_source: str,
    db_path: str,
    start_date: str | None,
    end_date: str | None,
    years: int,
):
    dataset = build_global_classification_dataset(
        data_source=data_source,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        years=years,
        train_split=train_split,
        val_split=val_split,
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
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train split")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio within training region",
    )
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU tree method when available")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "yfinance"],
        default="db",
        help="Training data source",
    )
    parser.add_argument("--db-path", type=str, default="data/financial_advisor.db", help="SQLite DB path")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=10, help="Fallback lookback years when start-date is omitted")
    args = parser.parse_args()

    train_xgboost(
        tune=not args.no_tune,
        train_split=args.train_split,
        val_split=args.val_split,
        output_dir=args.output_dir,
        seed=args.seed,
        use_gpu=args.gpu,
        data_source=args.data_source,
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        years=args.years,
    )
