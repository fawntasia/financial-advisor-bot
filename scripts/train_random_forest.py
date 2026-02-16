import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.stock_data import StockDataProcessor
from src.database.dal import DataAccessLayer
from src.models.random_forest_model import RandomForestModel
from src.models.reproducibility import set_global_seeds


def _to_market_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert DAL stock_prices rows to market-style columns used by processors."""
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
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj Close"] if c in df.columns]
    return df[keep_cols].astype(float)


def train_rf(
    ticker: str,
    tune: bool = True,
    train_split: float = 0.8,
    val_split: float = 0.1,
    save_dir: str = "models",
    seed: int = 42,
):
    ticker = ticker.upper()
    print(f"Starting Random Forest training for {ticker}...")
    set_global_seeds(seed=seed, include_tensorflow=False)

    dal = DataAccessLayer()
    print("Fetching DB-backed price history...")
    price_df = dal.get_stock_prices(ticker, "1900-01-01", datetime.now().strftime("%Y-%m-%d"))
    market_df = _to_market_dataframe(price_df)
    if market_df.empty:
        raise ValueError(f"No DB price history available for {ticker}.")

    processor = StockDataProcessor(ticker)
    enriched_df = processor.add_technical_indicators(market_df).dropna().copy()
    if len(enriched_df) < 30:
        raise ValueError("Not enough rows after indicator warm-up to train model.")

    print("Preparing leakage-safe train/validation/test splits...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        split_meta,
    ) = processor.prepare_for_classification_splits(
        enriched_df,
        target_col="Close",
        train_split=train_split,
        val_split=val_split,
        return_metadata=True,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    model = RandomForestModel(random_state=seed)
    print(f"Training model (Tuning: {tune})...")
    metrics = model.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparameters=tune,
        val_prices=split_meta.get("val_prices"),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"random_forest_{ticker}_{timestamp}.pkl")
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
    train_idx = split_meta.get("train_index")
    val_idx = split_meta.get("val_index")
    test_idx = split_meta.get("test_index")

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "seed": seed,
        "ticker": ticker,
        "model_name": model.get_name(),
        "model_path": model_path,
        "feature_columns": list(feature_cols),
        "train_split": train_split,
        "validation_split_within_train": val_split,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "coverage": {
            "train_start": str(train_idx.min()) if len(train_idx) else None,
            "train_end": str(train_idx.max()) if len(train_idx) else None,
            "val_start": str(val_idx.min()) if len(val_idx) else None,
            "val_end": str(val_idx.max()) if len(val_idx) else None,
            "test_start": str(test_idx.min()) if len(test_idx) else None,
            "test_end": str(test_idx.max()) if len(test_idx) else None,
        },
        "decision_threshold": float(model.decision_threshold),
        "metrics": metrics,
        "feature_importances": feature_importances,
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel artifact saved to {model_path}")
    print(f"Run metadata saved to {metadata_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model (DB-backed).")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train split")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio within training region",
    )
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_rf(
        args.ticker,
        tune=not args.no_tune,
        train_split=args.train_split,
        val_split=args.val_split,
        save_dir=args.save_dir,
        seed=args.seed,
    )
