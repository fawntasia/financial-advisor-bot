import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.stock_data import StockDataProcessor
from src.database.dal import DataAccessLayer
from src.models.lstm_model import LSTMModel

FEATURE_COLS = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


def _to_market_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DAL stock_prices rows to a DataFrame compatible with StockDataProcessor.
    Expected output columns: Open, High, Low, Close, Volume, Adj Close with DatetimeIndex.
    """
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
    df = df[keep_cols].astype(float)
    return df


def _create_sequences(dataset: np.ndarray, sequence_length: int, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences from scaled features."""
    X, y = [], []
    for i in range(sequence_length, len(dataset)):
        X.append(dataset[i - sequence_length : i, :])
        y.append(dataset[i, target_idx])

    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _prepare_ticker_sequences(
    ticker: str,
    raw_prices: pd.DataFrame,
    sequence_length: int,
    train_split: float,
) -> Optional[Dict]:
    """
    Prepare per-ticker train/test sequences from DB prices.
    Returns None when there is not enough history for sequence training.
    """
    if raw_prices.empty:
        return None

    processor = StockDataProcessor(ticker=ticker)
    df = processor.add_technical_indicators(raw_prices)
    df = df.dropna().copy()

    # Minimum rows: one train sequence and at least one held-out test point.
    min_rows = sequence_length + 2
    if len(df) < min_rows:
        return None

    split_idx = int(len(df) * train_split)
    min_split = sequence_length + 1
    max_split = len(df) - 1
    split_idx = max(min_split, min(split_idx, max_split))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[FEATURE_COLS].values)
    target_scaler.fit(train_df[["Close"]].values)

    train_scaled = feature_scaler.transform(train_df[FEATURE_COLS].values)
    test_scaled = feature_scaler.transform(test_df[FEATURE_COLS].values)

    target_idx = FEATURE_COLS.index("Close")

    X_train, y_train = _create_sequences(train_scaled, sequence_length, target_idx)
    if X_train.size == 0:
        return None

    # Keep sequence continuity into validation window.
    test_context = np.vstack([train_scaled[-sequence_length:], test_scaled])
    X_test, y_test = _create_sequences(test_context, sequence_length, target_idx)

    if X_test.size == 0:
        return None

    return {
        "ticker": ticker,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def _aggregate_sequences(per_ticker_batches: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack all per-ticker arrays into a single global training set."""
    X_train = np.concatenate([b["X_train"] for b in per_ticker_batches], axis=0).astype(np.float32)
    y_train = np.concatenate([b["y_train"] for b in per_ticker_batches], axis=0).astype(np.float32)
    X_test = np.concatenate([b["X_test"] for b in per_ticker_batches], axis=0).astype(np.float32)
    y_test = np.concatenate([b["y_test"] for b in per_ticker_batches], axis=0).astype(np.float32)

    test_ticker_labels = np.concatenate(
        [np.array([b["ticker"]] * len(b["y_test"]), dtype=object) for b in per_ticker_batches],
        axis=0,
    )
    return X_train, y_train, X_test, y_test, test_ticker_labels


def train_lstm(
    ticker: str = "ALL",
    epochs: int = 50,
    batch_size: int = 32,
    save_dir: str = "models",
    sequence_length: int = 60,
    train_split: float = 0.8,
    max_tickers: Optional[int] = None,
) -> Tuple[LSTMModel, object]:
    """
    Train a single Keras LSTM using SQLite-backed price history.
    By default, all tickers in the DB are included.
    """
    dal = DataAccessLayer()
    all_tickers = dal.get_all_tickers()

    if ticker.upper() == "ALL":
        selected_tickers = all_tickers
    else:
        selected_tickers = [ticker.upper()]

    if max_tickers is not None:
        selected_tickers = selected_tickers[:max_tickers]

    if not selected_tickers:
        raise ValueError("No tickers available for training.")

    print(f"Preparing DB-backed LSTM dataset from {len(selected_tickers)} ticker(s)...")

    per_ticker_batches = []
    skipped = []

    for i, t in enumerate(selected_tickers, start=1):
        if i % 25 == 0 or i == len(selected_tickers):
            print(f"Processed {i}/{len(selected_tickers)} tickers...")

        price_df = dal.get_stock_prices(t, "1900-01-01", datetime.now().strftime("%Y-%m-%d"))
        market_df = _to_market_dataframe(price_df)

        batch = _prepare_ticker_sequences(
            ticker=t,
            raw_prices=market_df,
            sequence_length=sequence_length,
            train_split=train_split,
        )

        if batch is None:
            skipped.append(t)
            continue
        per_ticker_batches.append(batch)

    if not per_ticker_batches:
        raise ValueError("No ticker produced valid LSTM sequences. Check DB data coverage.")

    X_train, y_train, X_test, y_test, test_ticker_labels = _aggregate_sequences(per_ticker_batches)
    used_tickers = [b["ticker"] for b in per_ticker_batches]
    scaler_by_ticker = {
        b["ticker"]: {
            "feature_scaler": b["feature_scaler"],
            "target_scaler": b["target_scaler"],
        }
        for b in per_ticker_batches
    }

    print(f"Tickers used: {len(used_tickers)} / {len(selected_tickers)}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = LSTMModel(sequence_length=sequence_length, n_features=X_train.shape[2])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "all" if ticker.upper() == "ALL" else ticker.upper()
    model_filename = f"lstm_{model_tag}_{timestamp}.keras"
    save_path = os.path.join(save_dir, model_filename)

    print("Training model...")
    history = model.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        patience=5,
        verbose=1,
    )

    print("Evaluating model...")
    test_mse = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Global Test Loss (scaled MSE): {test_mse:.6f}")

    predictions = model.predict(X_test).reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)

    per_ticker_rmse = {}
    for t in used_tickers:
        mask = test_ticker_labels == t
        if not np.any(mask):
            continue

        scaler = scaler_by_ticker[t]["target_scaler"]
        pred_real = scaler.inverse_transform(predictions[mask]).ravel()
        actual_real = scaler.inverse_transform(y_test_reshaped[mask]).ravel()
        rmse = float(np.sqrt(np.mean((pred_real - actual_real) ** 2)))
        per_ticker_rmse[t] = rmse

    avg_rmse = float(np.mean(list(per_ticker_rmse.values()))) if per_ticker_rmse else float("nan")
    print(f"Average per-ticker RMSE (real price): ${avg_rmse:.2f}")

    # Persist scalers and training metadata for inference/reproducibility.
    model_stem, _ = os.path.splitext(save_path)
    scaler_path = f"{model_stem}_scalers.joblib"
    metadata_path = f"{model_stem}_metadata.json"

    joblib.dump(scaler_by_ticker, scaler_path)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_path": save_path,
        "scaler_path": scaler_path,
        "mode": "all_tickers" if ticker.upper() == "ALL" else "single_ticker",
        "requested_tickers": len(selected_tickers),
        "used_tickers": len(used_tickers),
        "skipped_tickers": skipped,
        "sequence_length": sequence_length,
        "feature_columns": FEATURE_COLS,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "global_test_mse_scaled": float(test_mse),
        "avg_per_ticker_rmse_price": avg_rmse,
    }
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best model saved to {save_path}")
    print(f"Scalers saved to {scaler_path}")
    print(f"Run metadata saved to {metadata_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction (DB-backed)")
    parser.add_argument(
        "--ticker",
        type=str,
        default="ALL",
        help="Ticker symbol (e.g. AAPL) or ALL for all tickers in DB",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--sequence_length", type=int, default=60, help="LSTM lookback window")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split per ticker")
    parser.add_argument(
        "--max_tickers",
        type=int,
        default=None,
        help="Optional cap for quick test runs",
    )

    args = parser.parse_args()
    train_lstm(
        ticker=args.ticker,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        sequence_length=args.sequence_length,
        train_split=args.train_split,
        max_tickers=args.max_tickers,
    )
