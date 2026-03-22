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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
sys.path.append(PROJECT_ROOT)

from src.data.stock_data import StockDataProcessor
from src.models.data_sources import load_market_data, load_ticker_universe
from src.models.io_utils import ensure_parent_dir, get_library_versions
from src.models.reproducibility import set_global_seeds

FEATURE_COLS = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]

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


def _configure_tensorflow_runtime(
    use_gpu: bool = True,
    enable_mixed_precision: bool = True,
) -> Dict[str, object]:
    """Configure TensorFlow runtime to use available CPU/GPU resources."""
    runtime: Dict[str, object] = {
        "cpu_count": int(os.cpu_count() or 1),
        "use_gpu_requested": bool(use_gpu),
        "mixed_precision_requested": bool(enable_mixed_precision),
    }
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - optional dependency path
        runtime["status"] = f"tensorflow_unavailable: {exc}"
        return runtime

    cpu_count = int(runtime["cpu_count"])
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, min(8, cpu_count // 2 or 1)))
        runtime["cpu_threads_configured"] = True
    except RuntimeError as exc:
        runtime["cpu_threads_configured"] = False
        runtime["cpu_threads_note"] = str(exc)

    gpu_devices = tf.config.list_physical_devices("GPU")
    runtime["gpu_count"] = len(gpu_devices)
    runtime["gpu_enabled"] = bool(use_gpu and gpu_devices)

    if not use_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            runtime["gpu_visibility"] = "disabled_by_flag"
        except RuntimeError as exc:
            runtime["gpu_visibility"] = f"disable_failed: {exc}"
    else:
        for gpu in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                continue

    if runtime["gpu_enabled"] and enable_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            runtime["mixed_precision_policy"] = "mixed_float16"
        except Exception as exc:
            runtime["mixed_precision_policy"] = f"failed: {exc}"
    else:
        runtime["mixed_precision_policy"] = "float32"

    try:
        tf.config.optimizer.set_jit(True)
        runtime["xla_jit"] = True
    except Exception:
        runtime["xla_jit"] = False

    return runtime


def _create_sequences(
    dataset: np.ndarray,
    sequence_length: int,
    target_idx: int,
    target_dates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding window sequences from scaled features with aligned target dates."""
    X, y = [], []
    seq_dates = []
    for i in range(sequence_length, len(dataset)):
        X.append(dataset[i - sequence_length : i, :])
        y.append(dataset[i, target_idx])
        if target_dates is not None:
            seq_dates.append(target_dates[i])

    if not X:
        empty_dates = np.array([], dtype="datetime64[ns]")
        return np.array([]), np.array([]), empty_dates
    if target_dates is None:
        seq_dates_arr = np.array([], dtype="datetime64[ns]")
    else:
        seq_dates_arr = np.array(seq_dates, dtype="datetime64[ns]")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), seq_dates_arr


def _slice_by_date(df: pd.DataFrame, start_date: str, end_date: str | None) -> pd.DataFrame:
    """Slice a Date-indexed DataFrame by inclusive calendar bounds."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index).normalize()
    start_ts = pd.Timestamp(start_date).normalize()
    mask = idx >= start_ts
    if end_date is not None:
        end_ts = pd.Timestamp(end_date).normalize()
        mask &= idx <= end_ts
    return df.loc[mask].copy()


def _prepare_ticker_sequences(
    ticker: str,
    raw_prices: pd.DataFrame,
    sequence_length: int,
    train_start_date: str,
    train_end_date: str,
    val_start_date: str,
    val_end_date: str,
    test_start_date: str | None,
    test_end_date: str | None,
) -> Optional[Dict]:
    """
    Prepare per-ticker train/test sequences from market prices.
    Returns None when there is not enough history for sequence training.
    """
    if raw_prices.empty:
        return None

    processor = StockDataProcessor(ticker=ticker)
    df = processor.add_technical_indicators(raw_prices)
    df = df.dropna().copy()

    train_df = _slice_by_date(df, train_start_date, train_end_date)
    val_df = _slice_by_date(df, val_start_date, val_end_date)
    resolved_test_start = test_start_date
    if resolved_test_start is None:
        resolved_test_start = (pd.Timestamp(val_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test_df = _slice_by_date(df, resolved_test_start, test_end_date)

    if len(train_df) < (sequence_length + 1):
        return None
    if val_df.empty or test_df.empty:
        return None

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[FEATURE_COLS].values)
    target_scaler.fit(train_df[["Close"]].values)

    train_scaled = feature_scaler.transform(train_df[FEATURE_COLS].values)
    val_scaled = feature_scaler.transform(val_df[FEATURE_COLS].values)
    test_scaled = feature_scaler.transform(test_df[FEATURE_COLS].values)

    target_idx = FEATURE_COLS.index("Close")

    train_target_dates = pd.to_datetime(train_df.index).to_numpy(dtype="datetime64[ns]")
    val_target_dates = pd.to_datetime(val_df.index).to_numpy(dtype="datetime64[ns]")
    test_target_dates = pd.to_datetime(test_df.index).to_numpy(dtype="datetime64[ns]")

    X_train_fit, y_train_fit, train_fit_dates = _create_sequences(
        train_scaled,
        sequence_length,
        target_idx,
        target_dates=train_target_dates,
    )
    if X_train_fit.size == 0:
        return None

    val_context = np.vstack([train_scaled[-sequence_length:], val_scaled])
    val_context_dates = np.concatenate([train_target_dates[-sequence_length:], val_target_dates])
    X_val, y_val, val_dates = _create_sequences(
        val_context,
        sequence_length,
        target_idx,
        target_dates=val_context_dates,
    )
    if X_val.size == 0:
        return None

    pre_test_scaled = np.vstack([train_scaled, val_scaled])
    pre_test_dates = np.concatenate([train_target_dates, val_target_dates])
    test_context = np.vstack([pre_test_scaled[-sequence_length:], test_scaled])
    test_context_dates = np.concatenate([pre_test_dates[-sequence_length:], test_target_dates])
    X_test, y_test, test_dates = _create_sequences(
        test_context,
        sequence_length,
        target_idx,
        target_dates=test_context_dates,
    )

    if X_test.size == 0:
        return None

    return {
        "ticker": ticker,
        "X_train_fit": X_train_fit,
        "y_train_fit": y_train_fit,
        "train_fit_dates": train_fit_dates,
        "X_val": X_val,
        "y_val": y_val,
        "val_dates": val_dates,
        "X_test": X_test,
        "y_test": y_test,
        "test_dates": test_dates,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def _aggregate_sequences(
    per_ticker_batches: List[Dict],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Stack all per-ticker arrays into a single global training set."""
    X_train = np.concatenate([b["X_train_fit"] for b in per_ticker_batches], axis=0).astype(np.float32)
    y_train = np.concatenate([b["y_train_fit"] for b in per_ticker_batches], axis=0).astype(np.float32)
    train_dates = np.concatenate([b["train_fit_dates"] for b in per_ticker_batches], axis=0)
    train_ticker_labels = np.concatenate(
        [np.array([b["ticker"]] * len(b["y_train_fit"]), dtype=object) for b in per_ticker_batches],
        axis=0,
    )
    X_val = np.concatenate([b["X_val"] for b in per_ticker_batches], axis=0).astype(np.float32)
    y_val = np.concatenate([b["y_val"] for b in per_ticker_batches], axis=0).astype(np.float32)
    val_dates = np.concatenate([b["val_dates"] for b in per_ticker_batches], axis=0)
    val_ticker_labels = np.concatenate(
        [np.array([b["ticker"]] * len(b["y_val"]), dtype=object) for b in per_ticker_batches],
        axis=0,
    )
    X_test = np.concatenate([b["X_test"] for b in per_ticker_batches], axis=0).astype(np.float32)
    y_test = np.concatenate([b["y_test"] for b in per_ticker_batches], axis=0).astype(np.float32)
    test_dates = np.concatenate([b["test_dates"] for b in per_ticker_batches], axis=0)

    test_ticker_labels = np.concatenate(
        [np.array([b["ticker"]] * len(b["y_test"]), dtype=object) for b in per_ticker_batches],
        axis=0,
    )
    return (
        X_train,
        y_train,
        train_dates,
        train_ticker_labels,
        X_val,
        y_val,
        val_dates,
        val_ticker_labels,
        X_test,
        y_test,
        test_dates,
        test_ticker_labels,
    )


def train_lstm(
    ticker: str = "ALL",
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sequence_length: int = 60,
    max_tickers: Optional[int] = None,
    seed: int = 42,
    use_gpu: bool = True,
    enable_mixed_precision: bool = True,
    data_source: str = "db",
    db_path: str = "data/financial_advisor.db",
):
    """
    Train a single Keras LSTM using DB or yfinance price history.
    By default, all tickers from DB ticker universe are included.
    """
    from src.models.lstm_model import LSTMModel

    tf_seed_status = set_global_seeds(seed=seed, include_tensorflow=True)
    if tf_seed_status:
        print(f"Seed status: {tf_seed_status}")
    tf_runtime = _configure_tensorflow_runtime(use_gpu=use_gpu, enable_mixed_precision=enable_mixed_precision)
    print(f"TensorFlow runtime config: {tf_runtime}")

    if ticker.upper() == "ALL":
        selected_tickers = load_ticker_universe(db_path=db_path)
    else:
        selected_tickers = [ticker.upper()]

    if max_tickers is not None:
        selected_tickers = selected_tickers[:max_tickers]

    if not selected_tickers:
        raise ValueError("No tickers available for training.")

    print(f"Preparing LSTM dataset from {len(selected_tickers)} ticker(s), source={data_source}...")

    per_ticker_batches = []
    skipped = []

    for i, t in enumerate(selected_tickers, start=1):
        if i % 25 == 0 or i == len(selected_tickers):
            print(f"Processed {i}/{len(selected_tickers)} tickers...")

        market_df = load_market_data(
            ticker=t,
            source=data_source,
            db_path=db_path,
            start_date=DATA_WINDOW_START_DATE,
            end_date=DATA_WINDOW_END_DATE,
            years=10,
        )

        batch = _prepare_ticker_sequences(
            ticker=t,
            raw_prices=market_df,
            sequence_length=sequence_length,
            train_start_date=TRAIN_START_DATE,
            train_end_date=TRAIN_END_DATE,
            val_start_date=VALIDATION_START_DATE,
            val_end_date=VALIDATION_END_DATE,
            test_start_date=TEST_START_DATE,
            test_end_date=TEST_END_DATE,
        )

        if batch is None:
            skipped.append(t)
            continue
        per_ticker_batches.append(batch)

    if not per_ticker_batches:
        raise ValueError("No ticker produced valid LSTM sequences. Check data coverage.")

    (
        X_train,
        y_train,
        train_dates,
        train_ticker_labels,
        X_val,
        y_val,
        val_dates,
        val_ticker_labels,
        X_test,
        y_test,
        _test_dates,
        test_ticker_labels,
    ) = _aggregate_sequences(per_ticker_batches)

    used_tickers = [b["ticker"] for b in per_ticker_batches]
    scaler_by_ticker = {
        b["ticker"]: {
            "feature_scaler": b["feature_scaler"],
            "target_scaler": b["target_scaler"],
        }
        for b in per_ticker_batches
    }

    print(f"Tickers used: {len(used_tickers)} / {len(selected_tickers)}")
    print(
        "Train date range: "
        f"{pd.Timestamp(train_dates.min()).date()} -> {pd.Timestamp(train_dates.max()).date()}"
    )
    print(
        "Validation date range: "
        f"{pd.Timestamp(val_dates.min()).date()} -> {pd.Timestamp(val_dates.max()).date()}"
    )
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    model = LSTMModel(sequence_length=sequence_length, n_features=X_train.shape[2])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "all" if ticker.upper() == "ALL" else ticker.upper()
    model_filename = f"lstm_{model_tag}_{timestamp}.keras"
    model_path = os.path.join(output_dir, model_filename)

    print("Training model...")
    train_result = model.train(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_path,
        patience=5,
        verbose=1,
        X_val=X_val,
        y_val=y_val,
    )

    print("Evaluating model...")
    predictions = model.predict(X_test).reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    test_mse = float(np.mean((predictions.reshape(-1) - y_test.reshape(-1)) ** 2))
    print(f"Global Test Loss (scaled MSE): {test_mse:.6f}")

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

    model_stem, _ = os.path.splitext(model_path)
    scaler_path = f"{model_stem}_scalers.joblib"
    metadata_path = f"{model_stem}_metadata.json"
    manifest_path = f"{model_stem}.manifest.json"

    ensure_parent_dir(scaler_path)
    joblib.dump(scaler_by_ticker, scaler_path)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "mode": "all_tickers" if ticker.upper() == "ALL" else "single_ticker",
        "requested_tickers": len(selected_tickers),
        "used_tickers": len(used_tickers),
        "skipped_tickers": skipped,
        "sequence_length": sequence_length,
        "feature_columns": FEATURE_COLS,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "validation_split_basis": "hard_coded_calendar_dates",
        "hard_coded_split": {
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "validation_start": VALIDATION_START_DATE,
            "validation_end": VALIDATION_END_DATE,
            "test_start": TEST_START_DATE,
            "test_end": TEST_END_DATE,
        },
        "train_date_range": {
            "start": pd.Timestamp(train_dates.min()).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(train_dates.max()).strftime("%Y-%m-%d"),
        },
        "validation_date_range": {
            "start": pd.Timestamp(val_dates.min()).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(val_dates.max()).strftime("%Y-%m-%d"),
        },
        "train_ticker_count": int(np.unique(train_ticker_labels).size),
        "validation_ticker_count": int(np.unique(val_ticker_labels).size),
        "seed": seed,
        "tensorflow_runtime": tf_runtime,
        "global_test_mse_scaled": test_mse,
        "avg_per_ticker_rmse_price": avg_rmse,
        "train_metrics": train_result,
    }
    ensure_parent_dir(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    manifest = {
        "schema_version": 1,
        "model_kind": "lstm_regression",
        "model_path": model_path,
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "feature_columns": FEATURE_COLS,
        "data_source": {
            "source": data_source,
            "db_path": db_path if data_source == "db" else None,
            "start_date": DATA_WINDOW_START_DATE,
            "end_date": DATA_WINDOW_END_DATE,
            "years": None,
            "ticker": ticker.upper(),
        },
        "data_coverage": {
            "requested_tickers": len(selected_tickers),
            "used_tickers": len(used_tickers),
            "skipped_tickers": skipped,
        },
        "split_config": {
            "split_strategy": "hard_coded_calendar",
            "validation_split_basis": "hard_coded_calendar_dates",
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "validation_start": VALIDATION_START_DATE,
            "validation_end": VALIDATION_END_DATE,
            "test_start": TEST_START_DATE,
            "test_end": TEST_END_DATE,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "use_gpu_requested": bool(use_gpu),
            "mixed_precision_requested": bool(enable_mixed_precision),
        },
        "metrics": {
            "train_result": train_result,
            "global_test_mse_scaled": test_mse,
            "avg_per_ticker_rmse_price": avg_rmse,
        },
        "runtime": {"tensorflow": tf_runtime},
        "library_versions": get_library_versions(),
    }
    ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Best model saved to {model_path}")
    print(f"Scalers saved to {scaler_path}")
    print(f"Run metadata saved to {metadata_path}")
    print(f"Run manifest saved to {manifest_path}")

    return model, train_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction.")
    parser.add_argument(
        "--ticker",
        type=str,
        default="ALL",
        help="Ticker symbol (e.g. AAPL) or ALL for all tickers from DB ticker universe",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save model artifacts (default: project_root/models).",
    )
    parser.add_argument("--sequence-length", type=int, default=60, help="LSTM lookback window")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional cap for quick test runs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Use GPU acceleration when available (default).")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="Disable GPU and force CPU training.")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        action="store_true",
        help="Enable mixed precision when GPU is available (default).",
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable mixed precision and run float32.",
    )
    parser.set_defaults(mixed_precision=True)
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "yfinance"],
        default="db",
        help="Training data source",
    )
    parser.add_argument("--db-path", type=str, default="data/financial_advisor.db", help="SQLite DB path")

    args = parser.parse_args()
    train_lstm(
        ticker=args.ticker,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        max_tickers=args.max_tickers,
        seed=args.seed,
        use_gpu=args.gpu,
        enable_mixed_precision=args.mixed_precision,
        data_source=args.data_source,
        db_path=args.db_path,
    )
