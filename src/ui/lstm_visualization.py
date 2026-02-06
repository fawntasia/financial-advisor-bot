"""Utilities for loading LSTM artifacts and building visualization-ready datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.stock_data import StockDataProcessor
from src.database.dal import DataAccessLayer

DEFAULT_FEATURE_COLUMNS = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


@dataclass(frozen=True)
class LSTMArtifactPaths:
    """Resolved artifact paths for the newest LSTM run."""

    model_path: Path
    scaler_path: Path
    metadata_path: Optional[Path] = None


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_latest_lstm_artifacts(models_dir: Path = Path("models")) -> LSTMArtifactPaths:
    """Locate the latest LSTM model artifacts from the models directory."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    metadata_candidates = sorted(
        models_dir.glob("lstm_*_metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for metadata_path in metadata_candidates:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model_raw = metadata.get("model_path", "")
        scaler_raw = metadata.get("scaler_path", "")
        if not isinstance(model_raw, str) or not model_raw.strip():
            continue
        if not isinstance(scaler_raw, str) or not scaler_raw.strip():
            continue

        model_path = _resolve_path(model_raw)
        scaler_path = _resolve_path(scaler_raw)

        if model_path.is_file() and scaler_path.is_file():
            return LSTMArtifactPaths(
                model_path=model_path,
                scaler_path=scaler_path,
                metadata_path=metadata_path.resolve(),
            )

    model_candidates = sorted(
        models_dir.glob("lstm_*.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not model_candidates:
        raise FileNotFoundError("No LSTM `.keras` model artifacts were found in `models/`.")

    model_path = model_candidates[0].resolve()
    inferred_scaler = model_path.with_name(f"{model_path.stem}_scalers.joblib")
    if inferred_scaler.exists():
        return LSTMArtifactPaths(model_path=model_path, scaler_path=inferred_scaler, metadata_path=None)

    scaler_candidates = sorted(
        models_dir.glob("lstm_*_scalers.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not scaler_candidates:
        raise FileNotFoundError("No LSTM scaler artifact (`*_scalers.joblib`) found in `models/`.")

    return LSTMArtifactPaths(
        model_path=model_path,
        scaler_path=scaler_candidates[0].resolve(),
        metadata_path=None,
    )


@lru_cache(maxsize=2)
def _load_cached_lstm_artifacts(model_path: str, scaler_path: str, metadata_path: str) -> Dict:
    """
    Load and cache model/scalers to avoid reloading on every Streamlit rerun.
    """
    from src.models.lstm_model import LSTMModel

    model = LSTMModel()
    model.load(model_path)

    scaler_bundle = joblib.load(scaler_path)

    metadata: Dict = {}
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if "sequence_length" not in metadata:
        metadata["sequence_length"] = int(model.sequence_length)
    if "feature_columns" not in metadata:
        metadata["feature_columns"] = DEFAULT_FEATURE_COLUMNS

    return {
        "model": model,
        "scalers": scaler_bundle,
        "metadata": metadata,
    }


def _to_market_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert DAL price rows to a market DataFrame indexed by date."""
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


def _prepare_feature_frame(ticker: str, market_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Build feature matrix with indicators consistent with training."""
    processor = StockDataProcessor(ticker=ticker)
    feature_df = processor.add_technical_indicators(market_df.copy())
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    missing_cols = [col for col in feature_columns if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for inference: {missing_cols}")

    return feature_df


def _select_scalers_for_ticker(scaler_bundle: object, ticker: str):
    """Pick per-ticker scalers, with fallback for single-scaler artifacts."""
    ticker_upper = ticker.upper()

    if isinstance(scaler_bundle, dict):
        # New all-ticker format: {ticker: {"feature_scaler": ..., "target_scaler": ...}}
        if ticker_upper in scaler_bundle:
            item = scaler_bundle[ticker_upper]
            if isinstance(item, dict) and "feature_scaler" in item and "target_scaler" in item:
                return item["feature_scaler"], item["target_scaler"], ticker_upper

        # Single-object format fallback: {"feature_scaler": ..., "target_scaler": ...}
        if "feature_scaler" in scaler_bundle and "target_scaler" in scaler_bundle:
            return scaler_bundle["feature_scaler"], scaler_bundle["target_scaler"], ticker_upper

        # Fallback to first available ticker scaler.
        first_key = next(iter(scaler_bundle), None)
        if first_key is not None:
            item = scaler_bundle[first_key]
            if isinstance(item, dict) and "feature_scaler" in item and "target_scaler" in item:
                return item["feature_scaler"], item["target_scaler"], str(first_key)

    raise ValueError("Unable to find valid feature/target scalers in the LSTM scaler artifact.")


def _compute_rsi(closes: List[float], period: int = 14) -> float:
    """Compute a simple rolling RSI approximation from close prices."""
    if len(closes) <= period:
        return 50.0

    deltas = np.diff(np.asarray(closes[-(period + 1) :], dtype=float))
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    avg_gain = float(gains.mean()) if gains.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _build_backtest_frame(
    model,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int,
    feature_scaler,
    target_scaler,
    eval_window: int,
) -> pd.DataFrame:
    """Create a recent actual-vs-predicted window for model-fit visualization."""
    feature_values = feature_df[feature_columns].values
    scaled_features = feature_scaler.transform(feature_values)

    X_eval = []
    eval_dates = []
    for i in range(sequence_length, len(scaled_features)):
        X_eval.append(scaled_features[i - sequence_length : i, :])
        eval_dates.append(feature_df.index[i])

    if not X_eval:
        return pd.DataFrame(columns=["Actual_Close", "Predicted_Close"])

    X_eval_arr = np.asarray(X_eval, dtype=np.float32)
    predicted_scaled = model.predict(X_eval_arr).reshape(-1, 1)
    predicted_close = target_scaler.inverse_transform(predicted_scaled).ravel()
    actual_close = feature_df["Close"].iloc[sequence_length:].values

    backtest_df = pd.DataFrame(
        {
            "Actual_Close": actual_close,
            "Predicted_Close": predicted_close,
        },
        index=pd.DatetimeIndex(eval_dates),
    )

    if eval_window > 0:
        backtest_df = backtest_df.tail(eval_window)

    return backtest_df


def _build_forecast_frame(
    model,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int,
    feature_scaler,
    target_scaler,
    horizon: int,
) -> pd.DataFrame:
    """Generate recursive forward forecast prices for the next `horizon` business days."""
    if horizon <= 0:
        return pd.DataFrame(columns=["Predicted_Close"])

    last_sequence = feature_df[feature_columns].tail(sequence_length).values
    sequence_scaled = feature_scaler.transform(last_sequence).astype(np.float32)

    close_history = feature_df["Close"].tolist()

    ema12_prev = feature_df["Close"].ewm(span=12, adjust=False).mean().iloc[-1]
    ema26_prev = feature_df["Close"].ewm(span=26, adjust=False).mean().iloc[-1]
    macd_series = feature_df["Close"].ewm(span=12, adjust=False).mean() - feature_df["Close"].ewm(
        span=26, adjust=False
    ).mean()
    signal_prev = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]

    alpha12 = 2.0 / (12.0 + 1.0)
    alpha26 = 2.0 / (26.0 + 1.0)
    alpha9 = 2.0 / (9.0 + 1.0)

    last_known = feature_df.iloc[-1].to_dict()
    forecast_prices: List[float] = []

    for _ in range(horizon):
        pred_scaled = model.predict(sequence_scaled[np.newaxis, :, :]).reshape(-1, 1)
        pred_close = float(target_scaler.inverse_transform(pred_scaled)[0, 0])
        forecast_prices.append(pred_close)
        close_history.append(pred_close)

        ema12 = alpha12 * pred_close + (1 - alpha12) * ema12_prev
        ema26 = alpha26 * pred_close + (1 - alpha26) * ema26_prev
        macd = ema12 - ema26
        signal = alpha9 * macd + (1 - alpha9) * signal_prev

        feature_values = {
            "Close": pred_close,
            "SMA_20": float(np.mean(close_history[-20:])),
            "SMA_50": float(np.mean(close_history[-50:])),
            "RSI": _compute_rsi(close_history, period=14),
            "MACD": float(macd),
            "Signal_Line": float(signal),
        }

        next_row = np.array(
            [[feature_values.get(col, float(last_known.get(col, 0.0))) for col in feature_columns]],
            dtype=float,
        )

        next_scaled = feature_scaler.transform(next_row).astype(np.float32)[0]
        sequence_scaled = np.vstack([sequence_scaled[1:], next_scaled])

        ema12_prev = ema12
        ema26_prev = ema26
        signal_prev = signal

    future_dates = pd.bdate_range(feature_df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"Predicted_Close": forecast_prices}, index=future_dates)


def _build_technical_chart_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build chart-ready technical columns expected by ChartGenerator."""
    chart_df = feature_df.copy()

    chart_df["MACD_Signal"] = chart_df["Signal_Line"]
    chart_df["MACD_Hist"] = chart_df["MACD"] - chart_df["MACD_Signal"]

    bb_middle = chart_df["Close"].rolling(window=20).mean()
    bb_std = chart_df["Close"].rolling(window=20).std()
    chart_df["BB_Middle"] = bb_middle
    chart_df["BB_Upper"] = bb_middle + (2 * bb_std)
    chart_df["BB_Lower"] = bb_middle - (2 * bb_std)

    return chart_df.bfill().ffill()


def generate_lstm_visualization_data(
    ticker: str,
    dal: DataAccessLayer,
    history_window: int = 180,
    forecast_horizon: int = 7,
    eval_window: int = 90,
) -> Dict:
    """
    Build datasets and metrics needed by the stock-analysis frontend view.
    """
    latest_date = dal.get_latest_price_date(ticker)
    if not latest_date:
        raise ValueError(f"No stock price history found for ticker {ticker}.")

    raw_prices = dal.get_stock_prices(ticker, "1900-01-01", latest_date)
    market_df = _to_market_dataframe(raw_prices)
    if market_df.empty:
        raise ValueError(f"Unable to build market DataFrame for ticker {ticker}.")

    artifact_paths = resolve_latest_lstm_artifacts()
    loaded = _load_cached_lstm_artifacts(
        str(artifact_paths.model_path),
        str(artifact_paths.scaler_path),
        str(artifact_paths.metadata_path) if artifact_paths.metadata_path else "",
    )

    model = loaded["model"]
    scaler_bundle = loaded["scalers"]
    metadata = loaded["metadata"]

    feature_columns = list(metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS))
    sequence_length = int(metadata.get("sequence_length", 60))

    feature_df = _prepare_feature_frame(ticker, market_df, feature_columns)
    if len(feature_df) <= sequence_length:
        raise ValueError(
            f"Ticker {ticker} has insufficient history ({len(feature_df)} rows) for sequence length {sequence_length}."
        )

    feature_scaler, target_scaler, scaler_ticker = _select_scalers_for_ticker(scaler_bundle, ticker)

    backtest_df = _build_backtest_frame(
        model=model,
        feature_df=feature_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        eval_window=eval_window,
    )

    forecast_df = _build_forecast_frame(
        model=model,
        feature_df=feature_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        horizon=forecast_horizon,
    )

    technical_df = _build_technical_chart_frame(feature_df).tail(history_window)
    history_df = feature_df[["Close"]].tail(history_window)

    rmse = float("nan")
    mape = float("nan")
    if not backtest_df.empty:
        errors = backtest_df["Actual_Close"] - backtest_df["Predicted_Close"]
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        denom = backtest_df["Actual_Close"].replace(0, np.nan)
        mape = float(np.nanmean(np.abs(errors / denom)) * 100.0)

    last_close = float(feature_df["Close"].iloc[-1])
    next_pred_close = float(forecast_df["Predicted_Close"].iloc[0]) if not forecast_df.empty else float("nan")

    return {
        "history_df": history_df,
        "technical_df": technical_df,
        "backtest_df": backtest_df,
        "forecast_df": forecast_df,
        "last_close": last_close,
        "next_predicted_close": next_pred_close,
        "rmse": rmse,
        "mape": mape,
        "sequence_length": sequence_length,
        "feature_columns": feature_columns,
        "scaler_ticker": scaler_ticker,
        "artifact_paths": {
            "model_path": str(artifact_paths.model_path),
            "scaler_path": str(artifact_paths.scaler_path),
            "metadata_path": str(artifact_paths.metadata_path) if artifact_paths.metadata_path else "",
        },
    }
