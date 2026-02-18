import argparse
import os
import sys
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.stock_data import StockDataProcessor
from src.models.data_sources import load_market_data
from src.models.evaluation import calculate_metrics
from src.models.random_forest_model import RandomForestModel
from src.models.trading_config import predictions_to_signals
from src.models.xgboost_model import XGBoostModel

FEATURE_COLS = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


def _load_model(model_path: str):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pkl":
        model = RandomForestModel()
        model.load(model_path)
        return model, "classification"
    if ext == ".json":
        model = XGBoostModel()
        model.load(model_path)
        return model, "classification"
    if ext in {".keras", ".h5"}:
        from src.models.lstm_model import LSTMModel

        model = LSTMModel()
        model.load(model_path)
        return model, "lstm"
    raise ValueError("Unsupported model format. Use .pkl, .json, or .keras/.h5")


def _infer_lstm_scaler_path(model_path: str) -> str:
    stem, _ = os.path.splitext(model_path)
    return f"{stem}_scalers.joblib"


def _lstm_predictions(
    model,
    backtest_df: pd.DataFrame,
    ticker: str,
    scaler_path: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    scaler_payload = joblib.load(scaler_path)
    if not isinstance(scaler_payload, dict) or ticker not in scaler_payload:
        raise ValueError(
            f"Scaler payload missing ticker '{ticker}'. Provide scaler file produced by train_lstm.py."
        )

    feature_scaler = scaler_payload[ticker]["feature_scaler"]
    target_scaler = scaler_payload[ticker]["target_scaler"]
    sequence_length = model.sequence_length

    if len(backtest_df) <= sequence_length:
        raise ValueError("Not enough rows in backtest window for LSTM sequence length.")

    features = backtest_df[FEATURE_COLS].values
    scaled = feature_scaler.transform(features)

    X = []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length : i, :])
    X = np.asarray(X, dtype=np.float32)

    pred_scaled = model.predict(X).reshape(-1, 1)
    pred_price = target_scaler.inverse_transform(pred_scaled).reshape(-1)

    aligned = backtest_df.iloc[sequence_length:].copy()
    prev_close = backtest_df["Close"].shift(1).iloc[sequence_length:].values
    pred_direction = (pred_price > prev_close).astype(int)
    aligned["Predicted_Close"] = pred_price
    return aligned, pred_direction


def calculate_metrics_for_backtest(daily_returns: pd.Series) -> dict:
    metrics = calculate_metrics(daily_returns)
    in_market_returns = daily_returns[daily_returns != 0]
    metrics["win_rate"] = float((in_market_returns > 0).mean()) if len(in_market_returns) else 0.0
    return metrics


def run_backtest(
    ticker,
    model_path,
    start_date="2023-01-01",
    end_date="2023-12-31",
    transaction_cost=0.001,
    scaler_path: Optional[str] = None,
    data_source: str = "db",
    db_path: str = "data/financial_advisor.db",
    output_dir: str = "results",
    years: int = 10,
):
    """
    Run backtest for a given ticker and model.
    Supports RandomForest (.pkl), XGBoost (.json), and LSTM (.keras/.h5).
    """
    ticker = ticker.upper()
    print(f"Running backtest for {ticker} using model {model_path} (source={data_source})...")

    processor = StockDataProcessor(ticker)
    df = load_market_data(
        ticker=ticker,
        source=data_source,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        years=years,
    )
    if df.empty:
        print(f"No data found for {ticker} in range {start_date} to {end_date}")
        return
    df = processor.add_technical_indicators(df)

    backtest_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    if backtest_df.empty:
        print(f"No indicator-ready data found for {ticker} in range {start_date} to {end_date}")
        return

    model, model_kind = _load_model(model_path)

    if model_kind == "classification":
        X = backtest_df[FEATURE_COLS].values
        predictions = model.predict(X)
        aligned_df = backtest_df.copy()
    else:
        scaler_path = scaler_path or _infer_lstm_scaler_path(model_path)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"LSTM scaler file not found at {scaler_path}. Pass --scaler path generated during training."
            )
        aligned_df, predictions = _lstm_predictions(model, backtest_df, ticker, scaler_path)

    aligned_df["Prediction"] = predictions
    raw_signal = predictions_to_signals(predictions, index=aligned_df.index)
    aligned_df["Signal"] = raw_signal.shift(1).fillna(0)

    aligned_df["Market_Return"] = aligned_df["Close"].pct_change().fillna(0)
    aligned_df["Strategy_Return_Raw"] = aligned_df["Signal"] * aligned_df["Market_Return"]

    aligned_df["Trade"] = aligned_df["Signal"].diff().abs().fillna(0)
    if aligned_df["Signal"].iloc[0] != 0:
        aligned_df.iloc[0, aligned_df.columns.get_loc("Trade")] = 1

    aligned_df["Cost"] = aligned_df["Trade"] * transaction_cost
    aligned_df["Strategy_Return"] = aligned_df["Strategy_Return_Raw"] - aligned_df["Cost"]

    strategy_metrics = calculate_metrics_for_backtest(aligned_df["Strategy_Return"])
    benchmark_metrics = calculate_metrics_for_backtest(aligned_df["Market_Return"])

    aligned_df["Cumulative_Strategy"] = (aligned_df["Strategy_Return"] + 1).cumprod()
    aligned_df["Cumulative_Market"] = (aligned_df["Market_Return"] + 1).cumprod()

    print("\n=== Strategy Metrics ===")
    for key, value in strategy_metrics.items():
        if "sharpe" in key.lower() or "win_rate" in key.lower():
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.2%}")

    print("\n=== Benchmark Metrics ===")
    for key, value in benchmark_metrics.items():
        if "sharpe" in key.lower() or "win_rate" in key.lower():
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.2%}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"backtest_{ticker}_{start_date[:4]}.csv")
    aligned_df[
        [
            "Close",
            "Prediction",
            "Signal",
            "Strategy_Return",
            "Market_Return",
            "Cumulative_Strategy",
            "Cumulative_Market",
        ]
    ].to_csv(csv_path)
    print(f"\nBacktest rows saved to {csv_path}")

    return strategy_metrics, benchmark_metrics, aligned_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model backtesting.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model artifact")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost (0.001 = 0.1%%)")
    parser.add_argument(
        "--scaler",
        type=str,
        default=None,
        help="Optional scaler path for LSTM models (defaults to <model_stem>_scalers.joblib)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "yfinance"],
        default="db",
        help="Backtest data source",
    )
    parser.add_argument("--db-path", type=str, default="data/financial_advisor.db", help="SQLite DB path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for backtest CSV")
    parser.add_argument("--years", type=int, default=10, help="Fallback lookback years when start-date is omitted")

    args = parser.parse_args()
    run_backtest(
        ticker=args.ticker,
        model_path=args.model,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost=args.cost,
        scaler_path=args.scaler,
        data_source=args.data_source,
        db_path=args.db_path,
        output_dir=args.output_dir,
        years=args.years,
    )
