import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.backtest_models import FEATURE_COLS, _load_model, calculate_metrics_for_backtest
from src.data.stock_data import StockDataProcessor
from src.database.dal import DataAccessLayer
from src.models.data_sources import load_market_data
from src.models.trading_config import predictions_to_signals


def _resolve_model_path(model_arg: str) -> tuple[str, str]:
    normalized = model_arg.strip().lower()
    if normalized == "rf":
        return "models/random_forest_global.pkl", "rf"
    if normalized == "xgb":
        return "models/xgboost_global.json", "xgb"
    return model_arg, Path(model_arg).stem


def run_universe_backtest(
    model_arg: str,
    db_path: str = "data/financial_advisor.db",
    load_start: str = "2025-09-01",
    eval_start: str = "2026-01-01",
    end_date: str = "2026-03-20",
    transaction_cost: float = 0.001,
    output_dir: str = "results",
) -> tuple[pd.DataFrame, dict]:
    model_path, model_label = _resolve_model_path(model_arg)
    model, model_kind = _load_model(model_path)
    if model_kind != "classification":
        raise ValueError("Universe backtest currently supports Random Forest and XGBoost classification models only.")

    dal = DataAccessLayer(db_path=db_path)
    tickers = dal.get_all_tickers()

    rows = []
    errors = []

    for ticker in tickers:
        try:
            processor = StockDataProcessor(ticker)
            df = load_market_data(
                ticker=ticker,
                source="db",
                db_path=db_path,
                start_date=load_start,
                end_date=end_date,
                years=10,
            )
            if df.empty:
                errors.append({"ticker": ticker, "error": "no_data"})
                continue

            df = processor.add_technical_indicators(df)
            full_df = df[(df.index >= load_start) & (df.index <= end_date)].copy()
            if full_df.empty:
                errors.append({"ticker": ticker, "error": "no_indicator_ready_rows"})
                continue

            X = full_df[FEATURE_COLS].values
            predictions = model.predict(X)

            aligned_df = full_df.copy()
            aligned_df["Prediction"] = predictions
            raw_signal = predictions_to_signals(predictions, index=aligned_df.index)
            aligned_df["Signal"] = raw_signal.shift(1).fillna(0)
            aligned_df["Market_Return"] = aligned_df["Close"].pct_change().fillna(0)
            aligned_df["Strategy_Return_Raw"] = aligned_df["Signal"] * aligned_df["Market_Return"]
            aligned_df["Trade"] = aligned_df["Signal"].diff().abs().fillna(0)
            if len(aligned_df) and aligned_df["Signal"].iloc[0] != 0:
                aligned_df.iloc[0, aligned_df.columns.get_loc("Trade")] = 1
            aligned_df["Cost"] = aligned_df["Trade"] * transaction_cost
            aligned_df["Strategy_Return"] = aligned_df["Strategy_Return_Raw"] - aligned_df["Cost"]

            eval_df = aligned_df[(aligned_df.index >= eval_start) & (aligned_df.index <= end_date)].copy()
            if eval_df.empty:
                errors.append({"ticker": ticker, "error": "no_eval_rows"})
                continue

            strategy = calculate_metrics_for_backtest(eval_df["Strategy_Return"])
            benchmark = calculate_metrics_for_backtest(eval_df["Market_Return"])

            rows.append(
                {
                    "ticker": ticker,
                    "num_rows": int(len(eval_df)),
                    "strategy_total_return": float(strategy["total_return"]),
                    "strategy_annualized_return": float(strategy["annualized_return"]),
                    "strategy_annualized_volatility": float(strategy["annualized_volatility"]),
                    "strategy_sharpe_ratio": float(strategy["sharpe_ratio"]),
                    "strategy_max_drawdown": float(strategy["max_drawdown"]),
                    "strategy_win_rate": float(strategy["win_rate"]),
                    "benchmark_total_return": float(benchmark["total_return"]),
                    "benchmark_annualized_return": float(benchmark["annualized_return"]),
                    "benchmark_annualized_volatility": float(benchmark["annualized_volatility"]),
                    "benchmark_sharpe_ratio": float(benchmark["sharpe_ratio"]),
                    "benchmark_max_drawdown": float(benchmark["max_drawdown"]),
                    "benchmark_win_rate": float(benchmark["win_rate"]),
                    "beats_benchmark_total_return": int(strategy["total_return"] > benchmark["total_return"]),
                }
            )
        except Exception as exc:
            errors.append({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})

    results_df = pd.DataFrame(rows)
    summary = {
        "model": model_label,
        "model_path": model_path,
        "load_start": load_start,
        "eval_start": eval_start,
        "end_date": end_date,
        "requested_tickers": len(tickers),
        "processed_tickers": int(len(results_df)),
        "failed_tickers": int(len(errors)),
        "mean_strategy_total_return": float(results_df["strategy_total_return"].mean()),
        "median_strategy_total_return": float(results_df["strategy_total_return"].median()),
        "mean_strategy_sharpe_ratio": float(results_df["strategy_sharpe_ratio"].mean()),
        "median_strategy_sharpe_ratio": float(results_df["strategy_sharpe_ratio"].median()),
        "mean_strategy_max_drawdown": float(results_df["strategy_max_drawdown"].mean()),
        "mean_benchmark_total_return": float(results_df["benchmark_total_return"].mean()),
        "median_benchmark_total_return": float(results_df["benchmark_total_return"].median()),
        "mean_benchmark_sharpe_ratio": float(results_df["benchmark_sharpe_ratio"].mean()),
        "tickers_with_positive_strategy_return": int((results_df["strategy_total_return"] > 0).sum()),
        "tickers_with_positive_benchmark_return": int((results_df["benchmark_total_return"] > 0).sum()),
        "tickers_beating_benchmark_total_return": int(results_df["beats_benchmark_total_return"].sum()),
        "mean_num_rows": float(results_df["num_rows"].mean()),
        "failed_examples": errors[:10],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_stub = f"universe_backtest_{model_label}_eval_{eval_start.replace('-', '')}_to_{end_date.replace('-', '')}"
    csv_path = output_path / f"{file_stub}.csv"
    json_path = output_path / f"{file_stub}_summary.json"

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nPer-ticker backtest rows saved to {csv_path}")
    print(f"Summary saved to {json_path}")

    return results_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a classification backtest across the full ticker universe using existing global model artifacts."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        help="Model selector (`rf`, `xgb`) or direct model artifact path.",
    )
    parser.add_argument("--db-path", type=str, default="data/financial_advisor.db", help="SQLite DB path")
    parser.add_argument(
        "--load-start",
        type=str,
        default="2025-09-01",
        help="Start date used to load history and warm up indicators (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--eval-start",
        type=str,
        default="2026-01-01",
        help="Start date for the scored evaluation window (YYYY-MM-DD)",
    )
    parser.add_argument("--end-date", type=str, default="2026-03-20", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost (0.001 = 0.1%%)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for CSV and summary JSON")
    args = parser.parse_args()

    run_universe_backtest(
        model_arg=args.model,
        db_path=args.db_path,
        load_start=args.load_start,
        eval_start=args.eval_start,
        end_date=args.end_date,
        transaction_cost=args.cost,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
