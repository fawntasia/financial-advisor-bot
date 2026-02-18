import argparse
import json
import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.stock_data import StockDataProcessor
from src.models.data_sources import load_market_data
from src.models.random_forest_model import RandomForestModel
from src.models.reproducibility import set_global_seeds
from src.models.validation import WalkForwardValidator
from src.models.xgboost_model import XGBoostModel


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward validation.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb", "lstm"],
        help="Model type",
    )
    parser.add_argument("--train-years", type=int, default=3, help="Training window years")
    parser.add_argument("--val-months", type=int, default=3, help="Validation window months")
    parser.add_argument("--test-months", type=int, default=3, help="Testing window months")
    parser.add_argument("--step-months", type=int, default=3, help="Walk-forward step size in months")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for result artifacts")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "yfinance"],
        default="db",
        help="Evaluation data source",
    )
    parser.add_argument("--db-path", type=str, default="data/financial_advisor.db", help="SQLite DB path")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=10, help="Fallback lookback years when start-date is omitted")

    args = parser.parse_args()
    tf_seed_status = set_global_seeds(seed=args.seed, include_tensorflow=(args.model == "lstm"))

    print(
        f"Starting walk-forward validation for {args.ticker} using {args.model} model "
        f"(seed={args.seed}, source={args.data_source})..."
    )
    if tf_seed_status:
        print(f"Seed status: {tf_seed_status}")

    processor = StockDataProcessor(args.ticker)
    print("Loading data...")
    df = load_market_data(
        ticker=args.ticker,
        source=args.data_source,
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        years=args.years,
    )
    if df.empty:
        print("No data loaded. Check ticker/source/date range.")
        return

    print("Calculating technical indicators...")
    df = processor.add_technical_indicators(df)

    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})

    feature_cols = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]

    if args.model == "rf":
        model_factory = lambda: RandomForestModel(random_state=args.seed)
        task_type = "classification"
    elif args.model == "xgb":
        model_factory = lambda: XGBoostModel(random_state=args.seed)
        task_type = "classification"
    else:
        from src.models.lstm_model import LSTMModel

        model_factory = lambda: LSTMModel(sequence_length=args.sequence_length, n_features=len(feature_cols))
        task_type = "lstm_regression"

    validator = WalkForwardValidator(
        train_years=args.train_years,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    print("Running walk-forward validation...")
    results = validator.validate(
        model_factory=model_factory,
        data=df,
        feature_cols=feature_cols,
        target_col="Close",
        price_col="Close",
        task_type=task_type,
        sequence_length=args.sequence_length,
    )

    if not results:
        print("No results generated. Check your data range and window sizes.")
        return

    completed_results = [r for r in results if not r["metadata"].get("skipped_reason")]
    skipped_count = len(results) - len(completed_results)
    if skipped_count:
        print(f"Skipped {skipped_count} fold(s) due to insufficient data.")

    if not completed_results:
        print("All folds were skipped. No aggregate metrics available.")
        return

    print(f"\nCompleted {len(completed_results)} walk-forward steps ({len(results)} total folds).")

    avg_test_acc = np.mean([r["metrics"].get("test_accuracy", 0.0) for r in completed_results])
    avg_test_bal_acc = np.mean([r["metrics"].get("test_balanced_accuracy", 0.0) for r in completed_results])
    avg_test_f1 = np.mean([r["metrics"].get("test_f1", 0.0) for r in completed_results])
    avg_test_rmse = np.mean([r["metrics"].get("test_rmse", 0.0) for r in completed_results])
    avg_sharpe = np.mean([r["metrics"].get("sharpe_ratio", 0.0) for r in completed_results])
    avg_drawdown = np.mean([r["metrics"].get("max_drawdown", 0.0) for r in completed_results])

    print("\n=== Aggregated Results ===")
    print(f"Average Test Accuracy:          {avg_test_acc:.4f}")
    print(f"Average Test Balanced Accuracy: {avg_test_bal_acc:.4f}")
    print(f"Average Test F1:                {avg_test_f1:.4f}")
    print(f"Average Test RMSE:              {avg_test_rmse:.4f}")
    print(f"Average Sharpe Ratio:           {avg_sharpe:.4f}")
    print(f"Average Max Drawdown:           {avg_drawdown:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"wf_results_{args.ticker}_{args.model}.json")

    serializable_results = []
    for result in results:
        step_res = result.copy()
        metadata = step_res["metadata"].copy()
        for key, value in metadata.items():
            if hasattr(value, "isoformat"):
                metadata[key] = value.isoformat()
        step_res["metadata"] = metadata
        serializable_results.append(step_res)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4)

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
