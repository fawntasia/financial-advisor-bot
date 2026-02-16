import argparse
import json
import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.stock_data import StockDataProcessor
from src.models.lstm_model import LSTMModel
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
    parser.add_argument("--train_years", type=int, default=3, help="Training window years")
    parser.add_argument("--val_months", type=int, default=3, help="Validation window months")
    parser.add_argument("--test_months", type=int, default=3, help="Testing window months")
    parser.add_argument("--step_months", type=int, default=3, help="Walk-forward step size in months")
    parser.add_argument("--sequence_length", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    tf_seed_status = set_global_seeds(seed=args.seed, include_tensorflow=(args.model == "lstm"))

    print(
        f"Starting walk-forward validation for {args.ticker} using {args.model} model "
        f"(seed={args.seed})..."
    )
    if tf_seed_status:
        print(f"Seed status: {tf_seed_status}")

    processor = StockDataProcessor(args.ticker)
    print("Fetching data...")
    try:
        # Fetch 10 years to ensure enough rows for windowed folds.
        df = processor.fetch_data(years=10)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print("Calculating technical indicators...")
    df = processor.add_technical_indicators(df)

    # Normalized date column for validator, keep feature columns unchanged.
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        # fallback for uncommon index names
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})

    feature_cols = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]

    if args.model == "rf":
        model = RandomForestModel(random_state=args.seed)
        task_type = "classification"
    elif args.model == "xgb":
        model = XGBoostModel(random_state=args.seed)
        task_type = "classification"
    else:
        model = LSTMModel(sequence_length=args.sequence_length, n_features=len(feature_cols))
        task_type = "lstm_regression"

    validator = WalkForwardValidator(
        train_years=args.train_years,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    print("Running walk-forward validation...")
    results = validator.validate(
        model=model,
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

    print(f"\nCompleted {len(results)} walk-forward steps.")

    avg_test_acc = np.mean([r["metrics"].get("test_accuracy", 0.0) for r in results])
    avg_test_bal_acc = np.mean([r["metrics"].get("test_balanced_accuracy", 0.0) for r in results])
    avg_test_f1 = np.mean([r["metrics"].get("test_f1", 0.0) for r in results])
    avg_test_rmse = np.mean([r["metrics"].get("test_rmse", 0.0) for r in results])
    avg_sharpe = np.mean([r["metrics"].get("sharpe_ratio", 0.0) for r in results])
    avg_drawdown = np.mean([r["metrics"].get("max_drawdown", 0.0) for r in results])

    print("\n=== Aggregated Results ===")
    print(f"Average Test Accuracy:          {avg_test_acc:.4f}")
    print(f"Average Test Balanced Accuracy: {avg_test_bal_acc:.4f}")
    print(f"Average Test F1:                {avg_test_f1:.4f}")
    print(f"Average Test RMSE:              {avg_test_rmse:.4f}")
    print(f"Average Sharpe Ratio:           {avg_sharpe:.4f}")
    print(f"Average Max Drawdown:           {avg_drawdown:.4f}")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wf_results_{args.ticker}_{args.model}.json")

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
