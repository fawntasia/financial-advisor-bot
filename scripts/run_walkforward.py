import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.stock_data import StockDataProcessor
from src.models.validation import WalkForwardValidator
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel

def main():
    parser = argparse.ArgumentParser(description='Run Walk-Forward Validation')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'xgb', 'lstm'], help='Model type')
    parser.add_argument('--train_years', type=int, default=3, help='Training window years')
    parser.add_argument('--val_months', type=int, default=3, help='Validation window months')
    parser.add_argument('--test_months', type=int, default=3, help='Testing window months')
    
    args = parser.parse_args()
    
    print(f"Starting Walk-Forward Validation for {args.ticker} using {args.model} model...")
    
    # 1. Fetch & Prepare Data
    processor = StockDataProcessor(args.ticker)
    print("Fetching data...")
    try:
        # Fetch 10 years to ensure enough data for walk-forward splits
        df = processor.fetch_data(years=10)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    print("Calculating technical indicators...")
    df = processor.add_technical_indicators(df)
    
    # Create Target: 1 if Close(t+1) > Close(t), else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Reset index to have 'Date' as a column (yfinance returns it as index)
    df = df.reset_index()
    # Rename columns to lowercase to match validator's expectations
    df.columns = [c.lower() for c in df.columns]
    
    # Drop rows with NaNs from indicators and the last row with no target
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['close', 'sma_20', 'sma_50', 'rsi', 'macd', 'signal_line']
    target_col = 'target'
    
    # 2. Initialize Model
    if args.model == 'rf':
        model = RandomForestModel()
    elif args.model == 'xgb':
        model = XGBoostModel()
    elif args.model == 'lstm':
        # Need to handle LSTM's 3D input separately or use a wrapper that handles it
        # For now, let's just use RF/XGB to demonstrate the framework
        print("LSTM support in walk-forward requires special input formatting. Skipping for now.")
        return

    # 3. Initialize Validator
    validator = WalkForwardValidator(
        train_years=args.train_years,
        val_months=args.val_months,
        test_months=args.test_months
    )
    
    # 4. Run Validation
    print("Running walk-forward validation...")
    results = validator.validate(model, df, feature_cols, target_col)
    
    if not results:
        print("No results generated. Check your data range and window sizes.")
        return
        
    # 5. Summary and Export
    print(f"\nCompleted {len(results)} walk-forward steps.")
    
    avg_test_acc = np.mean([r['metrics']['test_accuracy'] for r in results])
    avg_sharpe = np.mean([r['metrics'].get('sharpe_ratio', 0) for r in results])
    avg_drawdown = np.mean([r['metrics'].get('max_drawdown', 0) for r in results])
    
    print("\n=== Aggregated Results ===")
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Sharpe Ratio:  {avg_sharpe:.4f}")
    print(f"Average Max Drawdown:  {avg_drawdown:.4f}")
    
    # Save results to JSON
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wf_results_{args.ticker}_{args.model}.json")
    
    # Convert results to JSON-serializable format
    serializable_results = []
    for r in results:
        step_res = r.copy()
        meta = step_res['metadata'].copy()
        for k, v in meta.items():
            if hasattr(v, 'isoformat'):
                meta[k] = v.isoformat()
        step_res['metadata'] = meta
        serializable_results.append(step_res)
        
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
