import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.stock_data import StockDataProcessor
from src.models.random_forest_model import RandomForestModel

def calculate_metrics(daily_returns):
    """
    Calculate performance metrics.
    """
    total_return = (daily_returns + 1).prod() - 1
    
    # Annualized Sharpe Ratio (assuming 252 trading days)
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
        
    # Max Drawdown
    cumulative_returns = (daily_returns + 1).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win Rate (Percentage of days with positive return)
    in_market_returns = daily_returns[daily_returns != 0]
    win_rate = (in_market_returns > 0).mean() if len(in_market_returns) > 0 else 0
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }

def run_backtest(ticker, model_path, start_date='2023-01-01', end_date='2023-12-31', transaction_cost=0.001):
    """
    Run backtest for a given ticker and model.
    """
    print(f"Running backtest for {ticker} using model {model_path}...")
    
    # 1. Load Data
    processor = StockDataProcessor(ticker)
    df = processor.fetch_data(years=5) 
    df = processor.add_technical_indicators(df)
    
    backtest_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    if backtest_df.empty:
        print(f"No data found for {ticker} in range {start_date} to {end_date}")
        return
        
    # 2. Load Model
    if model_path.endswith('.pkl'):
        model = RandomForestModel()
        model.load(model_path)
    else:
        raise ValueError("Unsupported model format. Only .pkl (Random Forest) is supported currently.")

    # 3. Prepare Features
    feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
    X = backtest_df[feature_cols].values
    
    # 4. Generate Predictions
    predictions = model.predict(X)
    
    # 5. Simulate Trading
    backtest_df['Prediction'] = predictions
    backtest_df['Signal'] = backtest_df['Prediction'].shift(1).fillna(0)
    
    backtest_df['Market_Return'] = backtest_df['Close'].pct_change().fillna(0)
    backtest_df['Strategy_Return_Raw'] = backtest_df['Signal'] * backtest_df['Market_Return']
    
    backtest_df['Trade'] = backtest_df['Signal'].diff().abs().fillna(0)
    if backtest_df['Signal'].iloc[0] == 1:
        backtest_df.iloc[0, backtest_df.columns.get_loc('Trade')] = 1
        
    backtest_df['Cost'] = backtest_df['Trade'] * transaction_cost
    backtest_df['Strategy_Return'] = backtest_df['Strategy_Return_Raw'] - backtest_df['Cost']
    
    # 6. Calculate Metrics
    strategy_metrics = calculate_metrics(backtest_df['Strategy_Return'])
    benchmark_metrics = calculate_metrics(backtest_df['Market_Return'])
    
    # 7. Equity Curve Data
    backtest_df['Cumulative_Strategy'] = (backtest_df['Strategy_Return'] + 1).cumprod()
    backtest_df['Cumulative_Market'] = (backtest_df['Market_Return'] + 1).cumprod()
    
    print("\n=== Strategy Metrics ===")
    for k, v in strategy_metrics.items():
        if "Ratio" in k or "Rate" in k:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2%}")

    print("\n=== Benchmark Metrics ===")
    for k, v in benchmark_metrics.items():
        if "Ratio" in k or "Rate" in k:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2%}")
            
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"backtest_{ticker}_{start_date[:4]}.csv")
    backtest_df[['Close', 'Signal', 'Strategy_Return', 'Market_Return', 'Cumulative_Strategy', 'Cumulative_Market']].to_csv(csv_path)
    
    return strategy_metrics, benchmark_metrics, backtest_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model backtesting")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pkl)")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost (0.001 = 0.1%)")
    
    args = parser.parse_args()
    
    run_backtest(args.ticker, args.model, args.start, args.end, args.cost)
