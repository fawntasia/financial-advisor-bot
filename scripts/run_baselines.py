import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Add src to path
sys.path.append(os.getcwd())

from src.database.dal import DataAccessLayer
from src.models.baselines import BuyAndHoldStrategy, RandomWalkStrategy, SMACrossoverStrategy
from src.models.evaluation import calculate_strategy_returns, calculate_metrics

def main():
    parser = argparse.ArgumentParser(description='Run baseline strategies.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2030-01-01', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    dal = DataAccessLayer()
    ticker = args.ticker
    
    print(f"Fetching data for {ticker}...")
    df = dal.get_stock_prices(ticker, start_date=args.start, end_date=args.end)
    
    if df.empty:
        print(f"No data found for {ticker} in range {args.start} to {args.end}")
        return
        
    # Ensure date is index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    
    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Define Strategies
    strategies = [
        ("Buy and Hold", BuyAndHoldStrategy()),
        ("Random Walk", RandomWalkStrategy(seed=42)),
        ("SMA Crossover (20/50)", SMACrossoverStrategy(fast_window=20, slow_window=50))
    ]
    
    # Run and Evaluate
    for name, strategy in strategies:
        print(f"\nRunning {name}...")
        try:
            signals = strategy.generate_signals(df)
            returns = calculate_strategy_returns(signals, df['close'])
            metrics = calculate_metrics(returns)
            
            print(f"Results for {name}:")
            if not metrics:
                print("  Insufficient data for metrics.")
                continue

            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
                
            # Save to DB
            start_date_str = df.index.min().strftime('%Y-%m-%d')
            end_date_str = df.index.max().strftime('%Y-%m-%d')
            
            dal.insert_model_performance(
                model_name=name,
                ticker=ticker,
                test_date_start=start_date_str,
                test_date_end=end_date_str,
                total_return=metrics.get('total_return'),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                max_drawdown=metrics.get('max_drawdown'),
                # Add other metrics if available in schema
                # accuracy, precision etc are for classification/direction, 
                # but here we have portfolio metrics. 
                # We can map total_return to total_return column.
            )
            print("  Saved to database.")
            
        except Exception as e:
            print(f"  Error running {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
