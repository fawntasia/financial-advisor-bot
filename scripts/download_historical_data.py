"""
Script to download 5 years of historical OHLCV data for all S&P 500 tickers.
"""

import sys
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json

# Add src to path
sys.path.insert(0, '.')

from src.data.yfinance_client import YFinanceClient
from src.database.dal import DataAccessLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/logs/download_historical_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("data/download_checkpoint.json")

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed_tickers": []}

def save_checkpoint(completed_tickers):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"completed_tickers": completed_tickers}, f)

def main():
    dal = DataAccessLayer()
    client = YFinanceClient(rate_limit_delay=0.5)
    
    # Get all tickers from DB
    tickers = dal.get_all_tickers()
    if not tickers:
        logger.error("No tickers found in database. Run scripts/init_db.py first.")
        return
    
    logger.info(f"Starting historical data download for {len(tickers)} tickers.")
    
    checkpoint = load_checkpoint()
    completed_tickers = checkpoint["completed_tickers"]
    
    tickers_to_download = [t for t in tickers if t not in completed_tickers]
    logger.info(f"Already completed: {len(completed_tickers)}. Remaining: {len(tickers_to_download)}.")
    
    for ticker in tqdm(tickers_to_download, desc="Downloading tickers"):
        try:
            # Fetch data from 2000-01-01 onwards
            df = client.get_ticker(ticker, start="2000-01-01")
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                completed_tickers.append(ticker)
                save_checkpoint(completed_tickers)
                continue
            
            # Prepare records for insertion
            records = []
            for timestamp, row in df.iterrows():
                # timestamp is the index (DatetimeIndex)
                date_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d')
                records.append({
                    "ticker": ticker,
                    "date": date_str,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "adj_close": float(row["Close"]) # yfinance history already provides adj prices by default if auto_adjust=True
                })
            
            # Bulk insert into DB
            dal.bulk_insert_prices(records)
            
            # Update checkpoint
            completed_tickers.append(ticker)
            if len(completed_tickers) % 10 == 0:
                save_checkpoint(completed_tickers)
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            # Continue with next ticker
    
    save_checkpoint(completed_tickers)
    logger.info("Download completed.")

if __name__ == "__main__":
    main()
