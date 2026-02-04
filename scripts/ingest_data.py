"""
Data ingestion script for Financial Advisor Bot.
Fetches historical data for all tickers in the database and stores it.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.dal import DataAccessLayer
from src.data.yfinance_client import YFinanceClient
from src.utils.logger import get_logger

logger = get_logger("ingest_data")

def ingest_data(days: int = 365 * 5):
    """
    Fetch and store data for all tickers.
    
    Args:
        days: Number of days of history to fetch (default 5 years)
    """
    dal = DataAccessLayer()
    client = YFinanceClient()
    
    # Get all tickers
    tickers = dal.get_all_tickers()
    logger.info(f"Found {len(tickers)} tickers in database.")
    
    # Statistics
    stats = {
        "processed": 0,
        "failed": 0,
        "records_inserted": 0
    }
    
    for ticker in tqdm(tickers, desc="Ingesting Data"):
        try:
            # Check last update
            last_date = dal.get_latest_price_date(ticker)
            start_date = None
            
            if last_date:
                # Start from the day after the last update
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                start_date = (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                logger.debug(f"Updating {ticker} from {start_date}")
            else:
                # Full history
                logger.debug(f"Fetching full history for {ticker}")
            
            # Fetch data
            # Use 'max' if no start date to get full history, or specific range
            # YFinanceClient handles start/end
            if start_date:
                # Check if start_date is in the future or today
                if start_date >= datetime.now().strftime("%Y-%m-%d"):
                    continue
                    
                df = client.get_ticker(ticker, start=start_date, end=datetime.now().strftime("%Y-%m-%d"))
            else:
                df = client.get_ticker(ticker, period="5y") # Default to 5y
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                stats["failed"] += 1
                continue
                
            # Prepare records
            records = []
            for index, row in df.iterrows():
                # yfinance returns index as Date (Timestamp)
                date_str = index.strftime("%Y-%m-%d")
                
                # Handle potential missing columns
                adj_close = row.get('Adj Close', row['Close']) # Fallback to Close if no Adj Close
                
                record = {
                    "ticker": ticker,
                    "date": date_str,
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "adj_close": float(adj_close)
                }
                records.append(record)
            
            # Bulk insert
            if records:
                dal.bulk_insert_prices(records)
                stats["processed"] += 1
                stats["records_inserted"] += len(records)
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            stats["failed"] += 1
            
    logger.info("Ingestion complete.")
    logger.info(f"Processed: {stats['processed']}, Failed: {stats['failed']}")
    logger.info(f"Total records inserted: {stats['records_inserted']}")

if __name__ == "__main__":
    ingest_data()
