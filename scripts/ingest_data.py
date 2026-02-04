"""
Data ingestion script for Financial Advisor Bot.
Fetches historical data for all tickers and financial news.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.dal import DataAccessLayer
from src.data.yfinance_client import YFinanceClient
from src.data.news_client import get_news_client
from src.utils.logger import get_logger

logger = get_logger("ingest_data")

def ingest_stock_data(dal: DataAccessLayer, days: int = 365 * 5):
    """Fetch and store stock price data."""
    client = YFinanceClient()
    tickers = dal.get_all_tickers()
    
    logger.info(f"Starting stock data ingestion for {len(tickers)} tickers...")
    
    stats = {"processed": 0, "failed": 0, "records_inserted": 0}
    
    for ticker in tqdm(tickers, desc="Stock Data"):
        try:
            last_date = dal.get_latest_price_date(ticker)
            start_date = None
            
            if last_date:
                # Assuming last_date is a valid date string YYYY-MM-DD
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                    # Ensure pd.Timedelta is handled correctly for type checker
                    delta = pd.Timedelta(days=1) 
                    start_date = (last_dt + delta).strftime("%Y-%m-%d")
                    
                    # If start_date is future, skip
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    if start_date >= current_date:
                        continue
                except ValueError:
                    logger.warning(f"Invalid date format for {ticker}: {last_date}")
                    pass
            
            if start_date:
                current_date = datetime.now().strftime("%Y-%m-%d")
                df = client.get_ticker(ticker, start=start_date, end=current_date)
            else:
                df = client.get_ticker(ticker, period="5y")
            
            if df.empty:
                stats["failed"] += 1
                continue
                
            records = []
            for index, row in df.iterrows():
                # Manually format the index as string if it's a Timestamp, otherwise use str()
                if isinstance(index, pd.Timestamp):
                    date_str = index.strftime("%Y-%m-%d")
                else:
                    date_str = str(index)
                
                # Handle potential missing or NaN values safely
                close_val = float(row['Close'])
                adj_close_raw = row.get('Adj Close', close_val)
                
                # Check for NaN explicitly
                if pd.isna(adj_close_raw):
                    adj_close_val = close_val
                else:
                    adj_close_val = float(adj_close_raw)
                
                record = {
                    "ticker": ticker,
                    "date": date_str,
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": close_val,
                    "volume": int(row['Volume']),
                    "adj_close": adj_close_val
                }
                records.append(record)
            
            if records:
                dal.bulk_insert_prices(records)
                stats["processed"] += 1
                stats["records_inserted"] += len(records)
                
        except Exception as e:
            logger.error(f"Error processing stock {ticker}: {e}")
            stats["failed"] += 1
            
    logger.info(f"Stock Data Complete. Processed: {stats['processed']}, Failed: {stats['failed']}")

def ingest_news_data(dal: DataAccessLayer, days: int = 7):
    """Fetch and store news headlines."""
    # Use NewsAPI by default, or configurable
    client = get_news_client(provider="newsapi") 
    tickers = dal.get_all_tickers()
    
    # NewsAPI free tier is limited, so we might want to be careful.
    # For now, let's just fetch for top tickers or general market news to save requests
    # Or fetch specifically for each ticker if we have a premium key.
    # Strategy: Fetch for top 10 tickers + "S&P 500" general query.
    
    # Simplified strategy for demo: Query "S&P 500" and major tickers
    queries = ["S&P 500", "Stock Market", "Economy"] + tickers[:5] 
    
    logger.info(f"Starting news ingestion for {len(queries)} queries...")
    
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    stats = {"processed": 0, "records_inserted": 0}
    
    for query in tqdm(queries, desc="News Data"):
        try:
            articles = client.fetch_news(query, start_date=start_date, end_date=end_date)
            
            for article in articles:
                # Map query to ticker if it's a ticker
                ticker = query if query in tickers else "UNKNOWN"
                
                # Check for duplicates or insert
                # Note: insert_news_headline returns ID or None
                dal.insert_news_headline(
                    ticker=ticker,
                    headline=article['title'],
                    source=article['source'],
                    url=article['url'],
                    published_at=article['published_at']
                )
                stats["records_inserted"] += 1
                
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Error fetching news for {query}: {e}")
            
    logger.info(f"News Data Complete. Queries: {stats['processed']}, Articles: {stats['records_inserted']}")

def main():
    """Main ingestion function."""
    dal = DataAccessLayer()
    
    # 1. Ingest Stock Data
    ingest_stock_data(dal)
    
    # 2. Ingest News Data
    ingest_news_data(dal)

if __name__ == "__main__":
    main()
