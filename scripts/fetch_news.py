"""
Script to fetch financial news for tickers and store them in the database.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import argparse

# Add src to path
sys.path.insert(0, '.')

from src.data.news_client import get_news_client
from src.database.dal import DataAccessLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/logs/fetch_news.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_and_store_news(tickers: List[str], days: int = 7, provider: str = "newsapi"):
    """Fetch news for tickers and store in DB."""
    dal = DataAccessLayer()
    client = get_news_client(provider)
    
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    total_new = 0
    for ticker in tickers:
        logger.info(f"Fetching news for {ticker} since {start_date}")
        articles = client.fetch_news(ticker, start_date=start_date)
        
        if not articles:
            logger.info(f"No news found for {ticker}")
            continue
            
        new_count = 0
        for article in articles:
            try:
                # Store in DB
                # Note: insert_news_headline returns ID or None
                headline_id = dal.insert_news_headline(
                    ticker=ticker,
                    headline=article["title"],
                    source=article["source"],
                    url=article["url"],
                    published_at=article["published_at"]
                )
                if headline_id:
                    new_count += 1
            except Exception as e:
                # Likely duplicate headline
                continue
        
        logger.info(f"Stored {new_count} new headlines for {ticker}")
        total_new += new_count
        
    logger.info(f"Total new headlines stored: {total_new}")

def main():
    parser = argparse.ArgumentParser(description="Fetch financial news for tickers.")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to fetch news for.")
    parser.add_argument("--all", action="store_true", help="Fetch news for all tickers in DB.")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back.")
    parser.add_argument("--provider", default="newsapi", choices=["newsapi", "alphavantage", "mock"], help="News provider to use.")
    
    args = parser.parse_args()
    
    dal = DataAccessLayer()
    
    if args.all:
        tickers = dal.get_all_tickers()
    elif args.tickers:
        tickers = args.tickers
    else:
        logger.error("Must specify --tickers or --all")
        return
        
    if not tickers:
        logger.error("No tickers to process.")
        return
        
    fetch_and_store_news(tickers, days=args.days, provider=args.provider)

if __name__ == "__main__":
    main()
