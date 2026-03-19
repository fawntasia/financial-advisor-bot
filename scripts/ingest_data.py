"""
Data ingestion script for Financial Advisor Bot.
Fetches historical data for all tickers and financial news.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.dal import DataAccessLayer
from src.data.yfinance_client import YFinanceClient
from src.data.news_client import get_news_client
from src.utils.logger import get_logger

logger = get_logger("ingest_data")
DEFAULT_LOOKBACK_DAYS = 365 * 5
DEFAULT_NEWS_DAYS = 7
DEFAULT_NEWS_LIMIT = 10
DEFAULT_NEWS_MAX_TICKERS = 25


def _select_tickers_round_robin(tickers: List[str], cursor: int, max_tickers: int) -> Tuple[List[str], int]:
    """Select a bounded ticker subset and return (selected, next_cursor)."""
    if not tickers or max_tickers <= 0:
        return [], 0

    total = len(tickers)
    start = cursor % total
    take = min(max_tickers, total)

    selected = [tickers[(start + i) % total] for i in range(take)]
    next_cursor = (start + take) % total
    return selected, next_cursor


def _get_news_cursor(dal: DataAccessLayer, total_tickers: int) -> int:
    """Load and sanitize persisted round-robin cursor."""
    if total_tickers <= 0:
        return 0
    raw_cursor = dal.get_user_preference("news_round_robin_cursor")
    if raw_cursor is None:
        return 0
    try:
        return int(raw_cursor) % total_tickers
    except ValueError:
        logger.warning("Invalid news_round_robin_cursor value %r. Resetting to 0.", raw_cursor)
        return 0

def ingest_stock_data(dal: DataAccessLayer, days: int = DEFAULT_LOOKBACK_DAYS):
    """Fetch and store stock price data (default: 5-year lookback for new tickers)."""
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
                # Initial backfill for new symbols: bounded 5-year window by default.
                current_dt = datetime.now()
                start_dt = current_dt - timedelta(days=max(1, int(days)))
                df = client.get_ticker(
                    ticker,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=current_dt.strftime("%Y-%m-%d"),
                )
            
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
    return stats

def ingest_news_data(
    dal: DataAccessLayer,
    days: int = DEFAULT_NEWS_DAYS,
    provider: str = "auto",
    limit: int = DEFAULT_NEWS_LIMIT,
    max_tickers: int = DEFAULT_NEWS_MAX_TICKERS,
):
    """Fetch and store news headlines for a round-robin ticker subset."""
    stats = {
        "processed": 0,
        "records_inserted": 0,
        "duplicates": 0,
        "failed": 0,
        "selected_tickers": 0,
        "cursor_start": 0,
        "cursor_end": 0,
    }

    client = get_news_client(provider=provider)
    tickers = dal.get_all_tickers()

    if not tickers:
        logger.warning("No tickers available. Skipping news ingestion.")
        return stats

    cursor = _get_news_cursor(dal, len(tickers))
    selected_tickers, next_cursor = _select_tickers_round_robin(tickers, cursor, max_tickers)
    if not selected_tickers:
        logger.warning("No tickers selected for news ingestion.")
        return stats

    stats["selected_tickers"] = len(selected_tickers)
    stats["cursor_start"] = cursor
    stats["cursor_end"] = next_cursor

    logger.info(
        "Starting news ingestion for %d tickers (provider=%s, cursor=%d, next_cursor=%d)...",
        len(selected_tickers),
        provider,
        cursor,
        next_cursor,
    )

    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for ticker in tqdm(selected_tickers, desc="News Data"):
        try:
            articles = client.fetch_news(
                ticker,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )

            for article in articles:
                headline = article.get("title")
                if not headline:
                    continue

                headline_id = dal.insert_news_headline(
                    ticker=ticker,
                    headline=headline,
                    source=article.get("source"),
                    url=article.get("url"),
                    published_at=article.get("published_at"),
                    summary=article.get("summary"),
                    provider=article.get("provider", provider),
                )
                if headline_id:
                    stats["records_inserted"] += 1
                else:
                    stats["duplicates"] += 1

            stats["processed"] += 1

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            stats["failed"] += 1

    dal.set_user_preference("news_round_robin_cursor", str(next_cursor))
    logger.info(
        "News Data Complete. Tickers processed: %d, failed: %d, inserted: %d, duplicates: %d, next_cursor: %d",
        stats["processed"],
        stats["failed"],
        stats["records_inserted"],
        stats["duplicates"],
        next_cursor,
    )
    return stats


def ingest_data(
    stock_days: int = DEFAULT_LOOKBACK_DAYS,
    news_days: int = DEFAULT_NEWS_DAYS,
    news_provider: str = "auto",
    news_limit: int = DEFAULT_NEWS_LIMIT,
    news_max_tickers: int = DEFAULT_NEWS_MAX_TICKERS,
):
    """Main ingestion function."""
    dal = DataAccessLayer()

    # 1. Ingest Stock Data
    stock_stats = ingest_stock_data(dal, days=stock_days)

    # 2. Ingest News Data
    news_stats = ingest_news_data(
        dal,
        days=news_days,
        provider=news_provider,
        limit=news_limit,
        max_tickers=news_max_tickers,
    )
    return {"stock": stock_stats, "news": news_stats}

def main():
    parser = argparse.ArgumentParser(description="Ingest stock prices and financial news into the SQLite DB.")
    parser.add_argument("--news-provider", type=str, default="auto", choices=["auto", "rss", "newsapi", "alphavantage", "mock"], help="News provider strategy.")
    parser.add_argument("--news-days", type=int, default=DEFAULT_NEWS_DAYS, help="Days of history to request per ticker.")
    parser.add_argument("--news-limit", type=int, default=DEFAULT_NEWS_LIMIT, help="Max articles to request per ticker.")
    parser.add_argument("--news-max-tickers", type=int, default=DEFAULT_NEWS_MAX_TICKERS, help="Round-robin ticker count per run.")

    args = parser.parse_args()
    ingest_data(
        news_days=max(1, args.news_days),
        news_provider=args.news_provider,
        news_limit=max(1, args.news_limit),
        news_max_tickers=max(1, args.news_max_tickers),
    )

if __name__ == "__main__":
    main()
