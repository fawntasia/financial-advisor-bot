"""
Data ingestion script for Financial Advisor Bot.
Fetches historical data for all tickers and financial news.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.news_client import get_news_client
from src.data.yfinance_client import YFinanceClient
from src.database.dal import DataAccessLayer
from src.features.indicators import calculate_indicators, prepare_indicators_for_db
from src.utils.logger import get_logger

logger = get_logger("ingest_data")

DEFAULT_LOOKBACK_DAYS = 365 * 5
DEFAULT_NEWS_PROVIDER = "auto"
DEFAULT_NEWS_LOOKBACK_DAYS = 10
DEFAULT_NEWS_ARTICLES_PER_TICKER = 4
DEFAULT_NEWS_RETENTION_DAYS = 10
DEFAULT_SENTIMENT_BATCH_SIZE = 32
DEFAULT_SENTIMENT_LIMIT = 50000
DEFAULT_SENTIMENT_MODEL_PATH = "models/finbert"


def ingest_stock_data(dal: DataAccessLayer, days: int = DEFAULT_LOOKBACK_DAYS):
    """Fetch and store stock price data (default: 5-year lookback for new tickers)."""
    client = YFinanceClient()
    tickers = dal.get_all_tickers()

    logger.info("Starting stock data ingestion for %d tickers...", len(tickers))

    stats = {"processed": 0, "failed": 0, "records_inserted": 0, "updated_tickers": []}

    for ticker in tqdm(tickers, desc="Stock Data"):
        try:
            last_date = dal.get_latest_price_date(ticker)
            start_date = None

            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                    start_date = (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                    current_date = datetime.now().strftime("%Y-%m-%d")
                    if start_date >= current_date:
                        continue
                except ValueError:
                    logger.warning("Invalid date format for %s: %s", ticker, last_date)

            if start_date:
                current_date = datetime.now().strftime("%Y-%m-%d")
                df = client.get_ticker(ticker, start=start_date, end=current_date)
            else:
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
                date_str = index.strftime("%Y-%m-%d") if isinstance(index, pd.Timestamp) else str(index)
                close_val = float(row["Close"])
                adj_close_raw = row.get("Adj Close", close_val)
                adj_close_val = close_val if pd.isna(adj_close_raw) else float(adj_close_raw)

                records.append(
                    {
                        "ticker": ticker,
                        "date": date_str,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": close_val,
                        "volume": int(row["Volume"]),
                        "adj_close": adj_close_val,
                    }
                )

            if records:
                dal.bulk_insert_prices(records)
                stats["processed"] += 1
                stats["records_inserted"] += len(records)
                stats["updated_tickers"].append(ticker)

        except Exception as e:
            logger.error("Error processing stock %s: %s", ticker, e)
            stats["failed"] += 1

    logger.info(
        "Stock Data Complete. Processed: %d, Failed: %d, Records Inserted: %d",
        stats["processed"],
        stats["failed"],
        stats["records_inserted"],
    )
    return stats


def refresh_indicators_for_tickers(dal: DataAccessLayer, tickers: List[str]):
    """Recompute and upsert technical indicators for provided tickers."""
    unique_tickers = sorted(set(tickers))
    stats = {"selected_tickers": len(unique_tickers), "processed": 0, "failed": 0, "records_upserted": 0}

    if not unique_tickers:
        logger.info("No tickers with new stock rows. Skipping technical indicator refresh.")
        return stats

    logger.info("Starting technical indicator refresh for %d tickers...", len(unique_tickers))

    start_date = "1900-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    for ticker in tqdm(unique_tickers, desc="Indicators"):
        try:
            df = dal.get_stock_prices(ticker, start_date, end_date)
            if df.empty:
                logger.warning("No stock prices found for %s during indicator refresh.", ticker)
                stats["failed"] += 1
                continue

            df_with_indicators = calculate_indicators(df.copy())
            records = prepare_indicators_for_db(df_with_indicators)
            if not records:
                logger.warning("No indicator records generated for %s.", ticker)
                stats["failed"] += 1
                continue

            dal.bulk_insert_indicators(records)
            stats["processed"] += 1
            stats["records_upserted"] += len(records)
        except Exception as e:
            logger.error("Error refreshing indicators for %s: %s", ticker, e)
            stats["failed"] += 1

    logger.info(
        "Technical Indicator Refresh Complete. Processed: %d, Failed: %d, Rows Upserted: %d",
        stats["processed"],
        stats["failed"],
        stats["records_upserted"],
    )
    return stats


def _run_automatic_sentiment(
    dal: DataAccessLayer,
    batch_size: int = DEFAULT_SENTIMENT_BATCH_SIZE,
    limit: int = DEFAULT_SENTIMENT_LIMIT,
    model_path: str = DEFAULT_SENTIMENT_MODEL_PATH,
) -> Dict[str, object]:
    """
    Process unscored headlines after ingestion.

    Returns status metadata when sentiment dependencies/model are unavailable
    so ingestion can still complete.
    """
    skipped = {
        "status": "skipped",
        "headlines": 0,
        "scores_inserted": 0,
        "aggregate_days": 0,
    }
    try:
        from src.nlp.sentiment_pipeline import SentimentPipeline
    except Exception as exc:
        logger.warning("Skipping automatic sentiment analysis: %s", exc)
        return {**skipped, "reason": "import_error"}

    pipeline = SentimentPipeline(dal=dal, model_path=model_path)
    if pipeline.loader.model is None or pipeline.loader.tokenizer is None:
        logger.warning(
            "Skipping automatic sentiment analysis: FinBERT model unavailable at '%s'.",
            model_path,
        )
        return {**skipped, "reason": "model_unavailable"}

    run_stats = pipeline.process_unprocessed(batch_size=max(1, int(batch_size)), limit=max(1, int(limit)))
    return {"status": "completed", **run_stats}


def ingest_news_data(
    dal: DataAccessLayer,
    provider: str = DEFAULT_NEWS_PROVIDER,
    run_sentiment: bool = True,
):
    """Fetch and store recent ticker headlines across the full universe."""
    stats = {
        "processed": 0,
        "records_inserted": 0,
        "duplicates": 0,
        "failed": 0,
        "selected_tickers": 0,
        "fetched_articles": 0,
        "lookback_days": DEFAULT_NEWS_LOOKBACK_DAYS,
        "articles_per_ticker": DEFAULT_NEWS_ARTICLES_PER_TICKER,
        "retention_days": DEFAULT_NEWS_RETENTION_DAYS,
        "pruned_headlines": 0,
        "pruned_scores": 0,
        "pruned_daily_sentiment": 0,
        "sentiment": {
            "status": "disabled",
            "headlines": 0,
            "scores_inserted": 0,
            "aggregate_days": 0,
        },
    }

    client = get_news_client(provider=provider)
    tickers = dal.get_all_tickers()
    if not tickers:
        logger.warning("No tickers available. Skipping news ingestion.")
        return stats

    stats["selected_tickers"] = len(tickers)
    start_date = (datetime.now() - timedelta(days=DEFAULT_NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(
        "Starting news ingestion for %d tickers (provider=%s, lookback=%d days, limit=%d/ticker).",
        len(tickers),
        provider,
        DEFAULT_NEWS_LOOKBACK_DAYS,
        DEFAULT_NEWS_ARTICLES_PER_TICKER,
    )

    for ticker in tqdm(tickers, desc="News Data"):
        try:
            articles = client.fetch_news(
                ticker,
                start_date=start_date,
                end_date=end_date,
                limit=DEFAULT_NEWS_ARTICLES_PER_TICKER,
            )
            stats["fetched_articles"] += len(articles)

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
            logger.error("Error fetching news for %s: %s", ticker, e)
            stats["failed"] += 1

    prune_stats = dal.prune_news_history(keep_days=DEFAULT_NEWS_RETENTION_DAYS, prune_daily_sentiment=True)
    stats.update(prune_stats)

    if run_sentiment:
        stats["sentiment"] = _run_automatic_sentiment(dal=dal)

    sentiment_status = stats["sentiment"].get("status")
    logger.info(
        "News ingestion complete. Tickers processed: %d/%d, failed: %d, fetched: %d, inserted: %d, duplicates: %d, pruned_headlines: %d, sentiment_status: %s",
        stats["processed"],
        stats["selected_tickers"],
        stats["failed"],
        stats["fetched_articles"],
        stats["records_inserted"],
        stats["duplicates"],
        stats["pruned_headlines"],
        sentiment_status,
    )
    return stats


def ingest_data(
    stock_days: int = DEFAULT_LOOKBACK_DAYS,
    news_provider: str = DEFAULT_NEWS_PROVIDER,
):
    """Run the full ingestion pipeline (stock -> indicators -> news -> sentiment)."""
    dal = DataAccessLayer()

    stock_stats = ingest_stock_data(dal, days=stock_days)
    indicator_stats = refresh_indicators_for_tickers(
        dal=dal,
        tickers=stock_stats.get("updated_tickers", []),
    )
    news_stats = ingest_news_data(dal=dal, provider=news_provider, run_sentiment=True)

    return {
        "stock": stock_stats,
        "indicators": indicator_stats,
        "news": news_stats,
        "sentiment": news_stats.get("sentiment", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest stock prices/news and run automatic sentiment scoring.")
    parser.add_argument(
        "--news-provider",
        type=str,
        default=DEFAULT_NEWS_PROVIDER,
        choices=["auto", "rss", "newsapi", "alphavantage", "mock"],
        help="News provider strategy.",
    )
    args = parser.parse_args()
    ingest_data(news_provider=args.news_provider)


if __name__ == "__main__":
    main()
