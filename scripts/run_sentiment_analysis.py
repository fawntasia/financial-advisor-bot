import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run batch sentiment analysis on news headlines.")
    parser.add_argument("--date", type=str, help="Date to process (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--ticker", type=str, help="Ticker to process (only unprocessed headlines for that symbol).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for FinBERT inference.")
    parser.add_argument("--all-unprocessed", action="store_true", help="Process all headlines that don't have scores.")
    parser.add_argument("--limit", type=int, default=1000, help="Max unprocessed headlines to score in a run.")
    parser.add_argument("--model-path", type=str, default="models/finbert", help="Path to local FinBERT model.")
    
    args = parser.parse_args()

    try:
        from src.nlp.sentiment_pipeline import SentimentPipeline
    except Exception as exc:
        logger.error(
            "Unable to import sentiment pipeline. Ensure transformers + torch are installed correctly. Error: %s",
            exc,
        )
        sys.exit(1)
    
    pipeline = SentimentPipeline(model_path=args.model_path)
    stats = {"headlines": 0, "scores_inserted": 0, "aggregate_days": 0}
    
    if args.ticker:
        ticker = args.ticker.upper().strip()
        logger.info("Running ticker-scoped sentiment analysis for: %s", ticker)
        stats = pipeline.process_unprocessed_for_ticker(
            ticker=ticker,
            batch_size=args.batch_size,
            limit=max(1, args.limit),
        )
    elif args.all_unprocessed:
        logger.info("Running sentiment analysis for all unprocessed headlines...")
        stats = pipeline.process_unprocessed(batch_size=args.batch_size, limit=max(1, args.limit))
    else:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Running sentiment analysis for date: {date_str}")
        stats = pipeline.process_date(date=date_str, batch_size=args.batch_size)

    logger.info(
        "Sentiment analysis run complete. Headline rows: %d, scores inserted: %d, aggregate days refreshed: %d",
        stats.get("headlines", 0),
        stats.get("scores_inserted", 0),
        stats.get("aggregate_days", 0),
    )

if __name__ == "__main__":
    main()
