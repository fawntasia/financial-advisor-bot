import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nlp.sentiment_pipeline import SentimentPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run batch sentiment analysis on news headlines.")
    parser.add_argument("--date", type=str, help="Date to process (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for FinBERT inference.")
    parser.add_argument("--all-unprocessed", action="store_true", help="Process all headlines that don't have scores.")
    parser.add_argument("--model-path", type=str, default="models/finbert", help="Path to local FinBERT model.")
    
    args = parser.parse_args()
    
    pipeline = SentimentPipeline(model_path=args.model_path)
    
    if args.all_unprocessed:
        logger.info("Running sentiment analysis for all unprocessed headlines...")
        pipeline.process_unprocessed(batch_size=args.batch_size)
        logger.warning("Aggregates are NOT automatically updated when using --all-unprocessed.")
    else:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Running sentiment analysis for date: {date_str}")
        pipeline.process_date(date=date_str, batch_size=args.batch_size)

    logger.info("Sentiment analysis run complete.")

if __name__ == "__main__":
    main()
