"""
Daily scheduler for data ingestion.
"""
import schedule
import time
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_data import ingest_data
from src.utils.logger import get_logger

logger = get_logger("scheduler")

def job():
    logger.info("Starting scheduled data ingestion...")
    try:
        ingest_data()
        logger.info("Scheduled ingestion complete.")
    except Exception as e:
        logger.error(f"Scheduled ingestion failed: {e}")

def main():
    logger.info("Scheduler started. Running daily at 18:00.")
    
    # Schedule to run every day at 6 PM (after market close)
    schedule.every().day.at("18:00").do(job)
    
    # Also run on startup? Optional.
    # job() 
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
