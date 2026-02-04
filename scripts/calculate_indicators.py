import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database.dal import DataAccessLayer
from src.features.indicators import calculate_indicators, prepare_indicators_for_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Calculate technical indicators for tickers.")
    parser.add_argument("--ticker", type=str, help="Specific ticker to process (optional)")
    args = parser.parse_args()

    dal = DataAccessLayer()
    
    # Determine which tickers to process
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = dal.get_all_tickers()
        
    logger.info(f"Starting indicator calculation for {len(tickers)} tickers.")
    
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}...")
            
            # Fetch all available stock prices
            # Using a wide date range to ensure we get everything
            start_date = "1900-01-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            df = dal.get_stock_prices(ticker, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No stock prices found for {ticker}. Skipping.")
                continue
            
            # Calculate indicators
            df_with_indicators = calculate_indicators(df)
            
            # Prepare for DB
            records = prepare_indicators_for_db(df_with_indicators)
            
            if not records:
                logger.warning(f"No records to insert for {ticker}.")
                continue
                
            # Bulk insert
            dal.bulk_insert_indicators(records)
            logger.info(f"Successfully updated indicators for {ticker} ({len(records)} records).")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}", exc_info=True)
            
    logger.info("Indicator calculation completed.")

if __name__ == "__main__":
    main()
