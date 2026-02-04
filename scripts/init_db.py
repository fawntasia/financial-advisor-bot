"""
Database initialization script.
Creates SQLite database with all tables and populates S&P 500 tickers.
"""

import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, '.')

DATABASE_PATH = Path("data/financial_advisor.db")

def create_tables(conn):
    """Create all database tables."""
    cursor = conn.cursor()
    
    # Tickers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            symbol VARCHAR(10) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            sector VARCHAR(50),
            industry VARCHAR(50),
            market_cap_category VARCHAR(10),
            added_to_sp500 DATE,
            is_active BOOLEAN DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Stock prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,4),
            high DECIMAL(10,4),
            low DECIMAL(10,4),
            close DECIMAL(10,4),
            volume BIGINT,
            adjusted_close DECIMAL(10,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES tickers(symbol)
        )
    """)
    
    # Technical indicators table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            rsi_14 DECIMAL(5,2),
            macd DECIMAL(10,4),
            macd_signal DECIMAL(10,4),
            macd_histogram DECIMAL(10,4),
            bb_upper DECIMAL(10,4),
            bb_middle DECIMAL(10,4),
            bb_lower DECIMAL(10,4),
            sma_20 DECIMAL(10,4),
            sma_50 DECIMAL(10,4),
            sma_200 DECIMAL(10,4),
            ema_12 DECIMAL(10,4),
            ema_26 DECIMAL(10,4),
            atr_14 DECIMAL(10,4),
            volume_sma_20 DECIMAL(10,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES tickers(symbol)
        )
    """)
    
    # News headlines table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT NOT NULL,
            source VARCHAR(50),
            url VARCHAR(500),
            published_at TIMESTAMP NOT NULL,
            tickers VARCHAR(500),
            raw_text TEXT,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Sentiment scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline_id INTEGER NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            sentiment_score DECIMAL(3,2),
            confidence DECIMAL(3,2),
            sentiment_label VARCHAR(10),
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (headline_id) REFERENCES news_headlines(id),
            FOREIGN KEY (ticker) REFERENCES tickers(symbol)
        )
    """)
    
    # Daily sentiment table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            avg_sentiment DECIMAL(3,2),
            sentiment_volatility DECIMAL(3,2),
            news_count INTEGER,
            bullish_count INTEGER,
            bearish_count INTEGER,
            neutral_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES tickers(symbol)
        )
    """)
    
    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            prediction_date DATE NOT NULL,
            target_date DATE NOT NULL,
            predicted_price DECIMAL(10,4),
            predicted_direction VARCHAR(4),
            confidence DECIMAL(5,2),
            actual_price DECIMAL(10,4),
            actual_direction VARCHAR(4),
            was_correct BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tickers(symbol)
        )
    """)
    
    # Model performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name VARCHAR(50) NOT NULL,
            training_date DATE NOT NULL,
            metric_name VARCHAR(30) NOT NULL,
            metric_value DECIMAL(10,6),
            evaluation_period_start DATE,
            evaluation_period_end DATE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # System logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_level VARCHAR(10) NOT NULL,
            component VARCHAR(50) NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # User preferences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_tolerance VARCHAR(10),
            preferred_timeframe VARCHAR(10),
            watchlist TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    print("[OK] Tables created successfully")

def create_indexes(conn):
    """Create indexes for query performance."""
    cursor = conn.cursor()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON stock_prices(ticker, date)",
        "CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(date)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_ticker_date ON technical_indicators(ticker, date)",
        "CREATE INDEX IF NOT EXISTS idx_news_published ON news_headlines(published_at)",
        "CREATE INDEX IF NOT EXISTS idx_news_tickers ON news_headlines(tickers)",
        "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_scores(ticker, analyzed_at)",
        "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_ticker_date ON daily_sentiment(ticker, date)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_ticker_model ON predictions(ticker, model_name, prediction_date)",
        "CREATE INDEX IF NOT EXISTS idx_model_performance_name_date ON model_performance(model_name, training_date)",
        "CREATE INDEX IF NOT EXISTS idx_logs_level_component ON system_logs(log_level, component)",
        "CREATE INDEX IF NOT EXISTS idx_logs_created ON system_logs(created_at)"
    ]
    
    for idx_sql in indexes:
        cursor.execute(idx_sql)
    
    conn.commit()
    print(f"[OK] {len(indexes)} indexes created successfully")

def populate_tickers(conn):
    """Populate S&P 500 tickers."""
    cursor = conn.cursor()
    
    # Top 100 S&P 500 tickers for initial setup
    tickers = [
        ("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", "Large"),
        ("MSFT", "Microsoft Corporation", "Technology", "Software", "Large"),
        ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "Internet Retail", "Large"),
        ("GOOGL", "Alphabet Inc.", "Technology", "Internet Services", "Large"),
        ("GOOG", "Alphabet Inc.", "Technology", "Internet Services", "Large"),
        ("META", "Meta Platforms Inc.", "Technology", "Internet Services", "Large"),
        ("TSLA", "Tesla Inc.", "Consumer Discretionary", "Automobiles", "Large"),
        ("NVDA", "NVIDIA Corporation", "Technology", "Semiconductors", "Large"),
        ("BRK.B", "Berkshire Hathaway Inc.", "Financials", "Insurance", "Large"),
        ("JPM", "JPMorgan Chase & Co.", "Financials", "Banks", "Large"),
        ("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals", "Large"),
        ("V", "Visa Inc.", "Financials", "Financial Services", "Large"),
        ("UNH", "UnitedHealth Group Inc.", "Health Care", "Health Care Providers", "Large"),
        ("HD", "Home Depot Inc.", "Consumer Discretionary", "Home Improvement", "Large"),
        ("PG", "Procter & Gamble Co.", "Consumer Staples", "Household Products", "Large"),
        ("MA", "Mastercard Inc.", "Financials", "Financial Services", "Large"),
        ("BAC", "Bank of America Corp.", "Financials", "Banks", "Large"),
        ("ABBV", "AbbVie Inc.", "Health Care", "Pharmaceuticals", "Large"),
        ("PFE", "Pfizer Inc.", "Health Care", "Pharmaceuticals", "Large"),
        ("KO", "Coca-Cola Co.", "Consumer Staples", "Beverages", "Large"),
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO tickers (symbol, name, sector, industry, market_cap_category) VALUES (?, ?, ?, ?, ?)",
        tickers
    )
    
    conn.commit()
    print(f"[OK] Populated {len(tickers)} tickers")

def enable_wal_mode(conn):
    """Enable WAL mode for better concurrency."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    result = cursor.fetchone()
    print(f"[OK] WAL mode enabled: {result[0]}")

def main():
    """Main initialization function."""
    print("Initializing Financial Advisor Bot database...")
    print(f"Database path: {DATABASE_PATH.absolute()}")
    
    # Ensure data directory exists
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DATABASE_PATH)
    
    try:
        # Create tables
        create_tables(conn)
        
        # Create indexes
        create_indexes(conn)
        
        # Enable WAL mode
        enable_wal_mode(conn)
        
        # Populate tickers
        populate_tickers(conn)
        
        print("\n[OK] Database initialization complete!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
