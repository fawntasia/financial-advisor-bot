import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.database.dal import DataAccessLayer

# Define schema creation SQL here to avoid dependency on scripts/init_db.py
# This ensures tests are self-contained and consistent with what the DAL expects
SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        sector TEXT,
        industry TEXT,
        date_added DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS stock_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        adj_close REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date),
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    );

    CREATE TABLE IF NOT EXISTS technical_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        rsi_14 REAL,
        macd_line REAL,
        macd_signal REAL,
        macd_histogram REAL,
        bb_upper REAL,
        bb_middle REAL,
        bb_lower REAL,
        sma_20 REAL,
        sma_50 REAL,
        sma_200 REAL,
        ema_12 REAL,
        ema_26 REAL,
        atr_14 REAL,
        volume_sma_20 REAL,
        price_roc_10 REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date),
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    );

    CREATE TABLE IF NOT EXISTS news_headlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        headline TEXT NOT NULL,
        source TEXT,
        url TEXT,
        published_at TIMESTAMP,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sentiment_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_id INTEGER NOT NULL,
        positive_score REAL,
        negative_score REAL,
        neutral_score REAL,
        sentiment_label TEXT,
        confidence REAL,
        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (news_id) REFERENCES news_headlines(id)
    );

    CREATE TABLE IF NOT EXISTS daily_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        avg_positive REAL,
        avg_negative REAL,
        avg_neutral REAL,
        overall_sentiment TEXT,
        confidence REAL,
        news_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date),
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        model_name TEXT NOT NULL,
        predicted_direction INTEGER,
        predicted_price REAL,
        confidence REAL,
        features_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    );

    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        ticker TEXT,
        train_date_start DATE,
        train_date_end DATE,
        test_date_start DATE,
        test_date_end DATE,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        total_return REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    );

    CREATE TABLE IF NOT EXISTS system_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT NOT NULL,
        component TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_name TEXT UNIQUE,
        setting_value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS schema_migrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version INTEGER UNIQUE,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT
    );
"""

@pytest.fixture
def dal(tmp_path):
    """Create a DataAccessLayer instance with a temporary database."""
    db_file = tmp_path / "test_financial_advisor.db"
    
    # Initialize schema
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    
    return DataAccessLayer(db_file)

class TestDataAccessLayer:
    
    def test_init(self, dal):
        """Test DAL initialization."""
        assert dal.db_path.exists()
        assert dal.db_path.name == "test_financial_advisor.db"

    # ==================== Tickers ====================
    
    def test_ticker_operations(self, dal):
        """Test insert, get info, and get all tickers."""
        # Insert
        dal.insert_ticker("AAPL", "Apple Inc.", "Tech", "Consumer Electronics", "2020-01-01")
        dal.insert_ticker("MSFT", "Microsoft", "Tech", "Software", "2020-01-01")
        
        # Get Info
        info = dal.get_ticker_info("AAPL")
        assert info is not None
        assert info["name"] == "Apple Inc."
        assert info["sector"] == "Tech"
        
        # Get All
        tickers = dal.get_all_tickers()
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert len(tickers) == 2
        
        # Update existing
        dal.insert_ticker("AAPL", "Apple Inc. Updated", "Technology", "Hardware")
        info_updated = dal.get_ticker_info("AAPL")
        assert info_updated["name"] == "Apple Inc. Updated"
        assert info_updated["sector"] == "Technology"

    def test_get_ticker_universe(self, dal):
        """Test retrieval of ticker list with metadata for UI browsing."""
        dal.insert_ticker("AAPL", "Apple Inc.", "Technology")
        dal.insert_ticker("MSFT", "Microsoft Corporation", "Technology")

        universe = dal.get_ticker_universe()
        assert len(universe) == 2
        assert universe[0]["ticker"] == "AAPL"
        assert universe[0]["name"] == "Apple Inc."
        assert universe[1]["ticker"] == "MSFT"
        assert universe[1]["sector"] == "Technology"

    def test_get_nonexistent_ticker(self, dal):
        """Test retrieving a ticker that doesn't exist."""
        info = dal.get_ticker_info("INVALID")
        assert info is None

    # ==================== Stock Prices ====================

    def test_stock_prices_operations(self, dal):
        """Test bulk insert and retrieval of stock prices."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        
        records = [
            {
                "ticker": "AAPL", "date": "2023-01-01", 
                "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, 
                "volume": 1000, "adj_close": 102.0
            },
            {
                "ticker": "AAPL", "date": "2023-01-02", 
                "open": 102.0, "high": 108.0, "low": 101.0, "close": 107.0, 
                "volume": 1200, "adj_close": 107.0
            }
        ]
        
        dal.bulk_insert_prices(records)
        
        # Test get_latest_price_date
        latest_date = dal.get_latest_price_date("AAPL")
        assert latest_date == "2023-01-02"
        
        # Test get_stock_prices
        df = dal.get_stock_prices("AAPL", "2023-01-01", "2023-01-02")
        assert len(df) == 2
        assert df.iloc[0]["close"] == 102.0
        assert df.iloc[1]["close"] == 107.0

    def test_stock_prices_empty(self, dal):
        """Test retrieving stock prices when none exist."""
        df = dal.get_stock_prices("AAPL", "2023-01-01", "2023-01-02")
        assert df.empty
        
        date = dal.get_latest_price_date("AAPL")
        assert date is None

    # ==================== Technical Indicators ====================

    def test_technical_indicators_operations(self, dal):
        """Test bulk insert and retrieval of technical indicators."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        
        records = [
            {
                "ticker": "AAPL", "date": "2023-01-01",
                "rsi_14": 50.5, "macd_line": 0.5, "macd_signal": 0.4, "macd_histogram": 0.1,
                "bb_upper": 110.0, "bb_middle": 105.0, "bb_lower": 100.0,
                "sma_20": 105.0, "sma_50": 102.0, "sma_200": 95.0,
                "ema_12": 106.0, "ema_26": 103.0, "atr_14": 2.5,
                "volume_sma_20": 1000.0, "price_roc_10": 0.05
            }
        ]
        
        dal.bulk_insert_indicators(records)
        
        indicators = dal.get_technical_indicators("AAPL", "2023-01-01")
        assert indicators is not None
        assert indicators["rsi_14"] == 50.5
        assert indicators["sma_20"] == 105.0

    def test_missing_technical_indicators(self, dal):
        """Test retrieving missing indicators."""
        indicators = dal.get_technical_indicators("AAPL", "2099-01-01")
        assert indicators is None

    # ==================== News Headlines ====================

    def test_news_operations(self, dal):
        """Test inserting and retrieving news."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        
        news_id = dal.insert_news_headline(
            "AAPL", "Apple releases new iPhone", "TechCrunch", 
            "http://example.com", "2023-09-01T10:00:00"
        )
        assert news_id is not None
        
        news_items = dal.get_news_by_ticker("AAPL")
        assert len(news_items) == 1
        assert news_items[0]["headline"] == "Apple releases new iPhone"
        
        # Test Limit
        dal.insert_news_headline("AAPL", "Another headline", "Reuters", "url", "2023-09-02")
        news_items = dal.get_news_by_ticker("AAPL", limit=1)
        assert len(news_items) == 1
        # Should be the latest one (2023-09-02) if sorted by date desc, but let's check sorting logic in DAL
        # DAL: ORDER BY published_at DESC
        assert news_items[0]["headline"] == "Another headline"

    def test_get_unprocessed_news(self, dal):
        """Test retrieving news without sentiment scores."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        news_id1 = dal.insert_news_headline("AAPL", "Unprocessed", "Source", "url", "2023-01-01")
        news_id2 = dal.insert_news_headline("AAPL", "Processed", "Source", "url", "2023-01-02")
        
        dal.insert_sentiment_score(news_id2, 0.5, 0.2, 0.3, "neutral", 0.9)
        
        unprocessed = dal.get_unprocessed_news()
        assert len(unprocessed) == 1
        assert unprocessed[0]["id"] == news_id1
        assert unprocessed[0]["headline"] == "Unprocessed"

    def test_get_news_for_date(self, dal):
        """Test retrieving news with sentiment for a specific date."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        news_id = dal.insert_news_headline("AAPL", "Daily News", "Source", "url", "2023-01-01 10:00:00")
        dal.insert_sentiment_score(news_id, 0.8, 0.1, 0.1, "positive", 0.9)
        
        # Another day
        news_id2 = dal.insert_news_headline("AAPL", "Other Day", "Source", "url", "2023-01-02 10:00:00")
        dal.insert_sentiment_score(news_id2, 0.1, 0.8, 0.1, "negative", 0.9)
        
        results = dal.get_news_for_date("2023-01-01")
        assert len(results) == 1
        assert results[0]["headline"] == "Daily News"
        assert results[0]["positive_score"] == 0.8
        assert results[0]["sentiment_label"] == "positive"

    # ==================== Sentiment Scores ====================

    def test_sentiment_operations(self, dal):
        """Test inserting and retrieving sentiment scores."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        news_id = dal.insert_news_headline("AAPL", "Good news", "Source", "url", "2023-01-01")
        
        dal.insert_sentiment_score(news_id, 0.8, 0.1, 0.1, "positive", 0.9)
        
        score = dal.get_sentiment_by_news_id(news_id)
        assert score is not None
        assert score["positive_score"] == 0.8
        assert score["sentiment_label"] == "positive"
        
        # Test Bulk Insert
        news_id2 = dal.insert_news_headline("AAPL", "Bad news", "Source", "url", "2023-01-02")
        records = [{
            "news_id": news_id2, "positive_score": 0.1, "negative_score": 0.8, 
            "neutral_score": 0.1, "sentiment_label": "negative", "confidence": 0.85
        }]
        dal.bulk_insert_sentiment_scores(records)
        
        score2 = dal.get_sentiment_by_news_id(news_id2)
        assert score2["sentiment_label"] == "negative"

    # ==================== Daily Sentiment ====================

    def test_daily_sentiment_operations(self, dal):
        """Test daily sentiment operations."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        
        dal.insert_daily_sentiment(
            "AAPL", "2023-01-01", 0.7, 0.1, 0.2, "positive", 0.8, 10
        )
        
        sentiment = dal.get_daily_sentiment("AAPL", "2023-01-01")
        assert sentiment is not None
        assert sentiment["overall_sentiment"] == "positive"
        assert sentiment["news_count"] == 10

    # ==================== Predictions ====================

    def test_prediction_operations(self, dal):
        """Test prediction operations."""
        dal.insert_ticker("AAPL", "Apple Inc.")
        
        dal.insert_prediction(
            "AAPL", "2023-01-02", "LSTM", 1, 150.0, 0.75, "hash123"
        )
        
        preds = dal.get_predictions("AAPL")
        assert len(preds) == 1
        assert preds[0]["model_name"] == "LSTM"
        assert preds[0]["predicted_price"] == 150.0
        
        # Test specific model retrieval
        dal.insert_prediction("AAPL", "2023-01-03", "Ensemble", 0, 148.0, 0.6)
        
        lstm_preds = dal.get_predictions("AAPL", model_name="LSTM")
        assert len(lstm_preds) == 1
        assert lstm_preds[0]["model_name"] == "LSTM"

    # ==================== Model Performance ====================

    def test_model_performance_operations(self, dal):
        """Test model performance tracking."""
        dal.insert_model_performance(
            "LSTM", "AAPL", "2022-01-01", "2022-12-31", 
            "2023-01-01", "2023-01-31", accuracy=0.65
        )
        
        # Test with ticker
        perf = dal.get_model_performance("LSTM", ticker="AAPL")
        assert len(perf) == 1
        assert perf[0]["accuracy"] == 0.65
        
        # Test without ticker (cover line 305)
        perf_all = dal.get_model_performance("LSTM")
        assert len(perf_all) == 1
        assert perf_all[0]["model_name"] == "LSTM"

    # ==================== System Logs ====================

    def test_system_logs(self, dal):
        """Test logging system."""
        dal.log_system_event("INFO", "DataIngestion", "Started ingestion")
        dal.log_system_event("ERROR", "DataIngestion", "Failed connection")
        
        # Test with both level and component (cover line 332)
        specific_logs = dal.get_system_logs(level="ERROR", component="DataIngestion")
        assert len(specific_logs) == 1
        assert specific_logs[0]["message"] == "Failed connection"
        
        logs = dal.get_system_logs()
        assert len(logs) == 2
        
        # Filter by level
        error_logs = dal.get_system_logs(level="ERROR")
        assert len(error_logs) == 1
        assert error_logs[0]["message"] == "Failed connection"
        
        # Filter by component
        component_logs = dal.get_system_logs(component="DataIngestion")
        assert len(component_logs) == 2

    # ==================== User Preferences ====================

    def test_user_preferences(self, dal):
        """Test user preferences."""
        dal.set_user_preference("theme", "dark")
        
        val = dal.get_user_preference("theme")
        assert val == "dark"
        
        # Update
        dal.set_user_preference("theme", "light")
        val_updated = dal.get_user_preference("theme")
        assert val_updated == "light"

    # ==================== Migrations ====================

    def test_migrations(self, dal):
        """Test migration version tracking."""
        current = dal.get_current_migration_version()
        # Should be 0 initially or whatever we set in schema if we had inserted one
        # Our manual schema init didn't insert a version, so 0 is expected
        assert current == 0
        
        dal.record_migration(1, "Init")
        assert dal.get_current_migration_version() == 1
