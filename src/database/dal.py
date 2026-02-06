"""
Data Access Layer (DAL) for the Financial Advisor Bot.
Provides abstraction over SQLite database operations.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager
from datetime import datetime

DATABASE_PATH = Path("data/financial_advisor.db")


class DataAccessLayer:
    """Data Access Layer for database operations."""
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ==================== Tickers ====================
    
    def get_all_tickers(self) -> List[str]:
        """Get all ticker symbols."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ticker FROM tickers ORDER BY ticker")
            return [row[0] for row in cursor.fetchall()]

    def get_ticker_universe(self) -> List[Dict]:
        """
        Get the full ticker universe with display metadata.

        Returns:
            List of dictionaries with keys: ticker, name, sector.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ticker, name, sector
                FROM tickers
                ORDER BY ticker
                """
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get information about a specific ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tickers WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def insert_ticker(self, ticker: str, name: str, sector: Optional[str] = None, 
                      industry: Optional[str] = None, date_added: Optional[str] = None):
        """Insert a new ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO tickers 
                   (ticker, name, sector, industry, date_added)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticker, name, sector, industry, date_added)
            )
            conn.commit()
    
    # ==================== Stock Prices ====================
    
    def get_stock_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get stock prices for a ticker in date range."""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM stock_prices 
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            return pd.read_sql_query(query, conn, params=[ticker, start_date, end_date])
    
    def bulk_insert_prices(self, records: List[Dict]):
        """Bulk insert price records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT OR REPLACE INTO stock_prices 
                   (ticker, date, open, high, low, close, volume, adj_close)
                   VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :adj_close)""",
                records
            )
            conn.commit()
    
    def get_latest_price_date(self, ticker: str) -> Optional[str]:
        """Get the latest price date for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(date) FROM stock_prices WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
    
    # ==================== Technical Indicators ====================
    
    def get_technical_indicators(self, ticker: str, date: str) -> Optional[Dict]:
        """Get technical indicators for a ticker on a specific date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM technical_indicators WHERE ticker = ? AND date = ?",
                (ticker, date)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def bulk_insert_indicators(self, records: List[Dict]):
        """Bulk insert technical indicator records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT OR REPLACE INTO technical_indicators 
                   (ticker, date, rsi_14, macd_line, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, sma_20, sma_50, sma_200,
                    ema_12, ema_26, atr_14, volume_sma_20, price_roc_10)
                   VALUES (:ticker, :date, :rsi_14, :macd_line, :macd_signal, :macd_histogram,
                           :bb_upper, :bb_middle, :bb_lower, :sma_20, :sma_50, :sma_200,
                           :ema_12, :ema_26, :atr_14, :volume_sma_20, :price_roc_10)""",
                records
            )
            conn.commit()
    
    # ==================== News Headlines ====================
    
    def insert_news_headline(self, ticker: str, headline: str, source: Optional[str] = None,
                             url: Optional[str] = None, published_at: Optional[str] = None) -> Optional[int]:
        """Insert a news headline and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO news_headlines 
                   (ticker, headline, source, url, published_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticker, headline, source, url, published_at)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_news_by_ticker(self, ticker: str, limit: int = 100) -> List[Dict]:
        """Get news headlines for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM news_headlines 
                   WHERE ticker = ? 
                   ORDER BY published_at DESC 
                   LIMIT ?""",
                (ticker, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_unprocessed_news(self, limit: int = 1000) -> List[Dict]:
        """Get news headlines that don't have sentiment scores yet."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT h.* FROM news_headlines h
                   LEFT JOIN sentiment_scores s ON h.id = s.news_id
                   WHERE s.id IS NULL
                   ORDER BY h.published_at DESC
                   LIMIT ?""",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_news_for_date(self, date: str) -> List[Dict]:
        """Get news headlines for a specific date (YYYY-MM-DD)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT h.*, s.positive_score, s.negative_score, s.neutral_score, s.sentiment_label, s.confidence
                   FROM news_headlines h
                   JOIN sentiment_scores s ON h.id = s.news_id
                   WHERE date(h.published_at) = ?""",
                (date,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Sentiment Scores ====================

    
    def insert_sentiment_score(self, news_id: int, positive_score: float,
                               negative_score: float, neutral_score: float,
                               sentiment_label: str, confidence: float):
        """Insert a sentiment score for a news item."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO sentiment_scores 
                   (news_id, positive_score, negative_score, neutral_score, 
                    sentiment_label, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (news_id, positive_score, negative_score, neutral_score,
                 sentiment_label, confidence)
            )
            conn.commit()
    
    def bulk_insert_sentiment_scores(self, records: List[Dict]):
        """Bulk insert sentiment scores."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT INTO sentiment_scores 
                   (news_id, positive_score, negative_score, neutral_score,
                    sentiment_label, confidence)
                   VALUES (:news_id, :positive_score, :negative_score, :neutral_score,
                           :sentiment_label, :confidence)""",
                records
            )
            conn.commit()
    
    def get_sentiment_by_news_id(self, news_id: int) -> Optional[Dict]:
        """Get sentiment score for a specific news item."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sentiment_scores WHERE news_id = ?",
                (news_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ==================== Daily Sentiment ====================
    
    def get_daily_sentiment(self, ticker: str, date: str) -> Optional[Dict]:
        """Get daily aggregated sentiment for a ticker on a date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_sentiment WHERE ticker = ? AND date = ?",
                (ticker, date)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def insert_daily_sentiment(self, ticker: str, date: str, avg_positive: float,
                               avg_negative: float, avg_neutral: float,
                               overall_sentiment: str, confidence: float,
                               news_count: int):
        """Insert daily aggregated sentiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO daily_sentiment 
                   (ticker, date, avg_positive, avg_negative, avg_neutral,
                    overall_sentiment, confidence, news_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, date, avg_positive, avg_negative, avg_neutral,
                 overall_sentiment, confidence, news_count)
            )
            conn.commit()
    
    # ==================== Predictions ====================
    
    def insert_prediction(self, ticker: str, date: str, model_name: str,
                          predicted_direction: int, predicted_price: Optional[float] = None,
                          confidence: Optional[float] = None, features_hash: Optional[str] = None):
        """Insert a model prediction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO predictions 
                   (ticker, date, model_name, predicted_direction, 
                    predicted_price, confidence, features_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ticker, date, model_name, predicted_direction,
                 predicted_price, confidence, features_hash)
            )
            conn.commit()
    
    def get_predictions(self, ticker: str, model_name: Optional[str] = None,
                        limit: int = 100) -> List[Dict]:
        """Get predictions for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if model_name:
                cursor.execute(
                    """SELECT * FROM predictions 
                       WHERE ticker = ? AND model_name = ?
                       ORDER BY date DESC 
                       LIMIT ?""",
                    (ticker, model_name, limit)
                )
            else:
                cursor.execute(
                    """SELECT * FROM predictions 
                       WHERE ticker = ?
                       ORDER BY date DESC 
                       LIMIT ?""",
                    (ticker, limit)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Model Performance ====================
    
    def insert_model_performance(self, model_name: str, ticker: Optional[str] = None,
                                  train_date_start: Optional[str] = None, train_date_end: Optional[str] = None,
                                  test_date_start: Optional[str] = None, test_date_end: Optional[str] = None,
                                  accuracy: Optional[float] = None, precision: Optional[float] = None,
                                  recall: Optional[float] = None, f1_score: Optional[float] = None,
                                  total_return: Optional[float] = None, sharpe_ratio: Optional[float] = None,
                                  max_drawdown: Optional[float] = None):
        """Insert model performance metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO model_performance 
                   (model_name, ticker, train_date_start, train_date_end,
                    test_date_start, test_date_end, accuracy, precision,
                    recall, f1_score, total_return, sharpe_ratio, max_drawdown)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_name, ticker, train_date_start, train_date_end,
                 test_date_start, test_date_end, accuracy, precision,
                 recall, f1_score, total_return, sharpe_ratio, max_drawdown)
            )
            conn.commit()
    
    def get_model_performance(self, model_name: str, ticker: Optional[str] = None) -> List[Dict]:
        """Get performance metrics for a model."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if ticker:
                cursor.execute(
                    """SELECT * FROM model_performance 
                       WHERE model_name = ? AND ticker = ?
                       ORDER BY created_at DESC""",
                    (model_name, ticker)
                )
            else:
                cursor.execute(
                    """SELECT * FROM model_performance 
                       WHERE model_name = ?
                       ORDER BY created_at DESC""",
                    (model_name,)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== System Logs ====================
    
    def log_system_event(self, level: str, component: str, message: str, 
                         details: Optional[str] = None):
        """Log a system event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_logs (level, component, message, details) VALUES (?, ?, ?, ?)",
                (level, component, message, details)
            )
            conn.commit()
    
    def get_system_logs(self, level: Optional[str] = None, component: Optional[str] = None,
                        limit: int = 100) -> List[Dict]:
        """Get system logs with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if level and component:
                cursor.execute(
                    """SELECT * FROM system_logs 
                       WHERE level = ? AND component = ?
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (level, component, limit)
                )
            elif level:
                cursor.execute(
                    """SELECT * FROM system_logs 
                       WHERE level = ?
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (level, limit)
                )
            elif component:
                cursor.execute(
                    """SELECT * FROM system_logs 
                       WHERE component = ?
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (component, limit)
                )
            else:
                cursor.execute(
                    """SELECT * FROM system_logs 
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== User Preferences ====================
    
    def get_user_preference(self, setting_name: str) -> Optional[str]:
        """Get a user preference value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT setting_value FROM user_preferences WHERE setting_name = ?",
                (setting_name,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def set_user_preference(self, setting_name: str, setting_value: str):
        """Set a user preference value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO user_preferences 
                   (setting_name, setting_value, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (setting_name, setting_value)
            )
            conn.commit()
    
    # ==================== Schema Migrations ====================
    
    def get_current_migration_version(self) -> int:
        """Get the current schema migration version."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(version) FROM schema_migrations")
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0
    
    def record_migration(self, version: int, description: str):
        """Record a schema migration."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO schema_migrations 
                   (version, description) VALUES (?, ?)""",
                (version, description)
            )
            conn.commit()
