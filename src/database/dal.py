"""
Data Access Layer (DAL) for the Financial Advisor Bot.
Provides abstraction over SQLite database operations.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

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
    
    def get_all_tickers(self) -> List[str]:
        """Get all ticker symbols."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM tickers WHERE is_active = 1 ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]
    
    def get_stock_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get stock prices for a ticker in date range."""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM stock_prices 
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            return pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
    
    def bulk_insert_prices(self, records: List[Dict]):
        """Bulk insert price records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT OR REPLACE INTO stock_prices 
                   (ticker, date, open, high, low, close, volume, adjusted_close)
                   VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :adjusted_close)""",
                records
            )
            conn.commit()
    
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
    
    def insert_sentiment_scores(self, scores: List[Dict]):
        """Insert sentiment scores."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT INTO sentiment_scores 
                   (headline_id, ticker, sentiment_score, confidence, sentiment_label)
                   VALUES (:headline_id, :ticker, :sentiment_score, :confidence, :sentiment_label)""",
                scores
            )
            conn.commit()
    
    def get_daily_sentiment(self, ticker: str, date: str) -> Optional[Dict]:
        """Get daily aggregated sentiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_sentiment WHERE ticker = ? AND date = ?",
                (ticker, date)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def insert_prediction(self, prediction: Dict):
        """Insert a model prediction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO predictions 
                   (ticker, model_name, prediction_date, target_date, predicted_price, 
                    predicted_direction, confidence)
                   VALUES (:ticker, :model_name, :prediction_date, :target_date, 
                           :predicted_price, :predicted_direction, :confidence)""",
                prediction
            )
            conn.commit()
    
    def get_model_performance(self, model_name: str) -> List[Dict]:
        """Get performance metrics for a model."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_performance WHERE model_name = ? ORDER BY training_date DESC",
                (model_name,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def log_system_event(self, level: str, component: str, message: str, details: str = None):
        """Log a system event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_logs (log_level, component, message, details) VALUES (?, ?, ?, ?)",
                (level, component, message, details)
            )
            conn.commit()
