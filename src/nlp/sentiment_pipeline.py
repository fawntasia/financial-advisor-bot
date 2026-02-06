import logging
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from src.database.dal import DataAccessLayer
from src.nlp.finbert_loader import FinBERTLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SentimentPipeline:
    """
    Pipeline for batch sentiment analysis of news headlines.
    """
    def __init__(self, dal: Optional[DataAccessLayer] = None, model_path: str = "models/finbert"):
        self.dal = dal or DataAccessLayer()
        self.loader = FinBERTLoader(model_path=model_path)
        
    def process_unprocessed(self, batch_size: int = 32, limit: int = 1000):
        """
        Fetches and processes news headlines that haven't been analyzed yet.
        """
        headlines = self.dal.get_unprocessed_news(limit=limit)
        if not headlines:
            logger.info("No unprocessed headlines found.")
            return
        
        logger.info(f"Processing {len(headlines)} headlines in batches of {batch_size}...")
        self._process_headlines(headlines, batch_size)

    def process_date(self, date: str, batch_size: int = 32):
        """
        Processes news headlines for a specific date that haven't been analyzed yet.
        
        Args:
            date: Date string in YYYY-MM-DD format.
            batch_size: Batch size for model inference.
        """
        with self.dal.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT h.* FROM news_headlines h
                   LEFT JOIN sentiment_scores s ON h.id = s.news_id
                   WHERE s.id IS NULL AND date(h.published_at) = ?""",
                (date,)
            )
            headlines = [dict(row) for row in cursor.fetchall()]
            
        if not headlines:
            logger.info(f"No unprocessed headlines found for date {date}.")
            # Even if no new headlines were processed, we might want to update aggregates 
            # if news were recently processed for this date but aggregates weren't updated.
            # But the prompt implies running the pipeline for a date means processing and storing.
            self.update_daily_aggregates(date)
            return
            
        logger.info(f"Processing {len(headlines)} headlines for date {date}...")
        self._process_headlines(headlines, batch_size)
        self.update_daily_aggregates(date)

    def _process_headlines(self, headlines: List[Dict], batch_size: int):
        """Internal helper to process a list of headline records."""
        texts = [h['headline'] for h in headlines]
        ids = [h['id'] for h in headlines]
        
        predictions = self.loader.predict(texts, batch_size=batch_size)
        
        sentiment_records = []
        for news_id, pred in zip(ids, predictions):
            if pred["label"] == "Error":
                continue
                
            sentiment_records.append({
                "news_id": news_id,
                "positive_score": pred["probs"].get("Positive", 0.0),
                "negative_score": pred["probs"].get("Negative", 0.0),
                "neutral_score": pred["probs"].get("Neutral", 0.0),
                "sentiment_label": pred["label"],
                "confidence": pred["score"]
            })
            
        if sentiment_records:
            self.dal.bulk_insert_sentiment_scores(sentiment_records)
            logger.info(f"Successfully stored {len(sentiment_records)} sentiment scores.")

    def update_daily_aggregates(self, date: str):
        """
        Calculates and stores daily sentiment aggregates for a specific date.
        """
        news_data = self.dal.get_news_for_date(date)
        if not news_data:
            logger.info(f"No sentiment data found to aggregate for date {date}.")
            return
            
        df = pd.DataFrame(news_data)
        
        # Aggregate by ticker
        # Note: Some headlines might not have a ticker if they are general news
        # We'll group by ticker and date
        
        summary = df.groupby('ticker').agg({
            'positive_score': 'mean',
            'negative_score': 'mean',
            'neutral_score': 'mean',
            'confidence': 'mean',
            'id': 'count'
        }).reset_index()
        
        for _, row in summary.iterrows():
            ticker = row['ticker']
            if not ticker: continue # Skip if no ticker
            
            avg_pos = row['positive_score']
            avg_neg = row['negative_score']
            avg_neu = row['neutral_score']
            
            # Determine overall sentiment label
            scores = {"Positive": avg_pos, "Negative": avg_neg, "Neutral": avg_neu}
            overall_sentiment = max(scores, key=scores.get)
            
            self.dal.insert_daily_sentiment(
                ticker=ticker,
                date=date,
                avg_positive=avg_pos,
                avg_negative=avg_neg,
                avg_neutral=avg_neu,
                overall_sentiment=overall_sentiment,
                confidence=row['confidence'],
                news_count=int(row['id'])
            )
            
        logger.info(f"Updated daily sentiment aggregates for {len(summary)} tickers on {date}.")
