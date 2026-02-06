"""
Context Builder for the Financial Advisor LLM.
Fetches data from DAL and formats it into natural language summaries.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from src.database.dal import DataAccessLayer

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builds context for LLM prompts by fetching and formatting data from the database.
    Includes a simple cache to avoid redundant queries in the same session.
    """

    def __init__(self, dal: DataAccessLayer):
        self.dal = dal
        self._cache = {}  # Format: {ticker: {"timestamp": datetime, "data": dict}}
        self._cache_ttl = timedelta(minutes=15)

    def build_context(self, ticker: str) -> Dict[str, str]:
        """
        Builds the context for a specific ticker.
        
        Args:
            ticker: The stock ticker symbol.
            
        Returns:
            A dictionary containing formatted summaries:
            - price: Latest price information.
            - indicators_summary: Natural language summary of technical indicators.
            - sentiment_summary: Natural language summary of recent sentiment.
            - prediction_summary: Natural language summary of AI predictions.
        """
        now = datetime.now()
        
        # Check cache
        if ticker in self._cache:
            cached_item = self._cache[ticker]
            if now - cached_item["timestamp"] < self._cache_ttl:
                logger.info(f"Using cached context for {ticker}")
                return cached_item["data"]

        logger.info(f"Building fresh context for {ticker}")
        
        latest_date = self.dal.get_latest_price_date(ticker)
        if not latest_date:
            return {
                "price": f"No price data available for {ticker}.",
                "indicators_summary": "No technical indicators available.",
                "sentiment_summary": "No sentiment data available.",
                "prediction_summary": "No AI predictions available."
            }

        price_data = self._get_latest_price(ticker, latest_date)
        indicators_data = self.dal.get_technical_indicators(ticker, latest_date)
        sentiment_data = self.dal.get_daily_sentiment(ticker, latest_date)
        predictions = self.dal.get_predictions(ticker, limit=5)

        # Get close price for indicator comparisons
        current_price = price_data.get('close') if price_data else None

        context = {
            "price": self._format_price(ticker, latest_date, price_data),
            "indicators_summary": self._format_indicators(indicators_data, current_price),
            "sentiment_summary": self._format_sentiment(sentiment_data),
            "prediction_summary": self._format_predictions(predictions)
        }

        # Update cache
        self._cache[ticker] = {
            "timestamp": now,
            "data": context
        }

        return context

    def _get_latest_price(self, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest price for a ticker on a specific date."""
        df = self.dal.get_stock_prices(ticker, date, date)
        if not df.empty:
            return df.iloc[0].to_dict()
        return None

    def _format_price(self, ticker: str, date: str, price_data: Optional[Dict[str, Any]]) -> str:
        """Format latest price into natural language."""
        if not price_data:
            return f"No price data available for {ticker} on {date}."
        
        price = price_data.get('close')
        change = price_data.get('close') - price_data.get('open') if price_data.get('open') else 0
        pct_change = (change / price_data.get('open') * 100) if price_data.get('open') else 0
        
        direction = "up" if change >= 0 else "down"
        return (f"As of {date}, {ticker} is trading at ${price:.2f}, "
                f"{direction} {abs(pct_change):.2f}% from the open.")

    def _format_indicators(self, indicators: Optional[Dict[str, Any]], current_price: Optional[float]) -> str:
        """Format technical indicators into natural language."""
        if not indicators:
            return "Technical indicator data is currently unavailable."

        date = indicators.get('date', 'Unknown')
        parts = [f"Technical analysis as of {date}:"]
        
        # RSI
        rsi = indicators.get('rsi_14')
        if rsi is not None:
            status = "neutral"
            if rsi > 70:
                status = "overbought"
            elif rsi < 30:
                status = "oversold"
            parts.append(f"The RSI is {rsi:.2f}, which is considered {status}.")

        # MACD
        macd_hist = indicators.get('macd_histogram')
        if macd_hist is not None:
            momentum = "bullish" if macd_hist > 0 else "bearish"
            parts.append(f"MACD histogram shows {momentum} momentum.")

        # Moving Averages
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        sma_200 = indicators.get('sma_200')
        
        if current_price and sma_20:
            rel = "above" if current_price > sma_20 else "below"
            parts.append(f"Price is currently {rel} its 20-day SMA (${sma_20:.2f}).")

        if sma_50 and sma_200:
            if sma_50 > sma_200:
                parts.append("A Golden Cross is present (SMA 50 > SMA 200), indicating a long-term bullish trend.")
            else:
                parts.append("A Death Cross is present (SMA 50 < SMA 200), indicating a long-term bearish trend.")
        
        # Bollinger Bands
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        if current_price and bb_upper and bb_lower:
            if current_price > bb_upper:
                parts.append("Price is above the upper Bollinger Band, suggesting it may be overextended.")
            elif current_price < bb_lower:
                parts.append("Price is below the lower Bollinger Band, suggesting it may be oversold.")
        
        return " ".join(parts) if parts else "Technical indicators are mixed or insufficient for a clear summary."

    def _format_sentiment(self, sentiment: Optional[Dict[str, Any]]) -> str:
        """Format sentiment data into natural language."""
        if not sentiment:
            return "No recent news sentiment data available."

        overall = sentiment.get('overall_sentiment', 'neutral').lower()
        score = sentiment.get('confidence', 0)
        count = sentiment.get('news_count', 0)
        date = sentiment.get('date')

        return (f"Market sentiment on {date} was {overall} (confidence: {score:.2f}) "
                f"based on {count} news articles.")

    def _format_predictions(self, predictions: List[Dict[str, Any]]) -> str:
        """Format model predictions into natural language."""
        if not predictions:
            return "No recent AI model predictions available."

        # Group predictions by model to get the latest for each
        latest_by_model = {}
        for p in predictions:
            m_name = p.get('model_name')
            if m_name and m_name not in latest_by_model:
                latest_by_model[m_name] = p
        
        summaries = []
        for m_name, p in latest_by_model.items():
            # direction 1: UP, 0: DOWN (or neutral)
            direction = "UP" if p.get('predicted_direction') == 1 else "DOWN"
            confidence = p.get('confidence')
            date = p.get('date')
            
            s = f"The {m_name} model predicts an {direction} move for {date}"
            if confidence:
                s += f" with {confidence:.2f} confidence"
            summaries.append(s + ".")
        
        return " ".join(summaries)

    def clear_cache(self):
        """Manually clear the context cache."""
        self._cache = {}
