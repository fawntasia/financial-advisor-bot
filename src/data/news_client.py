"""
News Client Module
Provides abstraction for fetching financial news from various sources.
"""

import os
import logging
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class NewsProvider(ABC):
    """Abstract base class for news providers."""
    
    @abstractmethod
    def fetch_news(self, query: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news for a given query (ticker or keyword).
        
        Args:
            query: Ticker symbol or keyword to search for.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            limit: Maximum number of articles to return.
            
        Returns:
            List of dictionaries containing news data.
            Format:
            [
                {
                    "title": "Article Title",
                    "source": "Source Name",
                    "url": "https://...",
                    "published_at": "YYYY-MM-DD HH:MM:SS",
                    "summary": "Brief summary..."
                },
                ...
            ]
        """
        pass

class NewsAPIClient(NewsProvider):
    """Client for NewsAPI.org."""
    
    BASE_URL = "https://newsapi.org/v2/everything"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            logger.warning("NewsAPI key not found. News fetching will fail.")
            
    def fetch_news(self, query: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI."""
        if not self.api_key:
            logger.error("Cannot fetch news: API key missing.")
            return []
            
        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
        }
        
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
            
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message')}")
                return []
                
            articles = []
            for item in data.get("articles", []):
                # Parse date
                published_at = item.get("publishedAt")
                if published_at:
                    # Convert ISO format to simpler format if needed
                    # "2024-02-04T12:00:00Z" -> "2024-02-04 12:00:00"
                    try:
                        dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                        published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass # Keep original if parsing fails
                
                articles.append({
                    "title": item.get("title"),
                    "source": item.get("source", {}).get("name"),
                    "url": item.get("url"),
                    "published_at": published_at,
                    "summary": item.get("description")
                })
                
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch news for {query}: {e}")
            return []

class AlphaVantageNewsClient(NewsProvider):
    """Client for Alpha Vantage News & Sentiment API."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage key not found.")

    def fetch_news(self, query: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news using Alpha Vantage Sentiment API."""
        if not self.api_key:
            logger.error("Cannot fetch news: API key missing.")
            return []
            
        # AV uses 'time_from' and 'time_to' in format YYYYMMDDTHHMM
        # We need to convert or just ignore for simple daily fetches
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": query, # AV expects tickers
            "apikey": self.api_key,
            "limit": limit,
            "sort": "LATEST"
        }
        
        if start_date:
            # Simple conversion YYYY-MM-DD -> YYYYMMDDT0000
            params["time_from"] = start_date.replace("-", "") + "T0000"
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error message
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return []
            if "Note" in data: # API limit reached
                 logger.warning(f"Alpha Vantage limit reached: {data['Note']}")
                 
            articles = []
            for item in data.get("feed", []):
                # Parse date "20240204T120000"
                published_at = item.get("time_published")
                if published_at:
                    try:
                        dt = datetime.strptime(published_at, "%Y%m%dT%H%M%S")
                        published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass

                articles.append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published_at": published_at,
                    "summary": item.get("summary"),
                    # AV provides sentiment too, we can capture it if we want
                    "av_sentiment_score": item.get("overall_sentiment_score"),
                    "av_sentiment_label": item.get("overall_sentiment_label")
                })
                
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch news for {query}: {e}")
            return []

def get_news_client(provider: str = "newsapi") -> NewsProvider:
    """Factory function to get a news client."""
    if provider.lower() == "alphavantage":
        return AlphaVantageNewsClient()
    return NewsAPIClient()
