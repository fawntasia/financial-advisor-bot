"""Abstractions for fetching financial news from multiple providers."""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def _format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime consistently for DB storage."""
    if dt is None:
        return None
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse common ISO datetime formats used by NewsAPI."""
    if not value:
        return None
    try:
        # Handles "...Z" and offsets.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_rfc822_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse RFC 822/2822 datetimes used in RSS feeds."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def _parse_date_only(value: Optional[str]) -> Optional[datetime.date]:
    """Parse date boundary input in YYYY-MM-DD format."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


class NewsProvider(ABC):
    """Abstract base class for news providers."""

    @abstractmethod
    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news for a query (usually a ticker).

        Returns:
            List[dict] with keys:
              - title
              - source
              - url
              - published_at (YYYY-MM-DD HH:MM:SS or None)
              - summary
              - provider
        """


class NewsAPIClient(NewsProvider):
    """Client for NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            logger.warning("NewsAPI key not found. NewsAPI client may return no data.")

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.error("Cannot fetch NewsAPI data: API key missing.")
            return []

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max(1, int(limit)),
        }
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "ok":
                logger.error("NewsAPI error: %s", data.get("message"))
                return []

            articles: List[Dict[str, Any]] = []
            for item in data.get("articles", []):
                dt = _parse_iso_datetime(item.get("publishedAt"))
                articles.append(
                    {
                        "title": item.get("title"),
                        "source": item.get("source", {}).get("name"),
                        "url": item.get("url"),
                        "published_at": _format_datetime(dt),
                        "summary": item.get("description"),
                        "provider": "newsapi",
                    }
                )
            return articles
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch NewsAPI news for %s: %s", query, exc)
            return []


class AlphaVantageNewsClient(NewsProvider):
    """Client for Alpha Vantage News & Sentiment API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage key not found. AlphaVantage client may return no data.")

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.error("Cannot fetch Alpha Vantage data: API key missing.")
            return []

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": query,
            "apikey": self.api_key,
            "limit": max(1, int(limit)),
            "sort": "LATEST",
        }
        if start_date:
            params["time_from"] = start_date.replace("-", "") + "T0000"
        if end_date:
            params["time_to"] = end_date.replace("-", "") + "T2359"

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                logger.error("Alpha Vantage error: %s", data["Error Message"])
                return []
            if "Note" in data:
                logger.warning("Alpha Vantage limit reached: %s", data["Note"])

            articles: List[Dict[str, Any]] = []
            for item in data.get("feed", []):
                published_raw = item.get("time_published")
                dt: Optional[datetime] = None
                if published_raw:
                    try:
                        dt = datetime.strptime(published_raw, "%Y%m%dT%H%M%S")
                    except ValueError:
                        dt = None

                articles.append(
                    {
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "published_at": _format_datetime(dt),
                        "summary": item.get("summary"),
                        "av_sentiment_score": item.get("overall_sentiment_score"),
                        "av_sentiment_label": item.get("overall_sentiment_label"),
                        "provider": "alphavantage",
                    }
                )
            return articles
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch Alpha Vantage news for %s: %s", query, exc)
            return []


class YahooFinanceRSSClient(NewsProvider):
    """No-key Yahoo Finance RSS client for ticker headlines."""

    BASE_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    REQUEST_HEADERS = {
        # Yahoo RSS commonly returns 429 to generic default clients.
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        ticker = query.upper().strip()
        if not ticker:
            return []

        start_bound = _parse_date_only(start_date)
        end_bound = _parse_date_only(end_date)
        url = self.BASE_URL.format(ticker=ticker)

        try:
            response = requests.get(url, headers=self.REQUEST_HEADERS, timeout=20)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (requests.exceptions.RequestException, ElementTree.ParseError) as exc:
            logger.error("Failed to fetch Yahoo RSS news for %s: %s", ticker, exc)
            return []

        articles: List[Dict[str, Any]] = []
        for item in root.findall("./channel/item"):
            title = item.findtext("title")
            link = item.findtext("link")
            source = item.findtext("source") or "Yahoo Finance"
            description = unescape(item.findtext("description") or "")
            pub_date_raw = item.findtext("pubDate")
            pub_dt = _parse_rfc822_datetime(pub_date_raw)
            pub_str = _format_datetime(pub_dt)

            if pub_dt:
                pub_day = pub_dt.date()
                if start_bound and pub_day < start_bound:
                    continue
                if end_bound and pub_day > end_bound:
                    continue

            articles.append(
                {
                    "title": title,
                    "source": source,
                    "url": link,
                    "published_at": pub_str,
                    "summary": description,
                    "provider": "rss",
                }
            )
            if len(articles) >= max(1, int(limit)):
                break

        return articles


class FallbackNewsClient(NewsProvider):
    """Try providers in order until one yields articles."""

    def __init__(self, providers: List[NewsProvider]):
        self.providers = providers

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        for provider in self.providers:
            articles = provider.fetch_news(
                query=query,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
            if articles:
                return articles
        return []


class MockNewsClient(NewsProvider):
    """Mock provider for tests and local development."""

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        logger.info("Using MockNewsClient for %s", query)
        return [
            {
                "title": f"Mock News for {query} {i}",
                "source": "Mock Finance",
                "url": f"https://mock.finance/{query}/{i}",
                "published_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": f"This is a mock summary for {query} article {i}.",
                "provider": "mock",
            }
            for i in range(max(1, int(limit)))
        ]


def get_news_client(provider: str = "auto") -> NewsProvider:
    """Return a news client instance by provider strategy name."""
    provider_key = provider.lower().strip()

    if provider_key == "alphavantage":
        return AlphaVantageNewsClient()
    if provider_key == "mock":
        return MockNewsClient()
    if provider_key == "rss":
        return YahooFinanceRSSClient()
    if provider_key == "auto":
        chain: List[NewsProvider] = [YahooFinanceRSSClient()]

        newsapi = NewsAPIClient()
        if newsapi.api_key:
            chain.append(newsapi)

        alphavantage = AlphaVantageNewsClient()
        if alphavantage.api_key:
            chain.append(alphavantage)

        return FallbackNewsClient(chain)

    return NewsAPIClient()
