import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import ingest_data
from scripts.init_db import create_tables
from src.database.dal import DataAccessLayer


class MockYFinanceClient:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def get_ticker(self, ticker, period="5y", start=None, end=None, use_cache=True):
        return self.data_frame


class MockNewsClient:
    def __init__(self, articles):
        self.articles = articles

    def fetch_news(self, query, start_date=None, end_date=None, limit=10):
        return [
            {
                **article,
                "title": article["title"].format(query=query)
            }
            for article in self.articles
        ]


@pytest.mark.integration
def test_data_pipeline_end_to_end(tmp_path, sample_prices, monkeypatch):
    db_path = tmp_path / "test_financial_advisor.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    conn.close()

    dal = DataAccessLayer(db_path=db_path)
    dal.insert_ticker("MSFT", "Microsoft Corporation")

    prices = sample_prices.copy()
    prices = prices.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }
    )
    prices["Adj Close"] = prices["Close"]
    prices = prices.set_index("date")

    mock_articles = [
        {
            "title": "Mock News for {query}",
            "source": "Mock Finance",
            "url": "https://example.com/mock",
            "published_at": "2024-01-02 00:00:00",
            "summary": "Mock summary"
        }
    ]

    def mock_yfinance_client():
        return MockYFinanceClient(prices)

    def mock_get_news_client(provider="newsapi"):
        return MockNewsClient(mock_articles)

    monkeypatch.setattr(ingest_data, "YFinanceClient", mock_yfinance_client)
    monkeypatch.setattr(ingest_data, "get_news_client", mock_get_news_client)

    ingest_data.ingest_stock_data(dal, days=5)
    ingest_data.ingest_news_data(dal, days=1)

    stored_prices = dal.get_stock_prices("MSFT", "2024-01-01", "2024-01-03")
    assert not stored_prices.empty
    assert set(stored_prices["ticker"]) == {"MSFT"}

    stored_news = dal.get_news_by_ticker("MSFT")
    assert stored_news
    assert stored_news[0]["headline"] == "Mock News for MSFT"
