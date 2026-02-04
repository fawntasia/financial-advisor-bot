"""
Pytest configuration and shared fixtures for Financial Advisor Bot tests.
"""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """Create a temporary database file for testing."""
    db_path = tmp_path_factory.mktemp("data") / "test_financial_advisor.db"
    return db_path


@pytest.fixture(scope="function")
def db_connection(test_db_path):
    """
    Create a database connection for testing.
    
    Yields a connection that rolls back after each test.
    Uses SQLite with WAL mode for better concurrency.
    """
    conn = sqlite3.connect(str(test_db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    
    # Start a transaction
    conn.execute("BEGIN")
    
    yield conn
    
    # Rollback after test to clean up
    conn.rollback()
    conn.close()


@pytest.fixture(scope="function")
def db_cursor(db_connection):
    """Provide a database cursor for testing."""
    cursor = db_connection.cursor()
    yield cursor


@pytest.fixture(scope="session")
def sample_ticker():
    """Load sample ticker data from fixture file."""
    fixture_path = FIXTURES_DIR / "sample_ticker.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_prices():
    """Load sample price data from fixture file as DataFrame."""
    fixture_path = FIXTURES_DIR / "sample_prices.csv"
    df = pd.read_csv(fixture_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture(scope="session")
def sample_news():
    """Load sample news data from fixture file."""
    fixture_path = FIXTURES_DIR / "sample_news.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="function")
def mock_ticker_data():
    """Provide mock ticker data for unit tests."""
    return {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software",
        "market_cap": 2500000000000,
        "exchange": "NASDAQ",
        "currency": "USD"
    }


@pytest.fixture(scope="function")
def mock_price_data():
    """Provide mock price data for unit tests."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'open': [100.0, 102.0, 101.0],
        'high': [105.0, 106.0, 104.0],
        'low': [99.0, 101.0, 100.0],
        'close': [103.0, 104.0, 102.0],
        'volume': [1000000, 1200000, 1100000]
    })


@pytest.fixture(scope="function")
def mock_news_data():
    """Provide mock news data for unit tests."""
    return [
        {
            "headline": "Test News Headline 1",
            "source": "Test Source",
            "date": "2024-01-01",
            "sentiment": "neutral",
            "url": "https://example.com/test/1"
        },
        {
            "headline": "Test News Headline 2",
            "source": "Test Source",
            "date": "2024-01-02",
            "sentiment": "positive",
            "url": "https://example.com/test/2"
        }
    ]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
