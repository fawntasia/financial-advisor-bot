"""
Example tests to verify pytest setup is working.
"""
import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd


@pytest.mark.unit
def test_example():
    """Simple test to verify pytest is working."""
    assert True


@pytest.mark.unit
def test_imports():
    """Test that key imports work."""
    from src.data.stock_data import StockDataProcessor
    assert StockDataProcessor is not None


@pytest.mark.unit
def test_sample_ticker_fixture(sample_ticker):
    """Test that sample ticker fixture loads correctly."""
    assert sample_ticker["ticker"] == "AAPL"
    assert sample_ticker["name"] == "Apple Inc."
    assert sample_ticker["sector"] == "Technology"
    assert "market_cap" in sample_ticker


@pytest.mark.unit
def test_sample_prices_fixture(sample_prices):
    """Test that sample prices fixture loads correctly."""
    assert isinstance(sample_prices, pd.DataFrame)
    assert len(sample_prices) == 5
    assert "date" in sample_prices.columns
    assert "open" in sample_prices.columns
    assert "high" in sample_prices.columns
    assert "low" in sample_prices.columns
    assert "close" in sample_prices.columns
    assert "volume" in sample_prices.columns


@pytest.mark.unit
def test_sample_news_fixture(sample_news):
    """Test that sample news fixture loads correctly."""
    assert isinstance(sample_news, list)
    assert len(sample_news) == 3
    assert "headline" in sample_news[0]
    assert "source" in sample_news[0]
    assert "sentiment" in sample_news[0]


@pytest.mark.unit
def test_mock_ticker_fixture(mock_ticker_data):
    """Test that mock ticker fixture works."""
    assert mock_ticker_data["ticker"] == "MSFT"
    assert mock_ticker_data["exchange"] == "NASDAQ"


@pytest.mark.unit
def test_mock_prices_fixture(mock_price_data):
    """Test that mock price data fixture works."""
    assert isinstance(mock_price_data, pd.DataFrame)
    assert len(mock_price_data) == 3
    assert mock_price_data["close"].iloc[0] == 103.0


@pytest.mark.unit
def test_mock_news_fixture(mock_news_data):
    """Test that mock news fixture works."""
    assert isinstance(mock_news_data, list)
    assert len(mock_news_data) == 2
    assert mock_news_data[0]["sentiment"] == "neutral"


@pytest.mark.database
@pytest.mark.integration
def test_database_fixture(db_connection):
    """Test that database connection fixture works."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1


@pytest.mark.unit
def test_environment_setup():
    """Test that test environment is properly configured."""
    import os
    assert os.environ.get("TESTING") == "true"
    assert "DATABASE_URL" in os.environ
