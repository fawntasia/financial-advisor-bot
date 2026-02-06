import sys

sys.path.insert(0, ".")

from datetime import datetime as real_datetime
from unittest.mock import Mock

import pandas as pd
import pytest

import src.llm.context_builder as context_builder_module
from src.llm.context_builder import ContextBuilder


@pytest.fixture
def mock_dal():
    return Mock()


def test_build_context_no_latest_date_returns_defaults(mock_dal):
    mock_dal.get_latest_price_date.return_value = None

    builder = ContextBuilder(mock_dal)
    result = builder.build_context("AAPL")

    assert result == {
        "price": "No price data available for AAPL.",
        "indicators_summary": "No technical indicators available.",
        "sentiment_summary": "No sentiment data available.",
        "prediction_summary": "No AI predictions available.",
    }
    mock_dal.get_stock_prices.assert_not_called()
    mock_dal.get_technical_indicators.assert_not_called()
    mock_dal.get_daily_sentiment.assert_not_called()
    mock_dal.get_predictions.assert_not_called()


def test_build_context_formats_with_data_includes_dates_and_prices(mock_dal):
    mock_dal.get_latest_price_date.return_value = "2024-01-03"
    mock_dal.get_stock_prices.return_value = pd.DataFrame(
        [{"open": 100.0, "close": 105.0}]
    )
    mock_dal.get_technical_indicators.return_value = {
        "date": "2024-01-03",
        "rsi_14": 55.0,
        "macd_histogram": 0.2,
        "sma_20": 102.0,
        "sma_50": 100.0,
        "sma_200": 90.0,
        "bb_upper": 106.0,
        "bb_lower": 96.0,
    }
    mock_dal.get_daily_sentiment.return_value = {
        "overall_sentiment": "positive",
        "confidence": 0.7,
        "news_count": 5,
        "date": "2024-01-03",
    }
    mock_dal.get_predictions.return_value = [
        {
            "model_name": "LSTM",
            "predicted_direction": 1,
            "confidence": 0.64,
            "date": "2024-01-04",
        },
        {
            "model_name": "Ensemble",
            "predicted_direction": 0,
            "confidence": 0.52,
            "date": "2024-01-04",
        },
    ]

    builder = ContextBuilder(mock_dal)
    result = builder.build_context("AAPL")

    assert "2024-01-03" in result["price"]
    assert "$105.00" in result["price"]
    assert "up 5.00%" in result["price"]
    assert "Technical analysis as of 2024-01-03" in result["indicators_summary"]
    assert "Market sentiment on 2024-01-03" in result["sentiment_summary"]
    assert "2024-01-04" in result["prediction_summary"]


def test_build_context_missing_supporting_data_returns_fallbacks(mock_dal):
    mock_dal.get_latest_price_date.return_value = "2024-01-03"
    mock_dal.get_stock_prices.return_value = pd.DataFrame([])
    mock_dal.get_technical_indicators.return_value = None
    mock_dal.get_daily_sentiment.return_value = None
    mock_dal.get_predictions.return_value = []

    builder = ContextBuilder(mock_dal)
    result = builder.build_context("AAPL")

    assert result["price"] == "No price data available for AAPL on 2024-01-03."
    assert result["indicators_summary"] == "Technical indicator data is currently unavailable."
    assert result["sentiment_summary"] == "No recent news sentiment data available."
    assert result["prediction_summary"] == "No recent AI model predictions available."


def test_build_context_cache_short_circuits_repeated_calls(mock_dal, monkeypatch):
    fixed_now = real_datetime(2024, 1, 3, 10, 0, 0)

    class FixedDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr(context_builder_module, "datetime", FixedDateTime)

    mock_dal.get_latest_price_date.return_value = "2024-01-03"
    mock_dal.get_stock_prices.return_value = pd.DataFrame(
        [{"open": 100.0, "close": 101.0}]
    )
    mock_dal.get_technical_indicators.return_value = None
    mock_dal.get_daily_sentiment.return_value = None
    mock_dal.get_predictions.return_value = []

    builder = ContextBuilder(mock_dal)
    first = builder.build_context("AAPL")
    second = builder.build_context("AAPL")

    assert first == second
    assert mock_dal.get_latest_price_date.call_count == 1
    assert mock_dal.get_stock_prices.call_count == 1
    assert mock_dal.get_technical_indicators.call_count == 1
    assert mock_dal.get_daily_sentiment.call_count == 1
    assert mock_dal.get_predictions.call_count == 1
