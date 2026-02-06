import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.context_builder import ContextBuilder
from src.llm.guardrails import Guardrails
from src.llm.prompts import PromptManager


class StubDAL:
    def __init__(self):
        self._tickers = ["ACME"]

    def get_latest_price_date(self, ticker):
        return "2024-01-02"

    def get_stock_prices(self, ticker, start, end):
        return pd.DataFrame([
            {
                "open": 100.0,
                "close": 101.0,
                "high": 102.0,
                "low": 99.0,
            }
        ])

    def get_technical_indicators(self, ticker, date):
        return {
            "date": date,
            "rsi_14": 55.0,
            "macd_histogram": 0.5,
            "sma_20": 99.5,
            "sma_50": 98.0,
            "sma_200": 95.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
        }

    def get_daily_sentiment(self, ticker, date):
        return {
            "overall_sentiment": "positive",
            "confidence": 0.82,
            "news_count": 12,
            "date": date,
        }

    def get_predictions(self, ticker, limit=5, model_name=None):
        return [
            {
                "model_name": "MockDirectional_v1",
                "predicted_direction": 1,
                "confidence": 0.9,
                "date": "2024-01-03",
            }
        ]

    def get_all_tickers(self):
        return list(self._tickers)


@pytest.mark.integration
def test_llm_flow_context_prompt_guardrails_offline():
    dal = StubDAL()
    context_builder = ContextBuilder(dal)
    prompt_manager = PromptManager()
    guardrails = Guardrails(dal)

    context = context_builder.build_context("ACME")
    assert context["price"]
    assert context["indicators_summary"]
    assert context["sentiment_summary"]
    assert context["prediction_summary"]

    prompt = prompt_manager.get_prompt(
        "general",
        ticker="ACME",
        price=context["price"],
        indicators_summary=context["indicators_summary"],
        sentiment_summary=context["sentiment_summary"],
        prediction_summary=context["prediction_summary"],
    )
    assert "ACME" in prompt
    assert "Current Price" in prompt
    assert context["sentiment_summary"] in prompt

    llm_mock = SimpleNamespace(generate=MagicMock())
    llm_mock.generate.return_value = "ACME looks stable at $101.00 based on recent signals."

    response = llm_mock.generate(prompt)
    llm_mock.generate.assert_called_once_with(prompt)

    validated = guardrails.validate("Tell me about ACME", response)
    assert validated["safe_to_display"] is True
    assert "Disclaimer" in validated["output"]
    assert validated["fact_check"]["is_factually_grounded"] is True
    assert validated["fact_check"]["price_verifications"]
