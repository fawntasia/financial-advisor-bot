
import sys
from typing import Any, cast

import pytest  # type: ignore
import pandas as pd

sys.path.insert(0, ".")

from src.llm.guardrails import Guardrails

class FakeDal:
    def __init__(self, tickers=None, latest_date="2023-01-01", close_price=100.0):
        self._tickers = tickers or []
        self._latest_date = latest_date
        self._close_price = close_price

    def get_all_tickers(self):
        return self._tickers

    def get_latest_price_date(self, ticker):
        return self._latest_date

    def get_stock_prices(self, ticker, start_date, end_date):
        return pd.DataFrame([{"close": self._close_price}])


@pytest.fixture
def guardrails():
    return Guardrails(cast(Any, FakeDal(tickers=["AAPL"])))


@pytest.fixture
def fake_guardrails():
    return Guardrails(cast(Any, FakeDal(tickers=["AAPL"])))


def test_pre_filter_safe(guardrails):
    result = guardrails.pre_filter("What is the price of AAPL?")
    assert result["is_safe"] is True


def test_pre_filter_unsafe(guardrails):
    result = guardrails.pre_filter("How to buy bitcoin?")
    assert result["is_safe"] is False
    assert result["reason"] == "disallowed_topic"


def test_post_filter_safe(guardrails):
    output = "AAPL is a solid investment. educational purposes only, not a financial advisor, consult with a qualified professional"
    result = guardrails.post_filter(output)
    assert result["is_safe"] is True
    assert result["has_all_disclaimers"] is True


def test_post_filter_unsafe_claim(guardrails):
    output = "AAPL offers guaranteed returns and no risk."
    result = guardrails.post_filter(output)
    assert result["is_safe"] is False
    assert result["reason"] == "unsafe_claim"


def test_post_filter_missing_disclaimers(guardrails):
    output = "AAPL is a solid investment."
    result = guardrails.post_filter(output)
    assert result["is_safe"] is True
    assert result["has_all_disclaimers"] is False
    assert len(result["missing_disclaimers"]) > 0


def test_fact_check_tickers(guardrails):
    result = guardrails.fact_check("Analysis of AAPL and XYZPD")
    assert "AAPL" in result["mentioned_tickers"]
    assert "XYZPD" in result["invalid_tickers"]
    assert result["is_factually_grounded"] is False


def test_validate_flow(guardrails):
    user_query = "Tell me about AAPL"
    llm_output = "AAPL is doing well."
    result = guardrails.validate(user_query, llm_output)

    assert result["safe_to_display"] is True
    assert "Disclaimer" in result["output"]
    assert result["report_issue"] == guardrails.report_issue_url


def test_validate_blocks_on_pre_filter(guardrails):
    result = guardrails.validate("How to buy bitcoin?", "Any output")
    assert result["safe_to_display"] is False
    assert result["reason"] == "disallowed_topic"
    assert result["report_issue"] == guardrails.report_issue_url


def test_validate_blocks_on_post_filter(guardrails):
    result = guardrails.validate("Tell me about AAPL", "Guaranteed returns, no risk.")
    assert result["safe_to_display"] is False
    assert result["reason"] == "unsafe_claim"
    assert result["report_issue"] == guardrails.report_issue_url


def test_validate_does_not_append_disclaimer_when_present(guardrails):
    llm_output = "AAPL looks steady. Not a financial advisor."
    result = guardrails.validate("Tell me about AAPL", llm_output)
    assert result["safe_to_display"] is True
    assert result["output"] == llm_output
    assert "Disclaimer" not in result["output"]


def test_fact_check_price_verification(fake_guardrails):
    result = fake_guardrails.fact_check("AAPL is trading at $110 today")
    assert result["mentioned_tickers"] == ["AAPL"]
    assert result["invalid_tickers"] == []
    assert result["price_verifications"]
    assert result["price_verifications"][0]["verified"] is True
    assert result["is_factually_grounded"] is True


def test_fact_check_no_price_mentions(fake_guardrails):
    result = fake_guardrails.fact_check("AAPL is trending upward")
    assert result["mentioned_tickers"] == ["AAPL"]
    assert result["price_verifications"] == []


def test_fact_check_ignores_price_without_valid_ticker():
    guardrails = Guardrails(cast(Any, FakeDal(tickers=["AAPL"])))
    result = guardrails.fact_check("XYZ is at $100")
    assert result["mentioned_tickers"] == []
    assert result["invalid_tickers"] == ["XYZ"]
    assert result["price_verifications"] == []
    assert result["is_factually_grounded"] is False
