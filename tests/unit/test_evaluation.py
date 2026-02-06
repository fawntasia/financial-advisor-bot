import sys

import pandas as pd
import pytest

sys.path.insert(0, ".")

from src.models.evaluation import (
    calculate_daily_returns,
    calculate_max_drawdown,
    calculate_metrics,
)


def test_calculate_daily_returns():
    prices = pd.Series([100.0, 105.0, 103.0], index=pd.date_range("2024-01-01", periods=3))
    returns = calculate_daily_returns(prices)

    expected = pd.Series([0.0, 0.05, -0.01904761904761905], index=prices.index)
    assert returns.tolist() == pytest.approx(expected.tolist())


def test_calculate_metrics_returns_and_sharpe():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015], index=pd.date_range("2024-01-01", periods=4))
    metrics = calculate_metrics(returns, risk_free_rate=0.0, periods_per_year=4)

    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (4 / len(returns)) - 1
    annualized_volatility = returns.std() * (4 ** 0.5)
    sharpe_ratio = annualized_return / annualized_volatility

    assert metrics["total_return"] == pytest.approx(cumulative_return)
    assert metrics["annualized_return"] == pytest.approx(annualized_return)
    assert metrics["annualized_volatility"] == pytest.approx(annualized_volatility)
    assert metrics["sharpe_ratio"] == pytest.approx(sharpe_ratio)


def test_calculate_max_drawdown():
    returns = pd.Series([0.1, -0.05, -0.1, 0.2], index=pd.date_range("2024-01-01", periods=4))
    max_drawdown = calculate_max_drawdown(returns)

    assert max_drawdown == pytest.approx(-0.145)
