import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import BuyAndHoldStrategy, RandomWalkStrategy, SMACrossoverStrategy


def _price_df(prices):
    dates = pd.date_range(start="2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.mark.unit
def test_buy_and_hold_generates_all_long_signals():
    df = _price_df([10, 11, 12, 13])
    signals = BuyAndHoldStrategy().generate_signals(df)

    assert (signals == 1).all()
    assert signals.index.equals(df.index)


@pytest.mark.unit
def test_random_walk_is_seeded_and_reproducible():
    df = _price_df([100, 101, 99, 102, 98, 103])
    strategy = RandomWalkStrategy(seed=123)
    signals = strategy.generate_signals(df)

    expected = np.random.default_rng(123).choice([-1, 1], size=len(df))

    assert signals.tolist() == expected.tolist()
    assert set(signals.unique()).issubset({-1, 1})
    assert signals.index.equals(df.index)


@pytest.mark.unit
def test_sma_crossover_generates_expected_signals():
    df = _price_df([10, 9, 8, 9, 10, 9, 8])
    strategy = SMACrossoverStrategy(fast_window=2, slow_window=3)
    signals = strategy.generate_signals(df)

    assert signals.tolist() == [0, 0, -1, -1, 1, 1, -1]
    assert signals.index.equals(df.index)
