import sys
sys.path.insert(0, ".")

import pytest
import numpy as np
import pandas as pd

from src.data import stock_data
from src.data.stock_data import StockDataProcessor


def _make_close_only_df(rows=120, start=100.0, step=0.5):
    dates = pd.date_range(start="2023-01-01", periods=rows)
    close = np.linspace(start, start + step * (rows - 1), rows)
    df = pd.DataFrame({"Close": close}, index=dates)
    df.index.name = "Date"
    return df


@pytest.mark.unit
def test_add_technical_indicators_adds_columns_and_fills():
    df = _make_close_only_df(rows=120)
    processor = StockDataProcessor("TEST")

    result = processor.add_technical_indicators(df)

    expected_cols = ["SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]
    for col in expected_cols:
        assert col in result.columns
        assert not result[col].isna().any()


@pytest.mark.unit
def test_prepare_for_lstm_shapes_and_target_scaler():
    df = _make_close_only_df(rows=100)
    processor = StockDataProcessor("TEST")

    X_train, y_train, X_test, y_test, target_scaler, data = processor.prepare_for_lstm(
        df, sequence_length=10, target_col="Close"
    )

    assert data.shape[0] == 100
    assert X_train.shape == (70, 10, 6)
    assert y_train.shape == (70,)
    assert X_test.shape == (10, 10, 6)
    assert y_test.shape == (10,)
    assert target_scaler is processor.target_scaler


@pytest.mark.unit
def test_get_latest_sequence_uses_indicators_and_scaler():
    df = _make_close_only_df(rows=120)
    processor = StockDataProcessor("TEST")
    processor.prepare_for_lstm(df, sequence_length=15, target_col="Close")

    latest = processor.get_latest_sequence(df, sequence_length=15, target_col="Close")

    assert latest.shape == (1, 15, 6)
    assert not np.isnan(latest).any()


@pytest.mark.unit
def test_prepare_for_classification_targets_and_shapes():
    df = _make_close_only_df(rows=51)
    processor = StockDataProcessor("TEST")

    X_train, y_train, X_test, y_test, feature_cols = processor.prepare_for_classification(
        df, target_col="Close"
    )

    total_rows = 50
    split_idx = int(total_rows * 0.8)
    assert X_train.shape == (split_idx, 6)
    assert y_train.shape == (split_idx,)
    assert X_test.shape == (total_rows - split_idx, 6)
    assert y_test.shape == (total_rows - split_idx,)
    assert set(np.unique(np.concatenate([y_train, y_test]))) <= {0, 1}
    assert feature_cols == ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


@pytest.mark.unit
def test_fetch_data_raises_when_empty(monkeypatch):
    class _EmptyTicker:
        def history(self, start=None, end=None):
            return pd.DataFrame()

    def _fake_ticker(_):
        return _EmptyTicker()

    monkeypatch.setattr(stock_data.yf, "Ticker", _fake_ticker)
    processor = StockDataProcessor("TEST")

    with pytest.raises(ValueError, match="No data found for ticker TEST"):
        processor.fetch_data(years=1)
