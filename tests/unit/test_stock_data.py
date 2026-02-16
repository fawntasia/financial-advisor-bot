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
def test_add_technical_indicators_adds_columns_and_drops_warmup():
    df = _make_close_only_df(rows=120)
    processor = StockDataProcessor("TEST")

    result = processor.add_technical_indicators(df)

    expected_cols = ["SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]
    for col in expected_cols:
        assert col in result.columns
        assert not result[col].isna().any()

    assert len(result) < len(df)
    assert result.index[0] > df.index[0]


@pytest.mark.unit
def test_prepare_for_lstm_shapes_and_target_scaler():
    df = _make_close_only_df(rows=100)
    processor = StockDataProcessor("TEST")

    X_train, y_train, X_test, y_test, target_scaler, data = processor.prepare_for_lstm(
        df, sequence_length=10, target_col="Close"
    )

    split_idx = int(len(data) * 0.8)
    expected_train_sequences = max(split_idx - 10, 0)
    expected_test_sequences = max((len(data) - split_idx) - 10, 0)

    assert data.shape[0] == 51
    assert X_train.shape == (expected_train_sequences, 10, 6)
    assert y_train.shape == (expected_train_sequences,)
    assert X_test.shape == (expected_test_sequences, 10, 6)
    assert y_test.shape == (expected_test_sequences,)
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
    df = _make_close_only_df(rows=220)
    processor = StockDataProcessor("TEST")

    X_train, y_train, X_test, y_test, feature_cols = processor.prepare_for_classification(
        df, target_col="Close"
    )

    enriched_rows = len(processor.add_technical_indicators(df))
    split_idx = int(enriched_rows * 0.8)
    expected_train_rows = max(split_idx - 1, 0)
    expected_test_rows = max((enriched_rows - split_idx) - 1, 0)

    assert X_train.shape == (expected_train_rows, 6)
    assert y_train.shape == (expected_train_rows,)
    assert X_test.shape == (expected_test_rows, 6)
    assert y_test.shape == (expected_test_rows,)
    assert set(np.unique(np.concatenate([y_train, y_test]))) <= {0, 1}
    assert feature_cols == ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


@pytest.mark.unit
def test_prepare_for_classification_splits_returns_leakage_safe_boundaries():
    df = _make_close_only_df(rows=220)
    processor = StockDataProcessor("TEST")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        meta,
    ) = processor.prepare_for_classification_splits(
        df,
        target_col="Close",
        train_split=0.8,
        val_split=0.1,
        return_metadata=True,
    )

    assert X_train.shape[1] == len(feature_cols)
    assert X_val.shape[1] == len(feature_cols)
    assert X_test.shape[1] == len(feature_cols)
    assert len(y_train) == len(meta["train_index"])
    assert len(y_val) == len(meta["val_index"])
    assert len(y_test) == len(meta["test_index"])

    # The training split should end before validation starts (no boundary leakage labels).
    assert meta["train_index"].max() < meta["val_index"].min()
    assert meta["val_index"].max() < meta["test_index"].min()


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
