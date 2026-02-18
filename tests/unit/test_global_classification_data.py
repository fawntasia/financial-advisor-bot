import numpy as np
import pandas as pd

from src.models import global_classification_data as gcd


def _market_df(rows: int, phase: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    trend = np.linspace(100.0, 180.0, rows)
    seasonal = 1.5 * np.sin((np.arange(rows) / 12.0) + phase)
    close = trend + seasonal
    df = pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 0.7,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.full(rows, 200_000, dtype=float),
            "Adj Close": close,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_build_global_classification_dataset_is_split_safe(monkeypatch):
    universe = ["AAA", "BBB"]

    def fake_load_ticker_universe(db_path: str = "unused"):
        return universe

    data_by_ticker = {
        "AAA": _market_df(220, phase=0.0),
        "BBB": _market_df(220, phase=0.8),
    }

    def fake_load_market_data(ticker: str, **kwargs):
        return data_by_ticker[ticker].copy()

    monkeypatch.setattr(gcd, "load_ticker_universe", fake_load_ticker_universe)
    monkeypatch.setattr(gcd, "load_market_data", fake_load_market_data)

    dataset = gcd.build_global_classification_dataset(train_split=0.8, val_split=0.1)
    meta = dataset["metadata"]
    split_rows = meta["split_rows"]
    split_cov = meta["split_ticker_coverage"]

    assert dataset["X_train"].shape[0] > 0
    assert dataset["X_test"].shape[0] > 0
    assert split_rows["train_labeled"] == split_rows["train_raw"] - split_cov["train_raw"]
    assert split_rows["val_labeled"] == split_rows["val_raw"] - split_cov["val_raw"]
    assert split_rows["test_labeled"] == split_rows["test_raw"] - split_cov["test_raw"]
    assert set(dataset["test_tickers"].tolist()) == {"AAA", "BBB"}


def test_build_global_classification_dataset_skips_sparse_tickers(monkeypatch):
    universe = ["AAA", "SPARSE"]

    def fake_load_ticker_universe(db_path: str = "unused"):
        return universe

    data_by_ticker = {
        "AAA": _market_df(220, phase=0.2),
        "SPARSE": _market_df(25, phase=0.1),
    }

    def fake_load_market_data(ticker: str, **kwargs):
        return data_by_ticker[ticker].copy()

    monkeypatch.setattr(gcd, "load_ticker_universe", fake_load_ticker_universe)
    monkeypatch.setattr(gcd, "load_market_data", fake_load_market_data)

    dataset = gcd.build_global_classification_dataset(train_split=0.8, val_split=0.1)
    meta = dataset["metadata"]
    skipped = {item["ticker"]: item["reason"] for item in meta["skipped_tickers"]}

    assert meta["requested_ticker_count"] == 2
    assert meta["used_ticker_count"] == 1
    assert skipped["SPARSE"] == "insufficient_indicator_rows"
    assert set(dataset["train_tickers"].tolist()) == {"AAA"}
    assert set(dataset["test_tickers"].tolist()) == {"AAA"}
