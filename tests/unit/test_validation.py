import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

import src.models.validation as validation
from src.models.validation import WalkForwardValidator


class DummyModel:
    def __init__(self, predictions_queue):
        self.predictions_queue = list(predictions_queue)
        self.trained = False

    def train(self, X_train, y_train, X_val, y_val):
        self.trained = True

    def predict(self, X_data):
        return self.predictions_queue.pop(0)


def make_monthly_data(start="2020-01-01", periods=17):
    dates = pd.date_range(start, periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "feature": np.arange(len(dates), dtype=float),
            "target": np.arange(len(dates)) % 2,
            "close": np.linspace(100.0, 100.0 + len(dates) - 1, len(dates)),
        }
    )


@pytest.mark.unit
def test_get_splits_generates_sequential_folds():
    data = make_monthly_data(periods=17)
    validator = WalkForwardValidator(train_years=1, val_months=1, test_months=1, step_months=1)

    splits = validator._get_splits(data)

    assert len(splits) == 3

    first = splits[0]
    assert len(first["train"]) == 12
    assert len(first["val"]) == 1
    assert len(first["test"]) == 1
    assert first["metadata"]["train_start"] == pd.Timestamp("2020-01-01")
    assert first["metadata"]["train_end"] == pd.Timestamp("2021-01-01")
    assert first["metadata"]["val_end"] == pd.Timestamp("2021-02-01")
    assert first["metadata"]["test_end"] == pd.Timestamp("2021-03-01")

    second = splits[1]
    assert second["metadata"]["train_start"] == pd.Timestamp("2020-02-01")
    assert second["metadata"]["train_end"] == pd.Timestamp("2021-02-01")
    assert second["metadata"]["val_end"] == pd.Timestamp("2021-03-01")
    assert second["metadata"]["test_end"] == pd.Timestamp("2021-04-01")

    third = splits[2]
    assert third["metadata"]["train_start"] == pd.Timestamp("2020-03-01")
    assert third["metadata"]["train_end"] == pd.Timestamp("2021-03-01")
    assert third["metadata"]["val_end"] == pd.Timestamp("2021-04-01")
    assert third["metadata"]["test_end"] == pd.Timestamp("2021-05-01")


@pytest.mark.unit
def test_validate_aggregates_metrics_and_signals(monkeypatch):
    data = make_monthly_data(periods=17)
    validator = WalkForwardValidator(train_years=1, val_months=1, test_months=2, step_months=12)
    splits = validator._get_splits(data)
    split = splits[0]

    model = DummyModel(
        [
            split["train"]["target"].values,
            split["val"]["target"].values,
            split["test"]["target"].values,
        ]
    )

    captured = {}

    def fake_strategy_returns(signals, prices):
        captured["signals"] = signals
        captured["prices"] = prices
        return pd.Series([0.01] * len(prices), index=prices.index)

    def fake_metrics(returns):
        captured["returns"] = returns
        return {"total_return": 0.123, "sharpe_ratio": 0.0}

    monkeypatch.setattr(validation, "calculate_strategy_returns", fake_strategy_returns)
    monkeypatch.setattr(validation, "calculate_metrics", fake_metrics)

    results = validator.validate(model, data, feature_cols=["feature"], target_col="target")

    assert model.trained is True
    assert len(results) == 1

    metrics = results[0]["metrics"]
    assert metrics["train_accuracy"] == pytest.approx(1.0)
    assert metrics["val_accuracy"] == pytest.approx(1.0)
    assert metrics["test_accuracy"] == pytest.approx(1.0)
    assert metrics["overfitting_ratio"] == pytest.approx(1.0)
    assert metrics["total_return"] == pytest.approx(0.123)
    assert metrics["sharpe_ratio"] == pytest.approx(0.0)

    expected_signals = (
        pd.Series(split["test"]["target"].values, index=split["test"].index)
        .map({1: 1, 0: -1})
        .tolist()
    )
    assert captured["signals"].tolist() == expected_signals
    assert captured["prices"].tolist() == split["test"]["close"].tolist()
