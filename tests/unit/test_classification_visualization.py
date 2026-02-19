import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from src.ui import classification_visualization as cv


class DummyModel:
    def __init__(self, *, threshold: float = 0.5, constant_prob: float | None = None, model_name: str = "DummyModel"):
        self.decision_threshold = threshold
        self.constant_prob = constant_prob
        self.model_name = model_name

    def predict_proba(self, X):
        n = len(X)
        if self.constant_prob is None:
            prob = np.clip(np.asarray(X)[:, 0] / 200.0, 0.0, 1.0)
        else:
            prob = np.full(n, self.constant_prob, dtype=float)
        return np.column_stack([1.0 - prob, prob])


class FakeDal:
    def __init__(self, price_df: pd.DataFrame):
        self.price_df = price_df
        self.insert_calls = []
        self.prediction_store = {}

    def get_latest_price_date(self, ticker: str):
        del ticker
        return str(self.price_df["date"].max())

    def get_stock_prices(self, ticker: str, start_date: str, end_date: str):
        del ticker
        del start_date
        del end_date
        return self.price_df.copy()

    def get_prediction_by_key(self, ticker: str, model_name: str, date: str):
        return self.prediction_store.get((ticker, model_name, date))

    def insert_prediction(self, **kwargs):
        self.insert_calls.append(kwargs)
        key = (kwargs["ticker"], kwargs["model_name"], kwargs["date"])
        self.prediction_store[key] = kwargs


def _make_raw_prices(dates, closes):
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(dates),
            "date": [str(d) for d in dates],
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
            "adj_close": closes,
        }
    )


def _patch_feature_builder(monkeypatch):
    def fake_add_indicators(self, df):
        del self
        out = df.copy()
        out["SMA_20"] = out["Close"].rolling(window=2, min_periods=1).mean()
        out["SMA_50"] = out["Close"].rolling(window=2, min_periods=1).mean()
        out["RSI"] = 50.0
        out["MACD"] = 0.1
        out["Signal_Line"] = 0.05
        return out

    monkeypatch.setattr(cv.StockDataProcessor, "add_technical_indicators", fake_add_indicators)


def test_resolve_classifier_artifacts_success(tmp_path):
    rf_model = tmp_path / "random_forest_global.pkl"
    rf_meta = tmp_path / "random_forest_global_metadata.json"
    xgb_model = tmp_path / "xgboost_global.json"
    xgb_meta = tmp_path / "xgboost_global_metadata.json"
    rf_model.write_text("rf", encoding="utf-8")
    rf_meta.write_text("{}", encoding="utf-8")
    xgb_model.write_text("xgb", encoding="utf-8")
    xgb_meta.write_text("{}", encoding="utf-8")

    rf_paths = cv.resolve_classifier_artifacts("rf", models_dir=tmp_path)
    xgb_paths = cv.resolve_classifier_artifacts("xgb", models_dir=tmp_path)

    assert rf_paths.model_type == "rf"
    assert rf_paths.model_path == rf_model.resolve()
    assert rf_paths.metadata_path == rf_meta.resolve()
    assert xgb_paths.model_type == "xgb"
    assert xgb_paths.model_path == xgb_model.resolve()
    assert xgb_paths.metadata_path == xgb_meta.resolve()


def test_resolve_classifier_artifacts_raises_for_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        cv.resolve_classifier_artifacts("rf", models_dir=tmp_path)


def test_generate_classification_signal_data_threshold_and_payload(monkeypatch, tmp_path):
    _patch_feature_builder(monkeypatch)
    model_path = tmp_path / "random_forest_global.pkl"
    metadata_path = tmp_path / "random_forest_global_metadata.json"
    model_path.write_text("rf", encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        cv,
        "resolve_classifier_artifacts",
        lambda model_type, models_dir=Path("models"): cv.ClassifierArtifactPaths(
            model_type="rf",
            model_path=model_path.resolve(),
            metadata_path=metadata_path.resolve(),
        ),
    )
    monkeypatch.setattr(
        cv,
        "_load_cached_classifier_artifacts",
        lambda **kwargs: {
            "model": DummyModel(threshold=0.5, constant_prob=0.49, model_name="RandomForest_v2"),
            "metadata": {
                "feature_columns": cv.DEFAULT_FEATURE_COLUMNS,
                "metrics": {"test": {"balanced_accuracy": 0.61, "f1": 0.58}},
                "per_ticker_test_metrics": {"by_ticker": {"AAPL": {"balanced_accuracy": 0.67}}},
            },
        },
    )

    prices = _make_raw_prices(
        dates=["2024-01-03", "2024-01-04", "2024-01-05"],
        closes=[100.0, 101.0, 102.0],
    )
    dal = FakeDal(prices)
    data = cv.generate_classification_signal_data(
        ticker="AAPL",
        model_type="rf",
        dal=dal,
        eval_window=2,
        persist_prediction=False,
    )

    assert data["model_type"] == "rf"
    assert data["model_name"] == "RandomForest_v2"
    assert data["predicted_direction"] == 0
    assert data["predicted_label"] == "DOWN"
    assert data["confidence"] == pytest.approx(0.51)
    assert data["prob_up"] == pytest.approx(0.49)
    assert data["prediction_date"] == "2024-01-08"
    assert set(data.keys()) >= {
        "global_test_metrics",
        "ticker_test_metrics",
        "recent_eval_metrics",
        "feature_columns",
        "artifact_paths",
    }
    assert data["global_test_metrics"]["balanced_accuracy"] == pytest.approx(0.61)
    assert data["ticker_test_metrics"]["balanced_accuracy"] == pytest.approx(0.67)


def test_generate_classification_signal_data_builds_recent_eval_metrics(monkeypatch, tmp_path):
    _patch_feature_builder(monkeypatch)
    model_path = tmp_path / "xgboost_global.json"
    model_path.write_text("xgb", encoding="utf-8")

    monkeypatch.setattr(
        cv,
        "resolve_classifier_artifacts",
        lambda model_type, models_dir=Path("models"): cv.ClassifierArtifactPaths(
            model_type="xgb",
            model_path=model_path.resolve(),
            metadata_path=None,
        ),
    )
    monkeypatch.setattr(
        cv,
        "_load_cached_classifier_artifacts",
        lambda **kwargs: {
            "model": DummyModel(threshold=0.5, constant_prob=0.9, model_name="XGBoost_v2"),
            "metadata": {"feature_columns": cv.DEFAULT_FEATURE_COLUMNS},
        },
    )

    prices = _make_raw_prices(
        dates=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    )
    dal = FakeDal(prices)
    data = cv.generate_classification_signal_data(
        ticker="AAPL",
        model_type="xgb",
        dal=dal,
        eval_window=4,
        persist_prediction=False,
    )

    assert data["recent_eval_metrics"]["accuracy"] == pytest.approx(1.0)
    assert data["recent_eval_metrics"]["f1"] == pytest.approx(1.0)


def test_generate_classification_signal_data_persistence_is_idempotent(monkeypatch, tmp_path):
    _patch_feature_builder(monkeypatch)
    model_path = tmp_path / "random_forest_global.pkl"
    model_path.write_text("rf", encoding="utf-8")

    monkeypatch.setattr(
        cv,
        "resolve_classifier_artifacts",
        lambda model_type, models_dir=Path("models"): cv.ClassifierArtifactPaths(
            model_type="rf",
            model_path=model_path.resolve(),
            metadata_path=None,
        ),
    )
    monkeypatch.setattr(
        cv,
        "_load_cached_classifier_artifacts",
        lambda **kwargs: {
            "model": DummyModel(threshold=0.5, constant_prob=0.7, model_name="RandomForest_v2"),
            "metadata": {"feature_columns": cv.DEFAULT_FEATURE_COLUMNS},
        },
    )

    prices = _make_raw_prices(
        dates=["2024-01-03", "2024-01-04", "2024-01-05"],
        closes=[100.0, 101.0, 102.0],
    )
    dal = FakeDal(prices)

    cv.generate_classification_signal_data(ticker="AAPL", model_type="rf", dal=dal, persist_prediction=True)
    cv.generate_classification_signal_data(ticker="AAPL", model_type="rf", dal=dal, persist_prediction=True)

    assert len(dal.insert_calls) == 1
    inserted = dal.insert_calls[0]
    assert inserted["ticker"] == "AAPL"
    assert inserted["model_name"] == "RandomForest_v2"
    assert inserted["predicted_direction"] == 1
