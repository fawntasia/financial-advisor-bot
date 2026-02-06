import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.init_db import create_tables
from src.data.stock_data import StockDataProcessor
from src.database.dal import DataAccessLayer


class MockDirectionalModel:
    def __init__(self):
        self.threshold = None
        self.model_name = "MockDirectional_v1"

    def train(self, X_train, y_train, X_test, y_test):
        self.threshold = float(np.mean(X_train[:, 0]))
        return {"train_mean": float(np.mean(y_train))}

    def predict(self, X_data):
        if self.threshold is None:
            raise ValueError("Model has not been trained.")
        return (X_data[:, 0] >= self.threshold).astype(int)

    def get_name(self):
        return self.model_name


def build_synthetic_prices(rows=80):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame({
        "Date": dates,
        "Open": close + 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.full(rows, 1000, dtype=int)
    })


@pytest.mark.integration
def test_model_train_predict_store_flow(tmp_path):
    db_path = tmp_path / "test_model_flow.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    conn.close()

    dal = DataAccessLayer(db_path=db_path)
    dal.insert_ticker("TEST", "Test Corp")

    prices = build_synthetic_prices()
    processor = StockDataProcessor("TEST")
    X_train, y_train, X_test, y_test, _ = processor.prepare_for_classification(prices)

    model = MockDirectionalModel()
    model.train(X_train, y_train, X_test, y_test)
    predictions = model.predict(X_test)

    assert predictions.shape[0] == y_test.shape[0]
    assert set(np.unique(predictions)).issubset({0, 1})

    enriched = processor.add_technical_indicators(prices)
    enriched["Target"] = (enriched["Close"].shift(-1) > enriched["Close"]).astype(int)
    enriched = enriched.iloc[:-1]
    split_idx = int(len(enriched) * 0.8)
    prediction_date = enriched["Date"].iloc[split_idx + len(predictions) - 1].strftime("%Y-%m-%d")

    dal.insert_prediction(
        "TEST",
        prediction_date,
        model.get_name(),
        int(predictions[-1]),
        confidence=0.9
    )

    stored = dal.get_predictions("TEST", model_name=model.get_name())
    assert len(stored) == 1
    assert stored[0]["model_name"] == model.get_name()
    assert stored[0]["predicted_direction"] == int(predictions[-1])
