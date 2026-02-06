"""
Unit tests for the StockPredictor base model interface.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from src.models.base_model import StockPredictor


class DummyPredictor(StockPredictor):
    def __init__(self):
        self.trained = False
        self.saved_path = None
        self.loaded_path = None

    def train(self, X_train, y_train, X_test, y_test, **kwargs):
        self.trained = True
        return {
            "status": "ok",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def predict(self, X_data):
        return np.zeros(len(X_data))

    def save(self, path: str):
        self.saved_path = path

    def load(self, path: str):
        self.loaded_path = path
        return self

    def get_name(self) -> str:
        return "dummy"


@pytest.mark.unit
def test_stock_predictor_is_abstract():
    with pytest.raises(TypeError):
        StockPredictor()


@pytest.mark.unit
def test_incomplete_subclass_is_abstract():
    class IncompletePredictor(StockPredictor):
        pass

    with pytest.raises(TypeError):
        IncompletePredictor()


@pytest.mark.unit
def test_dummy_predictor_interface_behavior():
    model = DummyPredictor()
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 0.0, 1.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([0.0, 1.0])

    result = model.train(X_train, y_train, X_test, y_test)
    assert model.trained is True
    assert result == {"status": "ok", "train_samples": 3, "test_samples": 2}

    predictions = model.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.tolist() == [0.0, 0.0]

    model.save("/tmp/model.bin")
    assert model.saved_path == "/tmp/model.bin"

    model.load("/tmp/loaded.bin")
    assert model.loaded_path == "/tmp/loaded.bin"

    assert model.get_name() == "dummy"
