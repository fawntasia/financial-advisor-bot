"""
Unit tests for LSTMStockPredictor compatibility wrapper.
"""

import importlib
import sys
import types

import pytest

sys.path.insert(0, ".")


class DummySequential:
    def __init__(self, layers):
        self.layers = list(layers)

    def compile(self, optimizer=None, loss=None):
        self.optimizer = optimizer
        self.loss = loss


class DummyAdam:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate


def dummy_load_model(path):
    return DummySequential([])


def DummyInput(shape=None):
    return {"type": "Input", "shape": shape}


def DummyLSTM(units, return_sequences=False):
    return {"type": "LSTM", "units": units, "return_sequences": return_sequences}


def DummyDense(units):
    return {"type": "Dense", "units": units}


def import_lstm_wrapper(monkeypatch):
    tf_module = types.ModuleType("tensorflow")
    keras_module = types.ModuleType("tensorflow.keras")
    models_module = types.ModuleType("tensorflow.keras.models")
    layers_module = types.ModuleType("tensorflow.keras.layers")
    callbacks_module = types.ModuleType("tensorflow.keras.callbacks")
    optimizers_module = types.ModuleType("tensorflow.keras.optimizers")

    models_module.Sequential = DummySequential
    models_module.load_model = dummy_load_model
    layers_module.LSTM = DummyLSTM
    layers_module.Dense = DummyDense
    layers_module.Input = DummyInput
    callbacks_module.EarlyStopping = object
    callbacks_module.ModelCheckpoint = object
    optimizers_module.Adam = DummyAdam

    keras_module.models = models_module
    keras_module.layers = layers_module
    keras_module.callbacks = callbacks_module
    keras_module.optimizers = optimizers_module
    tf_module.keras = keras_module

    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", models_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.layers", layers_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.callbacks", callbacks_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", optimizers_module)

    for module_name in ["src.models.lstm_model", "src.models.lstm_wrapper"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    return importlib.import_module("src.models.lstm_wrapper")


@pytest.mark.unit
def test_wrapper_maps_input_size_to_n_features(monkeypatch):
    lstm_wrapper = import_lstm_wrapper(monkeypatch)
    model = lstm_wrapper.LSTMStockPredictor(input_size=7, sequence_length=30, learning_rate=0.005)

    assert model.n_features == 7
    assert model.sequence_length == 30
    assert model.learning_rate == 0.005


@pytest.mark.unit
def test_wrapper_uses_single_keras_lstm_implementation(monkeypatch):
    lstm_wrapper = import_lstm_wrapper(monkeypatch)
    lstm_model = importlib.import_module("src.models.lstm_model")

    model = lstm_wrapper.LSTMStockPredictor()
    assert isinstance(model, lstm_model.LSTMModel)
