"""
Unit tests for LSTMModel.
"""
import sys
import types
import importlib

sys.path.insert(0, ".")

import numpy as np
import pytest


class DummyHistory:
    def __init__(self, history=None):
        self.history = history or {"loss": [1.0]}


class DummySequential:
    def __init__(self, layers):
        self.layers = list(layers)
        self.compiled = {}
        self.fit_calls = []
        self.predict_calls = []

    def compile(self, optimizer=None, loss=None):
        self.compiled = {"optimizer": optimizer, "loss": loss}

    def fit(
        self,
        X_train,
        y_train,
        validation_data=None,
        epochs=None,
        batch_size=None,
        callbacks=None,
        verbose=None,
    ):
        self.fit_calls.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "validation_data": validation_data,
                "epochs": epochs,
                "batch_size": batch_size,
                "callbacks": callbacks,
                "verbose": verbose,
            }
        )
        return DummyHistory()

    def predict(self, X_data, verbose=0):
        self.predict_calls.append({"X_data": X_data, "verbose": verbose})
        return np.full((len(X_data), 1), 0.5)


class DummyEarlyStopping:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyModelCheckpoint:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyAdam:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate


def dummy_load_model(path):
    return DummySequential([])


def DummyInput(shape=None):
    return {"type": "Input", "shape": shape}


def DummyLSTM(units, return_sequences=False):
    return {
        "type": "LSTM",
        "units": units,
        "return_sequences": return_sequences,
    }


def DummyDense(units):
    return {"type": "Dense", "units": units}


def import_lstm_model(monkeypatch):
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
    callbacks_module.EarlyStopping = DummyEarlyStopping
    callbacks_module.ModelCheckpoint = DummyModelCheckpoint
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

    if "src.models.lstm_model" in sys.modules:
        del sys.modules["src.models.lstm_model"]

    return importlib.import_module("src.models.lstm_model")


@pytest.mark.unit
def test_build_model_compiles_and_sets_layers(monkeypatch):
    lstm_model = import_lstm_model(monkeypatch)
    model = lstm_model.LSTMModel(sequence_length=5, n_features=2, learning_rate=0.01)

    assert isinstance(model.model, DummySequential)
    assert model.model.compiled["loss"] == "mean_squared_error"
    assert isinstance(model.model.compiled["optimizer"], DummyAdam)
    assert model.model.compiled["optimizer"].learning_rate == 0.01
    assert model.model.layers[0] == {"type": "Input", "shape": (5, 2)}
    assert model.model.layers[1]["type"] == "LSTM"
    assert model.model.layers[2]["type"] == "LSTM"
    assert model.model.layers[3] == {"type": "Dense", "units": 1}


@pytest.mark.unit
def test_train_invokes_fit_with_callbacks_and_validation(monkeypatch, tmp_path):
    lstm_model = import_lstm_model(monkeypatch)
    model = lstm_model.LSTMModel(sequence_length=3, n_features=1)

    X_train = np.zeros((4, 3, 1))
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    X_test = np.zeros((2, 3, 1))
    y_test = np.array([1.0, 2.0])
    save_path = tmp_path / "models" / "best.h5"

    history = model.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=2,
        batch_size=8,
        save_path=str(save_path),
        patience=3,
        verbose=0,
    )

    assert isinstance(history, DummyHistory)
    assert len(model.model.fit_calls) == 1
    fit_call = model.model.fit_calls[0]
    assert fit_call["validation_data"] == (X_test, y_test)
    assert fit_call["epochs"] == 2
    assert fit_call["batch_size"] == 8
    assert len(fit_call["callbacks"]) == 2
    assert isinstance(fit_call["callbacks"][0], DummyEarlyStopping)
    assert isinstance(fit_call["callbacks"][1], DummyModelCheckpoint)
    assert fit_call["callbacks"][0].kwargs["patience"] == 3
    assert fit_call["callbacks"][0].kwargs["restore_best_weights"] is True
    assert fit_call["callbacks"][1].kwargs["filepath"] == str(save_path)


@pytest.mark.unit
def test_predict_calls_underlying_model(monkeypatch):
    lstm_model = import_lstm_model(monkeypatch)
    model = lstm_model.LSTMModel(sequence_length=2, n_features=1)

    X_data = np.zeros((3, 2, 1))
    predictions = model.predict(X_data)

    assert predictions.shape == (3, 1)
    assert np.allclose(predictions, 0.5)
    assert model.model.predict_calls[0]["verbose"] == 0
