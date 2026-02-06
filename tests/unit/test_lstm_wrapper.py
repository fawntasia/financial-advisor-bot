"""
Unit tests for LSTMStockPredictor wrapper.
"""
import sys
import math

sys.path.insert(0, ".")

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models import lstm_wrapper


class DummyTensorDataset:
    def __init__(self, X_tensor, y_tensor):
        self.X_tensor = X_tensor
        self.y_tensor = y_tensor

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]


class DummyDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for start in range(0, len(self.dataset), self.batch_size):
            end = start + self.batch_size
            yield (
                self.dataset.X_tensor[start:end],
                self.dataset.y_tensor[start:end],
            )

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class DummyTorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, 1)
        self.train_calls = 0
        self.eval_calls = 0

    def forward(self, x):
        last_step = x[:, -1, :]
        return self.linear(last_step)

    def train(self, mode=True):
        self.train_calls += 1
        return super().train(mode)

    def eval(self):
        self.eval_calls += 1
        return super().eval()


class DummyAdam:
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


def configure_wrapper_mocks(monkeypatch):
    monkeypatch.setattr(lstm_wrapper, "PyTorchLSTM", DummyTorchModel)
    monkeypatch.setattr(lstm_wrapper, "TensorDataset", DummyTensorDataset)
    monkeypatch.setattr(lstm_wrapper, "DataLoader", DummyDataLoader)
    monkeypatch.setattr(lstm_wrapper.torch.optim, "Adam", DummyAdam)


@pytest.mark.unit
def test_train_updates_input_size_and_records_history(monkeypatch):
    configure_wrapper_mocks(monkeypatch)
    model = lstm_wrapper.LSTMStockPredictor(input_size=2, hidden_size=4, num_layers=1)

    X_train = np.zeros((4, 3, 3), dtype=np.float32)
    y_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    X_test = np.zeros((2, 3, 3), dtype=np.float32)
    y_test = np.array([1.0, 2.0], dtype=np.float32)

    history = model.train(X_train, y_train, X_test, y_test, epochs=1, batch_size=2)

    assert model.input_size == 3
    assert isinstance(model.model, DummyTorchModel)
    assert history["loss"] and history["val_loss"]
    assert len(history["loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert model.model.train_calls >= 1
    assert model.model.eval_calls >= 1


@pytest.mark.unit
def test_predict_returns_numpy_array(monkeypatch):
    configure_wrapper_mocks(monkeypatch)
    model = lstm_wrapper.LSTMStockPredictor(input_size=2, hidden_size=4, num_layers=1)

    with torch.no_grad():
        model.model.linear.weight.fill_(0.0)
        model.model.linear.bias.fill_(0.0)

    X_data = np.ones((2, 3, 2), dtype=np.float32)
    predictions = model.predict(X_data)

    assert predictions.shape == (2, 1)
    assert np.allclose(predictions, 0.0)
