"""
Unit tests for XGBoostModel.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from src.models import xgboost_model


class DummyXGBClassifier:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.fit_calls = []
        self.score_calls = []
        self.predict_calls = []
        self.predict_proba_calls = []
        self.set_params_calls = []
        self._feature_importances_accessed = False
        self._feature_importances = np.array([0.6, 0.4])

    @property
    def feature_importances_(self):
        self._feature_importances_accessed = True
        return self._feature_importances

    def set_params(self, **kwargs):
        self.set_params_calls.append(kwargs)
        return self

    def fit(self, X, y, **kwargs):
        self.fit_calls.append((X, y, kwargs))
        return self

    def score(self, X, y):
        self.score_calls.append((X, y))
        if len(y) == 3:
            return 0.9
        return 0.8

    def predict(self, X):
        self.predict_calls.append(X)
        return np.ones(len(X))

    def predict_proba(self, X):
        self.predict_proba_calls.append(X)
        return np.tile([0.2, 0.8], (len(X), 1))


@pytest.mark.unit
def test_train_without_tuning_runs_fit_scores_and_importances(monkeypatch):
    monkeypatch.setattr(xgboost_model, "XGBClassifier", DummyXGBClassifier)
    model = xgboost_model.XGBoostModel(n_estimators=10)

    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([0, 1, 0])
    X_val = np.array([[3.5, 4.5]])
    y_val = np.array([1])
    X_test = np.array([[4.0, 5.0], [5.0, 6.0]])
    y_test = np.array([1, 0])

    result = model.train(
        X_train,
        y_train,
        X_test,
        y_test,
        tune_hyperparameters=False,
        X_val=X_val,
        y_val=y_val,
    )

    assert model.is_tuned is False
    assert model.model.set_params_calls[0] == {"early_stopping_rounds": None}
    assert model.model.set_params_calls[-1] == {"early_stopping_rounds": 10}
    assert len(model.model.fit_calls) == 1
    assert model.model.fit_calls[0][2]["eval_set"] == [(X_train, y_train), (X_val, y_val)]
    assert model.model.fit_calls[0][2]["verbose"] is False
    assert model.model._feature_importances_accessed is True
    assert result == {"train_acc": 0.9, "test_acc": 0.8}


@pytest.mark.unit
def test_predict_and_predict_proba_use_underlying_model(monkeypatch):
    monkeypatch.setattr(xgboost_model, "XGBClassifier", DummyXGBClassifier)
    model = xgboost_model.XGBoostModel()

    X_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)

    assert predictions.tolist() == [1.0, 1.0]
    assert probabilities.shape == (2, 2)


@pytest.mark.unit
def test_train_without_external_validation_uses_internal_split(monkeypatch):
    monkeypatch.setattr(xgboost_model, "XGBClassifier", DummyXGBClassifier)
    model = xgboost_model.XGBoostModel()

    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[5.0, 6.0], [6.0, 7.0]])
    y_test = np.array([1, 0])

    model.train(X_train, y_train, X_test, y_test, tune_hyperparameters=False, validation_fraction=0.25)

    eval_set = model.model.fit_calls[0][2]["eval_set"]
    fit_X, fit_y = eval_set[0]
    val_X, val_y = eval_set[1]

    assert np.array_equal(fit_X, X_train[:3])
    assert np.array_equal(fit_y, y_train[:3])
    assert np.array_equal(val_X, X_train[3:])
    assert np.array_equal(val_y, y_train[3:])
