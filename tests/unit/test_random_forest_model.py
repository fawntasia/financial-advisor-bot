"""
Unit tests for RandomForestModel.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from src.models import random_forest_model


class DummyClassifier:
    def __init__(self, *args, **kwargs):
        self.fit_calls = []
        self.score_calls = []
        self.predict_calls = []
        self.predict_proba_calls = []
        self.feature_importances_ = np.array([0.7, 0.3])

    def fit(self, X, y):
        self.fit_calls.append((X, y))
        return self

    def score(self, X, y):
        self.score_calls.append((X, y))
        if len(y) == 3:
            return 0.9
        return 0.8

    def predict(self, X):
        self.predict_calls.append(X)
        return np.zeros(len(X))

    def predict_proba(self, X):
        self.predict_proba_calls.append(X)
        return np.tile([0.4, 0.6], (len(X), 1))


@pytest.mark.unit
def test_train_without_tuning_uses_model_fit_and_scores(monkeypatch):
    monkeypatch.setattr(random_forest_model, "RandomForestClassifier", DummyClassifier)
    model = random_forest_model.RandomForestModel(n_estimators=10)

    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[4.0, 5.0], [5.0, 6.0]])
    y_test = np.array([1, 0])

    result = model.train(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparameters=False,
    )

    assert model.is_tuned is False
    assert len(model.model.fit_calls) == 1
    assert "train" in result
    assert "test" in result
    assert result["train"]["accuracy"] == pytest.approx(1 / 3)
    assert result["test"]["accuracy"] == pytest.approx(0.5)


@pytest.mark.unit
def test_train_with_tuning_uses_search_best_estimator(monkeypatch):
    monkeypatch.setattr(random_forest_model, "RandomForestClassifier", DummyClassifier)
    created = {}

    class DummySearch:
        def __init__(
            self,
            estimator,
            param_distributions,
            n_iter,
            cv,
            n_jobs,
            verbose,
            random_state,
            scoring,
        ):
            created["instance"] = self
            self.estimator = estimator
            self.param_distributions = param_distributions
            self.n_iter = n_iter
            self.cv = cv
            self.n_jobs = n_jobs
            self.verbose = verbose
            self.random_state = random_state
            self.scoring = scoring
            self.fit_called = False
            self.best_estimator_ = DummyClassifier()
            self.best_params_ = {"n_estimators": 50}
            self.best_score_ = 0.77

        def fit(self, X, y):
            self.fit_called = True
            self.fit_args = (X, y)
            return self

    monkeypatch.setattr(random_forest_model, "RandomizedSearchCV", DummySearch)
    model = random_forest_model.RandomForestModel(n_estimators=10)

    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[4.0, 5.0], [5.0, 6.0]])
    y_test = np.array([1, 0])

    model.train(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparameters=True,
    )

    search = created["instance"]
    assert search.fit_called is True
    assert search.fit_args == (X_train, y_train)
    assert model.is_tuned is True
    assert model.model is search.best_estimator_


@pytest.mark.unit
def test_predict_and_predict_proba_use_underlying_model(monkeypatch):
    monkeypatch.setattr(random_forest_model, "RandomForestClassifier", DummyClassifier)
    model = random_forest_model.RandomForestModel()

    X_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)

    assert predictions.tolist() == [1, 1]
    assert probabilities.shape == (2, 2)
