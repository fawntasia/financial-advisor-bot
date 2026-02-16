import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from src.models.classification_utils import (
    compute_classification_metrics,
    tune_decision_threshold,
)


@pytest.mark.unit
def test_compute_classification_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.2, 0.9, 0.4, 0.1])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert set(metrics.keys()) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(1.0)


@pytest.mark.unit
def test_tune_decision_threshold_uses_balanced_accuracy_without_prices():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.6, 0.8, 0.4])

    threshold, objective = tune_decision_threshold(y_true=y_true, y_prob=y_prob)

    assert objective == "balanced_accuracy"
    assert threshold == pytest.approx(0.5)


@pytest.mark.unit
def test_tune_decision_threshold_uses_sharpe_with_prices():
    # With steadily rising prices, lower threshold keeps the strategy long and should be favored.
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.4, 0.4, 0.4, 0.4])
    prices = np.array([100.0, 101.0, 102.0, 103.0])

    threshold, objective = tune_decision_threshold(y_true=y_true, y_prob=y_prob, prices=prices)

    assert objective == "sharpe"
    assert threshold <= 0.4
