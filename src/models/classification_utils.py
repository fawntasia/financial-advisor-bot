"""Utilities for classification metrics and decision threshold tuning."""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.models.evaluation import calculate_metrics, calculate_strategy_returns
from src.models.trading_config import predictions_to_signals


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> dict:
    """Compute standard classification metrics."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob).reshape(-1)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _threshold_objective_from_prices(
    y_pred: np.ndarray,
    prices: np.ndarray,
) -> float:
    signals = predictions_to_signals(y_pred)
    returns = calculate_strategy_returns(signals, pd.Series(prices))
    stats = calculate_metrics(returns)
    return float(stats.get("sharpe_ratio", 0.0))


def tune_decision_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    prices: Optional[Sequence[float]] = None,
    thresholds: Optional[Iterable[float]] = None,
) -> Tuple[float, str]:
    """
    Tune threshold on validation data.

    Uses Sharpe ratio when prices are supplied; otherwise uses balanced accuracy.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob).reshape(-1)
    if y_true_arr.size == 0:
        return 0.5, "default_no_validation"

    if thresholds is None:
        thresholds = np.linspace(0.3, 0.7, 17)

    prices_arr = None
    if prices is not None:
        prices_arr = np.asarray(prices).reshape(-1)
        if prices_arr.size != y_true_arr.size:
            prices_arr = None

    best_threshold = 0.5
    best_score = float("-inf")
    objective_name = "sharpe" if prices_arr is not None else "balanced_accuracy"

    for threshold in thresholds:
        preds = (y_prob_arr >= threshold).astype(int)
        if prices_arr is not None:
            score = _threshold_objective_from_prices(preds, prices_arr)
        else:
            score = balanced_accuracy_score(y_true_arr, preds)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
        elif score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5):
            best_threshold = float(threshold)

    return best_threshold, objective_name
