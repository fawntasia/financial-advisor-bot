import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.models.base_model import ModelNotFittedError, StockPredictor
from src.models.classification_utils import (
    compute_classification_metrics,
    tune_decision_threshold,
)
from src.models.io_utils import ensure_parent_dir


class RandomForestModel(StockPredictor):
    """
    Random Forest model for stock direction prediction (classification).
    Predicts UP (1) or DOWN/FLAT (0).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        class_weight: Optional[str] = None,
    ):
        self.model_name = "RandomForest_v2"
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.is_tuned = False
        self._is_fitted = False
        self.decision_threshold = 0.5

    @staticmethod
    def _validate_no_nans(X, y, split_name: str):
        if X is None or y is None:
            return
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError(f"{split_name} data contains NaNs. Please clean data before training.")

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("RandomForestModel must be trained or loaded before inference.")

    def _predict_labels(self, X_data, threshold=None):
        self._ensure_fitted()
        if threshold is None:
            threshold = self.decision_threshold
        probabilities = self.model.predict_proba(X_data)[:, 1]
        return (probabilities >= threshold).astype(int), probabilities

    def _evaluate_split(self, X_data, y_data):
        labels, probabilities = self._predict_labels(X_data)
        return compute_classification_metrics(y_data, labels, probabilities)

    def train(
        self,
        X_train,
        y_train,
        *,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        tune_hyperparameters: bool = True,
        val_prices=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        Optionally performs hyperparameter tuning using RandomizedSearchCV.
        """
        self._validate_no_nans(X_train, y_train, "Training")
        self._validate_no_nans(X_val, y_val, "Validation")
        self._validate_no_nans(X_test, y_test, "Test")

        best_params = None
        best_cv_score = None
        threshold_objective = None

        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 8, 12, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced", "balanced_subsample"],
            }

            max_possible_splits = len(X_train) - 1
            n_splits = max(2, min(5, len(X_train) // 50))
            n_splits = min(n_splits, max_possible_splits)
            if n_splits < 2:
                raise ValueError("Not enough training rows for TimeSeriesSplit hyperparameter tuning.")
            tscv = TimeSeriesSplit(n_splits=n_splits)

            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_dist,
                n_iter=20,
                cv=tscv,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
                scoring="balanced_accuracy",
            )

            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            self.is_tuned = True
            best_params = dict(search.best_params_)
            best_cv_score = float(search.best_score_)
            print(f"Best parameters: {best_params}")
            print(f"Best CV balanced accuracy: {best_cv_score:.4f}")
        else:
            self.model.fit(X_train, y_train)

        self._is_fitted = True

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_prob = self.model.predict_proba(X_val)[:, 1]
            best_threshold, threshold_objective = tune_decision_threshold(
                y_true=y_val,
                y_prob=val_prob,
                prices=val_prices,
            )
            self.decision_threshold = best_threshold
            print(f"Selected threshold: {self.decision_threshold:.3f} ({threshold_objective})")

        train_metrics = self._evaluate_split(X_train, y_train)
        print(
            f"Train metrics: acc={train_metrics['accuracy']:.4f}, "
            f"bal_acc={train_metrics['balanced_accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}, roc_auc={train_metrics['roc_auc']:.4f}"
        )

        validation_metrics = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            validation_metrics = self._evaluate_split(X_val, y_val)
            print(
                f"Validation metrics: acc={validation_metrics['accuracy']:.4f}, "
                f"bal_acc={validation_metrics['balanced_accuracy']:.4f}, "
                f"f1={validation_metrics['f1']:.4f}, roc_auc={validation_metrics['roc_auc']:.4f}"
            )

        test_metrics = None
        if X_test is not None and y_test is not None and len(X_test) > 0:
            test_metrics = self._evaluate_split(X_test, y_test)
            print(
                f"Test metrics: acc={test_metrics['accuracy']:.4f}, "
                f"bal_acc={test_metrics['balanced_accuracy']:.4f}, "
                f"f1={test_metrics['f1']:.4f}, roc_auc={test_metrics['roc_auc']:.4f}"
            )

        if hasattr(self.model, "feature_importances_"):
            print("Feature Importances:", self.model.feature_importances_)

        metadata = {
            "model_name": self.model_name,
            "is_tuned": self.is_tuned,
            "best_params": best_params,
            "best_cv_balanced_accuracy": best_cv_score,
            "threshold_objective": threshold_objective,
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)) if X_val is not None else 0,
            "test_rows": int(len(X_test)) if X_test is not None else 0,
        }
        return {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "decision_threshold": float(self.decision_threshold),
            "metadata": metadata,
        }

    def predict(self, X_data):
        """Make thresholded class predictions."""
        labels, _ = self._predict_labels(X_data)
        return labels

    def predict_proba(self, X_data):
        """Make probability predictions."""
        self._ensure_fitted()
        return self.model.predict_proba(X_data)

    def save(self, path: str):
        """Save model and threshold using joblib."""
        self._ensure_fitted()
        ensure_parent_dir(path)
        payload = {
            "model": self.model,
            "decision_threshold": self.decision_threshold,
            "model_name": self.model_name,
        }
        joblib.dump(payload, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and threshold using joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.decision_threshold = float(loaded.get("decision_threshold", 0.5))
            self.model_name = loaded.get("model_name", self.model_name)
        else:
            self.model = loaded
            self.decision_threshold = 0.5
        self._is_fitted = True
        print(f"Model loaded from {path}")

    def get_name(self) -> str:
        return self.model_name
