import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.models.base_model import StockPredictor
from src.models.classification_utils import (
    compute_classification_metrics,
    tune_decision_threshold,
)


class RandomForestModel(StockPredictor):
    """
    Random Forest model for stock direction prediction (classification).
    Predicts UP (1) or DOWN/FLAT (0).
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight=None,
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
        self.decision_threshold = 0.5

    @staticmethod
    def _validate_no_nans(X, y, split_name: str):
        if X is None or y is None:
            return
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError(f"{split_name} data contains NaNs. Please clean data before training.")

    def _predict_labels(self, X_data, threshold=None):
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
        tune_hyperparameters=True,
        val_prices=None,
        **kwargs,
    ):
        """
        Train the Random Forest model.
        Optionally performs hyperparameter tuning using RandomizedSearchCV.
        """
        self._validate_no_nans(X_train, y_train, "Training")
        self._validate_no_nans(X_val, y_val, "Validation")
        self._validate_no_nans(X_test, y_test, "Test")

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

            # Use conservative split count for smaller datasets.
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
            print(f"Best parameters: {search.best_params_}")
            print(f"Best CV balanced accuracy: {search.best_score_:.4f}")
        else:
            self.model.fit(X_train, y_train)

        metrics = {"decision_threshold": self.decision_threshold}

        # Tune threshold on validation only.
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_prob = self.model.predict_proba(X_val)[:, 1]
            best_threshold, objective = tune_decision_threshold(
                y_true=y_val,
                y_prob=val_prob,
                prices=val_prices,
            )
            self.decision_threshold = best_threshold
            metrics["threshold_objective"] = objective
            metrics["decision_threshold"] = self.decision_threshold
            print(f"Selected threshold: {self.decision_threshold:.3f} ({objective})")

        train_metrics = self._evaluate_split(X_train, y_train)
        metrics["train"] = train_metrics
        print(
            f"Train metrics: acc={train_metrics['accuracy']:.4f}, "
            f"bal_acc={train_metrics['balanced_accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}, roc_auc={train_metrics['roc_auc']:.4f}"
        )

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_metrics = self._evaluate_split(X_val, y_val)
            metrics["validation"] = val_metrics
            print(
                f"Validation metrics: acc={val_metrics['accuracy']:.4f}, "
                f"bal_acc={val_metrics['balanced_accuracy']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, roc_auc={val_metrics['roc_auc']:.4f}"
            )

        if X_test is not None and y_test is not None and len(X_test) > 0:
            test_metrics = self._evaluate_split(X_test, y_test)
            metrics["test"] = test_metrics
            print(
                f"Test metrics: acc={test_metrics['accuracy']:.4f}, "
                f"bal_acc={test_metrics['balanced_accuracy']:.4f}, "
                f"f1={test_metrics['f1']:.4f}, roc_auc={test_metrics['roc_auc']:.4f}"
            )

        if hasattr(self.model, "feature_importances_"):
            print("Feature Importances:", self.model.feature_importances_)

        return metrics

    def predict(self, X_data):
        """Make thresholded class predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        labels, _ = self._predict_labels(X_data)
        return labels

    def predict_proba(self, X_data):
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict_proba(X_data)

    def save(self, path: str):
        """Save model and threshold using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
            # Backward compatibility with older payloads containing classifier only.
            self.model = loaded
            self.decision_threshold = 0.5
        print(f"Model loaded from {path}")

    def get_name(self) -> str:
        return self.model_name
