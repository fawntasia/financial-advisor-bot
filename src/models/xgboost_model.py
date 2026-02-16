import json
import os

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.models.base_model import StockPredictor
from src.models.classification_utils import (
    compute_classification_metrics,
    tune_decision_threshold,
)


class XGBoostModel(StockPredictor):
    """
    XGBoost model for stock direction prediction (classification).
    Predicts UP (1) or DOWN/FLAT (0).
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        random_state=42,
        use_gpu=False,
    ):
        self.model_name = "XGBoost_v2"
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.tree_method = "gpu_hist" if use_gpu else "hist"
        self.model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric="logloss",
            tree_method=self.tree_method,
            early_stopping_rounds=None,
        )
        self.is_tuned = False
        self.decision_threshold = 0.5

    @staticmethod
    def _validate_no_nans(X, y, split_name: str):
        if X is None or y is None:
            return
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError(f"{split_name} data contains NaNs. Please clean data before training.")

    @staticmethod
    def _split_train_validation(X_train, y_train, validation_fraction: float):
        """Create a chronological train/validation split from training data."""
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")

        n_samples = len(X_train)
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to create a validation split.")

        split_idx = int(n_samples * (1 - validation_fraction))
        split_idx = max(1, min(split_idx, n_samples - 1))

        X_fit = X_train[:split_idx]
        y_fit = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        return X_fit, y_fit, X_val, y_val

    def _new_estimator(self, **params):
        return XGBClassifier(
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method=self.tree_method,
            **params,
        )

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
        validation_fraction=0.1,
        **kwargs,
    ):
        """
        Train the XGBoost model.
        Supports hyperparameter tuning and validation-driven threshold tuning.
        """
        self.model.set_params(early_stopping_rounds=None)
        self._validate_no_nans(X_train, y_train, "Training")
        self._validate_no_nans(X_val, y_val, "Validation")
        self._validate_no_nans(X_test, y_test, "Test")

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both omitted.")

        if X_val is None:
            X_fit, y_fit, X_val, y_val = self._split_train_validation(
                X_train,
                y_train,
                validation_fraction=validation_fraction,
            )
            print(f"Using internal validation split: train={len(X_fit)}, val={len(X_val)}")
        else:
            X_fit, y_fit = X_train, y_train
            print(f"Using external validation split: train={len(X_fit)}, val={len(X_val)}")

        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_dist = {
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0.0, 0.1, 0.3, 0.5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "reg_lambda": [1.0, 2.0, 5.0, 10.0],
                "scale_pos_weight": [1.0, 1.5, 2.0],
            }

            max_possible_splits = len(X_fit) - 1
            n_splits = max(2, min(5, len(X_fit) // 50))
            n_splits = min(n_splits, max_possible_splits)
            if n_splits < 2:
                raise ValueError("Not enough training rows for TimeSeriesSplit hyperparameter tuning.")
            tscv = TimeSeriesSplit(n_splits=n_splits)
            search = RandomizedSearchCV(
                estimator=self._new_estimator(),
                param_distributions=param_dist,
                n_iter=25,
                cv=tscv,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
                scoring="balanced_accuracy",
            )

            search.fit(X_fit, y_fit)
            best_params = search.best_params_
            print(f"Best parameters: {best_params}")
            print(f"Best CV balanced accuracy: {search.best_score_:.4f}")
            self.model = self._new_estimator(**best_params)
            self.is_tuned = True
        else:
            self.model.set_params(early_stopping_rounds=None)

        print("Training final model with early stopping...")
        self.model.set_params(early_stopping_rounds=10)
        self.model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_fit, y_fit), (X_val, y_val)],
            verbose=False,
        )

        metrics = {"decision_threshold": self.decision_threshold}
        if len(X_val) > 0:
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

        train_metrics = self._evaluate_split(X_fit, y_fit)
        val_metrics = self._evaluate_split(X_val, y_val)
        metrics["train"] = train_metrics
        metrics["validation"] = val_metrics
        print(
            f"Train metrics: acc={train_metrics['accuracy']:.4f}, "
            f"bal_acc={train_metrics['balanced_accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}, roc_auc={train_metrics['roc_auc']:.4f}"
        )
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
        """Save model to JSON plus sidecar metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith(".json"):
            path = os.path.splitext(path)[0] + ".json"
        self.model.save_model(path)

        meta_path = os.path.splitext(path)[0] + ".meta.json"
        payload = {
            "model_name": self.model_name,
            "decision_threshold": self.decision_threshold,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from JSON with optional sidecar metadata."""
        if not os.path.exists(path):
            if os.path.exists(path + ".json"):
                path = path + ".json"
            else:
                raise FileNotFoundError(f"Model file not found at {path}")

        self.model = self._new_estimator()
        self.model.load_model(path)

        meta_path = os.path.splitext(path)[0] + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.model_name = payload.get("model_name", self.model_name)
            self.decision_threshold = float(payload.get("decision_threshold", 0.5))
        else:
            self.decision_threshold = 0.5
        print(f"Model loaded from {path}")

    def get_name(self) -> str:
        return self.model_name
