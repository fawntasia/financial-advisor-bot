import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib
import os
from src.models.base_model import StockPredictor

class XGBoostModel(StockPredictor):
    """
    XGBoost model for stock direction prediction (Classification).
    Predicts UP (1) or DOWN (0).
    """
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42, use_gpu=False):
        self.model_name = "XGBoost_v1"
        
        # Check for GPU availability if requested
        tree_method = 'auto'
        if use_gpu:
            try:
                # Simple check if CUDA is available via xgboost
                # Note: XGBoost usually handles 'gpu_hist' or 'cuda' gracefully if available
                tree_method = 'gpu_hist' 
            except:
                print("GPU not available or error configuring. Falling back to auto.")
                tree_method = 'auto'

        self.model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric='logloss',
            tree_method=tree_method,
            early_stopping_rounds=None  # Set during fit
        )
        self.is_tuned = False
        self.feature_names = None

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
        
    def train(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True, **kwargs):
        """
        Train the XGBoost model.
        Optionally performs hyperparameter tuning using RandomizedSearchCV.
        Uses a dedicated validation set for early stopping.
        Pass X_val/y_val to override the default internal validation split.
        """
        # Reset early stopping rounds to avoid issues with search
        self.model.set_params(early_stopping_rounds=None)

        X_val = kwargs.pop('X_val', None)
        y_val = kwargs.pop('y_val', None)
        validation_fraction = kwargs.pop('validation_fraction', 0.1)

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both omitted.")
        
        # Ensure no NaNs
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("Training data contains NaNs. Please clean data before training.")

        if X_val is None:
            X_fit, y_fit, X_val, y_val = self._split_train_validation(
                X_train,
                y_train,
                validation_fraction=validation_fraction
            )
            print(f"Using internal validation split: train={len(X_fit)}, val={len(X_val)}")
        else:
            X_fit, y_fit = X_train, y_train
            print(f"Using external validation set: train={len(X_fit)}, val={len(X_val)}")
            
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_dist,
                n_iter=15,
                cv=tscv,
                n_jobs=-1,
                verbose=1,
                random_state=42,
                scoring='accuracy'
            )
            
            # Note: We don't use early stopping inside CV to keep it simple, 
            # as it requires passing eval sets for each fold.
            search.fit(X_fit, y_fit)
            
            # Update model with best params but create a fresh instance to retrain with early stopping
            best_params = search.best_params_
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {search.best_score_:.4f}")
            
            self.model = XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='logloss'
            )
            self.is_tuned = True
        else:
             # Ensure early stopping is NOT set here
             self.model.set_params(early_stopping_rounds=None)

        # Train final model with early stopping using a dedicated validation set.
        print("Training final model with early stopping...")
        self.model.set_params(early_stopping_rounds=10)
        self.model.fit(
            X_fit, y_fit,
            eval_set=[(X_fit, y_fit), (X_val, y_val)],
            verbose=False
        )
            
        # Evaluate on train/validation/test sets
        train_score = self.model.score(X_fit, y_fit)
        val_score = self.model.score(X_val, y_val)
        test_score = self.model.score(X_test, y_test)
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Validation Accuracy: {val_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            print("Feature Importances:", importances)
            
        return {'train_acc': train_score, 'test_acc': test_score}

    def predict(self, X_data):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X_data)
        
    def predict_proba(self, X_data):
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict_proba(X_data)

    def save(self, path: str):
        """Save model to JSON (XGBoost native format)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Check if path ends with .json, if not replace/append
        if not path.endswith('.json'):
             path = os.path.splitext(path)[0] + '.json'
        
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from JSON."""
        if not os.path.exists(path):
            # Try appending .json if not present
            if os.path.exists(path + '.json'):
                path = path + '.json'
            else:
                raise FileNotFoundError(f"Model file not found at {path}")
        
        # Re-initialize model to load into
        self.model = XGBClassifier()
        self.model.load_model(path)
        print(f"Model loaded from {path}")

    def get_name(self) -> str:
        return self.model_name
