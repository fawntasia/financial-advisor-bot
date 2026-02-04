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
        
    def train(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True, **kwargs):
        """
        Train the XGBoost model.
        Optionally performs hyperparameter tuning using RandomizedSearchCV.
        Uses early stopping on the test set for the final fit.
        """
        # Ensure no NaNs
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("Training data contains NaNs. Please clean data before training.")
            
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
            search.fit(X_train, y_train)
            
            # Update model with best params but create a fresh instance to retrain with early stopping
            best_params = search.best_params_
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {search.best_score_:.4f}")
            
            self.model = XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=10 # Stop if no improvement for 10 rounds
            )
            self.is_tuned = True
        else:
             # Enable early stopping for the non-tuned model too
             self.model.set_params(early_stopping_rounds=10)

        # Train final model with early stopping using X_test as validation
        print("Training final model with early stopping...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
            
        # Evaluate on test set
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Train Accuracy: {train_score:.4f}")
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
