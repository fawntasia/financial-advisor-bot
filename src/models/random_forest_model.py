import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
from src.models.base_model import StockPredictor

class RandomForestModel(StockPredictor):
    """
    Random Forest model for stock direction prediction (Classification).
    Predicts UP (1) or DOWN (0).
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        self.model_name = "RandomForest_v1"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.is_tuned = False
        
    def train(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True, **kwargs):
        """
        Train the Random Forest model.
        Optionally performs hyperparameter tuning using RandomizedSearchCV.
        """
        # Ensure no NaNs
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("Training data contains NaNs. Please clean data before training.")
            
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Use a time-series friendly cv or just simple cv since we already split?
            # Standard CV shuffles, which is bad for time series if we just dump all data in.
            # However, we are provided with X_train/y_train which is already the "past" data.
            # Using CV within X_train (shuffled) assumes samples are independent. 
            # In financial data, they aren't fully independent, but for RF it's often accepted 
            # to just use standard CV on the training set if we are careful.
            # Better: TimeSeriesSplit.
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_dist,
                n_iter=10,
                cv=tscv,
                n_jobs=-1,
                verbose=1,
                random_state=42,
                scoring='accuracy'
            )
            
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            self.is_tuned = True
            print(f"Best parameters: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        else:
            self.model.fit(X_train, y_train)
            
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
        """Save model using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model using joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    def get_name(self) -> str:
        return self.model_name
