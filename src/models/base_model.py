from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class StockPredictor(ABC):
    """Abstract base class for all stock prediction models."""

    @abstractmethod
    def train(self, X_train, y_train, X_test, y_test, **kwargs):
        """
        Train the model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation/Test data
            **kwargs: Model-specific hyperparameters (epochs, etc.)
            
        Returns:
            Validation metrics or history
        """
        pass

    @abstractmethod
    def predict(self, X_data):
        """
        Make predictions on new data.
        
        Args:
            X_data: Input features
            
        Returns:
            Predictions (numpy array)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return distinct model name."""
        pass
