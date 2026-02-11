import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
from src.models.base_model import StockPredictor

class LSTMModel(StockPredictor):
    """
    LSTM-based model for stock price prediction using TensorFlow/Keras.
    Architecture: LSTM(128) -> LSTM(64) -> Dense(1)
    """
    
    def __init__(self, sequence_length=60, n_features=6, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.model = None
        self.model_name = "LSTM_Keras_v1"
        self._build_model()
        
    def _build_model(self):
        """Build the LSTM model architecture."""
        self.model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(128, return_sequences=True),
            LSTM(64, return_sequences=False),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=32,
        save_path=None,
        patience=5,
        verbose=1,
        X_val=None,
        y_val=None,
    ):
        """
        Train the model with Early Stopping and Checkpointing.
        
        Args:
            X_train: Training features (samples, seq_len, features)
            y_train: Training targets
            X_test: Held-out test features
            y_test: Held-out test targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            save_path: Path to save the best model (optional)
            patience: Patience for early stopping
            verbose: Verbosity level
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.
            
        Returns:
            History object containing training history
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both omitted.")

        validation_X = X_val if X_val is not None else X_test
        validation_y = y_val if y_val is not None else y_test

        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        )
        callbacks.append(early_stopping)
        
        # Model Checkpoint
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                save_best_only=True, # Save only the best model
                save_weights_only=False, # Save full model
                verbose=verbose
            )
            callbacks.append(checkpoint)
            
        history = self.model.fit(
            X_train, y_train,
            validation_data=(validation_X, validation_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
        
    def predict(self, X_data):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X_data, verbose=0)
        
    def save(self, path: str):
        """Save the full model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        self.model = load_model(path)
        print(f"Model loaded from {path}")
        
        # Update attributes based on loaded model input shape
        input_shape = self.model.input_shape
        if input_shape:
            # input_shape is (batch, seq_len, features)
            self.sequence_length = input_shape[1]
            self.n_features = input_shape[2]
            
    def get_name(self) -> str:
        return self.model_name
