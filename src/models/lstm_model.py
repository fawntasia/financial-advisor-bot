import os
from typing import Any, Dict

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from src.models.base_model import ModelNotFittedError, StockPredictor
from src.models.io_utils import ensure_parent_dir


class LSTMModel(StockPredictor):
    """
    LSTM-based model for stock price prediction using TensorFlow/Keras.
    Architecture: LSTM(128) -> LSTM(64) -> Dense(1)
    """

    def __init__(self, sequence_length: int = 60, n_features: int = 6, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.model = None
        self.model_name = "LSTM_Keras_v1"
        self._is_fitted = False
        self._build_model()

    def _build_model(self):
        """Build the LSTM model architecture."""
        self.model = Sequential(
            [
                Input(shape=(self.sequence_length, self.n_features)),
                LSTM(128, return_sequences=True),
                LSTM(64, return_sequences=False),
                Dense(1),
            ]
        )

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("LSTMModel must be trained or loaded before inference.")

    def train(
        self,
        X_train,
        y_train,
        *,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        epochs: int = 50,
        batch_size: int = 32,
        save_path=None,
        patience: int = 5,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model with Early Stopping and Checkpointing.

        Returns:
            Dict with train/validation/test metrics and metadata.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both omitted.")
        if (X_test is None) != (y_test is None):
            raise ValueError("X_test and y_test must both be provided or both omitted.")
        if X_val is None and X_test is None:
            raise ValueError("Provide either validation data or test data for early stopping.")

        validation_X = X_val if X_val is not None else X_test
        validation_y = y_val if y_val is not None else y_test

        callbacks = []

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=verbose,
        )
        callbacks.append(early_stopping)

        if save_path:
            ensure_parent_dir(save_path)
            checkpoint = ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=verbose,
            )
            callbacks.append(checkpoint)

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(validation_X, validation_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
        self._is_fitted = True

        train_loss = float(history.history.get("loss", [float("nan")])[-1])
        val_loss = float(history.history.get("val_loss", [float("nan")])[-1])
        train_metrics = {"loss": train_loss}
        validation_metrics = {"loss": val_loss}

        test_metrics = None
        if X_test is not None and y_test is not None and len(X_test) > 0:
            test_pred = self.predict(X_test).reshape(-1)
            y_test_arr = np.asarray(y_test).reshape(-1)
            test_mse = float(np.mean((test_pred - y_test_arr) ** 2))
            test_metrics = {"loss": test_mse}

        metadata = {
            "model_name": self.model_name,
            "epochs_requested": int(epochs),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "patience": int(patience),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(validation_X)),
            "test_rows": int(len(X_test)) if X_test is not None else 0,
        }
        return {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "metadata": metadata,
        }

    def predict(self, X_data):
        """Make predictions on new data."""
        self._ensure_fitted()
        return self.model.predict(X_data, verbose=0)

    def save(self, path: str):
        """Save the full model to disk."""
        self._ensure_fitted()
        ensure_parent_dir(path)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        self.model = load_model(path)
        self._is_fitted = True
        print(f"Model loaded from {path}")

        input_shape = self.model.input_shape
        if input_shape:
            self.sequence_length = input_shape[1]
            self.n_features = input_shape[2]

    def get_name(self) -> str:
        return self.model_name
