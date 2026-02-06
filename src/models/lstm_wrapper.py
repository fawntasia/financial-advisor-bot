"""
Compatibility layer for historical imports.

This project now uses a single LSTM implementation: `src.models.lstm_model.LSTMModel`
based on TensorFlow/Keras.
"""

from src.models.lstm_model import LSTMModel


class LSTMStockPredictor(LSTMModel):
    """
    Backward-compatible alias with the former wrapper class name.

    Args:
        input_size: Number of input features (mapped to `n_features`).
        sequence_length: Number of timesteps in each sequence.
        learning_rate: Optimizer learning rate.
    """

    def __init__(self, input_size: int = 6, sequence_length: int = 60, learning_rate: float = 0.001):
        super().__init__(sequence_length=sequence_length, n_features=input_size, learning_rate=learning_rate)
