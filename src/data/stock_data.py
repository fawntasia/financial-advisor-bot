"""Stock data fetching and preprocessing utilities."""

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class StockDataProcessor:
    """Fetch, enrich, and format stock data for model inputs."""
    
    def __init__(self, ticker: str = "AAPL"):
        """Initialize the processor with a ticker symbol."""
        self.ticker = ticker
        self.data: Optional[pd.DataFrame] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1)) # Scaler just for the target value
        
    def fetch_data(self, years: int = 5) -> pd.DataFrame:
        """Fetch historical stock data for the configured ticker."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        self.data = df
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA, RSI, and MACD features to a copy of the data."""
        df = df.copy()
        
        # Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Keep preprocessing forward-safe: fill only from past values, then drop warm-up rows.
        indicator_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.dropna(subset=indicator_cols, inplace=True)
        
        return df

    @staticmethod
    def _resolve_columns(df: pd.DataFrame, requested_cols: Sequence[str]) -> List[str]:
        """Resolve requested column names case-insensitively against a DataFrame."""
        col_lookup = {col.lower(): col for col in df.columns}
        resolved = []
        missing = []
        for col in requested_cols:
            match = col_lookup.get(col.lower())
            if match is None:
                missing.append(col)
            else:
                resolved.append(match)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return resolved

    @staticmethod
    def _build_direction_split(
        split_df: pd.DataFrame,
        price_col: str,
        feature_cols: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Build classification features/targets for one split only.
        The last row is dropped because its target would reference the next split.
        """
        if len(split_df) < 2:
            return (
                np.empty((0, len(feature_cols))),
                np.empty((0,), dtype=np.int64),
                split_df.iloc[0:0].copy(),
            )

        labeled = split_df.copy()
        labeled["Target"] = (labeled[price_col].shift(-1) > labeled[price_col]).astype(int)
        labeled = labeled.iloc[:-1].copy()

        X = labeled[list(feature_cols)].values
        y = labeled["Target"].values
        return X, y, labeled

    def prepare_for_lstm(self, df: pd.DataFrame, sequence_length: int = 60, target_col: str = 'Close'):
        """Prepare sequences for LSTM training and return scaled splits."""
        # Enrichment step
        data = self.add_technical_indicators(df)
        
        # Define features to use
        feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        
        # Save feature columns for later use
        self.feature_cols = feature_cols
        
        # SPLIT DATA FIRST (80/20) to prevent leakage
        split_idx = int(len(data) * 0.8)
        train_df = data.iloc[:split_idx]
        test_df = data.iloc[split_idx:]
        
        # FIT scalers ONLY on training data
        self.scaler.fit(train_df[feature_cols].values)
        target_values = train_df[target_col].values.reshape(-1, 1)
        self.target_scaler.fit(target_values)
        
        # Transform both sets using the Training scaler
        train_scaled = self.scaler.transform(train_df[feature_cols].values)
        test_scaled = self.scaler.transform(test_df[feature_cols].values)
        
        # Helper to create sequences
        def create_sequences(dataset):
            """Create sliding window sequences for features and target."""
            X, y = [], []
            target_idx = feature_cols.index(target_col)
            for i in range(sequence_length, len(dataset)):
                X.append(dataset[i - sequence_length:i, :]) # All features
                y.append(dataset[i, target_idx]) # Only target
            return np.array(X), np.array(y)
            
        # Create sequences for both sets
        X_train, y_train = create_sequences(train_scaled)
        
        # For Test set, we need to concatenate the last 'sequence_length' from train 
        # to ensure we don't lose the first few test points
        # But for strict separation, standard practice is just processing test_scaled
        X_test, y_test = create_sequences(test_scaled)
        
        # Reshape is automatic since X is [samples, seq_len, features]
        
        return X_train, y_train, X_test, y_test, self.target_scaler, data

    def get_latest_sequence(self, df: pd.DataFrame, sequence_length: int = 60, target_col: str = 'Close'):
        """Return the most recent scaled feature window for inference."""
        # Ensure indicators exist
        if 'RSI' not in df.columns:
            df = self.add_technical_indicators(df)
            
        feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        dataset = df[feature_cols].values[-sequence_length:]
        
        scaled = self.scaler.transform(dataset)
        return np.reshape(scaled, (1, sequence_length, len(feature_cols)))

    def prepare_for_classification_splits(
        self,
        df: pd.DataFrame,
        target_col: str = "Close",
        feature_cols: Optional[Sequence[str]] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        return_metadata: bool = False,
    ):
        """
        Prepare leakage-safe train/val/test splits for next-day direction classification.

        Split logic:
        1. Build chronological feature splits first.
        2. Create targets within each split.
        3. Drop each split's last row so labels never depend on the next split.
        """
        if not 0 < train_split < 1:
            raise ValueError("train_split must be between 0 and 1.")
        if not 0 <= val_split < 1:
            raise ValueError("val_split must be between 0 and 1.")

        data = df.copy()
        if "RSI" not in data.columns and "rsi" not in data.columns:
            data = self.add_technical_indicators(data)

        default_features = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]
        resolved_features = self._resolve_columns(data, feature_cols or default_features)
        resolved_target = self._resolve_columns(data, [target_col])[0]
        self.feature_cols = resolved_features

        if len(data) < 6:
            raise ValueError("Not enough rows to create train/validation/test splits.")

        split_idx = int(len(data) * train_split)
        split_idx = max(2, min(split_idx, len(data) - 2))
        train_region = data.iloc[:split_idx].copy()
        test_region = data.iloc[split_idx:].copy()

        if val_split > 0:
            val_size = max(2, int(len(train_region) * val_split))
            val_size = min(val_size, len(train_region) - 2)
            fit_region = train_region.iloc[:-val_size].copy()
            val_region = train_region.iloc[-val_size:].copy()
        else:
            fit_region = train_region.copy()
            val_region = train_region.iloc[0:0].copy()

        X_train, y_train, train_labeled = self._build_direction_split(
            fit_region,
            price_col=resolved_target,
            feature_cols=resolved_features,
        )
        X_val, y_val, val_labeled = self._build_direction_split(
            val_region,
            price_col=resolved_target,
            feature_cols=resolved_features,
        )
        X_test, y_test, test_labeled = self._build_direction_split(
            test_region,
            price_col=resolved_target,
            feature_cols=resolved_features,
        )

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough rows to create leakage-safe classification splits.")

        if not return_metadata:
            return X_train, y_train, X_val, y_val, X_test, y_test, resolved_features

        metadata: Dict[str, object] = {
            "train_index": train_labeled.index,
            "val_index": val_labeled.index,
            "test_index": test_labeled.index,
            "train_prices": train_labeled[resolved_target].values,
            "val_prices": val_labeled[resolved_target].values,
            "test_prices": test_labeled[resolved_target].values,
            "target_col": resolved_target,
        }
        return X_train, y_train, X_val, y_val, X_test, y_test, resolved_features, metadata

    def prepare_for_classification(self, df: pd.DataFrame, target_col: str = 'Close'):
        """
        Backward-compatible train/test preparation for direction classification.
        Uses leakage-safe split logic and omits an explicit validation split.
        """
        X_train, y_train, _, _, X_test, y_test, feature_cols = self.prepare_for_classification_splits(
            df=df,
            target_col=target_col,
            train_split=0.8,
            val_split=0.0,
            return_metadata=False,
        )
        return X_train, y_train, X_test, y_test, feature_cols

# Helper functions for backward compatibility or simple usage
def fetch_stock_data(ticker="AAPL", years=5):
    """Fetch historical data for a ticker using the processor."""
    processor = StockDataProcessor(ticker)
    return processor.fetch_data(years)

def prepare_data_for_lstm(df, sequence_length=60):
    """Prepare LSTM-ready sequences with default processor settings."""
    processor = StockDataProcessor()
    return processor.prepare_for_lstm(df, sequence_length)

def get_latest_sequence(df, scaler, sequence_length=60):
    """Return the latest scaled window using the provided scaler."""
    # Note: This helper requires the passed scaler, ignoring the processor's internal one
    # to maintain compatibility with existing app logic that passes scaler around
    processor = StockDataProcessor()
    processor.scaler = scaler 
    return processor.get_latest_sequence(df, sequence_length)
