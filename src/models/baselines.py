import numpy as np
import pandas as pd
from typing import Optional
from src.features.indicators import calculate_indicators

class BaselineStrategy:
    """Base class for baseline strategies."""
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals (-1, 0, 1) for the given data.
        
        Args:
            df: DataFrame containing price data (at least 'close')
            
        Returns:
            Series of signals aligned with the input DataFrame index.
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

class BuyAndHoldStrategy(BaselineStrategy):
    """Buy and Hold strategy: Long position always."""
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)

class RandomWalkStrategy(BaselineStrategy):
    """
    Random Walk strategy: Randomly chooses between Long (1) and Short (-1).
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Use numpy's modern generator for reproducibility
        rng = np.random.default_rng(self.seed)
        signals = rng.choice([-1, 1], size=len(df))
        return pd.Series(signals, index=df.index)

class SMACrossoverStrategy(BaselineStrategy):
    """
    SMA Crossover strategy: 
    Long (1) when Fast SMA > Slow SMA
    Short (-1) when Fast SMA < Slow SMA
    """
    def __init__(self, fast_window: int = 20, slow_window: int = 50):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Create a copy to avoid modifying the original dataframe
        df_calc = df.copy()
        
        fast_col = f'sma_{self.fast_window}'
        slow_col = f'sma_{self.slow_window}'
        
        # Check if we can use the existing features module for standard windows
        if self.fast_window == 20 and self.slow_window == 50:
            try:
                # calculate_indicators adds sma_20 and sma_50
                df_calc = calculate_indicators(df_calc)
            except Exception as e:
                # Fallback if ta library is missing or fails
                print(f"Warning: calculate_indicators failed ({e}), using manual calculation.")
                df_calc[fast_col] = df_calc['close'].rolling(window=self.fast_window).mean()
                df_calc[slow_col] = df_calc['close'].rolling(window=self.slow_window).mean()
        else:
            # Custom calculation for non-standard windows
            df_calc[fast_col] = df_calc['close'].rolling(window=self.fast_window).mean()
            df_calc[slow_col] = df_calc['close'].rolling(window=self.slow_window).mean()
        
        fast_sma = df_calc[fast_col]
        slow_sma = df_calc[slow_col]
        
        # Determine signals
        signals = np.where(fast_sma > slow_sma, 1, -1)
        signals = pd.Series(signals, index=df.index)
        
        # Neutral position where SMAs are not available (NaN)
        mask = fast_sma.isna() | slow_sma.isna()
        signals[mask] = 0
        
        return signals
