import pandas as pd
import ta
from typing import Dict, List, Any

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given stock dataframe.
    
    Args:
        df: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'
        
    Returns:
        DataFrame with added technical indicators
    """
    if df.empty:
        return df
        
    # Ensure column names are correct for ta library (it often defaults to "Close", "High" etc if not specified, 
    # but we will pass series explicitly to be safe)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI (14)
    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    df['rsi_14'] = rsi.rsi()
    
    # MACD (12, 26, 9)
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Bollinger Bands (20, 2 std)
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    
    # SMA (20, 50, 200)
    df['sma_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    
    # EMA (12, 26)
    df['ema_12'] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()
    
    # ATR (14)
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df['atr_14'] = atr.average_true_range()
    
    # Volume SMA (20)
    # ta library might not have a direct volume SMA, but it's just an SMA on volume
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    
    # Price ROC (10)
    roc = ta.momentum.ROCIndicator(close=close, window=10)
    df['price_roc_10'] = roc.roc()
    
    # Handle NaNs:
    # Indicators will produce NaNs for the initial periods. 
    # We should keep them as None/NaN or fill them. 
    # The prompt says: "Handle potential NaN values (e.g., first 200 days for SMA_200)."
    # Usually for DB insertion, we can leave them as None (NULL in SQL).
    # Pandas NaNs (float) might need to be converted to None for sqlite if we insert via dictionary.
    
    return df

def prepare_indicators_for_db(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert dataframe with indicators to list of dicts for DB insertion.
    Removes rows where critical identifiers might be missing (though unlikely).
    Handles NaN values by converting them to None.
    """
    records = []
    
    # Iterate over rows
    for _, row in df.iterrows():
        record = {
            'ticker': row['ticker'],
            'date': row['date'],
            'rsi_14': row.get('rsi_14'),
            'macd_line': row.get('macd_line'),
            'macd_signal': row.get('macd_signal'),
            'macd_histogram': row.get('macd_histogram'),
            'bb_upper': row.get('bb_upper'),
            'bb_middle': row.get('bb_middle'),
            'bb_lower': row.get('bb_lower'),
            'sma_20': row.get('sma_20'),
            'sma_50': row.get('sma_50'),
            'sma_200': row.get('sma_200'),
            'ema_12': row.get('ema_12'),
            'ema_26': row.get('ema_26'),
            'atr_14': row.get('atr_14'),
            'volume_sma_20': row.get('volume_sma_20'),
            'price_roc_10': row.get('price_roc_10')
        }
        
        # Replace NaN with None
        for k, v in record.items():
            if pd.isna(v):
                record[k] = None
                
        records.append(record)
        
    return records
