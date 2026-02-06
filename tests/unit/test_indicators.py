import pytest
import pandas as pd
import numpy as np
from src.features.indicators import calculate_indicators, prepare_indicators_for_db

@pytest.fixture
def sample_stock_data():
    """Create a sample dataframe with stock data."""
    dates = pd.date_range(start="2023-01-01", periods=250)
    data = {
        'open': np.linspace(100, 150, 250) + np.random.normal(0, 1, 250),
        'high': np.linspace(105, 155, 250) + np.random.normal(0, 1, 250),
        'low': np.linspace(95, 145, 250) + np.random.normal(0, 1, 250),
        'close': np.linspace(102, 152, 250) + np.random.normal(0, 1, 250),
        'volume': np.random.randint(1000, 5000, 250)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

class TestIndicators:
    
    def test_calculate_indicators(self, sample_stock_data):
        df = calculate_indicators(sample_stock_data.copy())
        
        # Check if indicators are added
        expected_columns = [
            'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'atr_14', 
            'volume_sma_20', 'price_roc_10'
        ]
        
        for col in expected_columns:
            assert col in df.columns
            
        # Check SMA_200 has NaNs for the first 199 rows
        assert df['sma_200'].iloc[0:199].isna().all()
        assert not df['sma_200'].iloc[200:].isna().any()
        
        # Check RSI is between 0 and 100
        valid_rsi = df['rsi_14'].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_calculate_indicators_empty(self):
        df = pd.DataFrame()
        result = calculate_indicators(df)
        assert result.empty

    def test_prepare_indicators_for_db(self, sample_stock_data):
        df = calculate_indicators(sample_stock_data.copy())
        df['ticker'] = 'AAPL'
        df['date'] = df.index.strftime('%Y-%m-%d')
        
        records = prepare_indicators_for_db(df)
        
        assert len(records) == 250
        assert records[0]['ticker'] == 'AAPL'
        # Check if NaN was converted to None
        assert records[0]['sma_200'] is None
        # Check if valid value is preserved
        assert records[249]['sma_200'] is not None
        assert isinstance(records[249]['sma_200'], float)
