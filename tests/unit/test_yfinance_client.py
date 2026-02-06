import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data.yfinance_client import YFinanceClient

@pytest.fixture
def client():
    # Use a high rate limit delay to ensure _rate_limit is called, 
    # but we'll mock time.sleep
    return YFinanceClient(rate_limit_delay=0.1)

class TestYFinanceClient:
    
    @patch('src.data.yfinance_client.yf.Ticker')
    @patch('src.data.yfinance_client.time.sleep')
    def test_get_ticker_success(self, mock_sleep, mock_yf_ticker, client):
        # Mocking yfinance
        mock_stock = MagicMock()
        mock_df = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [95.0], 
            'Close': [102.0], 'Volume': [1000]
        }, index=pd.to_datetime(['2023-01-01']))
        mock_stock.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_stock
        
        # Call with use_cache=False to avoid cache interference
        df = client.get_ticker("AAPL", period="1d", use_cache=False)
        
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]['Close'] == 102.0
        mock_yf_ticker.assert_called_with("AAPL")
        mock_stock.history.assert_called_once()

    @patch('src.data.yfinance_client.yf.Ticker')
    def test_get_ticker_empty_data(self, mock_yf_ticker, client):
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_yf_ticker.return_value = mock_stock
        
        with pytest.raises(ValueError, match="No data found for ticker AAPL"):
            client.get_ticker("AAPL", use_cache=False)

    @patch('src.data.yfinance_client.yf.Ticker')
    @patch('src.data.yfinance_client.time.sleep')
    def test_get_tickers_bulk(self, mock_sleep, mock_yf_ticker, client):
        mock_stock = MagicMock()
        mock_df = pd.DataFrame({'Close': [100.0]}, index=pd.to_datetime(['2023-01-01']))
        mock_stock.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_stock
        
        tickers = ["AAPL", "MSFT"]
        results = client.get_tickers(tickers, use_cache=False)
        
        assert len(results) == 2
        assert "AAPL" in results
        assert "MSFT" in results
        assert mock_yf_ticker.call_count == 2

    @patch('src.data.yfinance_client.yf.Ticker')
    @patch('src.data.yfinance_client.time.sleep')
    def test_get_ticker_dot_replacement(self, mock_sleep, mock_yf_ticker, client):
        # Mocking yfinance
        mock_stock = MagicMock()
        mock_df = pd.DataFrame({'Close': [100.0]}, index=pd.to_datetime(['2023-01-01']))
        mock_stock.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_stock
        
        client.get_ticker("BRK.B", use_cache=False)
        
        # Verify it was called with hyphen
        mock_yf_ticker.assert_called_with("BRK-B")

    def test_cache_logic(self, client, tmp_path):
        # Mock CACHE_DIR
        with patch('src.data.yfinance_client.CACHE_DIR', tmp_path):
            mock_df = pd.DataFrame({'Close': [100.0]}, index=pd.to_datetime(['2023-01-01']))
            cache_key = client._get_cache_key("AAPL", "1y")
            
            # Save to cache
            client._save_to_cache(cache_key, mock_df)
            
            # Load from cache
            loaded_df = client._load_from_cache(cache_key)
            assert loaded_df is not None
            assert loaded_df.iloc[0]['Close'] == 100.0
            
            # Test get_ticker uses cache
            with patch.object(client, '_fetch_from_yfinance') as mock_fetch:
                df = client.get_ticker("AAPL", period="1y", use_cache=True)
                assert not mock_fetch.called
                assert df.iloc[0]['Close'] == 100.0

    def test_clear_cache(self, client, tmp_path):
        with patch('src.data.yfinance_client.CACHE_DIR', tmp_path):
            # Create a fake cache file
            fake_cache = tmp_path / "test.pkl"
            fake_cache.write_text("data")
            
            assert len(list(tmp_path.glob("*.pkl"))) == 1
            client.clear_cache()
            assert len(list(tmp_path.glob("*.pkl"))) == 0

    def test_cache_info(self, client, tmp_path):
         with patch('src.data.yfinance_client.CACHE_DIR', tmp_path):
            info = client.get_cache_info()
            assert info["num_files"] == 0
            assert info["total_size_mb"] == 0
            
            # Add a file
            (tmp_path / "test.pkl").write_text("a" * 1024 * 1024) # 1MB roughly
            info = client.get_cache_info()
            assert info["num_files"] == 1
            assert info["total_size_mb"] > 0
