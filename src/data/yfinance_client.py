"""
yfinance API Client with caching, retry logic, and rate limiting.
"""

import yfinance as yf
import pandas as pd
import time
import logging
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.retry import retry_with_backoff

# Setup logging
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("data/cache/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class YFinanceClient:
    """Client for fetching stock data from yfinance with caching and retry logic."""
    
    def __init__(self, rate_limit_delay=0.5):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, ticker: str, period: str, start: Optional[str] = None, end: Optional[str] = None) -> str:
        """Generate cache key for a request."""
        key_data = f"{ticker}_{period}_{start}_{end}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return CACHE_DIR / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        # Check if cache is expired (1 day for recent data)
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > 86400:  # 24 hours
            logger.info(f"Cache expired for {cache_key}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded from cache: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    @retry_with_backoff(max_attempts=5, base_delay=1)
    def _fetch_from_yfinance(self, ticker: str, period: str = '5y', 
                           start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from yfinance with retry logic."""
        self._rate_limit()
        
        stock = yf.Ticker(ticker)
        
        if start and end:
            df = stock.history(start=start, end=end)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df
    
    def get_ticker(self, ticker: str, period: str = '5y', 
                   start: Optional[str] = None, end: Optional[str] = None,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = self._get_cache_key(ticker, period, start, end)
        
        # Try to load from cache
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch from yfinance
        try:
            df = self._fetch_from_yfinance(ticker, period, start, end)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, df)
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            raise
    
    def get_tickers(self, tickers: List[str], period: str = '5y',
                    use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Period to fetch
            use_cache: Whether to use caching
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        failed = []
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"Fetching {i+1}/{len(tickers)}: {ticker}")
                df = self.get_ticker(ticker, period, use_cache=use_cache)
                results[ticker] = df
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                failed.append(ticker)
        
        if failed:
            logger.warning(f"Failed to fetch {len(failed)} tickers: {failed}")
        
        return results
    
    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics."""
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(CACHE_DIR)
        }
