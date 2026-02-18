"""Canonical market data loaders used by model training/evaluation scripts."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.dal import DataAccessLayer


def _normalize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns and index for downstream processors."""
    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close",
    }
    normalized = df.rename(columns=rename_map).copy()

    if "Date" not in normalized.columns:
        if "date" in normalized.columns:
            normalized["Date"] = pd.to_datetime(normalized["date"])
        elif normalized.index.name and normalized.index.name.lower() == "date":
            normalized["Date"] = pd.to_datetime(normalized.index)
        elif isinstance(normalized.index, pd.DatetimeIndex):
            normalized["Date"] = pd.to_datetime(normalized.index)
        else:
            raise ValueError("Input data must contain a date column or datetime index.")

    normalized = normalized.sort_values("Date").set_index("Date")

    keep_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    for col in keep_cols:
        if col not in normalized.columns:
            if col == "Adj Close" and "Close" in normalized.columns:
                normalized[col] = normalized["Close"]
            else:
                raise ValueError(f"Missing required market column: {col}")

    return normalized[keep_cols].astype(float)


def _resolve_dates(
    start_date: Optional[str],
    end_date: Optional[str],
    years: int,
) -> tuple[str, str]:
    end_dt = datetime.now() if end_date is None else datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = (
        end_dt - timedelta(days=years * 365)
        if start_date is None
        else datetime.strptime(start_date, "%Y-%m-%d")
    )
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def load_market_data(
    ticker: str,
    source: str = "db",
    db_path: str = "data/financial_advisor.db",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 10,
) -> pd.DataFrame:
    """
    Load OHLCV data for a ticker from DB or yfinance.

    Returns a market-style DataFrame indexed by Date with:
    Open, High, Low, Close, Volume, Adj Close
    """
    resolved_start, resolved_end = _resolve_dates(start_date, end_date, years)
    ticker = ticker.upper()

    if source == "db":
        dal = DataAccessLayer(db_path=Path(db_path))
        raw = dal.get_stock_prices(ticker, resolved_start, resolved_end)
        return _normalize_market_dataframe(raw)

    if source == "yfinance":
        import yfinance as yf

        stock = yf.Ticker(ticker)
        raw = stock.history(start=resolved_start, end=resolved_end)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        return _normalize_market_dataframe(raw)

    raise ValueError(f"Unsupported data source: {source}. Use 'db' or 'yfinance'.")


def load_ticker_universe(db_path: str = "data/financial_advisor.db") -> list[str]:
    """Load all ticker symbols from the configured SQLite DB."""
    dal = DataAccessLayer(db_path=Path(db_path))
    return dal.get_all_tickers()
