import numpy as np
import pandas as pd
from typing import Dict, Optional

def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily percentage returns."""
    return prices.pct_change().fillna(0)

def calculate_strategy_returns(signals: pd.Series, prices: pd.Series) -> pd.Series:
    """
    Calculate returns of a strategy based on signals.
    
    Assumes signal at time t (based on close price at t) determines the position 
    held for the return from t to t+1.
    
    Args:
        signals: Series of positions (e.g., 1 for long, -1 for short, 0 for neutral)
        prices: Series of asset prices
        
    Returns:
        Series of strategy daily returns
    """
    asset_returns = calculate_daily_returns(prices)
    # Shift signals forward by 1 so signal at t aligns with return at t+1
    aligned_signals = signals.shift(1).fillna(0)
    strategy_returns = aligned_signals * asset_returns
    return strategy_returns

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate Maximum Drawdown."""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate performance metrics for a return series.
    
    Args:
        returns: Series of daily percentage returns
        risk_free_rate: Annualized risk-free rate (decimal, e.g., 0.02 for 2%)
        periods_per_year: Number of trading periods in a year (default 252)
        
    Returns:
        Dictionary containing:
        - total_return
        - annualized_return
        - annualized_volatility
        - sharpe_ratio
        - max_drawdown
    """
    if returns.empty:
        return {}

    # Total Return
    cumulative_return = (1 + returns).prod() - 1
    
    # Number of days (periods)
    n_periods = len(returns)
    if n_periods < 1:
        return {}

    # Annualized Return
    # (1 + total_return) ^ (periods_per_year / n_periods) - 1
    # Handle negative base for power operation if total return < -1 (bankruptcy)
    if cumulative_return <= -1:
        annualized_return = -1.0
    else:
        annualized_return = (1 + cumulative_return) ** (periods_per_year / n_periods) - 1
    
    # Annualized Volatility
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    # (Rp - Rf) / sigma_p
    # Adjust Rf to daily for the numerator or annulize both.
    # Usually: (Ann_Ret - Rf) / Ann_Vol
    if annualized_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
    # Max Drawdown
    max_drawdown = calculate_max_drawdown(returns)
    
    return {
        "total_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }
