# Model Backtesting Results

## Summary
- **Backtest Period**: 2023-01-01 to 2023-12-31
- **Assets**: AAPL (S&P 500)
- **Transaction Costs**: 0.1% per trade

## Performance Metrics

| Metric | Strategy | Buy & Hold (Benchmark) |
|--------|----------|-------------------------|
| Total Return | 40.02% | 54.80% |
| Sharpe Ratio | 2.2430 | 2.3126 |
| Max Drawdown | -7.54% | -14.93% |
| Win Rate | 55.41% | N/A |

## Equity Curve
The equity curve data is saved in `results/backtest_AAPL_2023.csv`.

### Key Observations
- **Risk Management**: The strategy outperformed the benchmark in terms of risk, with nearly half the maximum drawdown (-7.54% vs -14.93%).
- **Returns**: While the strategy returned 40.02%, it lagged behind the strong performance of AAPL in 2023 (54.80%).
- **Efficiency**: The Sharpe ratio is comparable (2.24 vs 2.31), suggesting the risk-adjusted returns are similar.
- **Win Rate**: A daily win rate of 55.4% when in the market is consistent with a profitable trend-following strategy.

## Conclusion
The model demonstrates a more conservative trading approach that successfully reduces downside risk while participating in most of the upward movement. In a strong bull market like 2023 for tech stocks, buy-and-hold is hard to beat on pure returns, but the strategy offers better capital preservation.
