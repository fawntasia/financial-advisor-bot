# Model Comparison and Selection Report

Date: 2026-02-04 14:27:20

## Selection Criteria

Models are evaluated using walk-forward validation. The following metrics are aggregated across all steps:
- **Average Accuracy**: Mean directional accuracy (predicting UP vs DOWN).
- **Average Sharpe Ratio**: Mean risk-adjusted return (using 0% risk-free rate).
- **Average Max Drawdown**: Mean of the maximum peak-to-trough decline in each step.

The **Production Model** is selected based on the highest **Average Sharpe Ratio**, as it represents the best risk-adjusted performance.

## Performance Summary

| ticker   | model   |   avg_accuracy |   avg_sharpe |   avg_max_drawdown |   num_steps |
|:---------|:--------|---------------:|-------------:|-------------------:|------------:|
| AAPL     | rf      |       0.485878 |     0.235285 |          -0.149917 |          34 |
| AAPL     | xgb     |       0.530804 |     1.49587  |          -0.124833 |          34 |

## Production Model Selection

The selected production model is **xgb** for **AAPL**.
- **Average Sharpe Ratio**: 1.4959
- **Average Accuracy**: 0.5308
- **Average Max Drawdown**: -0.1248

## Ensemble Strategy

A simple majority voting ensemble has been implemented to combine predictions from multiple models. This ensemble can be used to improve robustness by requiring agreement between different model architectures.
