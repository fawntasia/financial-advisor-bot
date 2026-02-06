# Chapter 5: Results

## 5.1 Overview
This chapter reports the empirical outcomes of the financial advisor bot evaluation. It summarizes model comparison outputs, walk-forward validation results, and backtesting performance. Where final metrics are pending, placeholders are provided for later insertion.

## 5.2 Data and Evaluation Windows
- **Validation Source**: `docs/validation_report.md`
- **Backtesting Source**: `docs/backtest_results.md`
- **Evaluation Assets**: {assets_list}
- **Evaluation Periods**: {evaluation_periods}

## 5.3 Model Comparison Results
This section consolidates comparative results across candidate models (e.g., LSTM vs. ensemble) using consistent evaluation metrics.

### 5.3.1 Summary Table (Model Comparison)
| Metric | Model A | Model B | Model C |
|--------|---------|---------|---------|
| Directional Accuracy | {model_a_acc} | {model_b_acc} | {model_c_acc} |
| RMSE | {model_a_rmse} | {model_b_rmse} | {model_c_rmse} |
| Sharpe Ratio | {model_a_sharpe} | {model_b_sharpe} | {model_c_sharpe} |
| Max Drawdown | {model_a_max_dd} | {model_b_max_dd} | {model_c_max_dd} |

### 5.3.2 Discussion
- [Insert narrative comparing model strengths and weaknesses.]
- [Highlight trade-offs between accuracy and risk-adjusted returns.]

## 5.4 Walk-Forward Validation Results
Walk-forward validation results are reported using the template in `docs/validation_report.md`. Final numeric results are pending insertion.

### 5.4.1 Aggregated Metrics (Placeholder)
| Metric | Value |
|--------|-------|
| Average Test Accuracy | {avg_test_acc} |
| Average Sharpe Ratio | {avg_sharpe} |
| Average Max Drawdown | {avg_max_drawdown} |
| Average RMSE | {avg_rmse} |

### 5.4.2 Overfitting Analysis (Placeholder)
- **Average Training Accuracy**: {avg_train_acc}
- **Average Validation Accuracy**: {avg_val_acc}
- **Accuracy Gap (Train - Val)**: {acc_gap}
- **Overfitting Ratio (Val / Train)**: {overfitting_ratio}

### 5.4.3 Step-by-Step Results (Template)
| Step | Test Period Start | Test Period End | Train Acc | Val Acc | Test Acc | Sharpe | Max DD |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ... | ... | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... | ... | ... |

## 5.5 Backtesting Results
Backtesting results are summarized from `docs/backtest_results.md`.

### 5.5.1 Summary Table (Backtest vs. Benchmark)
| Metric | Strategy | Buy & Hold (Benchmark) |
|--------|----------|-------------------------|
| Total Return | 40.02% | 54.80% |
| Sharpe Ratio | 2.2430 | 2.3126 |
| Max Drawdown | -7.54% | -14.93% |
| Win Rate | 55.41% | N/A |

### 5.5.2 Interpretation
- The strategy reduces downside risk relative to the benchmark while remaining competitive on risk-adjusted returns.
- Absolute returns lag in strong bull-market conditions, consistent with a more conservative allocation.

### 5.5.3 Equity Curve and Trade Logs
- **Equity Curve File**: `results/backtest_AAPL_2023.csv`
- **Trade Logs**: {trade_log_location}

## 5.6 Consolidated Results Summary
This section summarizes the combined implications of model comparison, validation, and backtesting.

### 5.6.1 Summary Table (Cross-Section)
| Category | Key Result | Evidence |
|----------|------------|----------|
| Predictive Accuracy | {accuracy_summary} | `docs/validation_report.md` |
| Risk Management | {risk_summary} | `docs/backtest_results.md` |
| Return Profile | {return_summary} | `docs/backtest_results.md` |

### 5.6.2 Key Findings
- [Insert key finding about predictive stability across regimes.]
- [Insert key finding about risk-adjusted performance.]
- [Insert key finding about practical trade-offs in live deployment.]

## 5.7 Limitations
- [Insert limitations related to data coverage, market regime dependence, or transaction costs.]

## 5.8 Chapter Conclusion
This chapter documents the observed performance of the proposed system. Final numeric values for validation and model comparison should be inserted once consolidated into the validation report and model comparison output.
