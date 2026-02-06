# Walk-Forward Validation Report Template

## Executive Summary
- **Ticker**: {ticker}
- **Model**: {model_name}
- **Validation Date Range**: {start_date} to {end_date}
- **Number of Walk-Forward Steps**: {num_steps}

## Aggregated Performance Metrics
| Metric | Value |
|--------|-------|
| Average Test Accuracy | {avg_test_acc} |
| Average Sharpe Ratio | {avg_sharpe} |
| Average Max Drawdown | {avg_max_drawdown} |
| Average RMSE | {avg_rmse} |

## Overfitting Analysis
- **Average Training Accuracy**: {avg_train_acc}
- **Average Validation Accuracy**: {avg_val_acc}
- **Accuracy Gap (Train - Val)**: {acc_gap}
- **Overfitting Ratio (Val / Train)**: {overfitting_ratio}

*Notes: A high gap or low overfitting ratio indicates potential overfitting.*

## Detailed Step-by-Step Results
| Step | Test Period Start | Test Period End | Train Acc | Val Acc | Test Acc | Sharpe | Max DD |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ... | ... | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... | ... | ... |

## Observations
- [Insert observations about model stability over different market regimes]
- [Discuss consistency of directional accuracy]
- [Evaluate risk-adjusted returns (Sharpe Ratio)]

## Recommendations
- [Retrain model frequency]
- [Feature adjustments]
- [Risk management tweaks]
