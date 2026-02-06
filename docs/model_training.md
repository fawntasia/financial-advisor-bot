# Model Training

## Purpose
Document the training setup for the LSTM, Random Forest, and XGBoost models, including data preparation, feature set, splits, and evaluation approach. Results are captured separately once training runs are finalized.

## Data Preparation
- **Source Data**: Historical OHLCV prices plus FinBERT-scored news sentiment stored in `data/financial_advisor.db`.
- **Indicator Pipeline**: Technical indicators are computed from price history and stored in the `technical_indicators` table.
- **Cleaning**: Remove rows with missing indicator values for the required lookback window; ensure chronological ordering per ticker.
- **Scaling**: Apply min-max scaling fit on the training segment only; reuse for validation/test segments to avoid leakage.
- **Sequence Windows (LSTM)**: Build 60-day lookback sequences to predict the next day target.

## Feature Set
Model inputs are aligned to the feature engineering pipeline described in `src/features/indicators.py` and `src/data/stock_data.py`.

- **Core Price Features**: Close price, volume.
- **Technical Indicators**: RSI, MACD, Bollinger Bands position, SMA/EMA ratios (as computed in the indicator pipeline).
- **Sentiment Features**: Daily aggregated FinBERT sentiment scores (where available).
- **Targets**:
  - **LSTM**: Next-day price (regression).
  - **Random Forest / XGBoost**: Next-day direction (UP=1, DOWN=0).

## Training Splits
All splits are chronological to prevent leakage.

- **Primary Split**:
  - **Train**: 2019-2021
  - **Validation**: 2022
  - **Test**: 2023
- **Walk-Forward Validation**:
  - Expanding window with rolling quarters in 2022-2023.
  - Each step trains on past data and evaluates on the next quarter.
  - Aggregated metrics are reported across steps.

## Model Training Details

### LSTM
- **Architecture**: LSTM(128) -> LSTM(64) -> Dense(1).
- **Loss**: Mean Squared Error.
- **Optimizer**: Adam.
- **Callbacks**: EarlyStopping and ModelCheckpoint.
- **Input Window**: 60-day sequences.

### Random Forest
- **Estimator**: `RandomForestClassifier`.
- **Tuning**: `RandomizedSearchCV` with `TimeSeriesSplit` to preserve temporal order.
- **Feature Importance**: Extracted post-training for interpretability.

### XGBoost
- **Estimator**: `XGBClassifier`.
- **Early Stopping**: Enabled with validation data to reduce overfitting.
- **Regularization**: Standard XGBoost regularization parameters (as configured in the training script).
- **Feature Importance**: Recorded for comparison with Random Forest.

## Hyperparameters
Hyperparameter values are recorded in training artifacts and should be updated here once runs are finalized.

### LSTM
- **Window Size**: 60
- **Units**: 128, 64
- **Dropout**: {TBD}
- **Batch Size**: {TBD}
- **Epochs**: {TBD}

### Random Forest
- **n_estimators**: {TBD}
- **max_depth**: {TBD}
- **min_samples_split**: {TBD}
- **min_samples_leaf**: {TBD}

### XGBoost
- **learning_rate**: {TBD}
- **n_estimators**: {TBD}
- **max_depth**: {TBD}
- **subsample**: {TBD}
- **colsample_bytree**: {TBD}

## Evaluation Metrics
Metrics align with the walk-forward validation and model selection docs.

- **Directional Accuracy**
- **RMSE**
- **MAE**
- **Sharpe Ratio**
- **Max Drawdown**

## Results (Placeholders)
Results will be populated after full training runs and walk-forward validation complete.

- **LSTM**:
  - **Average Accuracy**: {TBD}
  - **Average RMSE**: {TBD}
  - **Average Sharpe Ratio**: {TBD}
  - **Average Max Drawdown**: {TBD}
- **Random Forest**:
  - **Average Accuracy**: {TBD}
  - **Average Sharpe Ratio**: {TBD}
  - **Average Max Drawdown**: {TBD}
- **XGBoost**:
  - **Average Accuracy**: {TBD}
  - **Average Sharpe Ratio**: {TBD}
  - **Average Max Drawdown**: {TBD}

## Notes
- Walk-forward validation is required for all three models to avoid temporal leakage.
- Exact metrics and hyperparameters should be sourced from training logs and `docs/validation_report.md` once finalized.
