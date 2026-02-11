# Model Training

## Purpose
Document the current training setup for LSTM, Random Forest, and XGBoost models, including real data sources, feature preparation, splits, and artifacts produced by scripts.

## Current Data Sources
- **LSTM (`scripts/train_lstm.py`)**:
  - Reads historical OHLCV from SQLite via `DataAccessLayer.get_stock_prices`.
  - Default mode is `--ticker ALL`, meaning all tickers currently present in the `tickers` table are included.
  - Does not call yfinance during training.
- **Random Forest / XGBoost (`scripts/train_random_forest.py`, `scripts/train_xgboost.py`)**:
  - Use `StockDataProcessor.fetch_data()` and therefore still call yfinance.

## Feature Set
### LSTM Features
- Feature columns used in training:
  - `Close`
  - `SMA_20`
  - `SMA_50`
  - `RSI`
  - `MACD`
  - `Signal_Line`
- Target: next-step scaled `Close` (regression target built from sequence endpoint).
- Indicators are generated per ticker with `StockDataProcessor.add_technical_indicators`.
- Sentiment is not currently part of the LSTM feature tensor.

### RF/XGB Features
- Directional classification features from `StockDataProcessor.prepare_for_classification`.
- Target: next-day direction (`UP=1`, `DOWN=0`).

## Training Splits
All splits are chronological to reduce leakage.

### LSTM Split Logic
- Per ticker split: default `train_split=0.8`.
- Scalers are fitted on per-ticker train partition only.
- Sequence construction:
  - Train sequences from ticker train partition.
  - Test sequences include train tail context (`sequence_length`) to preserve continuity.
- A dedicated validation split is carved from aggregated training sequences (`--val_split`, default `0.1`) and used for early stopping, while the test split remains held out for final evaluation.
- Final model sees one aggregated dataset across all included tickers.

## Model Training Details

### LSTM
- **Architecture**: LSTM(128) -> LSTM(64) -> Dense(1).
- **Loss**: Mean Squared Error.
- **Optimizer**: Adam.
- **Callbacks**: EarlyStopping and ModelCheckpoint.
- **Input Window**: default 60 timesteps (`--sequence_length`).
- **Scope**: single global model trained across all tickers by default.
- **CLI**:
  - `python scripts/train_lstm.py --ticker ALL --epochs 50 --batch_size 32`
  - Optional: `--save_dir`, `--sequence_length`, `--train_split`, `--max_tickers`
- **Artifacts**:
  - `models/lstm_<all|ticker>_<timestamp>.keras`
  - `models/lstm_<all|ticker>_<timestamp>_scalers.joblib`
  - `models/lstm_<all|ticker>_<timestamp>_metadata.json`
- **Reported metrics**:
  - Global scaled test MSE.
  - Average per-ticker RMSE in original price space (via per-ticker target scaler).

### Random Forest
- **Estimator**: `RandomForestClassifier`.
- **Tuning**: `RandomizedSearchCV` with `TimeSeriesSplit` to preserve temporal order.
- **Feature Importance**: Extracted post-training for interpretability.

### XGBoost
- **Estimator**: `XGBClassifier`.
- **Early Stopping**: Enabled with a dedicated validation split (`--val-split`) to reduce overfitting without using test data.
- **Regularization**: Standard XGBoost regularization parameters (as configured in the training script).
- **Feature Importance**: Recorded for comparison with Random Forest.

## Evaluation Metrics
- **LSTM training script**: scaled MSE + per-ticker RMSE summary.
- **Walk-forward validation (`scripts/run_walkforward.py`)**:
  - Supports RF/XGB.
  - Produces directional accuracy, RMSE on labels, Sharpe ratio, and max drawdown aggregates.

## Notes
- LSTM walk-forward is not yet wired in `scripts/run_walkforward.py`.
- Ticker coverage is dynamic and depends on `tickers` table contents (not hardcoded in the trainer).
- For reproducibility, use metadata JSON generated per LSTM run to capture exact data coverage and run parameters.
