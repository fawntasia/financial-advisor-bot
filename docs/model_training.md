# Model Training

## Purpose
Document the current training setup for LSTM, Random Forest, and XGBoost models, including real data sources, feature preparation, splits, and artifacts produced by scripts.

## Current Data Sources
- **LSTM (`scripts/train_lstm.py`)**:
  - Reads historical OHLCV from the canonical loader (`src/models/data_sources.py`).
  - Default source is SQLite (`--data-source db`), with optional live fetch (`--data-source yfinance`).
  - Default mode is `--ticker ALL`, meaning all tickers currently present in the `tickers` table are included.
- **Random Forest / XGBoost (`scripts/train_random_forest.py`, `scripts/train_xgboost.py`)**:
  - Read historical OHLCV from the same canonical loader.
  - DB-first by default (`--data-source db`), with optional yfinance mode for ad-hoc experiments.
  - Use split-safe classification preparation (`prepare_for_classification_splits`) to avoid boundary leakage.
- **Walk-forward / Backtesting**:
  - `scripts/run_walkforward.py` and `scripts/backtest_models.py` now follow the same DB-first source contract.
  - SQLite path is explicit via `--db-path`.

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
- Directional classification features from `StockDataProcessor.prepare_for_classification_splits`.
- Target: next-day direction (`UP=1`, `DOWN/FLAT=0`).
- Trading signal policy (used consistently in validation/backtest):
  - Class `1` -> signal `1` (long)
  - Class `0` -> signal `0` (flat)

## Training Splits
All splits are chronological to reduce leakage.

### LSTM Split Logic
- Per ticker split: default `train_split=0.8`.
- Scalers are fitted on per-ticker train partition only.
- Sequence construction:
  - Train sequences from ticker train partition.
  - Test sequences include train tail context (`sequence_length`) to preserve continuity.
- A dedicated validation split is carved from aggregated training sequences (`--val-split`, default `0.1`) and used for early stopping, while the test split remains held out for final evaluation.
- Final model sees one aggregated dataset across all included tickers.

### RF/XGB Split Logic (Leakage-Safe)
- Data is split into chronological train/test regions first.
- Validation is split from the end of the train region (`--val-split`).
- Labels are created separately within each split:
  - `target(t) = 1 if close(t+1) > close(t) else 0`
  - Last row of each split is dropped, so labels never reference the next split.
- This prevents train labels from depending on validation/test prices.

## Model Training Details

### LSTM
- **Architecture**: LSTM(128) -> LSTM(64) -> Dense(1).
- **Loss**: Mean Squared Error.
- **Optimizer**: Adam.
- **Callbacks**: EarlyStopping and ModelCheckpoint.
- **Input Window**: default 60 timesteps (`--sequence-length`).
- **Scope**: single global model trained across all tickers by default.
- **Reproducibility**: global seeds set via `src/models/reproducibility.py` (`--seed`).
- **CLI**:
  - `python scripts/train_lstm.py --ticker ALL --epochs 50 --batch-size 32 --output-dir models`
  - Optional: `--sequence-length`, `--train-split`, `--val-split`, `--max-tickers`, `--seed`, `--data-source`, `--db-path`, `--start-date`, `--end-date`
- **Artifacts**:
  - `models/lstm_<all|ticker>_<timestamp>.keras`
  - `models/lstm_<all|ticker>_<timestamp>_scalers.joblib`
  - `models/lstm_<all|ticker>_<timestamp>_metadata.json`
  - `models/lstm_<all|ticker>_<timestamp>.manifest.json`
- **Reported metrics**:
  - Global scaled test MSE.
  - Average per-ticker RMSE in original price space (via per-ticker target scaler).

### Random Forest
- **Estimator**: `RandomForestClassifier`.
- **Tuning**: `RandomizedSearchCV` with `TimeSeriesSplit` and `balanced_accuracy` scoring.
- **Class imbalance**: hyperparameter search includes `class_weight` options.
- **Thresholding**: validation-based threshold tuning (Sharpe objective when validation prices are available; otherwise balanced accuracy).
- **Feature Importance**: extracted post-training for interpretability.
- **Artifacts**:
  - `models/random_forest_<ticker>_<timestamp>.pkl`
  - `models/random_forest_<ticker>_<timestamp>_metadata.json`
  - `models/random_forest_<ticker>_<timestamp>.manifest.json`

### XGBoost
- **Estimator**: `XGBClassifier`.
- **Early Stopping**: Enabled with dedicated validation split and `early_stopping_rounds=10`.
- **Regularization/Tuning Search**:
  - `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
  - plus core tree/learning-rate/subsample parameters and `scale_pos_weight`
- **Thresholding**: validation-based threshold tuning (Sharpe objective when validation prices are available; otherwise balanced accuracy).
- **Feature Importance**: Recorded for comparison with Random Forest.
- **Artifacts**:
  - `models/xgboost_<ticker>_<timestamp>.json`
  - `models/xgboost_<ticker>_<timestamp>.meta.json`
  - `models/xgboost_<ticker>_<timestamp>_metadata.json`
  - `models/xgboost_<ticker>_<timestamp>.manifest.json`

## Evaluation Metrics
- **LSTM training script**: scaled MSE + per-ticker RMSE summary.
- **Walk-forward validation (`scripts/run_walkforward.py`)**:
  - Supports RF, XGB, and LSTM.
  - Classification outputs include:
    - Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC
    - RMSE on direction labels
    - Sharpe ratio and max drawdown
  - LSTM walk-forward reports:
    - Regression RMSE (price space) and derived direction metrics
    - Sharpe ratio and max drawdown from mapped trading signals

## Notes
- Ticker coverage is dynamic and depends on `tickers` table contents (not hardcoded in the trainer).
- For reproducibility, use metadata JSON generated per LSTM/RF/XGB run to capture exact data coverage, split sizes, parameters, metrics, and artifact paths.

## Unified Manifest Sidecar
All model training scripts now emit a canonical sidecar:
- `<model_stem>.manifest.json`

Required keys:
- `schema_version`
- `model_kind`
- `model_path`
- `created_at`
- `seed`
- `feature_columns`
- `data_source`
- `data_coverage`
- `split_config`
- `metrics`
- `library_versions`

Native model artifacts remain unchanged (`.pkl`, `.json`, `.keras`), but the manifest is the standard machine-readable contract for downstream tooling.

## CLI Breaking Changes
| Old Flag | New Flag |
|---|---|
| `--save-dir` / `--save_dir` | `--output-dir` |
| `--start` | `--start-date` |
| `--end` | `--end-date` |
| `--batch_size` | `--batch-size` |
| `--sequence_length` | `--sequence-length` |
| `--max_tickers` | `--max-tickers` |
| `--train_split` | `--train-split` |
| `--val_split` | `--val-split` |
| *(new)* | `--data-source {db,yfinance}` |
| *(new)* | `--db-path` |

## Reproducible Runbook
```bash
# Random Forest (DB-first)
python scripts/train_random_forest.py --ticker AAPL --no-tune --output-dir models --data-source db --db-path data/financial_advisor.db

# XGBoost (DB-first)
python scripts/train_xgboost.py --ticker AAPL --no-tune --output-dir models --data-source db --db-path data/financial_advisor.db

# LSTM (DB-first)
python scripts/train_lstm.py --ticker ALL --epochs 10 --batch-size 32 --output-dir models --data-source db --db-path data/financial_advisor.db

# Walk-forward
python scripts/run_walkforward.py --ticker AAPL --model rf --output-dir results --data-source db --db-path data/financial_advisor.db

# Backtest
python scripts/backtest_models.py --ticker AAPL --model models/random_forest_AAPL_<timestamp>.pkl --start-date 2023-01-01 --end-date 2023-12-31 --output-dir results --data-source db --db-path data/financial_advisor.db
```
