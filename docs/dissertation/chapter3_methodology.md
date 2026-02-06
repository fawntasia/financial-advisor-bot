# Chapter 3: Methodology

## Overview
This chapter details the methodology used to design and evaluate a local-first financial advisory system for S&P 500 equities. The approach combines price-based technical indicators, FinBERT sentiment features, and three predictive models (LSTM, Random Forest, XGBoost) evaluated with time-aware validation.

## Data Sources
- **Market Data**: Historical OHLCV price data for S&P 500 tickers, stored in the local SQLite database.
- **News Sentiment**: Financial news headlines scored locally with FinBERT and aggregated to daily sentiment features.
- **Derived Indicators**: Technical indicators computed from price history and persisted alongside raw prices.

## Data Preprocessing
Preprocessing is structured to preserve temporal integrity and minimize leakage.

- **Chronological Ordering**: All records are sorted by ticker and date before feature construction.
- **Missing Values**: Rows with incomplete indicator lookback windows are removed.
- **Scaling**: Min-max scaling is fit only on the training segment and reused for validation and test segments.
- **Sequence Construction (LSTM)**: A 60-day lookback window is used to build input sequences for next-day prediction.
- **Sentiment Aggregation**: Multiple headlines per day are averaged into a single daily sentiment score, with counts retained for context.

## Feature Set
Features align with the indicator and data pipelines.

- **Core Price Features**: Close price and volume.
- **Technical Indicators**: RSI, MACD, Bollinger Bands position, SMA/EMA ratios.
- **Sentiment Features**: Daily aggregated FinBERT scores.
- **Targets**:
  - **LSTM**: Next-day price (regression).
  - **Random Forest / XGBoost**: Next-day direction (UP=1, DOWN=0).

## Model Design
Three complementary model types are used to balance sequence learning and interpretable ensemble baselines.

### LSTM
- **Architecture**: LSTM(128) -> LSTM(64) -> Dense(1).
- **Loss**: Mean Squared Error with Adam optimizer.
- **Purpose**: Capture temporal dependencies in price and indicator sequences.

### Random Forest
- **Estimator**: `RandomForestClassifier` for directional prediction.
- **Tuning**: Randomized search with time-series splits to avoid shuffling.
- **Purpose**: Provide a non-linear, interpretable ensemble baseline.

### XGBoost
- **Estimator**: `XGBClassifier` with early stopping on validation data.
- **Regularization**: Standard XGBoost regularization settings.
- **Purpose**: Strong gradient-boosted baseline for directional prediction.

## Validation Protocol
Validation emphasizes walk-forward evaluation to reflect real-world deployment.

- **Primary Split**: Chronological train/validation/test segments (2019-2021 train, 2022 validation, 2023 test).
- **Walk-Forward Validation**:
  - Expanding training window with rolling quarterly evaluation in 2022-2023.
  - Each step trains on past data and evaluates on the next period.
  - Metrics are aggregated across steps to assess stability.
- **Metrics**: Directional accuracy, RMSE, MAE, Sharpe ratio, max drawdown.

## Leakage Avoidance
Leakage avoidance is enforced throughout the pipeline.

- **No Shuffling**: All splits are strictly chronological.
- **Fit on Train Only**: Scaling and normalization parameters are learned from training data only.
- **Time-Aware Tuning**: Hyperparameter search uses time-series splits instead of random CV.
- **Walk-Forward Discipline**: Each evaluation window uses only historical data available at that time.

## Local-Only Runtime Constraint
The system is designed to run entirely offline during inference.

- **Local Models**: FinBERT and the local LLM are loaded from disk at runtime.
- **No External Calls**: Data ingestion may use external sources offline, but runtime inference does not depend on third-party APIs.
- **Reproducibility**: Local artifacts (database, models, and scripts) ensure consistent results.
