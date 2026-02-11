# Technical Documentation

## Purpose
This document summarizes the architecture and data flow for the Financial Advisor Bot. It is scoped to developer-facing structure and module responsibilities rather than implementation detail.

## Architecture Overview
- **Runtime**: Local-only execution with a Streamlit UI (`app.py`).
- **Storage**: SQLite database at `data/financial_advisor.db` with a DAL in `src/database`.
- **ML/NLP**: Technical indicators + ML models (LSTM, Random Forest, XGBoost) and FinBERT sentiment.
- **LLM**: Llama 3 (llama-cpp) for chat responses, grounded by database context.

## Data Flow Summary
1. **Ingestion**: Scripts fetch OHLCV data and news, then write to SQLite.
2. **Feature/Indicators**: Technical indicators computed and stored for each ticker/date.
3. **Sentiment**: FinBERT scores headlines and aggregates daily sentiment.
4. **Modeling**:
   - **LSTM** (`scripts/train_lstm.py`) trains from SQLite price history via DAL and saves local model artifacts.
   - **RF/XGB** training scripts currently still fetch data via `StockDataProcessor`/yfinance.
5. **LLM Context**: DAL pulls latest price, indicators, sentiment, and predictions to build prompt context.
6. **UI**: Streamlit displays dashboards and chat responses using the LLM context.

## Module Overview

### Data Pipeline
- `src/data`: Data fetching and preprocessing utilities (`stock_data.py`, `yfinance_client.py`, `news_client.py`).
- `scripts`: Operational entrypoints for ingestion, indicator calculation, sentiment runs, and scheduling.
- `src/features`: Technical indicator calculations (`indicators.py`).

### Model Layer
- `src/models`: Model definitions and utilities.
  - `lstm_model.py`: Canonical TensorFlow/Keras LSTM sequence model.
  - `lstm_wrapper.py`: Backward-compatible alias to `LSTMModel` (no separate PyTorch LSTM implementation).
  - `random_forest_model.py`: Classification model for direction.
  - `xgboost_model.py`: Gradient-boosted classifier with time-series splits.
  - `baselines.py`, `evaluation.py`, `validation.py`, `comparison.py`: baselines, metrics, walk-forward validation, and selection.
- `models/`: Saved model artifacts produced by training scripts.

### NLP + Sentiment
- `src/nlp/finbert_loader.py`: FinBERT model loading and inference.
- `src/nlp/sentiment_pipeline.py`: Batch scoring and daily aggregation to SQLite.

### LLM
- `src/llm/llama_loader.py`: Llama 3 loader (llama-cpp) with mock mode if model absent.
- `src/llm/context_builder.py`: Builds natural-language context from SQLite.
- `src/llm/prompts.py`: Prompt templates and system prompt.
- `src/llm/guardrails.py`: Input/output safety checks and basic fact validation.

### UI
- `app.py`: Streamlit entrypoint, session state, navigation.
- `src/ui/views.py`: Dashboard, analysis, and disclaimer views.
- `src/ui/chat.py`: Chat manager and LLM interaction flow.
- `src/ui/charts.py`: Plotly chart generation utilities.

### Database Access
- `src/database/dal.py`: Centralized read/write layer for SQLite tables.

## Database Schema Overview (Tables)
- `tickers`
- `stock_prices`
- `technical_indicators`
- `news_headlines`
- `sentiment_scores`
- `daily_sentiment`
- `predictions`
- `model_performance`
- `system_logs`
- `user_preferences`
- `schema_migrations`

## Key Artifacts and Storage Locations
- `data/financial_advisor.db`: Primary SQLite database.
- `data/cache/`: Cached data from data fetchers.
- `models/`: Trained model files and checkpoints.
- `results/`: Output artifacts from backtests and comparisons.

### LSTM Artifact Format
`scripts/train_lstm.py` now saves three artifacts per run:
- `models/lstm_<all|ticker>_<timestamp>.keras`: trained Keras model.
- `models/lstm_<all|ticker>_<timestamp>_scalers.joblib`: per-ticker feature/target scalers.
- `models/lstm_<all|ticker>_<timestamp>_metadata.json`: run metadata (coverage, shapes, metrics, and paths).

## Operational Scripts (Selected)
- `scripts/init_db.py`: Creates schema and seeds S&P 500 tickers.
- `scripts/ingest_data.py`: Main ingestion workflow for prices + news.
- `scripts/download_historical_data.py`, `scripts/fetch_news.py`: Data source fetchers.
- `scripts/run_sentiment_analysis.py`: FinBERT scoring and aggregation.
- `scripts/train_lstm.py`: DB-backed LSTM trainer (defaults to all tickers in `tickers` table).
- `scripts/train_random_forest.py`, `scripts/train_xgboost.py`: Classifier training scripts.
- `scripts/run_baselines.py`, `scripts/compare_models.py`, `scripts/run_walkforward.py`: Evaluation and comparison workflows.

## Notes on Runtime Behavior
- The Streamlit UI and LLM run locally with database-backed context.
- External APIs are used by ingestion scripts and by some training scripts (RF/XGB). The LSTM training path is DB-backed.
- Llama 3 requires a local GGUF model file in `models/llama3/` to enable full chat responses.
- FinBERT model files are expected under `models/finbert/` for sentiment runs.

## Current Implementation Notes
- Walk-forward validation currently supports RF/XGB only; `--model lstm` is not implemented in `scripts/run_walkforward.py`.
- LSTM training is DB-backed and API-independent, but RF/XGB training scripts still use yfinance through `StockDataProcessor`.
- LSTM and XGBoost training scripts now reserve a dedicated validation split for early stopping; held-out test sets are no longer reused as validation during training.
- Model artifacts are saved locally under `models/`; scripts do not automatically register LSTM outputs into `predictions`/`model_performance` tables.
