# Chapter 4: Implementation

## Overview
This chapter describes how the system is implemented as a set of modular components that execute locally. The implementation centers on a SQLite-backed data layer, a pipeline for data ingestion and feature construction, model training workflows, and a Streamlit interface that integrates a local LLM for explanatory outputs.

## System Components
Implementation is organized by subsystem, with each module responsible for a specific stage of the workflow.

### Data Pipeline and Storage
- **Operational entrypoints**: Scripts in `scripts/` orchestrate ingestion and preprocessing, including `scripts/init_db.py`, `scripts/ingest_data.py`, and specialized fetchers.
- **Data fetching**: `src/data/` provides market and news clients (`src/data/yfinance_client.py`, `src/data/news_client.py`) and a stock data processor (`src/data/stock_data.py`).
- **Database access**: The SQLite database is accessed exclusively through `src/database/dal.py`, which encapsulates reads and writes to tables such as `stock_prices`, `technical_indicators`, and `news_headlines`.
- **Persistence**: Artifacts are stored locally in `data/financial_advisor.db` and `data/cache/` to keep runtime dependencies offline.

### Feature Engineering
- **Indicators**: `src/features/indicators.py` computes technical indicators (RSI, MACD, Bollinger Bands, SMA/EMA ratios).
- **Integration**: Indicator outputs are persisted through the DAL so downstream modeling and UI modules can reuse a consistent feature set.

### Sentiment Scoring (FinBERT)
- **Model loading**: `src/nlp/finbert_loader.py` loads FinBERT from local model files in `models/finbert/`.
- **Batch processing**: `src/nlp/sentiment_pipeline.py` scores headlines in batches and aggregates them into daily sentiment summaries.
- **Storage**: Aggregated sentiment is written to `daily_sentiment` in SQLite for efficient join with price features.

### Model Training and Evaluation
- **Model definitions**: `src/models/` contains the LSTM (`src/models/lstm_model.py`) and ensemble classifiers (`src/models/random_forest_model.py`, `src/models/xgboost_model.py`).
- **Training scripts**: `scripts/train_lstm.py`, `scripts/train_random_forest.py`, and `scripts/train_xgboost.py` coordinate data loading, train/validation splits, and model persistence.
- **Evaluation utilities**: `src/models/evaluation.py`, `src/models/validation.py`, and `src/models/comparison.py` implement metrics, walk-forward validation, and model selection. Baselines are defined in `src/models/baselines.py`.
- **Artifacts**: Trained models and checkpoints are saved under `models/`, and performance summaries are stored in `model_performance`.

### LLM Context Injection
- **Context assembly**: `src/llm/context_builder.py` gathers recent prices, indicators, sentiment, and model predictions from the DAL to construct a compact, ticker-specific context block.
- **Prompt templates**: `src/llm/prompts.py` defines the system prompt and user message structure that the chat UI uses for consistent responses.
- **Model runtime**: `src/llm/llama_loader.py` loads the local Llama 3 GGUF model from `models/llama3/` using llama-cpp, with a mock mode if the model is absent.
- **Guardrails**: `src/llm/guardrails.py` performs basic input and output checks and ensures responses remain grounded in available data.

### Streamlit UI
- **Entrypoint**: `app.py` initializes the Streamlit application, manages session state, and routes to views.
- **Views**: `src/ui/views.py` renders the dashboard, analysis panels, and required disclaimer content.
- **Chat interface**: `src/ui/chat.py` coordinates user prompts, context injection, LLM calls, and error handling for the conversational assistant.
- **Visualization**: `src/ui/charts.py` uses Plotly to produce price, indicator, and sentiment charts.

## Integration Flow
The system executes as a linear flow from data acquisition to end-user interaction.

1. **Database initialization**: `scripts/init_db.py` creates the schema and seeds ticker metadata.
2. **Ingestion**: `scripts/ingest_data.py` fetches OHLCV prices and news headlines, writing raw records to SQLite.
3. **Feature construction**: `src/features/indicators.py` computes indicators that are stored alongside prices.
4. **Sentiment aggregation**: `src/nlp/sentiment_pipeline.py` scores headlines with FinBERT and persists daily sentiment.
5. **Model training**: Training scripts load aligned price, indicator, and sentiment features to fit LSTM and classifier models and save outputs.
6. **Prediction storage**: Model predictions and performance metrics are stored in `predictions` and `model_performance` for reuse.
7. **Context building**: When a user selects a ticker, `src/llm/context_builder.py` assembles the latest data and model summaries.
8. **UI presentation**: Streamlit renders charts and metrics, and the chat interface injects context into the LLM prompt.

## Local-Only Runtime Considerations
- **Offline inference**: FinBERT and Llama 3 model files are loaded from local disk; runtime does not require external APIs.
- **Reproducibility**: SQLite persistence and locally saved models ensure the UI and LLM responses can be regenerated from the same artifacts.
- **Separation of concerns**: Scripts handle data ingestion and training, while the Streamlit app focuses on consumption and explanation.
