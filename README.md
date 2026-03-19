# Financial Advisor Bot

A robust, AI-powered financial advisory system focusing on the S&P 500 market. This project integrates historical market data, technical analysis, and sentiment analysis from financial news to provide data-driven investment insights.

## Features

- **Data Pipeline**: Automated ingestion of S&P 500 OHLCV data and financial news.
- **Technical Analysis**: Calculation of key indicators (RSI, MACD, Bollinger Bands, etc.).
- **Sentiment Analysis**: FinBERT-based sentiment scoring of financial news headlines.
- **Machine Learning**: LSTM and Ensemble models for trend prediction.
- **Interactive UI**: Streamlit-based dashboard for visualization and LLM-powered chat.

## Project Structure

```
├── data/               # Data storage (SQLite DB, cache)
├── docs/               # Documentation
├── scripts/            # Utility scripts (init db, ingest data)
├── src/
│   ├── data/           # Data fetching and processing
│   ├── database/       # Database Access Layer (DAL)
│   ├── features/       # Feature engineering
│   ├── models/         # ML model definitions
│   ├── nlp/            # Sentiment analysis & text processing
│   ├── ui/             # Streamlit application
│   └── utils/          # Utilities (logging, config)
├── tests/              # Unit and integration tests
├── app.py              # Main application entry point
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/init_db.py
streamlit run app.py
```

Note: For full sentiment and chat responses, download local model files to `models/finbert/` (FinBERT) and `models/llama3/` (Llama 3 GGUF).

## Documentation

- User guide: `docs/user_guide.md`
- Technical documentation: `docs/technical_documentation.md`

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fawntasia/financial-advisor-bot.git
    cd financial-advisor-bot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize the database:**
    ```bash
    python scripts/init_db.py
    ```

## Usage

### Running the Application

To start the Streamlit dashboard:

```bash
streamlit run app.py
```

### Data Ingestion

To fetch the latest S&P 500 tickers from Wikipedia:

```bash
python scripts/update_tickers.py
```

`update_tickers.py` now performs a strict sync: it keeps only current S&P 500 constituents in the `tickers` table (stale symbols are removed).

To fetch the latest stock data and news (incremental update; new symbols are backfilled with a bounded 5-year history):

```bash
python scripts/ingest_data.py
```

News ingestion defaults are now free-first:
- `--news-provider auto` (default): tries Yahoo Finance RSS first (no key required), then falls back to NewsAPI / Alpha Vantage when keys are available.
- Round-robin ticker coverage: each run ingests a bounded subset and advances a persisted cursor (`news_round_robin_cursor`) in `user_preferences`.

Examples:

```bash
# Ingest prices + news using defaults (auto provider, 25 tickers/run, 10 articles/ticker)
python scripts/ingest_data.py

# Increase daily ticker coverage while staying round-robin
python scripts/ingest_data.py --news-max-tickers 50 --news-limit 5

# Force RSS-only mode
python scripts/ingest_data.py --news-provider rss
```

For targeted manual fetches (specific symbols or full list):

```bash
# Fetch for specific symbols
python scripts/fetch_news.py --tickers AAPL MSFT --provider auto --days 7

# Fetch for all DB tickers using RSS-only mode
python scripts/fetch_news.py --all --provider rss --days 3
```

To (re)download a 5-year OHLCV window for all current S&P 500 symbols:

```bash
python scripts/download_historical_data.py
```

### Model Training and Evaluation (DB-First)

All model scripts are SQLite-first by default and support optional live data via `--data-source yfinance`.

```bash
# Random Forest (global pooled model across all available tickers)
python scripts/train_random_forest.py --output-dir models --data-source db --db-path data/financial_advisor.db

# XGBoost (global pooled model across all available tickers)
python scripts/train_xgboost.py --output-dir models --data-source db --db-path data/financial_advisor.db

# LSTM
python scripts/train_lstm.py --ticker ALL --epochs 10 --batch-size 32 --output-dir models --data-source db --db-path data/financial_advisor.db

# Walk-forward validation
python scripts/run_walkforward.py --ticker AAPL --model rf --output-dir results --data-source db --db-path data/financial_advisor.db

# Backtest
python scripts/backtest_models.py --ticker AAPL --model models/random_forest_global.pkl --start-date 2023-01-01 --end-date 2023-12-31 --output-dir results --data-source db --db-path data/financial_advisor.db
```

Detailed model docs:
- `docs/model_training.md`
- `docs/technical_documentation.md`

## Safety Disclaimer

This tool is for educational and research purposes only and is not financial advice. Past performance does not guarantee future results.

## Development

- **Run Tests**: `pytest`
- **Linting**: `flake8 src tests`

## License

ALL RIGHTS RESERVED
