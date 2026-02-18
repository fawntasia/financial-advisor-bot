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

To fetch the latest stock data and news (incremental update or full history from 2000):

```bash
python scripts/ingest_data.py
```

### Model Training and Evaluation (DB-First)

All model scripts are SQLite-first by default and support optional live data via `--data-source yfinance`.

```bash
# Random Forest
python scripts/train_random_forest.py --ticker AAPL --output-dir models --data-source db --db-path data/financial_advisor.db

# XGBoost
python scripts/train_xgboost.py --ticker AAPL --output-dir models --data-source db --db-path data/financial_advisor.db

# LSTM
python scripts/train_lstm.py --ticker ALL --epochs 10 --batch-size 32 --output-dir models --data-source db --db-path data/financial_advisor.db

# Walk-forward validation
python scripts/run_walkforward.py --ticker AAPL --model rf --output-dir results --data-source db --db-path data/financial_advisor.db

# Backtest
python scripts/backtest_models.py --ticker AAPL --model models/random_forest_AAPL_<timestamp>.pkl --start-date 2023-01-01 --end-date 2023-12-31 --output-dir results --data-source db --db-path data/financial_advisor.db
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
