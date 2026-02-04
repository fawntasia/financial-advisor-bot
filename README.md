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

To fetch the latest stock data and news (once implemented):

```bash
python scripts/ingest_data.py
```

## Development

- **Run Tests**: `pytest`
- **Linting**: `flake8 src tests`

## License

[MIT License](LICENSE)
