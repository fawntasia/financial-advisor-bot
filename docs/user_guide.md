# User Guide

## Overview
- **Scope**: S&P 500-focused analysis and insights.
- **Interface**: Streamlit app for dashboards and an LLM-powered chat experience.
- **Runtime**: Local-only execution with no external APIs at runtime.

## Install
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize the database:

```bash
python scripts/init_db.py
```

## Run
Start the Streamlit app:

```bash
streamlit run app.py
```

## Key Features
- **Data pipeline** for S&P 500 OHLCV data and financial news ingestion.
- **Technical indicators** such as RSI, MACD, and Bollinger Bands.
- **Sentiment analysis** from financial news headlines (FinBERT-based).
- **ML forecasts** using LSTM and ensemble models.
- **Interactive UI** for exploration and chat.
- **Tabbed Stock Analysis** with:
  - `LSTM`: historical fit + forward forecast chart.
  - `Random Forest`: next-business-day direction signal cards.
  - `XGBoost`: next-business-day direction signal cards.

## Interpreting Outputs
- **Indicators**: Use technical signals as context, not single-decision triggers.
- **Model outputs**: Treat direction predictions as probabilistic signals, not guarantees.
- **Classifier signal cards**:
  - `Predicted Direction`: UP or DOWN for the next business day.
  - `UP Probability`: model probability of an upward move.
  - `Decision Threshold`: probability cutoff used for UP vs DOWN.
  - `Confidence`: probability aligned to the predicted class.
- **Backtests**: Compare strategy metrics to buy-and-hold benchmarks for context.
- **Risk metrics**: Sharpe ratio and max drawdown indicate risk-adjusted performance.

## Safety Disclaimer
- This tool is for educational and research purposes only.
- It is not financial advice and should not be relied on for trading decisions.
- Past performance does not guarantee future results.

## Data Limitations
- Coverage is limited to S&P 500 equities and available historical data.
- News sentiment reflects headline content and may miss broader context.
- Model performance can degrade in new market regimes.

## Troubleshooting
- If the app fails to start, confirm Python 3.9+ and that dependencies installed.
- Re-run database initialization if data appears missing.
