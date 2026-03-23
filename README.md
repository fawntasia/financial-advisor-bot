# Financial Advisor Bot

Local, end-to-end S&P 500 analytics app with:
- SQLite-backed market/news data pipeline
- technical indicator generation
- FinBERT sentiment scoring
- Random Forest, XGBoost, and LSTM training/evaluation
- Streamlit dashboard + local LLM chat (with mock fallback)

## What Is In This Repo

Key paths in the current project:

```text
.
├── app.py
├── config/
│   └── production_model.json
├── data/
│   ├── financial_advisor.db              # local SQLite DB (gitignored)
│   ├── download_checkpoint.json
│   └── cache/, logs/, archive/
├── models/
│   ├── random_forest_global.pkl
│   ├── random_forest_global_metadata.json
│   ├── xgboost_global.json
│   ├── xgboost_global_metadata.json
│   └── lstm_*.keras / *_metadata.json / *_scalers.joblib
├── results/
│   ├── training_runs/
│   └── fresh_eval_20260323/
├── scripts/
├── src/
└── tests/
```

Notes:
- The old `docs/` folder referenced in earlier README versions is not present in this project.
- FinBERT and GGUF LLM weights are not committed (`models/finbert/`, `models/llama3/` are gitignored).

## Requirements

- Python 3.10+ (3.10.11 validated in this workspace)
- `pip`
- Git (optional)

## Setup

From repository root:

```bash
python -m venv venv
```

Activate environment:

- Windows PowerShell: `.\venv\Scripts\Activate.ps1`
- Windows CMD: `venv\Scripts\activate.bat`
- macOS/Linux: `source venv/bin/activate`

Install dependencies and initialize DB:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/init_db.py
```

## Quick Run

Launch the app:

```bash
python -m streamlit run app.py
```

If the local LLM file is missing, chat runs in mock mode (the app still works).

## End-to-End Workflow

### 1) Sync ticker universe (strict S&P 500 sync)

```bash
python scripts/update_tickers.py
```

Use `--keep-stale` if you do not want removed constituents deleted from DB.

### 2) Ingest prices + news + automatic sentiment

```bash
python scripts/ingest_data.py --news-provider auto
```

Current ingestion behavior:
- prices are incremental per ticker (new symbols backfill ~5 years)
- indicators are recomputed for updated tickers
- news defaults to 10-day lookback, 4 articles/ticker
- old news/sentiment is pruned to 10-day retention
- sentiment scoring runs automatically when FinBERT is available

### 3) Optional targeted ingestion utilities

```bash
python scripts/fetch_news.py --tickers AAPL MSFT --provider auto --days 7
python scripts/fetch_news.py --all --provider rss --days 3
python scripts/run_sentiment_analysis.py --all-unprocessed --model-path models/finbert
python scripts/calculate_indicators.py --ticker AAPL
python scripts/download_historical_data.py
```

### 4) Train models

```bash
python scripts/train_random_forest.py --output-dir models --data-source db --db-path data/financial_advisor.db
python scripts/train_xgboost.py --output-dir models --data-source db --db-path data/financial_advisor.db
python scripts/train_lstm.py --ticker ALL --epochs 50 --batch-size 32 --output-dir models --data-source db --db-path data/financial_advisor.db
```

### 5) Evaluate / backtest

```bash
python scripts/run_walkforward.py --ticker AAPL --model rf --output-dir results --data-source db --db-path data/financial_advisor.db
python scripts/backtest_models.py --ticker AAPL --model models/random_forest_global.pkl --start-date 2026-01-01 --end-date 2026-03-20 --output-dir results --data-source db --db-path data/financial_advisor.db
python scripts/run_universe_backtest.py --model xgb --db-path data/financial_advisor.db --output-dir results
```

### 6) Compare walk-forward runs and refresh production config

```bash
python scripts/compare_models.py --results_dir results --output_report results/model_selection.md --output_csv results/comparison_summary.csv --production_config config/production_model.json
```

`config/production_model.json` is used by runtime production-model resolution in the app.

## Local Model Downloads (Optional but Recommended)

### FinBERT (for sentiment scoring)

```bash
python scripts/download_finbert.py
```

Expected path: `models/finbert/`.

### GGUF LLM (for non-mock chat)

```bash
python scripts/download_llama3.py --path models/llama3/
```

The chat loader currently looks for:

```text
models/llama3/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

If your downloaded GGUF has a different filename, rename it or update `DEFAULT_MODEL_PATH` in `src/llm/llama_loader.py`.

## Script Reference

Validated CLI entrypoints in `scripts/`:

- `init_db.py`: create schema/indexes, seed fallback ticker universe
- `update_tickers.py`: sync current S&P 500 constituents from Wikipedia
- `ingest_data.py`: unified stock + indicators + news + auto-sentiment pipeline
- `fetch_news.py`: manual ticker-scoped news fetch
- `run_sentiment_analysis.py`: manual batch sentiment scoring
- `train_random_forest.py`: global classifier training (`random_forest_global.*`)
- `train_xgboost.py`: global classifier training (`xgboost_global.*`)
- `train_lstm.py`: LSTM training (`lstm_*.keras`, metadata, scalers)
- `run_walkforward.py`: walk-forward metrics JSON (`wf_results_*`)
- `backtest_models.py`: ticker backtest CSV
- `run_universe_backtest.py`: full-universe classifier backtest
- `compare_models.py`: aggregate walk-forward results + production model config
- `run_baselines.py`: buy-and-hold/random-walk/SMA baselines
- `scheduler.py`: daily ingest scheduler (`18:00`)

## Testing

Run:

```bash
python -m pytest
```

Branch status observed on 2026-03-23:
- 176 collected tests (1 skipped)
- 13 failing tests currently in this branch

So treat the suite as non-green until those failures are resolved.

## Troubleshooting

- If `streamlit`/`transformers` import behavior is inconsistent, use the project venv interpreter explicitly:
  - Windows: `.\venv\Scripts\python.exe ...`
- If chat shows mock mode, verify GGUF exists at the loader path above.
- If sentiment is skipped, verify `models/finbert/` exists and `run_sentiment_analysis.py` works.

## Disclaimer

Educational/research project only. Not financial advice.

## License

ALL RIGHTS RESERVED
