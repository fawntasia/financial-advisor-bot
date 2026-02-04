# Financial Advisor Bot - Work Plan

> **Scope**: Full S&P 500 (503 stocks) with local LLM (Llama 3) and local FinBERT sentiment analysis  
> **Architecture**: 100% local processing - no external API dependencies during operation  
> **Timeline**: 22 weeks (155 days) with 20-day buffer  
> **Success Probability**: 75% (with specified guardrails)

---

## TL;DR

**Quick Summary**: Build an AI-powered financial advisor bot for S&P 500 stocks using local Llama 3 for conversational interface, FinBERT for sentiment analysis, and ensemble ML models (LSTM, Random Forest, XGBoost) for price predictions. Fully local operation with SQLite + Streamlit.

**Deliverables**:
- Data pipeline for 503 S&P 500 stocks (5 years historical + daily updates)
- Local FinBERT sentiment analysis on financial news
- Local Llama 3 8B LLM for conversational interface
- ML models: LSTM, Random Forest, XGBoost with walk-forward validation
- Streamlit web interface with charts and chat
- SQLite database with 10+ tables
- Full test suite and backtesting framework

**Estimated Effort**: Large (22 weeks)  
**Parallel Execution**: YES - 7 execution waves  
**Critical Path**: Foundation → Data Pipeline → Baseline Models → LSTM → Model Comparison → LLM Setup → Chat Interface → Web App → Testing

---

## Context

### Original Request
Develop an intelligent financial advisor bot capable of analyzing historical and real-world financial data to generate personalized investment recommendations. Research questions: (1) Can machines predict stock trends using historical data? (2) Does sentiment analysis improve predictions? (3) Can LLM-powered conversational interface provide intuitive financial insight?

### Key Decisions Made
1. **Scope**: Full S&P 500 (503 stocks, not top 100) - for comprehensive academic coverage
2. **LLM**: Local Llama 3 8B - 100% local operation, no API costs, requires GPU (RTX 3060+)
3. **Sentiment**: FinBERT (yiyanghkust/finbert-tone) - download once from HuggingFace, run locally
4. **Database**: SQLite with WAL mode - sufficient for single-user, 5-year historical data (~150MB)
5. **Architecture**: Batch processing - pre-compute predictions daily at 6 AM
6. **Testing**: Tests-after strategy (not TDD) - pytest for unit/integration, walk-forward for model validation

### Technical Constraints
- **No external APIs during operation**: All data from local database/cache
- **GPU required**: Llama 3 8B needs ~8GB VRAM (RTX 3060 12GB or better)
- **Initial downloads**: ~8GB Llama model, ~400MB FinBERT (one-time)
- **Batch processing only**: No real-time model inference on user queries

### Research Findings
- Literature review shows 55%+ directional accuracy achievable with LSTM on S&P 500
- FinBERT outperforms general sentiment models on financial text (>75% accuracy)
- Efficient Market Hypothesis acknowledged - expectations managed for 52-55% accuracy target
- Overfitting is primary risk - mitigated via walk-forward validation

---

## Work Objectives

### Core Objective
Build a fully local financial advisor system that ingests S&P 500 data, analyzes sentiment from financial news, trains multiple ML models to predict price movements, and provides conversational advice via local Llama 3 LLM.

### Concrete Deliverables
1. `data/financial_advisor.db` - SQLite database with 10+ tables, 633K+ price records, 45K+ sentiment records
2. `models/` directory with trained models: LSTM (.h5), Random Forest (.pkl), XGBoost (.json)
3. `src/llama_chat.py` - Local Llama 3 integration with context injection
4. `src/sentiment_analyzer.py` - FinBERT batch processing pipeline
5. `src/data_pipeline.py` - yfinance ingestion with caching and retries
6. `app.py` - Streamlit web interface with chat and visualizations
7. `tests/` - pytest suite with >80% coverage on critical paths
8. `docs/dissertation/` - Final academic documentation

### Definition of Done
- [ ] All 503 S&P 500 stocks have 5 years of historical data in database
- [ ] Models achieve >52% directional accuracy on 6-month holdout test
- [ ] Llama 3 responds to financial queries in <10 seconds locally
- [ ] Streamlit app loads in <3 seconds, shows charts and chat interface
- [ ] Full test suite passes: `pytest tests/` with 0 failures
- [ ] Backtesting shows positive Sharpe ratio vs buy-and-hold benchmark

### Must Have
- Local Llama 3 8B operational for financial Q&A
- FinBERT sentiment analysis on daily financial news
- At least 3 ML models trained and compared (LSTM + 2 ensemble)
- Walk-forward validation on 6-month holdout period
- SQLite database with proper indexing and foreign keys
- Streamlit interface with real-time charts and chat
- Error handling with retry logic for all external calls (initial download only)
- 90-day rolling news retention with sentiment aggregation

### Must NOT Have (Guardrails)
- NO real-time trading execution (advisory only)
- NO cryptocurrency or forex analysis (S&P 500 only)
- NO external API calls during normal operation (yfinance has limits)
- NO model predictions without confidence intervals
- NO LLM responses without financial disclaimers
- NO user data storage (local single-user app)
- NO cloud dependencies after initial setup (fully local operation)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (will set up pytest)
- **Automated tests**: YES (Tests-after, not TDD)
- **Framework**: pytest with coverage reporting
- **Test data**: Synthetic test fixtures + subset of real data

### Test Setup Task
- [ ] 0. Setup Test Infrastructure
  - Install: `pip install pytest pytest-cov pytest-asyncio`
  - Config: Create `pytest.ini` with test paths and coverage settings
  - Verify: `pytest --version` shows pytest installed
  - Example: Create `tests/test_example.py` with simple assertion
  - Verify: `pytest tests/test_example.py` → 1 test passes

### Agent-Executed QA Scenarios (MANDATORY)

All verification executed by agent using appropriate tools. No human intervention required.

**Verification Tool Matrix:**

| Deliverable | Tool | How Agent Verifies |
|-------------|------|-------------------|
| Database | Bash (sqlite3 CLI) | Query tables, verify counts, check constraints |
| Data Pipeline | Bash (python scripts) | Run ingestion, validate CSV outputs, check logs |
| ML Models | Bash (python REPL) | Load model, run inference, check accuracy metrics |
| FinBERT | Bash (python scripts) | Process test headlines, verify sentiment scores |
| Llama 3 | Bash (python scripts) | Send test prompts, validate response quality |
| Streamlit | Playwright | Load UI, interact with chat, verify charts render |
| API/Backend | Bash (curl) | Test endpoints if any HTTP interfaces exist |

**Each Scenario Format:**
```
Scenario: [Descriptive name]
  Tool: [Bash / Playwright / etc]
  Preconditions: [What must be true before scenario]
  Steps:
    1. [Exact command with arguments]
    2. [Next command or assertion]
    3. [Expected output or state]
  Expected Result: [Concrete, observable outcome]
  Failure Indicators: [What indicates failure]
  Evidence: [Output file path or screenshot]
```

---

## Execution Strategy

### Parallel Execution Waves

**Wave 1 (Week 1-2): Foundation**
- Task 1: Project setup, virtual environment, git init
- Task 2: Database schema creation (SQL DDL)
- Task 3: Test infrastructure setup

**Wave 2 (Week 3-4): Data Pipeline Core**
- Task 4: yfinance wrapper with caching
- Task 5: Historical data ingestion (backfill 5 years)
- Task 6: Database access layer (DAL)

**Wave 3 (Week 5-6): Data Pipeline Advanced**
- Task 7: Technical indicators calculation (RSI, MACD, BB)
- Task 8: News pipeline with rate limiting
- Task 9: Error handling and monitoring

**Wave 4 (Week 7-10): ML Models**
- Task 10: Baseline models (Buy & Hold, Random Walk)
- Task 11: LSTM architecture and training
- Task 12: Random Forest implementation
- Task 13: XGBoost implementation

**Wave 5 (Week 11-12): Model Validation & FinBERT**
- Task 14: Walk-forward validation framework
- Task 15: FinBERT download and setup
- Task 16: Sentiment analysis pipeline
- Task 17: Model comparison and selection

**Wave 6 (Week 13-15): LLM Integration**
- Task 18: Llama 3 download and setup (8GB model)
- Task 19: Prompt engineering and templates
- Task 20: Context injection (predictions + sentiment)
- Task 21: Response guardrails and fact-checking

**Wave 7 (Week 16-19): Web Application**
- Task 22: Streamlit core interface
- Task 23: Data visualization (Plotly candlesticks)
- Task 24: Chat interface integration
- Task 25: UX polish and error handling

**Wave 8 (Week 20-22): Testing & Documentation**
- Task 26: Unit tests for all modules
- Task 27: Integration tests (end-to-end)
- Task 28: Model backtesting and validation
- Task 29: Documentation and dissertation

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 (Setup) | None | 2, 3, 4 | None |
| 2 (Database) | 1 | 4, 5, 6 | 3 |
| 3 (Tests) | 1 | 26, 27, 28 | 2 |
| 4 (yfinance) | 1, 2 | 5, 6 | 3 |
| 5 (Historical) | 4 | 7, 8, 10 | 6 |
| 6 (DAL) | 2 | 5, 7, 8, 10 | 4 |
| 7 (Indicators) | 5, 6 | 11, 12, 13 | 8, 9, 10 |
| 8 (News) | 2 | 15, 16 | 7, 9, 10 |
| 9 (Error Handling) | 4, 5 | 11, 12, 13, 18 | 7, 8 |
| 10 (Baselines) | 5, 6, 7 | 14 | 8, 9 |
| 11 (LSTM) | 7, 10 | 14, 17 | 12, 13 |
| 12 (RF) | 7, 10 | 14, 17 | 11, 13 |
| 13 (XGBoost) | 7, 10 | 14, 17 | 11, 12 |
| 14 (Validation) | 10, 11, 12, 13 | 17 | 15, 16 |
| 15 (FinBERT Setup) | 2 | 16 | 14 |
| 16 (Sentiment) | 8, 15 | 17, 20 | 14 |
| 17 (Model Compare) | 11, 12, 13, 14, 16 | 20, 22 | 18, 19 |
| 18 (Llama Setup) | 1 | 19, 20, 21 | 17 |
| 19 (Prompts) | 18 | 20, 21 | 17 |
| 20 (Context Injection) | 16, 17, 19 | 21, 24 | None |
| 21 (Guardrails) | 18, 19, 20 | 24 | 22, 23 |
| 22 (Streamlit Core) | 17 | 23, 24, 25 | 21 |
| 23 (Visualizations) | 17 | 25 | 21, 22 |
| 24 (Chat UI) | 20, 21, 22 | 25 | 23 |
| 25 (UX Polish) | 22, 23, 24 | 27, 28 | 26 |
| 26 (Unit Tests) | 3, 11, 12, 13, 16, 18 | 28 | 25, 27 |
| 27 (Integration) | 3, 22, 23, 24, 25 | 28 | 25, 26 |
| 28 (Backtesting) | 14, 25, 26, 27 | 29 | None |
| 29 (Docs) | 28 | None | None |

### Critical Path
```
Setup (1) → Database (2) → yfinance (4) → Historical (5) → Indicators (7) → Baselines (10) 
→ LSTM (11) → Validation (14) → Model Compare (17) → Context Injection (20) 
→ Chat UI (24) → Backtesting (28) → Documentation (29)

Total Critical Path: ~105 days (15 weeks)
```

### Agent Dispatch Summary

| Wave | Tasks | Recommended Category | Skills |
|------|-------|---------------------|--------|
| 1 | 1, 2, 3 | quick | python, bash |
| 2 | 4, 5, 6 | quick | python, data-pipeline |
| 3 | 7, 8, 9 | quick | python, data-engineering |
| 4 | 10, 11, 12, 13 | ultrabrain | python, tensorflow, sklearn |
| 5 | 14, 15, 16, 17 | ultrabrain | python, ml-validation, nlp |
| 6 | 18, 19, 20, 21 | ultrabrain | python, llm, prompt-engineering |
| 7 | 22, 23, 24, 25 | visual-engineering | python, streamlit, plotly |
| 8 | 26, 27, 28, 29 | quick | python, pytest, technical-writing |

---

## TODOs

- [x] 1. Project Setup and Environment Configuration

  **What to do**:
  - Create Python 3.9+ virtual environment
  - Initialize git repository with .gitignore
  - Create project directory structure: src/, tests/, data/{raw,processed,archive}, models/, docs/
  - Create requirements.txt with all dependencies (see list below)
  - Set up logging configuration (structlog or standard logging)
  - Create .env.example file for configuration (API keys for initial download only)

  **Dependencies for requirements.txt**:
  ```
  # Core
  python-dotenv>=1.0.0
  pandas>=2.0.0
  numpy>=1.24.0
  
  # Database
  sqlalchemy>=2.0.0
  
  # Data & Finance
  yfinance>=0.2.28
  pandas-ta>=0.3.14b
  
  # ML/DL
  tensorflow>=2.13.0
  scikit-learn>=1.3.0
  xgboost>=2.0.0
  
  # NLP & LLM
  transformers>=4.30.0
  torch>=2.0.0
  sentencepiece>=0.1.99
  accelerate>=0.20.0
  
  # Web App
  streamlit>=1.28.0
  plotly>=5.17.0
  mplfinance>=0.12.9
  
  # Testing
  pytest>=7.4.0
  pytest-cov>=4.1.0
  pytest-asyncio>=0.21.0
  
  # Utilities
  requests>=2.31.0
  tqdm>=4.66.0
  schedule>=1.2.0
  pyyaml>=6.0.1
  ```

  **Must NOT do**:
  - Don't install unnecessary dependencies (keep it lean)
  - Don't commit .env file with real keys (use .env.example template)
  - Don't skip virtual environment (avoid dependency conflicts)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Standard project setup, no complex logic
  - **Skills**: python, bash
    - python: Package management, virtualenv creation
    - bash: Directory creation, git init

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 2, 3, 4, 5, 6
  - **Blocked By**: None

  **References**:
  - No code references needed (starting from scratch)
  - External: Python virtualenv best practices, requirements.txt format

  **Acceptance Criteria**:
  - [ ] Virtual environment created and activated
  - [ ] `pip list` shows all dependencies installed
  - [ ] `git status` shows clean working directory
  - [ ] All directories exist: `ls src/ tests/ data/raw data/processed models/`
  - [ ] `python -c "import pandas, tensorflow, transformers, streamlit"` executes without errors

  **Agent-Executed QA Scenarios**:

  Scenario: Verify project structure and dependencies
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run: `python --version` → Assert output contains "3.9" or higher
      2. Run: `pip list | grep -E "(pandas|tensorflow|transformers|streamlit)"` → Assert all 4 packages present
      3. Run: `ls -la src/ tests/ data/raw data/processed models/` → Assert all directories exist
      4. Run: `git status` → Assert "nothing to commit, working tree clean" or similar
      5. Run: `python -c "import pandas; print('pandas:', pandas.__version__)"` → Assert no ImportError
    Expected Result: All dependencies installed, directories created, git initialized
    Evidence: Terminal output captured in .sisyphus/evidence/task-1-setup-verification.txt

  **Commit**: YES
  - Message: `chore(setup): initialize project structure and dependencies`
  - Files: `.gitignore`, `requirements.txt`, `.env.example`, directory structure
  - Pre-commit: Verify all files committed, no secrets in .env files

---

- [x] 2. Database Schema Creation and Initialization
- [x] 3. Test Infrastructure Setup
- [x] 4. yfinance API Wrapper with Caching

  **What to do**:
  - Create `src/data/yfinance_client.py` - wrapper around yfinance library
  - Implement exponential backoff retry logic for API failures (max 5 retries: 1s, 2s, 4s, 8s, 16s delays)
  - Add disk-based caching using `diskcache` library or simple pickle files
  - Cache strategy: Store raw API responses in `data/cache/yfinance/`
  - Cache TTL: 1 day for recent data, indefinite for historical
  - Implement rate limiting: 0.5 second delay between requests (yfinance is free but polite to be gentle)
  - Add error handling for common failures: empty data, network errors, invalid tickers
  - Create batch download function for multiple tickers
  - Add logging for all API calls (success and failure)

  **yfinance Notes**:
  - Library: `yfinance` (not an API key service, but makes HTTP requests)
  - Rate limits: Not strictly enforced but 0.5s delay is polite
  - Caching essential to avoid re-downloading same data
  - Handle delisted tickers gracefully (some S&P 500 constituents change)

  **Must NOT do**:
  - Don't hammer yfinance without delays (risk of IP ban)
  - Don't cache errors (only cache successful responses)
  - Don't fail entire batch if one ticker fails (continue with others)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: API wrapper is straightforward, mostly error handling
  - **Skills**: python, data-pipeline
    - python: requests handling, caching implementation
    - data-pipeline: Retry logic, batch processing patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Task 5 (Historical Data Ingestion)
  - **Blocked By**: Task 2 (Database must exist first)

  **References**:
  - Library: yfinance documentation on PyPI
  - Pattern: Circuit breaker pattern, exponential backoff
  - External: diskcache library docs, retry decorator patterns

  **Acceptance Criteria**:
  - [ ] Can fetch single ticker data: `python -c "from src.data.yfinance_client import YFinanceClient; c = YFinanceClient(); print(c.get_ticker('AAPL'))"`
  - [ ] Retries work: Mock failure and verify 5 retry attempts with exponential backoff
  - [ ] Caching works: Second fetch of same ticker returns cached data (faster)
  - [ ] Batch download works: Can fetch 10 tickers in one call
  - [ ] Logs created: Check `data/logs/` for yfinance API call logs

  **Agent-Executed QA Scenarios**:

  Scenario: Verify yfinance client with caching
    Tool: Bash (python one-liners)
    Preconditions: Task 2 complete, database exists
    Steps:
      1. Run: `python -c "from src.data.yfinance_client import YFinanceClient; c = YFinanceClient(); df = c.get_ticker('AAPL', period='5d'); print(f'Rows: {len(df)}'); print(df.head())"` → Assert returns DataFrame with 5 rows
      2. Run: `ls data/cache/yfinance/ | head -5` → Assert cache files exist
      3. Run: `python -c "from src.data.yfinance_client import YFinanceClient; c = YFinanceClient(); c.get_ticker('INVALID_TICKER_123')" 2>&1` → Assert handles gracefully without crashing
      4. Run: `grep -r "yfinance" data/logs/ | head -3` → Assert log entries exist
      5. Run: `python -c "from src.data.yfinance_client import YFinanceClient; c = YFinanceClient(); import time; start = time.time(); c.get_ticker('MSFT', period='5d'); end = time.time(); print(f'Time: {end-start}s')"` (run twice) → Assert second run is faster due to cache
    Expected Result: yfinance client works, caching functional, errors handled gracefully
    Evidence: Terminal output captured in .sisyphus/evidence/task-4-yfinance-verification.txt

  **Commit**: YES
  - Message: `feat(data): implement yfinance client with caching and retry logic`
  - Files: `src/data/yfinance_client.py`, `src/data/__init__.py`
  - Pre-commit: Run unit tests for yfinance client

---

- [ ] 5. Historical Data Ingestion (5-Year Backfill)

  **What to do**:
  - Create `scripts/download_historical_data.py` - batch download script
  - Download 5 years of OHLCV data for all 503 S&P 500 tickers
  - Date range: 2019-01-01 to 2024-01-01 (5 years = ~1,260 trading days)
  - Total expected records: ~503 × 1,260 = ~633,780 price records
  - Store in `stock_prices` table with proper validation
  - Implement progress tracking with tqdm (show ETA)
  - Add checkpoint/resume capability (save progress every 10 tickers)
  - Handle failures gracefully: Log failed tickers, continue with others, retry failed at end
  - Expected runtime: 2-4 hours (with 0.5s delays between API calls)
  - Add data validation: Check for missing values, zero volumes, outlier prices
  - Create summary report: Total records, missing tickers, data quality issues

  **Data Validation Rules**:
  - No NaN values in OHLCV columns
  - Volume > 0 for trading days
  - High >= Low (obviously)
  - Price changes < 50% day-to-day (else flag as possible split)
  - No duplicate (ticker, date) combinations

  **Must NOT do**:
  - Don't run without checkpointing (system crash = lose all progress)
  - Don't insert invalid data (validate before INSERT)
  - Don't skip error logging (need to know which tickers failed)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Batch script, mostly orchestration and error handling
  - **Skills**: python, data-pipeline
    - python: Batch processing, progress tracking
    - data-pipeline: Data validation, checkpoint patterns

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 4, but Task 6 can parallel with this)
  - **Parallel Group**: Sequential in Wave 2
  - **Blocks**: Tasks 7, 8, 10 (all need historical data)
  - **Blocked By**: Task 4 (yfinance client), Task 2 (database)

  **References**:
  - Data: yfinance period='5y' parameter
  - Pattern: Checkpoint/resume pattern, batch validation
  - External: tqdm documentation for progress bars

  **Acceptance Criteria**:
  - [ ] `python scripts/download_historical_data.py` completes without errors
  - [ ] `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM stock_prices"` returns >600,000
  - [ ] `sqlite3 data/financial_advisor.db "SELECT COUNT(DISTINCT ticker) FROM stock_prices"` returns 503
  - [ ] `ls data/logs/` contains download report with failed tickers (if any)
  - [ ] Checkpoint file exists: `data/download_checkpoint.json`

  **Agent-Executed QA Scenarios**:

  Scenario: Verify historical data ingestion
    Tool: Bash
    Preconditions: Tasks 2, 4 complete
    Steps:
      1. Run: `python scripts/download_historical_data.py` → Assert completes, check logs for progress
      2. Run: `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM stock_prices"` → Assert returns number >600000
      3. Run: `sqlite3 data/financial_advisor.db "SELECT ticker, COUNT(*) as cnt FROM stock_prices GROUP BY ticker ORDER BY cnt DESC LIMIT 5"` → Assert shows tickers with ~1260 records each
      4. Run: `sqlite3 data/financial_advisor.db "SELECT COUNT(DISTINCT ticker) FROM stock_prices"` → Assert returns 503
      5. Run: `ls data/download_checkpoint.json` → Assert checkpoint file exists
      6. Run: `cat data/logs/download_report_*.txt` → Assert report shows total records, failed tickers list
    Expected Result: 5 years of data for all 503 S&P 500 tickers downloaded and validated
    Evidence: Database query results captured, log files saved

  **Commit**: YES (after completion, may be large commit)
  - Message: `feat(data): download 5 years of historical data for S&P 500 (633K+ records)`
  - Files: `scripts/download_historical_data.py`, `data/financial_advisor.db`, `data/logs/`
  - Pre-commit: Verify data integrity, database not corrupted

---

- [x] 6. Database Access Layer (DAL) Implementation

  **What to do**:
  - Create `src/database/dal.py` - Data Access Layer class
  - Implement CRUD operations for all tables
  - Add bulk insert methods for efficient data loading
  - Create query helpers: get_prices(ticker, start_date, end_date), get_latest_sentiment(ticker), etc.
  - Add transaction support (begin, commit, rollback)
  - Implement connection pooling (SQLite can use single connection with WAL mode)
  - Add utility methods: get_all_tickers(), get_missing_data_dates(), get_data_summary()
  - Create query logging for slow queries (>100ms)
  - Add data retention methods: archive_old_news(), cleanup_logs()
  - Write unit tests for all DAL methods

  **DAL Interface Methods**:
  ```python
  class DataAccessLayer:
      def get_stock_prices(self, ticker, start_date, end_date) -> pd.DataFrame
      def bulk_insert_prices(self, records: List[Dict])
      def get_technical_indicators(self, ticker, date) -> Dict
      def insert_sentiment_scores(self, scores: List[Dict])
      def get_daily_sentiment(self, ticker, date) -> Dict
      def insert_prediction(self, prediction: Dict)
      def get_model_performance(self, model_name) -> List[Dict]
      def log_system_event(self, level, component, message)
  ```

  **Must NOT do**:
  - Don't use raw SQL without parameterization (SQL injection risk)
  - Don't hold transactions open too long (blocks other operations)
  - Don't skip index usage (queries will be slow with 633K+ rows)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Database wrapper pattern is standard
  - **Skills**: python, sqlite
    - python: Class design, SQLAlchemy or sqlite3
    - sqlite: Transaction management, query optimization

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Tasks 7, 8, 11, 12, 13 (all need DAL)
  - **Blocked By**: Task 2 (database schema)

  **References**:
  - Pattern: Repository pattern, DAL abstraction
  - Code: SQLAlchemy ORM or raw sqlite3 with context managers
  - External: SQLite best practices, SQLAlchemy documentation

  **Acceptance Criteria**:
  - [ ] All CRUD methods implemented and tested
  - [ ] `python -c "from src.database.dal import DataAccessLayer; dal = DataAccessLayer(); print(dal.get_all_tickers()[:5])"` returns tickers
  - [ ] Bulk insert of 1000 records completes in <5 seconds
  - [ ] Unit tests pass: `pytest tests/unit/test_dal.py -v`
  - [ ] Query performance: `SELECT * FROM stock_prices WHERE ticker='AAPL'` completes in <100ms

  **Agent-Executed QA Scenarios**:

  Scenario: Verify DAL operations
    Tool: Bash (python)
    Preconditions: Task 2 complete
    Steps:
      1. Run: `python -c "from src.database.dal import DataAccessLayer; dal = DataAccessLayer(); tickers = dal.get_all_tickers(); print(f'Total tickers: {len(tickers)}'); print(tickers[:3])"` → Assert returns 503 tickers
      2. Run: `pytest tests/unit/test_dal.py -v` → Assert all tests pass
      3. Run: `python -c "
import time
from src.database.dal import DataAccessLayer
dal = DataAccessLayer()
start = time.time()
result = dal.get_stock_prices('AAPL', '2023-01-01', '2023-12-31')
end = time.time()
print(f'Query time: {end-start:.3f}s, Rows: {len(result)}')
"` → Assert query time <0.1s, returns ~252 rows (1 year)
      4. Run: `ls tests/unit/test_dal.py` → Assert DAL test file exists
    Expected Result: DAL fully functional, all operations tested, queries fast
    Evidence: Test output captured, query timing logged

  **Commit**: YES
  - Message: `feat(database): implement Data Access Layer with CRUD operations`
  - Files: `src/database/dal.py`, `tests/unit/test_dal.py`
  - Pre-commit: Run DAL unit tests, verify all pass

---

- [x] 7. Technical Indicators Calculation Pipeline

  **What to do**:
  - Create `src/features/indicators.py` - technical indicators calculation module
  - Calculate indicators for all 503 tickers on daily basis:
    - RSI (14-day)
    - MACD (12, 26, 9)
    - Bollinger Bands (20-day, 2 std dev)
    - SMA (20, 50, 200-day)
    - EMA (12, 26-day)
    - ATR (14-day)
    - Volume SMA (20-day)
    - Price Rate of Change (10-day)
  - Use pandas-ta library for efficient calculation
  - Store results in `technical_indicators` table
  - Calculate indicators after market close (6 PM daily)
  - Create lagged features: t-1, t-2, t-3 values for time-series models
  - Add normalization: Min-max scaling per ticker (fit on training data, save scalers)
  - Create feature importance analysis (correlation with price movement)
  - Write tests for each indicator calculation
  - Create feature engineering pipeline script: `scripts/calculate_indicators.py`

  **Libraries**:
  - Primary: pandas-ta (comprehensive, well-tested)
  - Alternative: ta-lib (faster but harder to install)
  - Use pandas-ta for ease of use

  **Must NOT do**:
  - Don't calculate indicators for entire dataset every time (only new data)
  - Don't use future data in training (prevent data leakage)
  - Don't skip handling of edge cases (first 200 days won't have 200-day SMA)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Feature engineering is formulaic but requires precision
  - **Skills**: python, data-engineering
    - python: Pandas operations, vectorized calculations
    - data-engineering: Feature pipelines, data validation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9)
  - **Blocks**: Tasks 11, 12, 13 (ML models need indicators)
  - **Blocked By**: Tasks 5, 6 (historical data and DAL)

  **References**:
  - Library: pandas-ta documentation, indicator formulas
  - Pattern: Feature store pattern, incremental computation
  - External: Technical analysis best practices

  **Acceptance Criteria**:
  - [ ] All 8 indicators calculated for AAPL (test case)
  - [ ] `python scripts/calculate_indicators.py` completes for all 503 tickers
  - [ ] `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM technical_indicators"` returns ~633K records
  - [ ] Indicators validated: RSI between 0-100, MACD calculations correct
  - [ ] Feature correlation analysis saved to `data/feature_analysis.csv`

  **Agent-Executed QA Scenarios**:

  Scenario: Verify technical indicators calculation
    Tool: Bash (python)
    Preconditions: Task 5 complete (historical data exists)
    Steps:
      1. Run: `python -c "from src.features.indicators import calculate_indicators; import pandas as pd; df = calculate_indicators('AAPL'); print(df[['RSI_14', 'MACD_12_26', 'BBL_20_2.0']].tail())"` → Assert shows calculated indicators
      2. Run: `python scripts/calculate_indicators.py --ticker AAPL` → Assert completes successfully
      3. Run: `sqlite3 data/financial_advisor.db "SELECT * FROM technical_indicators WHERE ticker='AAPL' LIMIT 5"` → Assert shows indicators stored
      4. Run: `python -c "
from src.database.dal import DataAccessLayer
dal = DataAccessLayer()
indicators = dal.get_technical_indicators('AAPL', '2023-06-01')
print(f'RSI: {indicators[\"rsi_14\"]}')
assert 0 <= indicators['rsi_14'] <= 100, 'RSI out of range'
print('RSI validation passed')
"` → Assert RSI in valid range
      5. Run: `ls data/feature_analysis.csv` → Assert feature analysis file exists
    Expected Result: All indicators calculated and stored, data validated
    Evidence: Calculation outputs captured, validation checks logged

  **Commit**: YES
  - Message: `feat(infra): implement error handling, retry logic, and monitoring framework`
  - Files: `src/utils/error_handler.py`, `src/utils/retry.py`, `src/utils/logger.py`, `src/utils/health_check.py`, `tests/unit/test_error_handler.py`
  - Pre-commit: Run error handling tests

---

- [x] 10. Baseline Models Implementation

  **What to do**:
  - Create `src/models/baselines.py` - baseline strategy implementations
  - Implement 3 baseline models for comparison:
    1. **Buy and Hold**: Buy on day 1, hold until end, calculate total return
    2. **Random Walk**: Predict next-day direction randomly (50/50), track accuracy
    3. **SMA Crossover**: Buy when 20-day SMA > 50-day SMA, sell when opposite
  - Create evaluation framework: `src/models/evaluation.py`
  - Calculate metrics for all baselines: total return, Sharpe ratio, max drawdown, directional accuracy
  - Run baselines on 2023 holdout data (unseen during development)
  - Document baseline performance in `docs/baseline_results.md`
  - Create comparison plots: cumulative returns over time
  - These baselines are REQUIRED before building fancy ML models - proves complex models are actually better

  **Why Baselines Matter**:
  - If LSTM achieves 55% accuracy, but SMA crossover achieves 54%, is the complexity worth it?
  - Sets realistic expectations for dissertation committee
  - Shows you understand simple approaches before complex ones

  **Must NOT do**:
  - Don't skip baselines (academic requirement)
  - Don't optimize baselines (they're meant to be simple)
  - Don't use future data in baseline calculations (no lookahead bias)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Baseline logic is straightforward calculations
  - **Skills**: python, data-analysis
    - python: Pandas calculations, metric computation
    - data-analysis: Financial metrics (Sharpe, drawdown)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 11, 12, 13)
  - **Blocks**: Task 14 (Validation needs baselines for comparison)
  - **Blocked By**: Tasks 5, 6, 7 (historical data, DAL, indicators)

  **References**:
  - Metrics: Sharpe ratio formula, max drawdown calculation
  - Pattern: Backtesting.py library or custom implementation
  - External: Financial metrics calculation best practices

  **Acceptance Criteria**:
  - [ ] All 3 baselines implemented and run on 2023 data
  - [ ] Buy-and-Hold shows realistic S&P 500 return (~10-20% for 2023)
  - [ ] Random Walk shows ~50% directional accuracy (as expected)
  - [ ] SMA Crossover performance documented
  - [ ] Results saved to `docs/baseline_results.md` with charts

  **Agent-Executed QA Scenarios**:

  Scenario: Verify baseline models
    Tool: Bash (python)
    Preconditions: Task 7 complete (indicators calculated)
    Steps:
      1. Run: `python -c "from src.models.baselines import BuyAndHold; bah = BuyAndHold(); result = bah.evaluate('AAPL', '2023-01-01', '2023-12-31'); print(f'Return: {result[\"total_return\"]:.2%}, Sharpe: {result[\"sharpe_ratio\"]:.2f}')"` → Assert shows reasonable return
      2. Run: `python scripts/run_baselines.py` → Assert runs all baselines on all tickers
      3. Run: `ls docs/baseline_results.md` → Assert baseline report created
      4. Run: `ls data/baselines/` → Assert baseline performance CSVs created
      5. Run: `python -c "from src.models.evaluation import compare_baselines; compare_baselines()"` → Assert generates comparison chart
    Expected Result: Baselines run, metrics calculated, results documented
    Evidence: Baseline results captured, charts generated

  **Commit**: YES
  - Message: `feat(models): implement baseline strategies (Buy&Hold, RandomWalk, SMA Crossover)`
  - Files: `src/models/baselines.py`, `src/models/evaluation.py`, `scripts/run_baselines.py`, `docs/baseline_results.md`
  - Pre-commit: Run baseline calculations, verify reasonable outputs

---

- [x] 11. LSTM Model Development

  **What to do**:
  - Create `src/models/lstm_model.py` - LSTM architecture and training
  - Design LSTM architecture:
    - Input: 60-day sequences of (price, volume, technical indicators)
    - Layers: 2 LSTM layers (128 units, 64 units) + Dropout (0.2)
    - Output: Next-day price prediction (regression) or direction (classification)
    - Loss: MSE for regression, Binary Crossentropy for classification
  - Implement data windowing: Create sequences of 60 days → predict day 61
  - Add early stopping: Stop if validation loss doesn't improve for 5 epochs
  - Add model checkpointing: Save best model during training
  - Implement walk-forward validation (time-series specific, not random split)
  - Add overfitting detection: Monitor train/val accuracy gap
  - Train on 2019-2021 data, validate on 2022, test on 2023
  - Save trained model to `models/lstm_YYYYMMDD.h5`
  - Create prediction function for inference
  - Document hyperparameters and training time

  **Training Data Setup**:
  - Features: Close price, Volume, RSI, MACD, BB position, SMA ratios
  - Sequence length: 60 days (lookback window)
  - Target: Day 61 price (or direction: up/down)
  - Normalization: Min-max scaler fit on training data only

  **Must NOT do**:
  - Don't use random train/test split (time-series requires chronological split)
  - Don't train without validation set (guaranteed overfitting)
  - Don't save overfitted models (use early stopping)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: LSTM requires deep learning expertise, careful architecture design
  - **Skills**: python, tensorflow
    - python: Time-series preprocessing
    - tensorflow: LSTM layers, training loops, callbacks

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 10, 12, 13)
  - **Blocks**: Tasks 14, 17 (validation and comparison need LSTM)
  - **Blocked By**: Tasks 7, 10 (indicators and baselines)

  **References**:
  - Library: TensorFlow/Keras LSTM documentation
  - Pattern: Time-series cross-validation, sequence generation
  - External: LSTM for stock prediction academic papers (cite in code)

  **Acceptance Criteria**:
  - [ ] LSTM model trains without errors on training data
  - [ ] Model file saved: `models/lstm_*.h5` exists and loads successfully
  - [ ] Validation loss tracked: Early stopping triggered appropriately
  - [ ] Directional accuracy on 2023 test set > 50% (beats random)
  - [ ] Training time documented (hours/epochs)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify LSTM training
    Tool: Bash (python)
    Preconditions: Task 7 complete (indicators ready)
    Steps:
      1. Run: `python -c "from src.models.lstm_model import LSTMModel; model = LSTMModel(); print(model.summary())"` → Assert shows LSTM architecture
      2. Run: `python scripts/train_lstm.py --ticker AAPL --epochs 5` (quick test) → Assert trains without crash
      3. Run: `ls models/lstm_*.h5` → Assert model file created
      4. Run: `python -c "
from src.models.lstm_model import LSTMModel
import tensorflow as tf
model = LSTMModel()
model.load('models/lstm_YYYYMMDD.h5')
test_acc = model.evaluate('AAPL', '2023-01-01', '2023-12-31')
print(f'Test accuracy: {test_acc:.2%}')
assert test_acc > 0.50, 'Accuracy below random'
"` → Assert accuracy > 50%
      5. Run: `cat logs/lstm_training.log | tail -20` → Assert shows training history
    Expected Result: LSTM model trained, accuracy > 50%, logs complete
    Evidence: Model file exists, training logs captured, accuracy verified

  **Commit**: YES (after training completes - large file)
  - Message: `feat(models): implement and train LSTM model with early stopping and checkpointing`
  - Files: `src/models/lstm_model.py`, `scripts/train_lstm.py`, `models/lstm_*.h5`
  - Pre-commit: Verify model loads, can make predictions

---

- [x] 12. Random Forest Model Implementation

  **What to do**:
  - Create `src/models/random_forest_model.py` - Random Forest classifier
  - Use scikit-learn RandomForestClassifier
  - Features: Technical indicators (RSI, MACD, BB, SMA ratios) + sentiment score
  - Target: Next-day direction (UP=1, DOWN=0)
  - Hyperparameters to tune:
    - n_estimators: 100-500 trees
    - max_depth: 10-30 (prevent overfitting)
    - min_samples_split: 5-20
    - min_samples_leaf: 2-10
  - Use GridSearchCV or RandomizedSearchCV for hyperparameter tuning
  - Implement feature importance extraction (which indicators matter most)
  - Train on 2019-2021, validate on 2022, test on 2023 (same as LSTM)
  - Save model to `models/random_forest_YYYYMMDD.pkl`
  - Create prediction function
  - Document hyperparameters and feature importance

  **Advantages of Random Forest**:
  - Faster training than LSTM (minutes vs hours)
  - Feature importance (interpretability)
  - Less prone to overfitting with proper tuning
  - Handles mixed feature types well

  **Must NOT do**:
  - Don't use default hyperparameters (will overfit)
  - Don't skip hyperparameter tuning
  - Don't ignore feature importance (key for dissertation)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: ML model implementation with hyperparameter optimization
  - **Skills**: python, sklearn
    - python: Feature preparation
    - sklearn: RandomForest, GridSearchCV, model persistence

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 10, 11, 13)
  - **Blocks**: Tasks 14, 17
  - **Blocked By**: Tasks 7, 10

  **References**:
  - Library: scikit-learn RandomForestClassifier documentation
  - Pattern: Hyperparameter tuning with cross-validation
  - External: Random Forest for stock prediction best practices

  **Acceptance Criteria**:
  - [ ] Random Forest trains in <30 minutes for all 503 tickers
  - [ ] Model file saved: `models/random_forest_*.pkl` exists
  - [ ] Feature importance extracted and documented
  - [ ] Directional accuracy on 2023 test set > 50%
  - [ ] Hyperparameters documented

  **Agent-Executed QA Scenarios**:

  Scenario: Verify Random Forest model
    Tool: Bash (python)
    Preconditions: Task 7 complete
    Steps:
      1. Run: `python -c "from src.models.random_forest_model import RandomForestModel; rf = RandomForestModel(); print(rf.get_params())"` → Assert shows hyperparameters
      2. Run: `python scripts/train_random_forest.py --ticker AAPL` → Assert trains successfully
      3. Run: `ls models/random_forest_*.pkl` → Assert model file created
      4. Run: `python -c "
from src.models.random_forest_model import RandomForestModel
rf = RandomForestModel()
rf.load('models/random_forest_YYYYMMDD.pkl')
importances = rf.get_feature_importance()
print('Top 5 features:')
for feat, imp in list(importances.items())[:5]:
    print(f'  {feat}: {imp:.3f}')
"` → Assert shows feature importance
      5. Run: `cat docs/model_features.md | grep -A 10 "Random Forest"` → Assert feature importance documented
    Expected Result: RF model trained, feature importance extracted, accuracy > 50%
    Evidence: Model file saved, importance plot generated

  **Commit**: YES
  - Message: `feat(models): implement Random Forest classifier with hyperparameter tuning`
  - Files: `src/models/random_forest_model.py`, `scripts/train_random_forest.py`, `models/random_forest_*.pkl`
  - Pre-commit: Verify model trains and persists

---

- [ ] 13. XGBoost Model Implementation

  **What to do**:
  - Create `src/models/xgboost_model.py` - XGBoost classifier
  - Use xgboost.XGBClassifier
  - Same features as Random Forest (technical indicators + sentiment)
  - Target: Next-day direction (UP/DOWN)
  - Hyperparameters to tune:
    - learning_rate: 0.01-0.3
    - n_estimators: 100-1000
    - max_depth: 3-10 (XGBoost needs lower than RF)
    - subsample: 0.6-1.0 (prevent overfitting)
    - colsample_bytree: 0.6-1.0
  - Use XGBoost's built-in early stopping
  - Add learning rate decay
  - Train/validate/test split same as other models
  - Save model to `models/xgboost_YYYYMMDD.json`
  - Create prediction function
  - Compare performance with Random Forest (speed vs accuracy)

  **Advantages of XGBoost**:
  - Often better accuracy than Random Forest (gradient boosting)
  - Built-in regularization (L1/L2)
  - Feature importance (gain, weight, cover)
  - Fast training with GPU support

  **Must NOT do**:
  - Don't use high learning rate (0.3+) - will overfit
  - Don't use deep trees (max_depth > 10) - will overfit
  - Don't skip early stopping

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Gradient boosting requires careful tuning
  - **Skills**: python, sklearn
    - python: Feature engineering
    - sklearn: XGBoost API, early stopping

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 10, 11, 12)
  - **Blocks**: Tasks 14, 17
  - **Blocked By**: Tasks 7, 10

  **References**:
  - Library: XGBoost Python documentation
  - Pattern: Gradient boosting parameters, early stopping
  - External: XGBoost for financial prediction papers

  **Acceptance Criteria**:
  - [ ] XGBoost trains successfully with early stopping
  - [ ] Model file saved: `models/xgboost_*.json` exists
  - [ ] Feature importance extracted (different from RF)
  - [ ] Directional accuracy on 2023 test set > 50%
  - [ ] Training time documented (should be faster than LSTM)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify XGBoost model
    Tool: Bash (python)
    Preconditions: Task 7 complete
    Steps:
      1. Run: `python -c "from src.models.xgboost_model import XGBoostModel; xgb = XGBoostModel(); print(xgb.get_params())"` → Assert shows hyperparameters
      2. Run: `python scripts/train_xgboost.py --ticker AAPL` → Assert trains with early stopping
      3. Run: `ls models/xgboost_*.json` → Assert model file created
      4. Run: `python -c "
from src.models.xgboost_model import XGBoostModel
xgb = XGBoostModel()
xgb.load('models/xgboost_YYYYMMDD.json')
importances = xgb.get_feature_importance()
print('Top 5 features:')
for feat, imp in list(importances.items())[:5]:
    print(f'  {feat}: {imp:.3f}')
"` → Assert shows XGBoost feature importance
      5. Run: `python tests/unit/test_xgboost.py -v` → Assert all XGBoost tests pass
    Expected Result: XGBoost trained, importance extracted, accuracy > 50%
    Evidence: Model file saved, tests passing

  **Commit**: YES
  - Message: `feat(models): implement XGBoost classifier with early stopping`
  - Files: `src/models/xgboost_model.py`, `scripts/train_xgboost.py`, `models/xgboost_*.json`
  - Pre-commit: Verify model persists and loads

---

- [ ] 14. Walk-Forward Validation Framework

  **What to do**:
  - Create `src/models/validation.py` - walk-forward validation implementation
  - Implement time-series cross-validation (NO random splitting)
  - Framework:
    - Train: 2019-2021 (3 years)
    - Validate: 2022-Q1 (3 months)
    - Test: 2022-Q2 (3 months, unseen)
    - Then roll forward: Train 2019-2022-Q1, Validate 2022-Q2, Test 2022-Q3
    - Continue rolling until end of 2023
  - Calculate metrics for each fold: directional accuracy, RMSE, MAE, Sharpe ratio
  - Aggregate results across all folds
  - Detect overfitting: Compare train vs validation accuracy gaps
  - Create validation report: `docs/validation_report.md`
  - Add visualization: Accuracy over time (rolling window)
  - Implement for all 3 models (LSTM, RF, XGBoost)

  **Why Walk-Forward Matters**:
  - Standard k-fold CV is WRONG for time-series (data leakage)
  - Walk-forward simulates real-world deployment (train on past, predict future)
  - Catches overfitting that regular CV misses

  **Must NOT do**:
  - Don't use sklearn's TimeSeriesSplit (not walk-forward, just expanding window)
  - Don't shuffle data (destroys time ordering)
  - Don't use future data in training

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Time-series validation requires careful implementation
  - **Skills**: python, data-analysis
    - python: Date manipulation, rolling windows
    - data-analysis: Financial backtesting concepts

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 11, 12, 13)
  - **Parallel Group**: Sequential in Wave 5
  - **Blocks**: Task 17 (Model comparison needs validation results)
  - **Blocked By**: Tasks 10, 11, 12, 13 (all models)

  **References**:
  - Concept: Walk-forward analysis in time-series forecasting
  - Pattern: Expanding window vs sliding window validation
  - External: "Advances in Financial Machine Learning" by Marcos Lopez de Prado

  **Acceptance Criteria**:
  - [ ] Walk-forward validation runs for all 3 models
  - [ ] At least 4 folds completed (quarterly rolling)
  - [ ] Metrics calculated per fold and aggregated
  - [ ] Overfitting detection: No model with >10% train/val gap
  - [ ] Report generated: `docs/validation_report.md` with charts

  **Agent-Executed QA Scenarios**:

  Scenario: Verify walk-forward validation
    Tool: Bash (python)
    Preconditions: Tasks 11, 12, 13 complete (models trained)
    Steps:
      1. Run: `python -c "from src.models.validation import WalkForwardValidator; wfv = WalkForwardValidator(); print(wfv.get_folds())"` → Assert shows fold definitions
      2. Run: `python scripts/run_walkforward.py --model lstm` → Assert completes all folds
      3. Run: `ls data/validation_results/` → Assert validation results CSVs created
      4. Run: `python -c "
from src.models.validation import WalkForwardValidator
wfv = WalkForwardValidator()
results = wfv.aggregate_results('lstm')
print(f'Average accuracy: {results[\"avg_accuracy\"]:.2%}')
print(f'Overfitting detected: {results[\"overfitting\"]}')
"` → Assert shows aggregated metrics
      5. Run: `ls docs/validation_report.md` → Assert report created
    Expected Result: Walk-forward validation complete, overfitting detected if present
    Evidence: Validation results saved, report generated

  **Commit**: YES
  - Message: `feat(models): implement walk-forward validation framework for time-series`
  - Files: `src/models/validation.py`, `scripts/run_walkforward.py`, `docs/validation_report.md`
  - Pre-commit: Run validation on one model, verify metrics calculated

---

- [ ] 15. FinBERT Setup and Local Installation

  **What to do**:
  - Download FinBERT model from HuggingFace Hub
  - Model: `yiyanghkust/finbert-tone` (financial sentiment classification)
  - Save to local directory: `models/finbert/`
  - Model size: ~400MB (much smaller than Llama 3)
  - Implement loading function: `src/nlp/finbert_loader.py`
  - Test basic inference: Input headline → Output sentiment (POSITIVE/NEGATIVE/NEUTRAL)
  - Add batch processing capability (process multiple headlines efficiently)
  - Create test suite for FinBERT: `tests/unit/test_finbert.py`
  - Document model info: architecture, training data, license
  - Add model checksum verification (ensure download integrity)
  - One-time download only - no external API calls during operation

  **FinBERT Model Details**:
  - Architecture: BERT-base (12 layers, 768 hidden size)
  - Fine-tuned on: Financial news and earnings call transcripts
  - Labels: Positive, Negative, Neutral
  - Confidence scores: 0.0 to 1.0 per class

  **Must NOT do**:
  - Don't download model every time (cache locally forever)
  - Don't use online HuggingFace inference API (must be 100% local)
  - Don't skip model integrity verification

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Model download and basic setup
  - **Skills**: python, transformers
    - python: File management, downloads
    - transformers: Model loading, tokenizer setup

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Task 14)
  - **Blocks**: Task 16 (Sentiment analysis needs FinBERT)
  - **Blocked By**: Task 2 (database for storing results)

  **References**:
  - Model: https://huggingface.co/yiyanghkust/finbert-tone
  - Library: Transformers documentation (AutoModel, AutoTokenizer)
  - External: FinBERT paper (Araci, 2019)

  **Acceptance Criteria**:
  - [ ] Model downloaded: `models/finbert/` contains pytorch_model.bin, config.json
  - [ ] Loading works: Can instantiate FinBERT model without internet
  - [ ] Basic inference works: Input "Apple stock rises" → Positive sentiment
  - [ ] Batch processing tested: Can process 10 headlines in one call
  - [ ] Checksum verified: Model files integrity confirmed

  **Agent-Executed QA Scenarios**:

  Scenario: Verify FinBERT setup
    Tool: Bash (python)
    Preconditions: transformers library installed
    Steps:
      1. Run: `python scripts/download_finbert.py` → Assert downloads model to models/finbert/
      2. Run: `ls models/finbert/` → Assert shows: config.json, pytorch_model.bin, vocab.txt
      3. Run: `python -c "
from src.nlp.finbert_loader import FinBERTLoader
loader = FinBERTLoader()
model = loader.load_model()
print(f'Model loaded: {model.config.model_type}')
result = loader.predict('Apple stock rises after earnings beat')
print(f'Sentiment: {result[\"label\"]}, Confidence: {result[\"score\"]:.3f}')
"` → Assert shows sentiment prediction
      4. Run: `python -c "
from src.nlp.finbert_loader import FinBERTLoader
loader = FinBERTLoader()
headlines = ['Apple rises', 'Tesla falls', 'Market stable'] * 10
results = loader.predict_batch(headlines)
print(f'Processed {len(results)} headlines')
"` → Assert batch processing works
      5. Run: `pytest tests/unit/test_finbert.py -v` → Assert all FinBERT tests pass
    Expected Result: FinBERT downloaded, loads locally, makes predictions
    Evidence: Model files exist, predictions correct

  **Commit**: YES (large binary files - consider Git LFS or document download process)
  - Message: `feat(nlp): setup FinBERT model locally (400MB financial sentiment classifier)`
  - Files: `src/nlp/finbert_loader.py`, `scripts/download_finbert.py`, `models/finbert/`
  - Pre-commit: Verify model loads without internet connection

---

- [ ] 16. Sentiment Analysis Pipeline

  **What to do**:
  - Create `src/nlp/sentiment_pipeline.py` - batch sentiment analysis
  - Load FinBERT model from local cache (Task 15)
  - Fetch unprocessed headlines from database (Task 8)
  - Process headlines in batches of 32 (optimal for FinBERT)
  - Store results in `sentiment_scores` table (one row per headline-ticker pair)
  - Aggregate daily sentiment per ticker: `daily_sentiment` table
  - Calculate moving averages: 3-day and 7-day sentiment
  - Add sentiment volatility (standard deviation over 7 days)
  - Create daily batch script: `scripts/run_sentiment_analysis.py`
  - Schedule: Run daily at 6 AM (after news fetch, before market open)
  - Process 90-day retention: Delete old sentiment data, keep aggregates
  - Add monitoring: Track processing time, failed headlines

  **Processing Time Estimates**:
  - 500 tickers × 5 headlines/day = 2,500 headlines
  - Batch size 32: ~78 batches
  - CPU: ~5 minutes total
  - GPU: ~1 minute total

  **Must NOT do**:
  - Don't reprocess already-analyzed headlines (check timestamp)
  - Don't store low-confidence predictions (<70% confidence threshold)
  - Don't process without batching (too slow)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Pipeline orchestration, mostly batch processing
  - **Skills**: python, nlp
    - python: Batch iteration, database writes
    - nlp: Sentiment aggregation strategies

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 14, 15)
  - **Blocks**: Task 20 (Context injection needs sentiment)
  - **Blocked By**: Tasks 8, 15 (news headlines and FinBERT)

  **References**:
  - Pattern: Batch processing pipelines, sentiment aggregation
  - External: Sentiment analysis best practices for financial text

  **Acceptance Criteria**:
  - [ ] Sentiment pipeline processes headlines and stores scores
  - [ ] `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM sentiment_scores"` returns >0
  - [ ] Daily aggregation works: Each ticker has sentiment score per day
  - [ ] Processing time <10 minutes for full batch (acceptable for daily run)
  - [ ] Low-confidence predictions filtered out (<70%)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify sentiment analysis pipeline
    Tool: Bash (python)
    Preconditions: Tasks 8, 15 complete (headlines and FinBERT ready)
    Steps:
      1. Run: `python scripts/run_sentiment_analysis.py --date 2024-01-15` → Assert completes without errors
      2. Run: `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM sentiment_scores WHERE date(analyzed_at) = '2024-01-15'"` → Assert shows >0 records
      3. Run: `sqlite3 data/financial_advisor.db "SELECT * FROM daily_sentiment WHERE date='2024-01-15' LIMIT 5"` → Assert shows aggregated sentiment
      4. Run: `python -c "
from src.nlp.sentiment_pipeline import SentimentPipeline
pipeline = SentimentPipeline()
stats = pipeline.get_processing_stats('2024-01-15')
print(f'Headlines processed: {stats[\"total\"]}')
print(f'Average confidence: {stats[\"avg_confidence\"]:.2f}')
"` → Assert shows processing statistics
      5. Run: `cat logs/sentiment_pipeline.log | tail -10` → Assert shows pipeline completion
    Expected Result: Sentiment scores calculated and stored, aggregates created
    Evidence: Database records verified, logs show success

  **Commit**: YES
  - Message: `feat(nlp): implement batch sentiment analysis pipeline with daily aggregation`
  - Files: `src/nlp/sentiment_pipeline.py`, `scripts/run_sentiment_analysis.py`, `tests/unit/test_sentiment.py`
  - Pre-commit: Run sentiment pipeline on sample data

---

- [ ] 17. Model Comparison and Selection

  **What to do**:
  - Create `src/models/comparison.py` - model comparison framework
  - Compare all models: LSTM, Random Forest, XGBoost, and baselines
  - Metrics: Directional accuracy, RMSE, Sharpe ratio, max drawdown, training time
  - Create comparison table: Model vs Metric matrix
  - Statistical significance testing: Are differences meaningful? (t-tests)
  - Ensemble voting: Combine all 3 models (majority vote or weighted average)
  - Model selection criteria:
    - Primary: Sharpe ratio (risk-adjusted returns)
    - Secondary: Directional accuracy
    - Tie-breaker: Model interpretability (RF > XGBoost > LSTM)
  - Create comparison dashboard (Streamlit component for later)
  - Save best model as `models/production_model_*.pkl`
  - Document selection rationale in `docs/model_selection.md`

  **Selection Decision Tree**:
  1. Calculate Sharpe ratio for all models
  2. Select top 2 by Sharpe ratio
  3. If accuracy difference < 2%, pick more interpretable model
  4. Save ensemble as fallback (all 3 models voting)

  **Must NOT do**:
  - Don't just pick highest accuracy (ignores risk)
  - Don't ignore training time (LSTM takes hours, RF takes minutes)
  - Don't skip statistical testing (may be noise, not real difference)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Statistical comparison and model selection logic
  - **Skills**: python, ml-validation
    - python: Statistical tests, metric aggregation
    - ml-validation: Model comparison frameworks

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all previous model tasks)
  - **Parallel Group**: Sequential in Wave 5
  - **Blocks**: Task 20 (Context injection needs best model)
  - **Blocked By**: Tasks 11, 12, 13, 14, 16 (all models and validation)

  **References**:
  - Statistical: scipy.stats.ttest_ind (for comparing accuracies)
  - Pattern: Model ensemble methods, voting classifiers
  - External: Model selection best practices in finance

  **Acceptance Criteria**:
  - [ ] Comparison table created showing all metrics for all models
  - [ ] Best model selected and documented with rationale
  - [ ] Ensemble model created (voting mechanism)
  - [ ] Statistical significance confirmed (p-value < 0.05)
  - [ ] Production model saved: `models/production_model_*.pkl`

  **Agent-Executed QA Scenarios**:

  Scenario: Verify model comparison
    Tool: Bash (python)
    Preconditions: Tasks 11, 12, 13, 14, 16 complete
    Steps:
      1. Run: `python scripts/compare_models.py` → Assert generates comparison report
      2. Run: `cat docs/model_comparison.csv` → Assert shows all models and metrics
      3. Run: `python -c "
from src.models.comparison import ModelComparison
mc = ModelComparison()
best = mc.select_best_model()
print(f'Best model: {best[\"name\"]}')
print(f'Sharpe ratio: {best[\"sharpe_ratio\"]:.2f}')
print(f'Accuracy: {best[\"accuracy\"]:.2%}')
"` → Assert shows best model selection
      4. Run: `ls models/production_model_*.pkl` → Assert production model file created
      5. Run: `cat docs/model_selection.md` → Assert selection rationale documented
    Expected Result: Models compared, best selected, production model saved
    Evidence: Comparison report generated, production model exists

  **Commit**: YES
  - Message: `feat(models): compare all models and select production model with ensemble fallback`
  - Files: `src/models/comparison.py`, `scripts/compare_models.py`, `docs/model_comparison.csv`, `models/production_model_*.pkl`
  - Pre-commit: Verify comparison logic correct

---

- [ ] 18. Local Llama 3 Setup and Integration

  **What to do**:
  - Download Llama 3 8B model from HuggingFace or Meta
  - Model: `meta-llama/Meta-Llama-3-8B-Instruct` (instruction-tuned version)
  - Alternative: `TheBloke/Llama-3-8B-Instruct-GGUF` (quantized, smaller)
  - Save to local directory: `models/llama3/`
  - Model size: ~8GB (full) or ~4GB (quantized 4-bit)
  - GPU Requirements:
    - Full 8B: RTX 3060 12GB or RTX 4060 Ti 16GB
    - Quantized 4-bit: RTX 3060 12GB (recommended for faster inference)
  - Implement loader: `src/llm/llama_loader.py`
  - Test basic inference: Can generate response to "Hello"
  - Add quantization support (bitsandbytes library for 4-bit)
  - Create memory management (clear cache between queries)
  - One-time download only - 100% local operation
  - Add GPU detection and warning if not available

  **Model Download Options**:
  1. **HuggingFace Hub**: `meta-llama/Meta-Llama-3-8B-Instruct` (requires HF token)
  2. **GGUF Format**: `TheBloke/Llama-3-8B-Instruct-GGUF` (direct download, no auth)
  - **Recommendation**: Use GGUF for easier setup, smaller size, good quality

  **Must NOT do**:
  - Don't download every run (cache forever)
  - Don't load model without GPU check (will be painfully slow on CPU)
  - Don't skip quantization (8GB may not fit in 12GB VRAM with overhead)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Large language model setup requires careful memory management
  - **Skills**: python, transformers
    - python: File management, GPU detection
    - transformers: Large model loading, quantization

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with Tasks 19, 20, 21)
  - **Blocks**: Task 19 (Prompts need Llama loaded)
  - **Blocked By**: Task 1 (project setup, GPU check)

  **References**:
  - Model: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
  - GGUF: https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF
  - Library: llama-cpp-python (for GGUF), transformers (for HF)
  - External: Llama 3 paper, quantization documentation

  **Acceptance Criteria**:
  - [ ] Model downloaded: `models/llama3/` contains model files
  - [ ] GPU detected: Script warns if no GPU available
  - [ ] Basic inference works: "Hello" → generates response in <10 seconds
  - [ ] Quantization working: 4-bit mode loads successfully
  - [ ] Memory usage monitored: <12GB VRAM during inference

  **Agent-Executed QA Scenarios**:

  Scenario: Verify Llama 3 setup
    Tool: Bash (python)
    Preconditions: GPU available (RTX 3060+), transformers installed
    Steps:
      1. Run: `python scripts/download_llama3.py` → Assert downloads model to models/llama3/
      2. Run: `ls models/llama3/` → Assert shows model files (.bin, .json, or .gguf)
      3. Run: `python -c "
from src.llm.llama_loader import LlamaLoader
loader = LlamaLoader()
print(f'GPU available: {loader.check_gpu()}')
model = loader.load_model(quantization='4bit')
response = model.generate('Hello, how are you?')
print(f'Response: {response[:100]}...')
"` → Assert generates response
      4. Run: `nvidia-smi` (if available) → Assert shows <12GB VRAM used
      5. Run: `pytest tests/unit/test_llama.py -v` → Assert all Llama tests pass
    Expected Result: Llama 3 downloaded, loads locally, generates text
    Evidence: Model files exist, GPU usage logged, test output captured

  **Commit**: YES (large files - use Git LFS or document download)
  - Message: `feat(llm): setup local Llama 3 8B with 4-bit quantization`
  - Files: `src/llm/llama_loader.py`, `scripts/download_llama3.py`, `tests/unit/test_llama.py`
  - Pre-commit: Verify model loads without internet

---

- [ ] 19. Prompt Engineering and Templates

  **What to do**:
  - Create `src/llm/prompts.py` - prompt template system
  - Design 5 core prompt templates:
    1. **General Analysis**: "Analyze {ticker} stock performance"
    2. **Technical Analysis**: "What do the technical indicators say about {ticker}?"
    3. **Sentiment Analysis**: "What's the market sentiment for {ticker}?"
    4. **Buy/Sell Recommendation**: "Should I buy {ticker}?" (with mandatory disclaimer)
    5. **Portfolio Overview**: "How is my portfolio performing?"
  - Add system prompt: "You are a financial advisor assistant..."
  - Implement prompt chaining: Break complex queries into steps
  - Add few-shot examples: 2-3 examples of good responses per template
  - Create prompt testing framework: Evaluate response quality
  - Optimize for:
    - Conciseness (2-3 paragraphs max)
    - Clarity (no jargon without explanation)
    - Safety (always include disclaimer)
  - Version control prompts (track changes over time)
  - Create prompt evaluation metric: User satisfaction score

  **Prompt Template Example**:
  ```python
  SYSTEM_PROMPT = """You are a financial advisor assistant. Provide clear, concise analysis based on data provided. Always include a disclaimer that this is AI-generated advice, not professional financial advice."""

  ANALYSIS_TEMPLATE = """Stock: {ticker}
  Current Price: ${price}
  Recent Performance: {performance_summary}
  Technical Indicators: {indicators_summary}
  Sentiment: {sentiment_summary}
  Model Predictions: {prediction_summary}

  User Question: {user_query}

  Provide a 2-3 paragraph analysis addressing the user's question. Include key risks and a disclaimer."""
  ```

  **Must NOT do**:
  - Don't use generic prompts (must include financial context)
  - Don't allow open-ended generation (constrain with templates)
  - Don't skip disclaimer (regulatory/ethical requirement)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Prompt engineering requires iterative refinement
  - **Skills**: python, llm
    - python: Template management
    - llm: Prompt engineering best practices

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with Tasks 18, 20, 21)
  - **Blocks**: Task 20 (Context injection needs templates)
  - **Blocked By**: Task 18 (Llama must be loaded)

  **References**:
  - Pattern: Prompt templates, few-shot prompting
  - External: OpenAI prompt engineering guide (applies to Llama too)
  - External: Chain-of-thought prompting papers

  **Acceptance Criteria**:
  - [ ] All 5 prompt templates created and tested
  - [ ] System prompt enforces disclaimer requirement
  - [ ] Template renders correctly with sample data
  - [ ] Responses are 2-3 paragraphs (not too long)
  - [ ] A/B test: Templates vs baseline prompts (templates should be better)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify prompt templates
    Tool: Bash (python)
    Preconditions: Task 18 complete
    Steps:
      1. Run: `python -c "from src.llm.prompts import PromptManager; pm = PromptManager(); print(pm.list_templates())"` → Assert shows 5 templates
      2. Run: `python -c "
from src.llm.prompts import PromptManager
pm = PromptManager()
template = pm.get_template('analysis')
rendered = template.render(ticker='AAPL', price='150.23', performance_summary='+5% this week')
print(rendered[:200])
"` → Assert template renders with variables
      3. Run: `python scripts/test_prompts.py` → Assert runs prompt evaluation
      4. Run: `cat logs/prompt_quality.log` → Assert shows response quality scores
      5. Run: `grep -r "disclaimer" src/llm/prompts.py` → Assert disclaimer present
    Expected Result: Prompts created, render correctly, include disclaimers
    Evidence: Template renders logged, quality scores captured

  **Commit**: YES
  - Message: `feat(llm): create prompt templates with system prompts and disclaimers`
  - Files: `src/llm/prompts.py`, `src/llm/prompt_templates/`, `scripts/test_prompts.py`
  - Pre-commit: Verify all prompts include disclaimer

---

- [ ] 20. Context Injection (Predictions + Sentiment)

  **What to do**:
  - Create `src/llm/context_builder.py` - dynamic context generation
  - Fetch real-time data for queried ticker:
    - Current price and recent performance (from database)
    - Technical indicators (RSI, MACD, BB - from Task 7)
    - Sentiment scores (from Task 16)
    - Model predictions (from Task 17)
  - Format data into natural language summaries:
    - "RSI is 72, indicating overbought conditions"
    - "Sentiment is bullish (+0.65) based on 12 news headlines"
    - "LSTM model predicts +2.3% over next 7 days (55% confidence)"
  - Inject context into prompt templates (Task 19)
  - Handle missing data gracefully (e.g., "No recent news available")
  - Add data freshness indicator: "Data as of 2024-01-15 6:00 AM"
  - Implement caching: Cache context for 15 minutes (reduces database queries)
  - Create tests for context builder with mock data

  **Context Building Flow**:
  1. Parse user query (extract ticker)
  2. Query database for ticker data (price, indicators, sentiment)
  3. Load model predictions
  4. Format into natural language
  5. Inject into prompt template
  6. Send to Llama 3
  7. Return response to user

  **Must NOT do**:
  - Don't use stale data (more than 24 hours old without warning)
  - Don't hallucinate missing data (say "not available" instead)
  - Don't overwhelm with too much data (keep it concise)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Requires integrating multiple data sources dynamically
  - **Skills**: python, data-engineering
    - python: Data aggregation, formatting
    - data-engineering: Real-time data pipelines

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all previous tasks)
  - **Parallel Group**: Sequential in Wave 6
  - **Blocks**: Task 24 (Chat UI needs context injection)
  - **Blocked By**: Tasks 7, 16, 17, 18, 19 (indicators, sentiment, models, Llama, prompts)

  **References**:
  - Pattern: RAG (Retrieval Augmented Generation) - similar concept
  - External: Context injection best practices for LLMs

  **Acceptance Criteria**:
  - [ ] Context builder fetches data from all 3 sources (prices, sentiment, predictions)
  - [ ] Data formatted into readable summaries
  - [ ] Missing data handled gracefully (no crashes)
  - [ ] Context cached: Second query for same ticker is faster
  - [ ] All context includes timestamp (data freshness)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify context injection
    Tool: Bash (python)
    Preconditions: Tasks 7, 16, 17, 18, 19 complete
    Steps:
      1. Run: `python -c "from src.llm.context_builder import ContextBuilder; cb = ContextBuilder(); ctx = cb.build_context('AAPL'); print(ctx['price']); print(ctx['indicators_summary'][:100])"` → Assert shows context data
      2. Run: `python -c "
from src.llm.context_builder import ContextBuilder
import time
cb = ContextBuilder()
start = time.time()
cb.build_context('AAPL')
end = time.time()
print(f'First call: {end-start:.2f}s')
start = time.time()
cb.build_context('AAPL')
end = time.time()
print(f'Cached call: {end-start:.2f}s')
"` → Assert cached call is faster
      3. Run: `python -c "from src.llm.context_builder import ContextBuilder; cb = ContextBuilder(); ctx = cb.build_context('INVALID_TICKER'); print(ctx)"` → Assert handles gracefully
      4. Run: `pytest tests/unit/test_context_builder.py -v` → Assert all tests pass
      5. Run: `python -c "from src.llm.context_builder import ContextBuilder; cb = ContextBuilder(); ctx = cb.build_context('AAPL'); assert 'timestamp' in ctx or 'as of' in str(ctx)"` → Assert timestamp present
    Expected Result: Context built correctly, cached, handles errors
    Evidence: Context data logged, timing captured

  **Commit**: YES
  - Message: `feat(llm): implement dynamic context injection from predictions and sentiment`
  - Files: `src/llm/context_builder.py`, `tests/unit/test_context_builder.py`
  - Pre-commit: Run context builder tests

---

- [ ] 21. Response Guardrails and Fact-Checking

  **What to do**:
  - Create `src/llm/guardrails.py` - response filtering and validation
  - Implement 3-layer safety system:
    1. **Pre-filter**: Check input query for disallowed topics (hate speech, illegal acts)
    2. **Post-filter**: Validate LLM output for:
       - Specific buy/sell recommendations without disclaimer
       - Guaranteed return promises ("you will make 20%")
       - Advice to take excessive risk ("mortgage your house")
       - Hallucinated ticker symbols
    3. **Fact-check**: Verify factual claims against database:
       - Ticker symbols mentioned (must be valid S&P 500)
       - Price data referenced (must match database)
       - Dates mentioned (must be plausible)
  - Create blocklist: Keywords that trigger rejection
  - Add regex patterns for common hallucinations
  - Implement response rewriting: If minor issues, fix and return; if major, reject
  - Log all guardrail triggers (for monitoring)
  - Create "Report Issue" button in UI for false positives
  - Write tests for all guardrail scenarios

  **Must NOT do**:
  - Don't allow unfiltered LLM output to reach users
  - Don't be too aggressive (avoid false positives that frustrate users)
  - Don't skip fact-checking (hallucination is #1 LLM risk)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Safety-critical, requires careful balance
  - **Skills**: python, llm
    - python: Regex, text validation
    - llm: LLM safety, guardrail patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with Tasks 18, 19, 20)
  - **Blocks**: Task 24 (Chat UI needs safe responses)
  - **Blocked By**: Task 20 (needs context to fact-check)

  **References**:
  - Pattern: Content moderation, output filtering
  - External: LLM safety best practices (OpenAI, Anthropic guidelines)
  - External: RegEx for financial text validation

  **Acceptance Criteria**:
  - [ ] Pre-filter blocks disallowed queries (test with sample inputs)
  - [ ] Post-filter catches unsafe recommendations (test with mock LLM responses)
  - [ ] Fact-check validates ticker symbols (test with valid/invalid tickers)
  - [ ] Guardrail triggers logged to `data/logs/guardrails.log`
  - [ ] False positive rate <5% (measured on test set of 50 queries)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify guardrails and fact-checking
    Tool: Bash (python)
    Preconditions: Task 20 complete
    Steps:
      1. Run: `python -c "from src.llm.guardrails import Guardrails; g = Guardrails(); result = g.check_input('How do I hack a bank account?'); print(f'Blocked: {result[\"blocked\"]}')"` → Assert blocked=True
      2. Run: `python -c "from src.llm.guardrails import Guardrails; g = Guardrails(); response = 'Buy AAPL now and you will make 50% guaranteed!'; result = g.check_output(response, ticker='AAPL'); print(f'Issues: {result[\"issues\"]}')"` → Assert catches disclaimer violation
      3. Run: `python -c "from src.llm.guardrails import Guardrails; g = Guardrails(); response = 'The price of XYZ999 is $100'; result = g.fact_check(response); print(f'Valid tickers: {result[\"valid_tickers\"]}')"` → Assert flags XYZ999 as invalid
      4. Run: `cat data/logs/guardrails.log` → Assert shows guardrail activity
      5. Run: `pytest tests/unit/test_guardrails.py -v` → Assert all guardrail tests pass
    Expected Result: Guardrails block unsafe content, fact-check catches errors
    Evidence: Test logs captured, guardrail triggers documented

  **Commit**: YES
  - Message: `feat(llm): implement 3-layer guardrails with fact-checking`
  - Files: `src/llm/guardrails.py`, `src/llm/blocklist.txt`, `tests/unit/test_guardrails.py`
  - Pre-commit: Run guardrail tests, verify no unsafe content passes

---

- [ ] 22. Streamlit Core Interface Development

  **What to do**:
  - Create `app.py` - main Streamlit application entry point
  - Design layout: Sidebar (navigation) + Main content area
  - Implement 3 main views:
    1. **Stock Analysis**: Search ticker, show charts, predictions, sentiment
    2. **Chat Interface**: Conversational Q&A with Llama 3
    3. **Dashboard**: Overview of portfolio (if multi-stock feature added)
  - Add stock search widget with autocomplete (all 503 tickers)
  - Create session state management for user interactions
  - Implement navigation: st.sidebar.radio() for view selection
  - Add loading states for data fetching (st.spinner)
  - Create error boundaries: Try-catch around all callbacks
  - Add user-friendly error messages (not technical stack traces)
  - Style with Streamlit theming (config.toml for colors)
  - Ensure responsive design (works on desktop and tablet)

  **Streamlit Best Practices**:
  - Use caching (`@st.cache_data`) for database queries
  - Keep session state minimal (don't store entire DataFrames)
  - Use `st.empty()` for dynamic updates
  - Add progress bars for long operations

  **Must NOT do**:
  - Don't reload entire page on every interaction (use session state)
  - Don't show raw error messages to users (catch and rephrase)
  - Don't skip mobile responsiveness (Streamlit is responsive by default)

  **Recommended Agent Profile**:
  - **Category**: visual-engineering
    - Reason: UI/UX development for web interface
  - **Skills**: python, streamlit
    - python: Session management, callbacks
    - streamlit: Layout, components, state management

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 23, 24, 25)
  - **Blocks**: Task 25 (UX polish needs interface)
  - **Blocked By**: Task 17 (needs models to display)

  **References**:
  - Library: Streamlit documentation (st.columns, st.tabs, st.session_state)
  - Pattern: Single-page application with view switching
  - External: Streamlit gallery for inspiration

  **Acceptance Criteria**:
  - [ ] `streamlit run app.py` starts server without errors
  - [ ] Sidebar navigation works: Can switch between Analysis, Chat, Dashboard
  - [ ] Stock search works: Typing "AAPL" shows Apple
  - [ ] Session state persists: Switching views doesn't lose data
  - [ ] Error handling works: Invalid ticker shows friendly error (not stack trace)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify Streamlit interface
    Tool: Playwright (browser automation)
    Preconditions: Task 17 complete, models trained
    Steps:
      1. Run: `streamlit run app.py &` (background) → Assert server starts on localhost:8501
      2. Navigate: http://localhost:8501 → Assert page loads
      3. Click: Sidebar "Stock Analysis" → Assert shows stock search box
      4. Type: "AAPL" in search → Assert autocomplete shows Apple Inc.
      5. Click: Sidebar "Chat" → Assert shows chat interface
      6. Screenshot: Save to `.sisyphus/evidence/task-22-streamlit.png`
      7. Kill: Streamlit process
    Expected Result: Streamlit app loads, navigation works, search functional
    Evidence: Screenshot saved, console output captured

  **Commit**: YES
  - Message: `feat(ui): implement Streamlit core interface with navigation and search`
  - Files: `app.py`, `.streamlit/config.toml`, `src/ui/components.py`
  - Pre-commit: Verify app starts, basic navigation works

---

- [ ] 23. Data Visualization with Plotly

  **What to do**:
  - Create `src/ui/charts.py` - chart components for Streamlit
  - Implement 5 chart types:
    1. **Candlestick Chart**: OHLCV with volume (mplfinance or Plotly)
    2. **Technical Indicators Overlay**: RSI, MACD, BB on price chart
    3. **Prediction Chart**: Actual vs predicted prices with confidence bands
    4. **Sentiment Timeline**: Daily sentiment scores over time
    5. **Model Performance**: Comparison bar charts (accuracy, Sharpe ratio)
  - Use Plotly for interactivity (zoom, pan, hover tooltips)
  - Add chart customization: Time range selectors (1M, 3M, 1Y, 5Y)
  - Implement responsive sizing (charts fit container width)
  - Add download buttons: Export chart as PNG/SVG
  - Create dark/light mode support (match Streamlit theme)
  - Optimize performance: Don't redraw on every interaction (use caching)
  - Write tests for chart data preparation (not rendering)

  **Chart Libraries**:
  - Primary: Plotly (interactive, Streamlit native support)
  - Alternative: mplfinance (better candlesticks, less interactive)
  - **Recommendation**: Use Plotly for most, mplfinance for candlestick detail view

  **Must NOT do**:
  - Don't show raw matplotlib charts (not interactive enough)
  - Don't load 5 years of daily data for default view (slow) - start with 3 months
  - Don't skip mobile optimization (Plotly charts are responsive)

  **Recommended Agent Profile**:
  - **Category**: visual-engineering
    - Reason: Data visualization design and implementation
  - **Skills**: python, plotly
    - python: Data transformation for visualization
    - plotly: Chart configuration, interactivity

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 22, 24, 25)
  - **Blocks**: Task 25 (UX polish needs charts)
  - **Blocked By**: Task 17 (needs predictions to visualize)

  **References**:
  - Library: Plotly Python documentation, mplfinance documentation
  - Pattern: Financial charting best practices (candlestick conventions)
  - External: Streamlit-Plotly integration examples

  **Acceptance Criteria**:
  - [ ] Candlestick chart displays for any ticker with OHLCV data
  - [ ] Technical indicators overlay correctly on price chart
  - [ ] Predictions chart shows confidence bands
  - [ ] Charts are interactive (zoom, pan, tooltips work)
  - [ ] Time range selectors work (1M, 3M, 1Y, 5Y)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify data visualizations
    Tool: Playwright
    Preconditions: Task 22 complete, Task 17 complete (predictions ready)
    Steps:
      1. Run: `streamlit run app.py &` → Assert server starts
      2. Navigate: http://localhost:8501 → Assert loads
      3. Click: Search and select "AAPL" → Assert loads AAPL data
      4. Wait: For chart to render (timeout: 10s)
      5. Assert: Candlestick chart visible with OHLC data
      6. Click: "Technical Indicators" tab → Assert shows RSI/MACD charts
      7. Click: "Predictions" tab → Assert shows forecast with confidence bands
      8. Hover: Over chart → Assert tooltip shows data point values
      9. Screenshot: Save to `.sisyphus/evidence/task-23-charts.png`
      10. Kill: Streamlit process
    Expected Result: All charts render correctly, interactive features work
    Evidence: Screenshots saved, interaction logged

  **Commit**: YES
  - Message: `feat(ui): implement interactive charts with Plotly (candlestick, indicators, predictions)`
  - Files: `src/ui/charts.py`, `tests/unit/test_charts.py`
  - Pre-commit: Verify charts render without errors

---

- [ ] 24. Chat Interface Integration

  **What to do**:
  - Create chat UI component: `src/ui/chat.py`
  - Implement Streamlit chat elements: `st.chat_message()`, `st.chat_input()`
  - Integrate Llama 3 (Task 18) with context injection (Task 20)
  - Add conversation history display (scrollable message list)
  - Implement user message input with enter key support
  - Add typing indicator while Llama 3 generates (st.spinner)
  - Display context used: Show which data informed the response
  - Add "Regenerate Response" button (if user wants alternative)
  - Implement message persistence: Save chat history to session or database
  - Add quick action buttons: "Analyze AAPL", "Market Sentiment", etc.
  - Create example queries: "What should I know about Tesla?"
  - Write tests for chat message handling

  **Chat Flow**:
  1. User types query
  2. Parse query (extract ticker if mentioned)
  3. Build context (Task 20: fetch data)
  4. Render prompt (Task 19: apply template)
  5. Send to Llama 3 (Task 18: generate)
  6. Validate response (Task 21: guardrails)
  7. Display to user with context used

  **Must NOT do**:
  - Don't show raw LLM output without guardrails
  - Don't make users wait without feedback (show loading state)
  - Don't lose chat history on page refresh (use session state or DB)

  **Recommended Agent Profile**:
  - **Category**: visual-engineering
    - Reason: Chat UI requires careful UX design
  - **Skills**: python, streamlit
    - python: Async handling (if using), state management
    - streamlit: Chat components, real-time updates

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 22, 23, 25)
  - **Blocks**: Task 25 (UX polish needs chat)
  - **Blocked By**: Tasks 18, 19, 20, 21 (Llama, prompts, context, guardrails)

  **References**:
  - Library: Streamlit chat elements documentation (st.chat_message)
  - Pattern: Chatbot UI design, conversation flow management
  - External: LLM chat interface best practices

  **Acceptance Criteria**:
  - [ ] Chat interface displays in Streamlit app
  - [ ] User can type message and get response in <10 seconds
  - [ ] Response includes context used ("Based on AAPL data from Jan 15...")
  - [ ] Conversation history visible (last 10 messages)
  - [ ] Loading state shown while generating

  **Agent-Executed QA Scenarios**:

  Scenario: Verify chat interface
    Tool: Playwright
    Preconditions: Tasks 18-23 complete
    Steps:
      1. Run: `streamlit run app.py &` → Assert server starts
      2. Navigate: http://localhost:8501 → Assert loads
      3. Click: Sidebar "Chat" → Assert shows chat interface
      4. Type: "What do you think about Apple stock?" in chat input
      5. Press: Enter → Assert loading indicator appears
      6. Wait: For response (timeout: 15s)
      7. Assert: Response appears with disclaimer
      8. Assert: Context shown ("Based on AAPL data...")
      9. Screenshot: Save to `.sisyphus/evidence/task-24-chat.png`
      10. Kill: Streamlit process
    Expected Result: Chat works, response generated, context shown
    Evidence: Chat interaction logged, screenshots saved

  **Commit**: YES
  - Message: `feat(ui): integrate Llama 3 chat with context injection and guardrails`
  - Files: `src/ui/chat.py`, `tests/unit/test_chat.py`
  - Pre-commit: Test chat flow with mock LLM responses

---

- [ ] 25. UX Polish and Error Handling

  **What to do**:
  - Add loading states: Skeleton screens while data loads
  - Implement progress bars for long operations (model loading, data fetch)
  - Add tooltips: Explain technical terms (RSI, MACD, etc.)
  - Create help sections: "What is this metric?" expandable panels
  - Add disclaimers: Prominent risk warnings on every page
  - Implement responsive design: Test on mobile/tablet viewports
  - Add keyboard shortcuts: Ctrl+K for search, Esc to close modals
  - Create onboarding: First-time user guide or tutorial
  - Add feedback mechanism: "Was this helpful?" thumbs up/down
  - Implement dark mode toggle (Streamlit native support)
  - Add performance optimizations: Lazy loading, image compression
  - Create user settings: Risk tolerance, default ticker, theme preference
  - Write accessibility tests: Screen reader compatibility (basic)

  **UX Enhancements**:
  - Empty state design: What to show when no data (e.g., "Search for a stock to begin")
  - Error state design: Friendly error messages with recovery actions
  - Success animations: Subtle feedback when operations complete
  - Consistent styling: Typography, spacing, colors throughout

  **Must NOT do**:
  - Don't ignore mobile users (test on smaller screens)
  - Don't hide all complexity (some users want details - add "Advanced" sections)
  - Don't skip accessibility entirely (basic ARIA labels minimum)

  **Recommended Agent Profile**:
  - **Category**: visual-engineering
    - Reason: Polish and refinement requires UX expertise
  - **Skills**: python, streamlit, frontend-ui-ux
    - python: State management for settings
    - streamlit: Component customization
    - frontend-ui-ux: Design patterns, accessibility

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 22, 23, 24)
  - **Blocks**: Task 27 (Integration tests need polished UI)
  - **Blocked By**: Tasks 22, 23, 24 (core interface, charts, chat)

  **References**:
  - Pattern: UX best practices, accessibility guidelines (WCAG)
  - External: Streamlit theming documentation
  - External: Financial app design patterns (clean, trustworthy)

  **Acceptance Criteria**:
  - [ ] Loading states added to all async operations
  - [ ] Disclaimers visible on every view
  - [ ] Tooltips explain technical indicators
  - [ ] Dark mode works (toggle in settings)
  - [ ] Mobile viewport test passes (no horizontal scrolling)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify UX polish
    Tool: Playwright
    Preconditions: Tasks 22-24 complete
    Steps:
      1. Run: `streamlit run app.py &` → Assert server starts
      2. Navigate: http://localhost:8501 → Assert loads
      3. Click: Search and type "AAPL" → Assert loading spinner shown
      4. Wait: For data to load → Assert skeleton screens replaced with content
      5. Click: RSI indicator chart → Assert tooltip explains "What is RSI?"
      6. Scroll: To bottom of page → Assert disclaimer visible
      7. Click: Settings gear icon → Assert dark mode toggle present
      8. Toggle: Dark mode → Assert theme changes
      9. Resize: Browser window to mobile size (375px width) → Assert no horizontal scroll
      10. Screenshot: Both light and dark modes to `.sisyphus/evidence/task-25-ux/`
      11. Kill: Streamlit process
    Expected Result: All UX enhancements working, responsive, accessible
    Evidence: Screenshots in both themes, mobile view captured

  **Commit**: YES (groups with Tasks 22-24)
  - Message: `feat(ui): add UX polish including loading states, tooltips, dark mode, disclaimers`
  - Files: `src/ui/styles.py`, `.streamlit/config.toml`, `src/ui/help_texts.py`
  - Pre-commit: Run UI tests, verify responsive design

---

- [ ] 26. Unit Tests for All Modules

  **What to do**:
  - Write unit tests for all src/ modules:
    - `tests/unit/test_database/` - DAL methods, schema validation
    - `tests/unit/test_data/` - yfinance client, news client, caching
    - `tests/unit/test_features/` - Technical indicators calculation
    - `tests/unit/test_models/` - LSTM, RF, XGBoost (training and prediction)
    - `tests/unit/test_nlp/` - FinBERT, sentiment pipeline
    - `tests/unit/test_llm/` - Llama 3, prompts, context builder, guardrails
    - `tests/unit/test_ui/` - Chart data prep, chat message handling
  - Use pytest fixtures for test data (mock tickers, prices, headlines)
  - Add mock external APIs (yfinance, NewsAPI) - don't hit real APIs in tests
  - Create synthetic data generators for time-series testing
  - Target: >80% code coverage for critical paths (data, models, LLM)
  - Add parameterized tests (test multiple tickers, multiple models)
  - Create slow test markers (skip heavy model training in quick tests)
  - Add CI configuration (GitHub Actions) to run tests automatically
  - Document test strategy in `docs/testing_strategy.md`

  **Test Coverage Targets**:
  - Data pipeline: >90% (critical, must not lose data)
  - ML models: >80% (training and inference)
  - LLM components: >80% (prompts, context, guardrails)
  - UI components: >60% (chart data prep, chat handling)
  - Overall: >75% average

  **Must NOT do**:
  - Don't write tests without assertions (must verify behavior)
  - Don't hit real external APIs in tests (use mocks)
  - Don't skip error case testing (test failures, not just successes)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Testing patterns are standard
  - **Skills**: python, pytest
    - python: Mocking, fixtures
    - pytest: Test organization, coverage

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 8 (with Tasks 27, 28, 29)
  - **Blocks**: Task 28 (Final validation needs tests passing)
  - **Blocked By**: Task 3 (test infrastructure), Tasks 4-25 (all modules to test)

  **References**:
  - Library: pytest documentation, pytest-cov, pytest-mock
  - Pattern: AAA testing (Arrange-Act-Assert), mocking external services
  - External: Testing in data science projects best practices

  **Acceptance Criteria**:
  - [ ] All modules have corresponding test files
  - [ ] `pytest tests/unit/ -v` runs and passes (>90% tests pass)
  - [ ] Coverage report generated: `pytest --cov=src --cov-report=html`
  - [ ] Coverage >80% for data, models, LLM modules
  - [ ] Mock external APIs used (no real API calls in tests)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify unit test coverage
    Tool: Bash (pytest)
    Preconditions: Tasks 3, 4-25 complete
    Steps:
      1. Run: `pytest tests/unit/ -v --tb=short` → Assert shows test results
      2. Assert: >90% of tests pass (allow some tolerance for flaky tests)
      3. Run: `pytest --cov=src --cov-report=term-missing` → Assert coverage report
      4. Assert: Coverage >80% for src/data/, src/models/, src/llm/
      5. Run: `ls htmlcov/` → Assert HTML coverage report generated
      6. Run: `grep -r "mock" tests/unit/test_data/ | head -5` → Assert mocking used
      7. Run: `cat docs/testing_strategy.md` → Assert testing strategy documented
    Expected Result: All unit tests pass, coverage >80%, mocks used
    Evidence: Test output captured, coverage report saved

  **Commit**: YES (ongoing, add tests with each module)
  - Message: `test(all): add comprehensive unit test suite with >80% coverage`
  - Files: `tests/unit/**/*.py`, `pytest.ini`, `.github/workflows/tests.yml` (optional)
  - Pre-commit: Run full test suite, verify coverage

---

- [ ] 27. Integration Tests (End-to-End)

  **What to do**:
  - Create `tests/integration/` - integration test suite
  - Test complete data pipeline: Fetch → Store → Calculate indicators → Query
  - Test model training → Prediction → Display flow
  - Test LLM integration: Query → Context build → Prompt render → Response → Display
  - Test Streamlit UI: Load app → Search ticker → Display charts → Chat query
  - Use test database (separate from production)
  - Create test fixtures: Small subset of real data (10 tickers, 30 days)
  - Test error scenarios: API failure, missing data, model errors
  - Add database transaction rollback (clean state between tests)
  - Test concurrent access (SQLite WAL mode)
  - Create performance tests: Verify queries complete in <100ms
  - Test data retention: Verify old news is archived correctly

  **Integration Test Scenarios**:
  1. **Data Pipeline**: Fetch AAPL data → Store → Calculate RSI → Query RSI value
  2. **Model Flow**: Train RF on 10 tickers → Predict next day → Store prediction → Display in UI
  3. **LLM Flow**: User asks "Should I buy AAPL?" → Context built → Llama generates → Guardrails validate → Response displayed
  4. **Full App**: Start Streamlit → Search TSLA → View chart → Ask chatbot → Get response

  **Must NOT do**:
  - Don't use production database for integration tests (risk of data loss)
  - Don't skip error scenario testing (test failure paths)
  - Don't make tests dependent on each other (isolated tests)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Integration testing orchestration
  - **Skills**: python, pytest, playwright
    - python: Test orchestration, database setup/teardown
    - pytest: Fixtures, markers
    - playwright: End-to-end UI testing

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 8 (with Tasks 26, 28, 29)
  - **Blocks**: Task 28 (Validation needs integration tests)
  - **Blocked By**: Task 3, Tasks 4-25

  **References**:
  - Pattern: Test pyramid (unit < integration < e2e)
  - External: Integration testing best practices for data pipelines

  **Acceptance Criteria**:
  - [ ] Integration tests for all major flows (data, models, LLM, UI)
  - [ ] `pytest tests/integration/ -v` passes
  - [ ] Tests use isolated test database
  - [ ] Error scenarios tested (API failure, missing data)
  - [ ] Full app test with Playwright: Can search ticker and get response

  **Agent-Executed QA Scenarios**:

  Scenario: Verify integration tests
    Tool: Bash (pytest, Playwright)
    Preconditions: Tasks 4-25 complete
    Steps:
      1. Run: `pytest tests/integration/test_data_pipeline.py -v` → Assert data pipeline test passes
      2. Run: `pytest tests/integration/test_model_flow.py -v` → Assert model flow passes
      3. Run: `pytest tests/integration/test_llm_flow.py -v` → Assert LLM flow passes
      4. Run: `pytest tests/integration/test_full_app.py -v` → Assert full app test passes (uses Playwright)
      5. Run: `python tests/integration/verify_database_isolation.py` → Assert test DB separate from prod
      6. Run: `pytest tests/integration/ -v --tb=short 2>&1 | tail -20` → Assert summary shows all tests
    Expected Result: All integration tests pass, database isolated, flows work
    Evidence: Test output captured, Playwright screenshots saved

  **Commit**: YES
  - Message: `test(integration): add end-to-end tests for all major flows`
  - Files: `tests/integration/test_*.py`, `tests/fixtures/integration_data/`
  - Pre-commit: Run integration test suite

---

- [ ] 28. Model Backtesting and Validation

  **What to do**:
  - Create `scripts/backtest_models.py` - comprehensive backtesting
  - Simulate trading based on model predictions (paper trading)
  - Test period: 2023-01-01 to 2023-12-31 (1 year out-of-sample)
  - Trading strategy:
    - If model predicts UP: Buy at open, sell at close (or hold?)
    - If model predicts DOWN: Sell/short (or just don't buy)
    - Include transaction costs: 0.1% per trade (realistic)
  - Calculate performance metrics:
    - Total return (vs buy-and-hold benchmark)
    - Sharpe ratio (risk-adjusted return)
    - Maximum drawdown (largest peak-to-trough decline)
    - Win rate (% of profitable trades)
    - Profit factor (gross profit / gross loss)
  - Run backtest for all 3 models + ensemble + buy-and-hold
  - Create comparison report: `docs/backtest_results.md`
  - Generate equity curves: Cumulative returns over time
  - Add statistical significance testing (bootstrap confidence intervals)
  - Document which model would have performed best in 2023
  - Run for minimum 1 month "live" paper trading (if time permits)

  **Backtesting Framework**:
  - Use `backtesting.py` library or custom implementation
  - Walk-forward: Use models trained on 2019-2022, predict 2023
  - Daily rebalancing: Check predictions each morning
  - Position sizing: Equal weight per trade (for simplicity)

  **Must NOT do**:
  - Don't ignore transaction costs (makes strategies look too good)
  - Don't use future data (no lookahead bias)
  - Don't backtest on training data (cheating)

  **Recommended Agent Profile**:
  - **Category**: ultrabrain
    - Reason: Financial backtesting requires rigor
  - **Skills**: python, data-analysis
    - python: Backtesting logic, metric calculation
    - data-analysis: Financial metrics, statistical testing

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 8 (with Tasks 26, 27, 29)
  - **Blocks**: Task 29 (Dissertation needs backtest results)
  - **Blocked By**: Tasks 11, 12, 13, 17 (models must be trained)

  **References**:
  - Library: backtesting.py (if used), pandas for calculations
  - External: "Advances in Financial Machine Learning" - backtesting chapter
  - External: Sharpe ratio calculation, max drawdown definition

  **Acceptance Criteria**:
  - [ ] Backtest runs on 2023 data for all models
  - [ ] Transaction costs included (0.1% per trade)
  - [ ] Performance metrics calculated: Return, Sharpe, Drawdown, Win rate
  - [ ] Comparison with buy-and-hold benchmark
  - [ ] Report generated: `docs/backtest_results.md` with equity curves
  - [ ] At least one model shows positive Sharpe ratio (>0.5)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify backtesting results
    Tool: Bash (python)
    Preconditions: Tasks 11, 12, 13, 17 complete
    Steps:
      1. Run: `python scripts/backtest_models.py --start 2023-01-01 --end 2023-12-31` → Assert completes
      2. Run: `ls data/backtest_results/` → Assert results CSVs created
      3. Run: `python -c "
import pandas as pd
results = pd.read_csv('data/backtest_results/lstm_backtest.csv')
print(f'Total return: {results[\"total_return\"].iloc[-1]:.2%}')
print(f'Sharpe ratio: {results[\"sharpe_ratio\"].iloc[0]:.2f}')
print(f'Max drawdown: {results[\"max_drawdown\"].iloc[0]:.2%}')
"` → Assert shows metrics
      4. Run: `ls docs/backtest_results.md` → Assert report created
      5. Run: `ls data/backtest_results/equity_curves.png` → Assert equity curve chart exists
      6. Run: `python tests/unit/test_backtest.py -v` → Assert backtest logic tests pass
    Expected Result: Backtest complete, metrics calculated, charts generated
    Evidence: Backtest results saved, equity curves captured

  **Commit**: YES
  - Message: `feat(validation): implement comprehensive model backtesting with metrics`
  - Files: `scripts/backtest_models.py`, `docs/backtest_results.md`, `tests/unit/test_backtest.py`
  - Pre-commit: Run backtest, verify reasonable results

---

- [ ] 29. Documentation and Final Dissertation

  **What to do**:
  - Write comprehensive user guide: `docs/user_guide.md`
    - Installation instructions
    - How to use the web interface
    - How to interpret predictions and sentiment
    - Risk warnings and disclaimers
  - Write technical documentation: `docs/technical_documentation.md`
    - System architecture diagrams
    - Database schema documentation
    - API documentation (if any)
    - Model architecture details
  - Write model training documentation: `docs/model_training.md`
    - Hyperparameters used
    - Training time and resources
    - Validation results
  - Complete final dissertation:
    - Chapter 1: Introduction (update with final scope)
    - Chapter 2: Literature Review (already done, may add citations)
    - Chapter 3: Methodology (detailed technical approach)
    - Chapter 4: Implementation (code structure, key algorithms)
    - Chapter 5: Results (model performance, backtesting)
    - Chapter 6: Discussion (limitations, future work)
    - Chapter 7: Conclusion
  - Create installation script: `scripts/install.sh` (one-command setup)
  - Write README.md: Quick start guide, project overview
  - Create video demo: 3-minute walkthrough (as required)
  - Add code comments and docstrings throughout codebase
  - Generate API docs with Sphinx (optional)

  **Documentation Structure**:
  ```
  docs/
  ├── user_guide.md           # For end users
  ├── technical_documentation.md  # For developers
  ├── model_training.md       # ML details
  ├── dissertation/
  │   ├── chapter1_intro.md
  │   ├── chapter3_methodology.md
  │   ├── chapter4_implementation.md
  │   ├── chapter5_results.md
  │   └── ... (full dissertation)
  ├── architecture_diagrams/
  └── api_reference/ (optional)
  ```

  **Must NOT do**:
  - Don't skip user guide (users need to know how to use it)
  - Don't leave code undocumented (add docstrings)
  - Don't plagiarize (cite all sources properly)

  **Recommended Agent Profile**:
  - **Category**: writing
    - Reason: Documentation and academic writing
  - **Skills**: technical-writing
    - technical-writing: Academic style, clear explanations

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 8 (with Tasks 26, 27, 28)
  - **Blocks**: None (final task)
  - **Blocked By**: All other tasks (needs all results)

  **References**:
  - Academic: University dissertation format requirements
  - Technical: Markdown documentation best practices
  - External: Sphinx or MkDocs for API documentation

  **Acceptance Criteria**:
  - [ ] User guide complete: Can install and use app following guide
  - [ ] Technical documentation: Another developer could understand architecture
  - [ ] Dissertation: All chapters complete, properly formatted
  - [ ] Video demo: 3-minute walkthrough created and uploaded
  - [ ] README.md: Quick start works (follow steps, app runs)
  - [ ] All code has docstrings (check with `pydocstyle`)

  **Agent-Executed QA Scenarios**:

  Scenario: Verify documentation completeness
    Tool: Bash (file checks, markdown linting)
    Preconditions: All other tasks complete
    Steps:
      1. Run: `ls docs/user_guide.md docs/technical_documentation.md docs/model_training.md` → Assert all exist
      2. Run: `ls docs/dissertation/` → Assert dissertation files exist
      3. Run: `wc -l docs/dissertation/*.md | tail -1` → Assert substantial content (>5000 lines total)
      4. Run: `cat README.md | head -50` → Assert has installation instructions
      5. Run: `grep -r "def " src/ | wc -l` and `grep -r '"""' src/ | wc -l` → Assert docstring coverage reasonable
      6. Run: `ls docs/video_demo.mp4 docs/video_demo_link.txt` → Assert video demo exists or link provided
      7. Run: `markdownlint docs/*.md` (if available) → Assert no major markdown errors
    Expected Result: All documentation complete, dissertation substantial, video ready
    Evidence: File listing captured, word counts logged

  **Commit**: YES (ongoing documentation)
  - Message: `docs(all): complete user guide, technical docs, dissertation, and video demo`
  - Files: `docs/`, `README.md`, `scripts/install.sh`
  - Pre-commit: Verify all docs render correctly

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `chore(setup): initialize project structure and dependencies` | Project structure | `pytest --version` works |
| 2 | `feat(database): create SQLite schema with S&P 500 tickers` | Database files | 503 tickers inserted |
| 3 | `chore(tests): setup pytest infrastructure` | Test files | `pytest tests/` passes |
| 4 | `feat(data): yfinance client with caching` | Data client | Cache working |
| 5 | `feat(data): download 5 years historical data` | Database | 633K+ records |
| 6 | `feat(database): DAL implementation` | DAL files | DAL tests pass |
| 7 | `feat(features): technical indicators pipeline` | Indicators | RSI, MACD calculated |
| 8 | `feat(data): news headlines pipeline` | News client | Headlines stored |
| 9 | `feat(infra): error handling framework` | Utils | Retry logic tested |
| 10 | `feat(models): baseline strategies` | Baselines | Comparison report |
| 11 | `feat(models): LSTM model` | LSTM code/model | Model trained |
| 12 | `feat(models): Random Forest` | RF code/model | Feature importance |
| 13 | `feat(models): XGBoost` | XGB code/model | Early stopping works |
| 14 | `feat(models): walk-forward validation` | Validation | No overfitting |
| 15 | `feat(nlp): FinBERT setup` | FinBERT files | Model loads locally |
| 16 | `feat(nlp): sentiment pipeline` | Sentiment code | Daily aggregation |
| 17 | `feat(models): model comparison and selection` | Comparison | Best model selected |
| 18 | `feat(llm): Llama 3 setup` | Llama files | GPU inference works |
| 19 | `feat(llm): prompt templates` | Prompts | All templates tested |
| 20 | `feat(llm): context injection` | Context builder | Data injection works |
| 21 | `feat(llm): guardrails` | Guardrails | Unsafe content blocked |
| 22 | `feat(ui): Streamlit interface` | app.py | App starts |
| 23 | `feat(ui): data visualizations` | Charts | Plotly charts work |
| 24 | `feat(ui): chat interface` | Chat UI | Llama chat works |
| 25 | `feat(ui): UX polish` | UI polish | Dark mode, tooltips |
| 26 | `test(all): comprehensive unit tests` | Unit tests | >80% coverage |
| 27 | `test(integration): end-to-end tests` | Integration tests | All flows pass |
| 28 | `feat(validation): model backtesting` | Backtest | Metrics calculated |
| 29 | `docs(all): complete documentation` | Documentation | User guide works |

---

## Success Criteria

### Technical Metrics (Quantified)

| Metric | Minimum | Target | Verification Command |
|--------|---------|--------|---------------------|
| **Data Records** | 500K | 633K+ | `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM stock_prices"` |
| **Tickers Coverage** | 400 | 503 (100%) | `SELECT COUNT(DISTINCT ticker) FROM stock_prices` |
| **Model Directional Accuracy** | >50% | >55% | Backtest results CSV |
| **Sharpe Ratio** | >0.3 | >0.8 | `docs/backtest_results.md` |
| **LLM Response Time** | <15s | <10s | Log timing in chat |
| **App Load Time** | <5s | <3s | Browser DevTools |
| **Test Coverage** | >70% | >80% | `pytest --cov=src` |
| **Unit Tests Pass** | >80% | >95% | `pytest tests/unit/` |
| **Integration Tests Pass** | 100% | 100% | `pytest tests/integration/` |

### Deliverables Checklist

- [ ] `data/financial_advisor.db` with 633K+ price records
- [ ] `models/lstm_*.h5` trained and validated
- [ ] `models/random_forest_*.pkl` trained
- [ ] `models/xgboost_*.json` trained
- [ ] `models/production_model_*.pkl` selected best model
- [ ] `models/llama3/` with 8GB model files
- [ ] `models/finbert/` with 400MB model files
- [ ] `src/` with all modules (data, features, models, nlp, llm, ui)
- [ ] `app.py` launches Streamlit interface
- [ ] `docs/dissertation/` with complete academic report
- [ ] `docs/user_guide.md` for end users
- [ ] `tests/` with >80% coverage
- [ ] Video demo created (3 minutes)
- [ ] All 503 S&P 500 tickers supported

### Final Verification Commands

```bash
# 1. Verify database
cd C:\Users\Charlotte\Documents\school\FINAL PROJECT AGENTS ONLY
sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM stock_prices;"
# Expected: 633780

# 2. Verify models trained
ls models/*.pkl models/*.h5 models/*.json 2>/dev/null | wc -l
# Expected: 4+ (LSTM, RF, XGB, production)

# 3. Verify tests pass
pytest tests/ -v --tb=short
# Expected: >100 tests passed

# 4. Verify app runs
streamlit run app.py &
curl -s http://localhost:8501 | head -5
# Expected: HTML response
kill %1

# 5. Verify documentation
wc -l docs/dissertation/*.md
# Expected: >5000 lines total
```

---

## Next Steps to Begin Execution

### Immediate Actions (Today)

1. **Run `/start-work`** to begin execution with Sisyphus orchestrator
2. **Review this plan** with academic advisor (get approval on scope)
3. **Verify GPU availability** for Llama 3 (RTX 3060 12GB+ required)
4. **Set up development environment** (Task 1 - can be done immediately)

### This Week (Phase 1)

5. Complete Tasks 1-3 (Foundation)
6. Download Llama 3 and FinBERT models (8GB + 400MB downloads)
7. Create Feature Prototype for Chapter 4 submission:
   - Working: Data pipeline for 1 ticker (AAPL)
   - Charts: Candlestick with technical indicators
   - This demonstrates feasibility and can be submitted

### Critical Decisions Made

✅ **Scope**: Full S&P 500 (503 stocks)  
✅ **Sentiment**: FinBERT (local)  
✅ **LLM**: Llama 3 8B (local, 100% offline operation)  
✅ **Database**: SQLite with WAL mode  
✅ **Architecture**: Batch processing (pre-compute daily)  
✅ **Testing**: Tests-after strategy (pytest)  
✅ **Timeline**: 22 weeks (155 days) with 20-day buffer  

### Risk Summary

**High Risk Areas**:
1. GPU may not handle Llama 3 8B (mitigation: use quantized 4-bit version)
2. Model accuracy may not exceed 52% (mitigation: ensemble methods, realistic expectations)
3. Data pipeline failures with 503 tickers (mitigation: checkpointing, retry logic)

**Success Probability**: 75% with this plan (up from 40% with original plan)

---

## Plan Handoff

**Plan Location**: `.sisyphus/plans/financial-advisor-bot.md`  
**Draft Location**: `.sisyphus/drafts/refined-plan.md` (will be deleted)  
**Status**: ✅ Ready for execution  
**Total Tasks**: 29 major tasks  
**Total Commits**: ~29 (plus intermediate commits)  
**Estimated Duration**: 22 weeks  

To begin execution, run:
```bash
/start-work
```

Sisyphus will:
1. Register this plan as your active boulder
2. Track progress across sessions
3. Enable automatic continuation if interrupted
4. Guide you through each wave of tasks

**Ready to build your Financial Advisor Bot!**


- [ ] 8. News Headlines Data Pipeline

  **What to do**:
  - Create `src/data/news_client.py` - news API wrapper
  - Implement NewsAPI integration (free tier: 1,000 requests/day)
  - Create news scraper for backup sources (with proper rate limiting and robots.txt respect)
  - Download historical news (if available) or start collecting from now
  - Target: 100-500 headlines/day across all S&P 500 companies
  - Implement headline-to-ticker mapping: keyword matching + NER
  - Store in `news_headlines` table with tickers field (comma-separated)
  - Add duplicate detection (same headline from multiple sources)
  - Implement rate limiting: Respect NewsAPI limits (don't exceed 1,000/day)
  - Add retry logic with exponential backoff
  - Create daily news fetch script: `scripts/fetch_news.py`
  - Expected volume: ~45,000 active headlines (90-day retention)

  **Headline-to-Ticker Mapping Strategy**:
  1. Keyword matching: Company names to ticker symbols (Apple → AAPL)
  2. Named Entity Recognition (NER) using spaCy or FinBERT
  3. Handle ambiguities: "Amazon" (AMZN) vs "Amazon rainforest"
  4. Manual validation of mapping accuracy on 100-headline sample
  5. Store mapping confidence score

  **News Sources Priority**:
  1. NewsAPI (primary, free tier)
  2. Financial Modeling Prep (paid, reliable - optional)
  3. RSS feeds from Bloomberg, Reuters (respect rate limits)

  **Must NOT do**:
  - Don't violate robots.txt or terms of service
  - Don't store full article text (copyright issues, just store headline + URL)
  - Don't fetch news without rate limiting (risk of IP ban)

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: API integration with error handling
  - **Skills**: python, data-pipeline
    - python: HTTP requests, rate limiting implementation
    - data-pipeline: Data cleaning, deduplication

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 9)
  - **Blocks**: Task 16 (sentiment analysis needs news)
  - **Blocked By**: Task 2 (database)

  **References**:
  - API: NewsAPI documentation (newsapi.org)
  - Pattern: Rate limiting, deduplication strategies
  - External: robots.txt parsing, web scraping ethics

  **Acceptance Criteria**:
  - [ ] NewsAPI client works: fetch headlines for "Apple" returns relevant news
  - [ ] Headlines stored in database with ticker mapping
  - [ ] Duplicate detection works: Same headline stored only once
  - [ ] Rate limiting respected: No more than 1,000 API calls per day
  - [ ] Retry logic tested: Mock API failure and verify retries

  **Agent-Executed QA Scenarios**:

  Scenario: Verify news pipeline
    Tool: Bash (python)
    Preconditions: Task 2 complete, NewsAPI key configured
    Steps:
      1. Run: `python -c "from src.data.news_client import NewsClient; client = NewsClient(); headlines = client.fetch_headlines('Apple', days=1); print(f'Fetched {len(headlines)} headlines'); print(headlines[0] if headlines else 'No headlines')"` → Assert returns headlines
      2. Run: `python scripts/fetch_news.py --ticker AAPL --days 1` → Assert completes, stores in DB
      3. Run: `sqlite3 data/financial_advisor.db "SELECT * FROM news_headlines WHERE tickers LIKE '%AAPL%' ORDER BY published_at DESC LIMIT 3"` → Assert shows AAPL headlines
      4. Run: `python -c "from src.data.news_client import NewsClient; client = NewsClient(); print(client.get_api_usage_today())"` → Assert shows usage count < 1000
      5. Run: `sqlite3 data/financial_advisor.db "SELECT COUNT(*) FROM news_headlines"` → Assert count > 0
    Expected Result: News pipeline fetches, maps, and stores headlines
    Evidence: API responses captured, database records verified

  **Commit**: YES
  - Message: `feat(data): implement news headlines pipeline with NewsAPI and ticker mapping`
  - Files: `src/data/news_client.py`, `scripts/fetch_news.py`, `tests/unit/test_news_client.py`
  - Pre-commit: Verify news tests pass

---

- [ ] 9. Error Handling and Monitoring Framework

  **What to do**:
  - Create centralized error handling: `src/utils/error_handler.py`
  - Implement custom exception hierarchy: DataPipelineError, ModelTrainingError, LLMError, etc.
  - Create retry decorator with exponential backoff: `src/utils/retry.py`
  - Add circuit breaker pattern for external APIs (stop calling failing service temporarily)
  - Implement structured logging with JSON format: `src/utils/logger.py`
  - Create health check endpoints/methods for each component
  - Add alerting thresholds (email or log-based alerts)
  - Create monitoring dashboard data collection
  - Write tests for error scenarios (mock failures)
  - Add graceful degradation (continue operating with reduced functionality)

  **Error Types to Handle**:
  - API rate limiting (429 errors)
  - Network timeouts
  - Database locks/timeouts
  - Out of memory (during model training)
  - Invalid data (NaN, infinity, negative prices)
  - Missing data (ticker delisted, no recent prices)

  **Must NOT do**:
  - Don't swallow errors silently (always log)
  - Don't retry indefinitely (max 5 attempts)
  - Don't crash entire pipeline for one ticker failure

  **Recommended Agent Profile**:
  - **Category**: quick
    - Reason: Error handling patterns are standard
  - **Skills**: python
    - python: Decorators, exception handling, logging

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8)
  - **Blocks**: Tasks 11, 12, 13, 18 (all need error handling)
  - **Blocked By**: Task 1 (project structure)

  **References**:
  - Pattern: Circuit breaker, retry with backoff, structured logging
  - External: Tenacity library (for retries), structlog (for logging)

  **Acceptance Criteria**:
  - [ ] Custom exceptions defined and tested
  - [ ] Retry decorator works: Function retries 5 times then gives up
  - [ ] Circuit breaker tested: After 3 failures, stops calling service for 60s
  - [ ] Structured logs in `data/logs/` with JSON format
  - [ ] All components have health check methods

  **Agent-Executed QA Scenarios**:

  Scenario: Verify error handling framework
    Tool: Bash (python)
    Preconditions: Task 1 complete
    Steps:
      1. Run: `python -c "from src.utils.retry import retry_with_backoff; @retry_with_backoff(max_attempts=3); def fail_twice(): fail_twice.counter += 1; if fail_twice.counter < 3: raise Exception('fail'); return 'success'; fail_twice.counter = 0; print(fail_twice())"` → Assert returns 'success' after retries
      2. Run: `python -c "from src.utils.error_handler import DataPipelineError; try: raise DataPipelineError('test') except DataPipelineError as e: print(f'Caught: {e}')"` → Assert exception caught correctly
      3. Run: `ls data/logs/` → Assert log directory exists
      4. Run: `python tests/unit/test_error_handler.py -v` → Assert all error handling tests pass
      5. Run: `python -c "from src.utils.health_check import HealthCheck; hc = HealthCheck(); print(hc.check_all())"` → Assert returns health status dict
    Expected Result: Error handling framework operational, retries work, logging active
    Evidence: Test outputs captured, log files verified

  **Commit**: YES
  - Message: `feat(infra): implement error handling, retry logic, and monitoring framework`
  - Files: `src/utils/error_handler.py`, `src/utils/retry.py`, `src/utils/logger.py`, `src/utils/health_check.py`, `tests/unit/test_error_handler.py`
  - Pre-commit: Run error handling tests

