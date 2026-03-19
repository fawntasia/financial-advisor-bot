# ML Pipeline Audit - 2026-03-19

## Scope
- Audit focus: training/validation methodology for classifier (`RF`, `XGB`) and sequence model (`LSTM`).
- Unit tests were intentionally not run for this audit pass.
- Data source inspected: `data/financial_advisor.db` (max stock price date observed: `2026-03-18`).

## Findings (Pre-fix)
1. Pooled classifier rows were reordered by ticker blocks during label generation, which broke strict chronological ordering before `TimeSeriesSplit`.
2. LSTM validation split used the tail of concatenated pooled training sequences, so validation composition depended on ticker ordering.

## Implemented Fixes

### 1) Chronological pooled ordering for RF/XGB CV
- File: `src/models/global_classification_data.py`
- Change:
  - Replaced ticker-group concat labeling with split-local grouped shift on a globally sorted frame (`date`, `ticker`).
  - Kept split-safe target creation (`next_close` within ticker only, drop boundary rows where next value is unavailable).
- Outcome:
  - Labeled pooled training rows now remain globally time-ordered for downstream `TimeSeriesSplit`.

### 2) Chronological per-ticker validation for LSTM
- File: `scripts/train_lstm.py`
- Changes:
  - Added date-aware sequence construction (`target_dates` carried with sequences).
  - Added per-ticker chronological fit/validation split (`_split_sequences_by_date`) before global pooling.
  - Aggregation now pools pre-split per-ticker fit/validation/test arrays.
  - Metadata/manifest now record validation split basis as `per_ticker_latest_train_dates`.
- Outcome:
  - Validation no longer depends on ticker concatenation order.
  - Each ticker's validation dates are strictly after its own fit-training dates.

## Verification Checks (Post-fix)

### Classifier pooled chronology check
- Result:
  - `rows = 440325`
  - `is_monotonic(date) = True`
  - `num_backsteps = 0`

### LSTM per-ticker boundary check
- Result:
  - `used_tickers = 502`, `skipped = 1`
  - `train_ticker_count = 502`, `validation_ticker_count = 502`
  - Strict per-ticker temporal boundaries validated (`fit_max_date < val_min_date` for all used tickers).

## Notes
- Existing saved classifier artifacts (`models/random_forest_global_metadata.json`, `models/xgboost_global_metadata.json`) were trained up to `2026-02-05`; DB now contains newer data through `2026-03-18`.
- Retraining is recommended if you want artifacts aligned with current data coverage.
