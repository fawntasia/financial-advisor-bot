import json
import sqlite3
from pathlib import Path

import pytest


DB_PATH = Path("data/financial_advisor.db")
CHECKPOINT_PATH = Path("data/download_checkpoint.json")


def _connect_live_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        pytest.skip(f"Live DB not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)


@pytest.mark.integration
@pytest.mark.database
def test_live_db_download_coverage_and_checkpoint_sync():
    conn = _connect_live_db()
    cur = conn.cursor()

    ticker_count = cur.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
    price_ticker_count = cur.execute("SELECT COUNT(DISTINCT ticker) FROM stock_prices").fetchone()[0]
    missing_price_count = cur.execute(
        """
        SELECT COUNT(*)
        FROM tickers t
        WHERE NOT EXISTS (
            SELECT 1 FROM stock_prices s WHERE s.ticker = t.ticker
        )
        """
    ).fetchone()[0]
    conn.close()

    assert ticker_count >= 500, f"Unexpected ticker universe size: {ticker_count}"
    assert price_ticker_count == ticker_count
    assert missing_price_count == 0

    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint file: {CHECKPOINT_PATH}"
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    completed = checkpoint.get("completed_tickers", [])
    assert isinstance(completed, list)
    assert len(set(completed)) == ticker_count


@pytest.mark.integration
@pytest.mark.database
def test_live_db_ohlcv_integrity():
    conn = _connect_live_db()
    cur = conn.cursor()

    epsilon = 1e-6

    # Enforce hard errors only (impossible ranges, missing core fields with non-placeholder pattern).
    impossible_range_rows = cur.execute(
        """
        SELECT COUNT(*)
        FROM stock_prices
        WHERE high < low
           OR volume < 0
        """
    ).fetchone()[0]

    close_outside_rows = cur.execute(
        f"""
        SELECT COUNT(*)
        FROM stock_prices
        WHERE close IS NOT NULL
          AND low IS NOT NULL
          AND high IS NOT NULL
          AND (close < low - {epsilon} OR close > high + {epsilon})
        """
    ).fetchone()[0]

    # Open can occasionally be slightly inconsistent around splits/adjustments; keep a tight threshold.
    open_outside_rows = cur.execute(
        f"""
        SELECT COUNT(*)
        FROM stock_prices
        WHERE open IS NOT NULL
          AND low IS NOT NULL
          AND high IS NOT NULL
          AND (open < low - {epsilon} OR open > high + {epsilon})
        """
    ).fetchone()[0]

    partial_null_rows = cur.execute(
        """
        SELECT COUNT(*)
        FROM stock_prices
        WHERE (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL)
          AND NOT (
              open IS NULL
              AND high IS NULL
              AND low IS NULL
              AND close IS NULL
              AND IFNULL(volume, 0) = 0
              AND adj_close IS NULL
          )
        """
    ).fetchone()[0]

    placeholder_null_rows = cur.execute(
        """
        SELECT COUNT(*)
        FROM stock_prices
        WHERE open IS NULL
          AND high IS NULL
          AND low IS NULL
          AND close IS NULL
          AND IFNULL(volume, 0) = 0
          AND adj_close IS NULL
        """
    ).fetchone()[0]

    duplicate_rows = cur.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT ticker, date, COUNT(*) AS c
            FROM stock_prices
            GROUP BY ticker, date
            HAVING c > 1
        )
        """
    ).fetchone()[0]

    future_rows = cur.execute(
        """
        SELECT COUNT(*)
        FROM stock_prices
        WHERE date > date('now')
        """
    ).fetchone()[0]

    conn.close()

    assert impossible_range_rows == 0
    assert close_outside_rows == 0
    assert open_outside_rows <= 5
    assert partial_null_rows == 0
    assert placeholder_null_rows <= 5
    assert duplicate_rows == 0
    assert future_rows == 0
