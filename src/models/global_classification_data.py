"""Global pooled dataset preparation for RF/XGBoost classification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.data.stock_data import StockDataProcessor
from src.models.data_sources import load_market_data, load_ticker_universe

FEATURE_COLS = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"]


@dataclass
class SplitPayload:
    X: np.ndarray
    y: np.ndarray
    ticker_labels: np.ndarray
    labeled_df: pd.DataFrame


def _to_iso_date(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _resolve_split_dates(unique_dates: np.ndarray, train_split: float, val_split: float) -> Dict[str, pd.DatetimeIndex]:
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1.")
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")
    if len(unique_dates) < 8:
        raise ValueError("Not enough unique dates to build global train/validation/test splits.")

    split_idx = int(len(unique_dates) * train_split)
    split_idx = max(2, min(split_idx, len(unique_dates) - 2))

    train_all = pd.DatetimeIndex(unique_dates[:split_idx])
    test_dates = pd.DatetimeIndex(unique_dates[split_idx:])

    if val_split > 0:
        val_size = max(2, int(len(train_all) * val_split))
        val_size = min(val_size, len(train_all) - 2)
        train_dates = train_all[:-val_size]
        val_dates = train_all[-val_size:]
    else:
        train_dates = train_all
        val_dates = pd.DatetimeIndex([])

    if len(train_dates) < 2:
        raise ValueError("Not enough training dates after validation carve-out.")
    if len(test_dates) < 2:
        raise ValueError("Not enough test dates.")

    return {
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }


def _label_within_split(
    split_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "Close",
) -> SplitPayload:
    if split_df.empty:
        return SplitPayload(
            X=np.empty((0, len(feature_cols))),
            y=np.empty((0,), dtype=np.int64),
            ticker_labels=np.empty((0,), dtype=object),
            labeled_df=split_df.copy(),
        )

    labeled_groups: List[pd.DataFrame] = []
    for _, grp in split_df.groupby("ticker", sort=False):
        grp_sorted = grp.sort_values("date").copy()
        if len(grp_sorted) < 2:
            continue
        grp_sorted["Target"] = (grp_sorted[target_col].shift(-1) > grp_sorted[target_col]).astype(int)
        grp_sorted = grp_sorted.iloc[:-1].copy()
        if grp_sorted.empty:
            continue
        labeled_groups.append(grp_sorted)

    if not labeled_groups:
        return SplitPayload(
            X=np.empty((0, len(feature_cols))),
            y=np.empty((0,), dtype=np.int64),
            ticker_labels=np.empty((0,), dtype=object),
            labeled_df=split_df.iloc[0:0].copy(),
        )

    labeled_df = pd.concat(labeled_groups, axis=0, ignore_index=True)
    X = labeled_df[list(feature_cols)].values
    y = labeled_df["Target"].values.astype(np.int64)
    ticker_labels = labeled_df["ticker"].values.astype(object)
    return SplitPayload(X=X, y=y, ticker_labels=ticker_labels, labeled_df=labeled_df)


def build_global_classification_dataset(
    *,
    data_source: str = "db",
    db_path: str = "data/financial_advisor.db",
    start_date: str | None = None,
    end_date: str | None = None,
    years: int = 10,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Dict[str, object]:
    """
    Build a pooled classification dataset across the full ticker universe.

    Date split policy:
    - train/test is split by global unique dates.
    - validation is carved from the end of the train-date region.
    - labels are built per ticker within each split only.
    """

    requested_tickers = load_ticker_universe(db_path=db_path)
    if not requested_tickers:
        raise ValueError("No tickers available from ticker universe.")

    collected: List[pd.DataFrame] = []
    used_tickers: List[str] = []
    skipped_tickers: List[Dict[str, str]] = []

    for ticker in requested_tickers:
        market_df = load_market_data(
            ticker=ticker,
            source=data_source,
            db_path=db_path,
            start_date=start_date,
            end_date=end_date,
            years=years,
        )
        if market_df.empty:
            skipped_tickers.append({"ticker": ticker, "reason": "no_market_data"})
            continue

        processor = StockDataProcessor(ticker=ticker)
        enriched = processor.add_technical_indicators(market_df).dropna().copy()
        if len(enriched) < 2:
            skipped_tickers.append({"ticker": ticker, "reason": "insufficient_indicator_rows"})
            continue

        enriched = enriched.reset_index()
        if "Date" in enriched.columns:
            enriched = enriched.rename(columns={"Date": "date"})
        elif "date" not in enriched.columns:
            first_col = enriched.columns[0]
            enriched = enriched.rename(columns={first_col: "date"})

        enriched["date"] = pd.to_datetime(enriched["date"]).dt.normalize()
        enriched["ticker"] = ticker.upper()
        required_cols = ["ticker", "date", *FEATURE_COLS]
        enriched = enriched[required_cols].copy()

        collected.append(enriched)
        used_tickers.append(ticker.upper())

    if not collected:
        raise ValueError("No ticker produced valid indicator-ready rows for global training.")

    pooled = pd.concat(collected, axis=0, ignore_index=True)
    pooled = pooled.sort_values(["date", "ticker"]).reset_index(drop=True)
    unique_dates = np.array(sorted(pooled["date"].unique()))

    split_dates = _resolve_split_dates(unique_dates=unique_dates, train_split=train_split, val_split=val_split)
    train_dates = split_dates["train_dates"]
    val_dates = split_dates["val_dates"]
    test_dates = split_dates["test_dates"]

    train_df = pooled[pooled["date"].isin(train_dates)].copy()
    val_df = pooled[pooled["date"].isin(val_dates)].copy()
    test_df = pooled[pooled["date"].isin(test_dates)].copy()

    train_payload = _label_within_split(train_df, FEATURE_COLS, target_col="Close")
    val_payload = _label_within_split(val_df, FEATURE_COLS, target_col="Close")
    test_payload = _label_within_split(test_df, FEATURE_COLS, target_col="Close")

    if len(train_payload.X) == 0:
        raise ValueError("Training split produced no labeled rows in pooled dataset.")
    if len(test_payload.X) == 0:
        raise ValueError("Test split produced no labeled rows in pooled dataset.")

    date_coverage = {
        "overall_start": _to_iso_date(unique_dates[0]),
        "overall_end": _to_iso_date(unique_dates[-1]),
        "train_start": _to_iso_date(train_dates.min()),
        "train_end": _to_iso_date(train_dates.max()),
        "val_start": _to_iso_date(val_dates.min()) if len(val_dates) else None,
        "val_end": _to_iso_date(val_dates.max()) if len(val_dates) else None,
        "test_start": _to_iso_date(test_dates.min()),
        "test_end": _to_iso_date(test_dates.max()),
    }

    split_rows = {
        "train_raw": int(len(train_df)),
        "val_raw": int(len(val_df)),
        "test_raw": int(len(test_df)),
        "train_labeled": int(len(train_payload.X)),
        "val_labeled": int(len(val_payload.X)),
        "test_labeled": int(len(test_payload.X)),
    }

    split_ticker_coverage = {
        "train_raw": int(train_df["ticker"].nunique()) if not train_df.empty else 0,
        "val_raw": int(val_df["ticker"].nunique()) if not val_df.empty else 0,
        "test_raw": int(test_df["ticker"].nunique()) if not test_df.empty else 0,
        "train_labeled": int(train_payload.labeled_df["ticker"].nunique()) if not train_payload.labeled_df.empty else 0,
        "val_labeled": int(val_payload.labeled_df["ticker"].nunique()) if not val_payload.labeled_df.empty else 0,
        "test_labeled": int(test_payload.labeled_df["ticker"].nunique()) if not test_payload.labeled_df.empty else 0,
    }

    metadata: Dict[str, object] = {
        "scope": "global",
        "requested_ticker_count": len(requested_tickers),
        "used_ticker_count": len(used_tickers),
        "used_tickers": used_tickers,
        "skipped_tickers": skipped_tickers,
        "date_coverage": date_coverage,
        "split_rows": split_rows,
        "split_ticker_coverage": split_ticker_coverage,
        "train_dates_count": int(len(train_dates)),
        "val_dates_count": int(len(val_dates)),
        "test_dates_count": int(len(test_dates)),
    }

    return {
        "X_train": train_payload.X,
        "y_train": train_payload.y,
        "train_tickers": train_payload.ticker_labels,
        "X_val": val_payload.X,
        "y_val": val_payload.y,
        "val_tickers": val_payload.ticker_labels,
        "X_test": test_payload.X,
        "y_test": test_payload.y,
        "test_tickers": test_payload.ticker_labels,
        "feature_cols": FEATURE_COLS,
        "metadata": metadata,
    }
