import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.dal import DataAccessLayer
from scripts.init_db import SP500_TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("update_tickers")
CHECKPOINT_FILE = Path("data/download_checkpoint.json")


def _clean_cell_text(value: str) -> str:
    """Normalize table cell text by removing references and whitespace noise."""
    # Remove inline reference markers like "[1]" and compact whitespace.
    cleaned = " ".join(value.replace("\xa0", " ").split())
    if "[" in cleaned:
        cleaned = cleaned.split("[", 1)[0].strip()
    return cleaned


def fetch_sp500_from_wikipedia(url: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Fetch S&P 500 constituents from Wikipedia using BeautifulSoup.

    Returns:
        List of tuples: (ticker, name, sector, industry, date_added)
    """
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    with urlopen(request, timeout=30) as response:
        html = response.read()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", class_="wikitable")
    if table is None:
        raise ValueError("Could not find constituents table on Wikipedia page.")

    rows = table.find_all("tr")
    if not rows:
        raise ValueError("Wikipedia constituents table is empty.")

    header_cells = rows[0].find_all(["th", "td"])
    headers = [_clean_cell_text(cell.get_text(strip=True)) for cell in header_cells]

    normalized_headers = {h.replace(" ", "").lower(): i for i, h in enumerate(headers)}

    def idx(*names: str) -> int:
        for name in names:
            key = name.replace(" ", "").lower()
            if key in normalized_headers:
                return normalized_headers[key]
        raise ValueError(f"Missing required columns {names}. Found headers: {headers}")

    i_symbol = idx("Symbol")
    i_name = idx("Security")
    i_sector = idx("GICS Sector", "GICSSector")
    i_industry = idx("GICS Sub-Industry", "GICSSub-Industry")
    i_date_added = idx("Date added", "Dateadded")

    tickers: List[Tuple[str, str, str, str, str]] = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if not cells:
            continue

        values = [_clean_cell_text(td.get_text(" ", strip=True)) for td in cells]
        max_idx = max(i_symbol, i_name, i_sector, i_industry, i_date_added)
        if len(values) <= max_idx:
            continue

        ticker = values[i_symbol]
        name = values[i_name]
        sector = values[i_sector]
        industry = values[i_industry]
        date_added = values[i_date_added]

        if ticker:
            tickers.append((ticker, name, sector, industry, date_added))

    if not tickers:
        raise ValueError("No tickers parsed from Wikipedia table.")

    return tickers

def _dedupe_by_symbol(records: List[Tuple[str, str, str, str, str]]) -> List[Tuple[str, str, str, str, str]]:
    """Keep one row per symbol (latest occurrence wins) to protect against duplicate source rows."""
    by_symbol: Dict[str, Tuple[str, str, str, str, str]] = {}
    for record in records:
        by_symbol[record[0]] = record
    return list(by_symbol.values())


def _delete_stale_tickers(dal: DataAccessLayer, stale_tickers: List[str]) -> None:
    """Delete stale symbols and their dependent rows."""
    if not stale_tickers:
        return

    placeholders = ",".join("?" for _ in stale_tickers)
    with dal.get_connection() as conn:
        cursor = conn.cursor()

        # Delete child rows before parent rows to keep integrity even when foreign keys are enabled.
        cursor.execute(
            f"""
            DELETE FROM sentiment_scores
            WHERE news_id IN (
                SELECT id FROM news_headlines WHERE ticker IN ({placeholders})
            )
            """,
            stale_tickers,
        )
        cursor.execute(f"DELETE FROM news_headlines WHERE ticker IN ({placeholders})", stale_tickers)
        cursor.execute(f"DELETE FROM stock_prices WHERE ticker IN ({placeholders})", stale_tickers)
        cursor.execute(f"DELETE FROM technical_indicators WHERE ticker IN ({placeholders})", stale_tickers)
        cursor.execute(f"DELETE FROM daily_sentiment WHERE ticker IN ({placeholders})", stale_tickers)
        cursor.execute(f"DELETE FROM predictions WHERE ticker IN ({placeholders})", stale_tickers)
        cursor.execute(f"DELETE FROM model_performance WHERE ticker IN ({placeholders})", stale_tickers)
        cursor.execute(f"DELETE FROM tickers WHERE ticker IN ({placeholders})", stale_tickers)
        conn.commit()


def _sync_download_checkpoint(valid_tickers: Set[str]) -> None:
    """Keep checkpoint entries aligned to the active ticker universe."""
    if not CHECKPOINT_FILE.exists():
        return

    try:
        payload = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not parse checkpoint file {CHECKPOINT_FILE}: {exc}")
        return

    completed = payload.get("completed_tickers", [])
    if not isinstance(completed, list):
        logger.warning("Checkpoint format invalid: completed_tickers is not a list.")
        return

    filtered = [ticker for ticker in completed if ticker in valid_tickers]
    # Preserve order while deduplicating.
    deduped = list(dict.fromkeys(filtered))
    if deduped == completed:
        return

    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed_tickers": deduped}, ensure_ascii=True),
        encoding="utf-8",
    )
    logger.info(
        "Checkpoint synced: %d -> %d completed tickers.",
        len(completed),
        len(deduped),
    )


def update_sp500_tickers(prune_stale: bool = True):
    """Fetch current S&P 500 constituents and sync the local DB ticker universe."""
    dal = DataAccessLayer()

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 tickers from {url}...")

    try:
        records = fetch_sp500_from_wikipedia(url)
    except Exception as e:
        logger.warning(f"Online fetch failed ({e}). Falling back to local ticker source from scripts/init_db.py.")
        records = list(SP500_TICKERS)

    try:
        records = _dedupe_by_symbol(records)
        target_symbols = {ticker for ticker, *_ in records}

        count = 0
        inserted = 0

        for ticker, name, sector, industry, date_added in records:
            existing = dal.get_ticker_info(ticker)
            if not existing:
                dal.insert_ticker(ticker, name, sector, industry, date_added)
                inserted += 1
            else:
                dal.insert_ticker(ticker, name, sector, industry, date_added)
            count += 1

        stale_tickers: List[str] = []
        if prune_stale:
            db_tickers = set(dal.get_all_tickers())
            stale_tickers = sorted(db_tickers - target_symbols)
            _delete_stale_tickers(dal, stale_tickers)

        final_tickers = set(dal.get_all_tickers())
        _sync_download_checkpoint(final_tickers)

        logger.info(
            "S&P 500 sync complete. Source=%d, inserted=%d, removed=%d, final_universe=%d",
            count,
            inserted,
            len(stale_tickers),
            len(final_tickers),
        )

    except Exception as e:
        logger.error(f"Failed to update tickers: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Sync DB ticker universe to current S&P 500 constituents.")
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help="Do not delete tickers that are no longer in the S&P 500.",
    )
    args = parser.parse_args()
    update_sp500_tickers(prune_stale=not args.keep_stale)

if __name__ == "__main__":
    main()
