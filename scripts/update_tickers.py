
import sys
import logging
from pathlib import Path
from typing import List, Tuple
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

def update_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia and update the database."""
    dal = DataAccessLayer()
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 tickers from {url}...")
    
    try:
        records = fetch_sp500_from_wikipedia(url)
    except Exception as e:
        logger.warning(f"Online fetch failed ({e}). Falling back to local ticker source from scripts/init_db.py.")
        records = list(SP500_TICKERS)

    try:
        count = 0
        updated = 0
        
        for ticker, name, sector, industry, date_added in records:
            # Wikipedia uses dots (BRK.B), yfinance prefers hyphens (BRK-B).
            # We store the canonical wiki symbol and convert in yfinance client.
            
            # Check if exists
            existing = dal.get_ticker_info(ticker)
            if not existing:
                dal.insert_ticker(ticker, name, sector, industry, date_added)
                updated += 1
            else:
                # Optionally update details? For now just skip or overwrite if we want to refresh metadata
                dal.insert_ticker(ticker, name, sector, industry, date_added)
            
            count += 1
            
        logger.info(f"Processed {count} tickers. Added/Updated {updated} new tickers.")
        
    except Exception as e:
        logger.error(f"Failed to update tickers: {e}")
        # Fallback to init_db if Wikipedia fails?
        # For now, just log error.

if __name__ == "__main__":
    update_sp500_tickers()
