import os
import re
import sys
from typing import Dict, List

import streamlit as st

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.database.dal import DataAccessLayer
from src.ui.chat import ChatManager
from src.ui.views import (
    show_chat_interface,
    show_dashboard,
    show_disclaimer,
    show_stock_analysis,
)

# Page configuration
st.set_page_config(
    page_title="Financial Advisor Bot",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Visual/UI Context applied)
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

    :root {
        --primary: #00f2fe;
        --secondary: #4facfe;
        --accent: #f093fb;
        --bg: #0e1117;
        --card-bg: #1a1c23;
        --text: #e0e0e0;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stSidebar {
        background-color: var(--bg);
        border-right: 1px solid #30363d;
    }

    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--secondary);
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(79, 172, 254, 0.3);
    }

    /* Card styling */
    .stMetric {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-5px);
        border-color: var(--secondary);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg);
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize necessary session state variables."""
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Dashboard"
    if "dal" not in st.session_state:
        st.session_state.dal = DataAccessLayer()
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager(st.session_state.dal)
    if "ticker_universe" not in st.session_state:
        try:
            st.session_state.ticker_universe = st.session_state.dal.get_ticker_universe()
        except Exception:
            st.session_state.ticker_universe = []


def _format_ticker_option(row: Dict) -> str:
    """Create a searchable display label for ticker browse mode."""
    ticker = str(row.get("ticker", "")).upper()
    name = str(row.get("name", "") or "Unknown Company")
    sector = str(row.get("sector", "") or "Unknown Sector")
    return f"{ticker} | {name} | {sector}"


def _extract_ticker_from_option(option: str) -> str:
    """Extract ticker symbol from formatted option text."""
    return option.split(" | ", 1)[0].strip().upper()


def _sync_ticker_from_text_input():
    """Allow manual ticker entry for quick direct jumps."""
    ticker_input = st.text_input("Enter Ticker (e.g., TSLA, MSFT)", value=st.session_state.ticker)
    new_ticker = ticker_input.upper().strip() if ticker_input else ""

    if new_ticker and new_ticker != st.session_state.ticker:
        ticker_pattern = r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$"
        if re.fullmatch(ticker_pattern, new_ticker):
            st.session_state.ticker = new_ticker
            st.toast(f"Ticker updated to: {new_ticker}")
        else:
            st.error(f"Invalid ticker format: {new_ticker}. Example valid symbols: AAPL, MSFT, BRK.B")


def _render_ticker_selector():
    """
    Render ticker interaction controls:
    1) Manual ticker input
    2) Searchable browse list of the full S&P 500 universe
    """
    _sync_ticker_from_text_input()

    universe: List[Dict] = st.session_state.get("ticker_universe", [])
    if not universe:
        st.warning("Ticker list unavailable. Initialize and ingest the database to browse all stocks.")
        return

    options = [_format_ticker_option(row) for row in universe if row.get("ticker")]
    if not options:
        st.warning("No ticker entries were found in the database.")
        return

    current_ticker = st.session_state.ticker.upper()
    if not any(opt.startswith(f"{current_ticker} |") for opt in options):
        options = [f"{current_ticker} | Custom Symbol | Manual Entry"] + options

    default_option = next((opt for opt in options if opt.startswith(f"{current_ticker} |")), options[0])

    selected_option = st.selectbox(
        "Browse S&P 500 Constituents",
        options=options,
        index=options.index(default_option),
        help="Type to search by ticker, company name, or sector.",
    )
    selected_ticker = _extract_ticker_from_option(selected_option)

    if selected_ticker != st.session_state.ticker:
        st.session_state.ticker = selected_ticker
        st.toast(f"Ticker updated to: {selected_ticker}")


def main():
    initialize_session_state()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Antigravity Finance</div>', unsafe_allow_html=True)

        st.subheader("Stock Search")
        _render_ticker_selector()

        st.markdown("---")

        # View Selection
        st.subheader("Navigation")
        view_options = ["Dashboard", "Stock Analysis", "Chat Interface"]
        selected_view = st.radio("Go to:", view_options, index=view_options.index(st.session_state.current_view))
        st.session_state.current_view = selected_view

        st.markdown("---")
        st.info(f"Currently tracking: **{st.session_state.ticker}**")

    # Main Content Area
    if st.session_state.current_view == "Dashboard":
        show_dashboard()
    elif st.session_state.current_view == "Stock Analysis":
        show_stock_analysis()
    elif st.session_state.current_view == "Chat Interface":
        show_chat_interface()

    # Financial Disclaimer (Mandatory on all pages)
    show_disclaimer()


if __name__ == "__main__":
    main()
