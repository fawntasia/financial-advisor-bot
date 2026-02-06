import streamlit as st
import sys
import os
import re

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.database.dal import DataAccessLayer
from src.ui.chat import ChatManager
from src.ui.views import (
    show_stock_analysis, 
    show_chat_interface, 
    show_dashboard, 
    show_disclaimer
)

# Page configuration
st.set_page_config(
    page_title="Financial Advisor Bot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Visual/UI Context applied)
st.markdown("""
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
""", unsafe_allow_html=True)

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

def main():
    initialize_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Antigravity Finance</div>', unsafe_allow_html=True)
        
        # Ticker Search
        st.subheader("🔍 Stock Search")
        ticker_input = st.text_input("Enter Ticker (e.g., TSLA, MSFT)", value=st.session_state.ticker)
        new_ticker = ticker_input.upper() if ticker_input else ""
        
        if new_ticker and new_ticker != st.session_state.ticker:
            # Support standard symbols and class-share tickers like BRK.B
            ticker_pattern = r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$"
            if re.fullmatch(ticker_pattern, new_ticker):
                st.session_state.ticker = new_ticker
                st.toast(f"Ticker updated to: {new_ticker}")
            else:
                st.error(f"Invalid ticker format: {new_ticker}. Example valid symbols: AAPL, MSFT, BRK.B")
            
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
