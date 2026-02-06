import streamlit as st

def show_stock_analysis():
    st.header("📈 Stock Analysis")
    ticker = st.session_state.get("ticker", "AAPL")
    st.write(f"Displaying analysis for: **{ticker}**")
    
    with st.spinner(f"Fetching data for {ticker}..."):
        # Simulate loading with a small delay if needed, 
        # but here we just wrap the existing logic
        st.info("Placeholder: Ticker-specific charts, technical indicators, and sentiment summary will go here.")
        
        # Placeholder for chart
        st.subheader("Technical Overview", help="A visual representation of stock price movements and technical indicators.")
        st.image("https://via.placeholder.com/800x400.png?text=Stock+Chart+Placeholder", use_container_width=True)

def show_chat_interface():
    """
    Displays the AI chat interface using ChatManager.
    """
    if "chat_manager" in st.session_state:
        st.session_state.chat_manager.display_chat()
    else:
        st.error("Chat Manager not initialized. Please check the application setup.")

def show_dashboard():
    st.header("📊 Market Dashboard")
    st.write("Overview of S&P 500 and your tracked assets.")
    
    with st.status("Loading market data...", expanded=True) as status:
        st.write("Fetching S&P 500 index...")
        # (Mock loading steps)
        st.write("Analyzing market volatility...")
        st.write("Calculating global sentiment...")
        status.update(label="Market Data Loaded", state="complete", expanded=False)

    st.info("Placeholder: Global market trends, top gainers/losers, and news feed.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("S&P 500", "5,000.00", "+1.2%", help="Standard & Poor's 500 Index tracking 500 large companies.")
    col2.metric("Market Volatility", "15.4", "-2.1%", help="VIX Index measuring market expectations of near-term volatility.")
    col3.metric("Sentiment Score", "0.65", "Bullish", help="Aggregated sentiment score from financial news (0-1).")

def show_disclaimer():
    st.markdown("---")
    st.caption("**Mandatory Financial Disclaimer**: This application is for educational and informational purposes only. "
               "The content provided here does not constitute financial, investment, tax, or legal advice. "
               "Investing in financial markets involves risks. Always consult with a qualified professional before making any investment decisions.")
