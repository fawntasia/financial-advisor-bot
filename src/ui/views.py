"""Streamlit view components for the Financial Advisor Bot UI."""

from math import isnan
from pathlib import Path

import streamlit as st

from src.database.dal import DataAccessLayer
from src.ui.charts import ChartGenerator
from src.ui.classification_visualization import generate_classification_signal_data
from src.ui.lstm_visualization import generate_lstm_visualization_data


def _format_ratio(value) -> str:
    """Format a ratio value as percentage text, or N/A when unavailable."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if isnan(numeric):
        return "N/A"
    return f"{numeric:.2%}"


def _format_decimal(value, digits: int = 3) -> str:
    """Format a decimal value, or N/A when unavailable."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if isnan(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}"


def _render_lstm_tab(ticker: str, dal: DataAccessLayer, chart_generator: ChartGenerator):
    """Render the existing LSTM chart and forecast section."""
    with st.spinner(f"Fetching data and model output for {ticker}..."):
        try:
            viz_data = generate_lstm_visualization_data(
                ticker=ticker,
                dal=dal,
                history_window=180,
                forecast_horizon=7,
                eval_window=90,
            )
        except Exception as exc:
            st.error(f"Unable to load analysis for {ticker}: {exc}")
            st.info(
                "Confirm that price history exists in `data/financial_advisor.db` and that LSTM artifacts are available in `models/`."
            )
            return

    st.subheader(
        "Technical Overview",
        help="Candlestick, volume, RSI, MACD, and Bollinger Bands from recent price history.",
    )
    technical_fig = chart_generator.create_ohlcv_chart(viz_data["technical_df"], ticker)
    st.plotly_chart(technical_fig, use_container_width=True)

    st.subheader(
        "LSTM Forecast Visualization",
        help="Recent model fit (actual vs predicted) and a 7-day forward recursive forecast.",
    )
    forecast_fig = chart_generator.create_lstm_prediction_chart(
        history_df=viz_data["history_df"],
        backtest_df=viz_data["backtest_df"],
        forecast_df=viz_data["forecast_df"],
        ticker=ticker,
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"${viz_data['last_close']:.2f}")

    next_close = viz_data["next_predicted_close"]
    if isnan(next_close):
        col2.metric("Next Predicted Close", "N/A")
    else:
        delta = next_close - viz_data["last_close"]
        col2.metric("Next Predicted Close", f"${next_close:.2f}", f"{delta:+.2f}")

    rmse = viz_data["rmse"]
    mape = viz_data["mape"]
    col3.metric("Evaluation RMSE", "N/A" if isnan(rmse) else f"${rmse:.2f}")
    col4.metric("Evaluation MAPE", "N/A" if isnan(mape) else f"{mape:.2f}%")

    artifact_name = Path(viz_data["artifact_paths"]["model_path"]).name
    feature_text = ", ".join(viz_data["feature_columns"])
    st.caption(
        f"Model artifact: `{artifact_name}` | Sequence length: `{viz_data['sequence_length']}` | Features: `{feature_text}`"
    )

    if viz_data["scaler_ticker"] != ticker:
        st.warning(
            f"Ticker-specific scalers were not found for {ticker}. Using scaler from {viz_data['scaler_ticker']} as fallback."
        )


def _render_classifier_tab(ticker: str, dal: DataAccessLayer, model_type: str, label: str):
    """Render concise classifier signal cards for RF/XGBoost."""
    with st.spinner(f"Fetching {label} signal for {ticker}..."):
        try:
            signal = generate_classification_signal_data(
                ticker=ticker,
                model_type=model_type,
                dal=dal,
                eval_window=90,
                persist_prediction=True,
            )
        except Exception as exc:
            st.error(f"Unable to load {label} signal for {ticker}: {exc}")
            if model_type == "rf":
                st.info(
                    "Expected artifacts in `models/`: `random_forest_global.pkl` and `random_forest_global_metadata.json`."
                )
            else:
                st.info("Expected artifacts in `models/`: `xgboost_global.json` and `xgboost_global_metadata.json`.")
            return

    st.subheader(
        f"{label} Signal",
        help="Direction and confidence for the next business day, inferred from local model artifacts.",
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Direction", signal["predicted_label"])
    col2.metric("UP Probability", _format_ratio(signal["prob_up"]))
    col3.metric("Decision Threshold", _format_decimal(signal["decision_threshold"], digits=2))
    col4.metric("Confidence", _format_ratio(signal["confidence"]))

    global_bal = _format_ratio(signal["global_test_metrics"].get("balanced_accuracy"))
    global_f1 = _format_ratio(signal["global_test_metrics"].get("f1"))
    ticker_bal = _format_ratio(signal["ticker_test_metrics"].get("balanced_accuracy"))
    artifact_name = Path(signal["artifact_paths"]["model_path"]).name
    st.caption(
        f"Prediction date: `{signal['prediction_date']}` | Artifact: `{artifact_name}` | "
        f"Global test balanced accuracy: `{global_bal}` | Global test F1: `{global_f1}` | "
        f"{ticker} test balanced accuracy: `{ticker_bal}`"
    )


def show_stock_analysis():
    """Render ticker-specific technical charts and model signal tabs."""
    st.header("Stock Analysis")
    ticker = st.session_state.get("ticker", "AAPL")
    st.write(f"Displaying analysis for: **{ticker}**")

    dal = st.session_state.get("dal")
    if dal is None:
        dal = DataAccessLayer()
        st.session_state.dal = dal

    chart_generator = ChartGenerator()
    lstm_tab, rf_tab, xgb_tab = st.tabs(["LSTM", "Random Forest", "XGBoost"])

    with lstm_tab:
        _render_lstm_tab(ticker=ticker, dal=dal, chart_generator=chart_generator)

    with rf_tab:
        _render_classifier_tab(ticker=ticker, dal=dal, model_type="rf", label="Random Forest")

    with xgb_tab:
        _render_classifier_tab(ticker=ticker, dal=dal, model_type="xgb", label="XGBoost")


def show_chat_interface():
    """
    Displays the AI chat interface using ChatManager.
    """
    if "chat_manager" in st.session_state:
        st.session_state.chat_manager.display_chat()
    else:
        st.error("Chat Manager not initialized. Please check the application setup.")


def show_dashboard():
    st.header("Market Dashboard")
    st.write("Overview of S&P 500 and your tracked assets.")

    with st.status("Loading market data...", expanded=True) as status:
        st.write("Fetching S&P 500 index...")
        st.write("Analyzing market volatility...")
        st.write("Calculating global sentiment...")
        status.update(label="Market Data Loaded", state="complete", expanded=False)

    st.info("Placeholder: Global market trends, top gainers/losers, and news feed.")

    col1, col2, col3 = st.columns(3)
    col1.metric("S&P 500", "5,000.00", "+1.2%", help="Standard & Poor's 500 Index tracking 500 large companies.")
    col2.metric(
        "Market Volatility",
        "15.4",
        "-2.1%",
        help="VIX Index measuring market expectations of near-term volatility.",
    )
    col3.metric(
        "Sentiment Score",
        "0.65",
        "Bullish",
        help="Aggregated sentiment score from financial news (0-1).",
    )


def show_disclaimer():
    st.markdown("---")
    st.caption(
        "**Mandatory Financial Disclaimer**: This application is for educational and informational purposes only. "
        "The content provided here does not constitute financial, investment, tax, or legal advice. "
        "Investing in financial markets involves risks. Always consult with a qualified professional before making any investment decisions."
    )
