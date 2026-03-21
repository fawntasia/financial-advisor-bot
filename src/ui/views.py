"""Streamlit view components for the Financial Advisor Bot UI."""

from datetime import datetime
from math import isnan
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from scripts.ingest_data import ingest_data, ingest_news_data
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


def _format_currency(value, digits: int = 2) -> str:
    """Format numeric value as USD text, or N/A when unavailable."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if isnan(numeric):
        return "N/A"
    return f"${numeric:.{digits}f}"


def _format_signed_percent(value, digits: int = 2) -> str:
    """Format a decimal ratio as signed percentage text."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if isnan(numeric):
        return "N/A"
    return f"{numeric:+.{digits}%}"


def _signed_sentiment_score(avg_positive, avg_negative) -> float:
    """Map average sentiment probabilities to a signed score in [-1, 1]."""
    try:
        pos = float(avg_positive or 0.0)
    except (TypeError, ValueError):
        pos = 0.0
    try:
        neg = float(avg_negative or 0.0)
    except (TypeError, ValueError):
        neg = 0.0
    return pos - neg


def _run_sentiment_job(
    dal: DataAccessLayer,
    scope: str,
    batch_size: int,
    limit: int,
    model_path: str = "models/finbert",
    ticker: str = "",
    date_str: str = "",
):
    """Execute a sentiment analysis job and return pipeline stats."""
    try:
        from src.nlp.sentiment_pipeline import SentimentPipeline
    except Exception as exc:
        raise RuntimeError(
            "Sentiment dependencies are not available. Install/repair transformers + torch, then retry."
        ) from exc

    pipeline = SentimentPipeline(dal=dal, model_path=model_path)
    if pipeline.loader.model is None or pipeline.loader.tokenizer is None:
        raise RuntimeError(
            "FinBERT model is not loaded. Ensure files exist in models/finbert (or provide a valid model path)."
        )

    if scope == "ticker":
        return pipeline.process_unprocessed_for_ticker(
            ticker=ticker,
            batch_size=batch_size,
            limit=limit,
        )

    if scope == "date":
        return pipeline.process_date(date=date_str, batch_size=batch_size)

    return pipeline.process_unprocessed(batch_size=batch_size, limit=limit)


def _empty_market_snapshot() -> Dict:
    """Return default structure for market snapshot rendering."""
    return {
        "latest_date": None,
        "tracked_tickers": 0,
        "tickers_with_prices": 0,
        "tickers_with_returns": 0,
        "advancers": 0,
        "decliners": 0,
        "unchanged": 0,
        "breadth_ratio": None,
        "cross_sectional_volatility": None,
        "average_move": None,
        "median_move": None,
        "top_gainers": [],
        "top_losers": [],
    }


def _fetch_market_snapshot(dal: DataAccessLayer) -> Dict:
    """Build market breadth + top mover snapshot from latest stock prices."""
    with dal.get_connection() as conn:
        ticker_count_df = pd.read_sql_query(
            "SELECT COUNT(*) AS count FROM tickers",
            conn,
        )
        tracked_tickers = int(ticker_count_df.iloc[0]["count"]) if not ticker_count_df.empty else 0

        latest_df = pd.read_sql_query(
            """
            WITH latest_market_date AS (
                SELECT MAX(date) AS latest_date
                FROM stock_prices
            ),
            latest_prices AS (
                SELECT
                    s.ticker,
                    s.date,
                    s.close,
                    LAG(s.close) OVER (
                        PARTITION BY s.ticker
                        ORDER BY s.date
                    ) AS prev_close
                FROM stock_prices s
            )
            SELECT
                lp.ticker,
                lp.date,
                lp.close,
                lp.prev_close,
                CASE
                    WHEN lp.prev_close IS NULL OR lp.prev_close = 0 THEN NULL
                    ELSE (lp.close - lp.prev_close) / lp.prev_close
                END AS pct_change
            FROM latest_prices lp
            JOIN latest_market_date lm
                ON lp.date = lm.latest_date
            """,
            conn,
        )

    if latest_df.empty:
        snapshot = _empty_market_snapshot()
        snapshot["tracked_tickers"] = tracked_tickers
        return snapshot

    latest_df["pct_change"] = pd.to_numeric(latest_df["pct_change"], errors="coerce")
    latest_df["close"] = pd.to_numeric(latest_df["close"], errors="coerce")
    valid_returns = latest_df.dropna(subset=["pct_change"]).copy()

    total = int(len(valid_returns))
    advancers = int((valid_returns["pct_change"] > 0).sum()) if total else 0
    decliners = int((valid_returns["pct_change"] < 0).sum()) if total else 0
    unchanged = max(total - advancers - decliners, 0)

    top_gainers: List[Dict] = []
    top_losers: List[Dict] = []
    if total:
        top_n = 10
        top_gainers = valid_returns.nlargest(top_n, "pct_change")[["ticker", "close", "pct_change"]].to_dict("records")
        top_losers = valid_returns.nsmallest(top_n, "pct_change")[["ticker", "close", "pct_change"]].to_dict("records")

    return {
        "latest_date": str(latest_df["date"].max()) if "date" in latest_df.columns else None,
        "tracked_tickers": tracked_tickers,
        "tickers_with_prices": int(latest_df["ticker"].nunique()),
        "tickers_with_returns": int(valid_returns["ticker"].nunique()),
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "breadth_ratio": (advancers / total) if total else None,
        "cross_sectional_volatility": float(valid_returns["pct_change"].std(ddof=0)) if total else None,
        "average_move": float(valid_returns["pct_change"].mean()) if total else None,
        "median_move": float(valid_returns["pct_change"].median()) if total else None,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
    }


def _sentiment_label_from_score(score: Optional[float]) -> str:
    """Map signed sentiment score to a directional label."""
    if score is None:
        return "N/A"
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return "N/A"
    if isnan(numeric):
        return "N/A"
    if numeric >= 0.05:
        return "Bullish"
    if numeric <= -0.05:
        return "Bearish"
    return "Neutral"


def _empty_sentiment_snapshot() -> Dict:
    """Return default structure for global sentiment rendering."""
    return {
        "date": None,
        "ticker_count": 0,
        "news_count": 0,
        "avg_confidence": None,
        "avg_net_score": None,
        "weighted_net_score": None,
        "label": "N/A",
    }


def _fetch_global_sentiment_snapshot(dal: DataAccessLayer) -> Dict:
    """Build a global sentiment snapshot from the latest aggregate day."""
    with dal.get_connection() as conn:
        sentiment_df = pd.read_sql_query(
            """
            WITH latest_sentiment_date AS (
                SELECT MAX(date) AS latest_date
                FROM daily_sentiment
            )
            SELECT
                ls.latest_date AS date,
                COUNT(*) AS ticker_count,
                COALESCE(SUM(COALESCE(ds.news_count, 0)), 0) AS news_count,
                AVG(ds.confidence) AS avg_confidence,
                AVG(ds.avg_positive - ds.avg_negative) AS avg_net_score,
                CASE
                    WHEN SUM(COALESCE(ds.news_count, 0)) > 0 THEN
                        SUM((ds.avg_positive - ds.avg_negative) * COALESCE(ds.news_count, 0))
                        / SUM(COALESCE(ds.news_count, 0))
                    ELSE AVG(ds.avg_positive - ds.avg_negative)
                END AS weighted_net_score
            FROM daily_sentiment ds
            JOIN latest_sentiment_date ls
                ON ds.date = ls.latest_date
            GROUP BY ls.latest_date
            """,
            conn,
        )

    if sentiment_df.empty:
        return _empty_sentiment_snapshot()

    row = sentiment_df.iloc[0]
    weighted_score = row.get("weighted_net_score")

    return {
        "date": str(row.get("date")) if row.get("date") is not None else None,
        "ticker_count": int(row.get("ticker_count") or 0),
        "news_count": int(row.get("news_count") or 0),
        "avg_confidence": row.get("avg_confidence"),
        "avg_net_score": row.get("avg_net_score"),
        "weighted_net_score": weighted_score,
        "label": _sentiment_label_from_score(weighted_score),
    }


def _fetch_recent_scored_headlines(dal: DataAccessLayer, limit: int = 5) -> List[Dict]:
    """Get latest scored headlines across all tickers for dashboard feed."""
    with dal.get_connection() as conn:
        headlines_df = pd.read_sql_query(
            """
            SELECT
                h.ticker,
                h.headline,
                h.source,
                h.published_at,
                s.sentiment_label,
                s.confidence
            FROM news_headlines h
            JOIN sentiment_scores s
                ON h.id = s.news_id
            ORDER BY COALESCE(h.published_at, h.fetched_at) DESC
            LIMIT ?
            """,
            conn,
            params=[int(limit)],
        )
    return headlines_df.to_dict("records")


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


def _render_sentiment_tab(ticker: str, dal: DataAccessLayer, chart_generator: ChartGenerator):
    """Render per-ticker sentiment controls and visualizations."""
    st.subheader(
        "Ticker Sentiment",
        help="Daily aggregate sentiment from FinBERT-scored headlines for this ticker.",
    )

    with st.expander("Run FinBERT For This Ticker", expanded=False):
        c1, c2, c3 = st.columns(3)
        batch_size = c1.number_input(
            "Batch Size",
            min_value=1,
            max_value=256,
            value=32,
            step=1,
            key=f"sent_batch_{ticker}",
        )
        limit = c2.number_input(
            "Unprocessed Headline Limit",
            min_value=1,
            max_value=5000,
            value=500,
            step=50,
            key=f"sent_limit_{ticker}",
        )
        model_path = c3.text_input(
            "FinBERT Model Path",
            value="models/finbert",
            key=f"sent_model_path_{ticker}",
        )

        if st.button(
            f"Analyze {ticker} Headlines",
            type="primary",
            use_container_width=True,
            key=f"run_sentiment_{ticker}",
        ):
            with st.spinner(f"Running FinBERT sentiment analysis for {ticker}..."):
                start = perf_counter()
                try:
                    stats = _run_sentiment_job(
                        dal=dal,
                        scope="ticker",
                        batch_size=int(batch_size),
                        limit=int(limit),
                        model_path=(model_path or "models/finbert").strip(),
                        ticker=ticker,
                    )
                except Exception as exc:
                    st.error(f"Sentiment run failed: {exc}")
                else:
                    elapsed = perf_counter() - start
                    st.success(f"Sentiment analysis completed in {elapsed:.1f}s.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Headlines Processed", str(stats.get("headlines", 0)))
                    m2.metric("Scores Inserted", str(stats.get("scores_inserted", 0)))
                    m3.metric("Aggregate Days Refreshed", str(stats.get("aggregate_days", 0)))

    controls = st.columns(2)
    history_days = controls[0].slider(
        "Sentiment History Window (Days)",
        min_value=7,
        max_value=365,
        value=60,
        step=1,
        key=f"sent_history_days_{ticker}",
    )
    headline_rows = controls[1].slider(
        "Recent Scored Headlines",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key=f"sent_rows_{ticker}",
    )

    latest = dal.get_latest_daily_sentiment(ticker)
    if latest:
        net_score = _signed_sentiment_score(
            latest.get("avg_positive"),
            latest.get("avg_negative"),
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Sentiment", str(latest.get("overall_sentiment", "N/A")))
        m2.metric("Net Score", _format_decimal(net_score, digits=3))
        m3.metric("Confidence", _format_ratio(latest.get("confidence")))
        m4.metric("News Count", str(latest.get("news_count", 0)))
        st.caption(f"Latest aggregate date: `{latest.get('date', 'N/A')}`")
    else:
        st.info(
            f"No daily sentiment aggregates are available for {ticker} yet. "
            "Run news ingestion first, then run FinBERT scoring."
        )

    history_rows = dal.get_daily_sentiment_history(ticker=ticker, days=int(history_days), limit=400)
    if history_rows:
        sentiment_df = pd.DataFrame(history_rows)
        sentiment_df["Date"] = pd.to_datetime(sentiment_df.get("date"), errors="coerce")
        sentiment_df = sentiment_df.dropna(subset=["Date"]).sort_values("Date")
        sentiment_df["SentimentScore"] = sentiment_df.apply(
            lambda row: _signed_sentiment_score(row.get("avg_positive"), row.get("avg_negative")),
            axis=1,
        )

        if not sentiment_df.empty:
            st.caption("Sentiment score formula: `avg_positive - avg_negative`.")
            fig = chart_generator.create_sentiment_chart(sentiment_df[["Date", "SentimentScore"]], ticker)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment timeline to plot yet for this ticker.")

    st.subheader("Recent Scored Headlines")
    scored_rows = dal.get_recent_scored_news_by_ticker(ticker=ticker, limit=int(headline_rows))
    if not scored_rows:
        st.info("No scored headlines found for this ticker.")
        return

    scored_df = pd.DataFrame(scored_rows)
    if "confidence" in scored_df.columns:
        scored_df["confidence"] = scored_df["confidence"].apply(_format_ratio)
    columns = [
        col
        for col in [
            "published_at",
            "headline",
            "sentiment_label",
            "confidence",
            "source",
            "provider",
            "url",
        ]
        if col in scored_df.columns
    ]
    if columns:
        scored_df = scored_df[columns]
    st.dataframe(scored_df, use_container_width=True, hide_index=True)


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
    lstm_tab, rf_tab, xgb_tab, sentiment_tab = st.tabs(["LSTM", "Random Forest", "XGBoost", "Sentiment"])

    with lstm_tab:
        _render_lstm_tab(ticker=ticker, dal=dal, chart_generator=chart_generator)

    with rf_tab:
        _render_classifier_tab(ticker=ticker, dal=dal, model_type="rf", label="Random Forest")

    with xgb_tab:
        _render_classifier_tab(ticker=ticker, dal=dal, model_type="xgb", label="XGBoost")

    with sentiment_tab:
        _render_sentiment_tab(ticker=ticker, dal=dal, chart_generator=chart_generator)


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

    dal = st.session_state.get("dal")
    if dal is None:
        dal = DataAccessLayer()
        st.session_state.dal = dal

    market_snapshot = _empty_market_snapshot()
    sentiment_snapshot = _empty_sentiment_snapshot()
    recent_headlines: List[Dict] = []

    with st.status("Loading market data...", expanded=True) as status:
        st.write("Loading latest S&P 500 breadth snapshot from stock prices...")
        try:
            market_snapshot = _fetch_market_snapshot(dal)
        except Exception as exc:
            st.warning(f"Stock dashboard metrics unavailable: {exc}")

        st.write("Loading aggregated global news sentiment...")
        try:
            sentiment_snapshot = _fetch_global_sentiment_snapshot(dal)
        except Exception as exc:
            st.warning(f"Sentiment dashboard metrics unavailable: {exc}")

        st.write("Loading latest scored headlines feed...")
        try:
            recent_headlines = _fetch_recent_scored_headlines(dal, limit=5)
        except Exception as exc:
            st.warning(f"Scored headlines feed unavailable: {exc}")

        status.update(label="Market Data Loaded", state="complete", expanded=False)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "S&P 500 Breadth",
        _format_ratio(market_snapshot.get("breadth_ratio")),
        f"{market_snapshot.get('advancers', 0)} up / {market_snapshot.get('decliners', 0)} down",
        help="Share of S&P 500 tickers that closed higher on the latest market date in your database.",
    )
    col2.metric(
        "Market Volatility",
        _format_ratio(market_snapshot.get("cross_sectional_volatility")),
        _format_signed_percent(market_snapshot.get("average_move")),
        help="Cross-sectional volatility of latest daily returns across tracked S&P 500 tickers.",
    )
    col3.metric(
        "Sentiment Score",
        _format_decimal(sentiment_snapshot.get("weighted_net_score"), digits=3),
        sentiment_snapshot.get("label", "N/A"),
        help="Latest weighted net sentiment from daily FinBERT aggregates (avg_positive - avg_negative).",
    )

    st.caption(
        f"Latest stock date: `{market_snapshot.get('latest_date') or 'N/A'}` | "
        f"Tickers covered: `{market_snapshot.get('tickers_with_prices', 0)}/{market_snapshot.get('tracked_tickers', 0)}` | "
        f"Latest sentiment date: `{sentiment_snapshot.get('date') or 'N/A'}` | "
        f"Sentiment tickers: `{sentiment_snapshot.get('ticker_count', 0)}` | "
        f"Headlines in sentiment snapshot: `{sentiment_snapshot.get('news_count', 0)}`"
    )

    movers_col1, movers_col2, movers_col3 = st.columns(3)
    top_gainer = (market_snapshot.get("top_gainers") or [{}])[0]
    top_loser = (market_snapshot.get("top_losers") or [{}])[0]
    movers_col1.metric(
        "Top Gainer",
        str(top_gainer.get("ticker", "N/A")),
        _format_signed_percent(top_gainer.get("pct_change")),
    )
    movers_col2.metric(
        "Top Loser",
        str(top_loser.get("ticker", "N/A")),
        _format_signed_percent(top_loser.get("pct_change")),
    )
    movers_col3.metric(
        "Sentiment Confidence",
        _format_ratio(sentiment_snapshot.get("avg_confidence")),
        f"{sentiment_snapshot.get('news_count', 0)} headlines",
    )

    st.subheader("Top Movers")
    top_gainers = market_snapshot.get("top_gainers") or []
    top_losers = market_snapshot.get("top_losers") or []
    if top_gainers:
        st.write("Top Gainers (latest session):")
        for idx, row in enumerate(top_gainers, start=1):
            st.write(
                f"{idx}. {row.get('ticker', 'N/A')} | "
                f"{_format_signed_percent(row.get('pct_change'))} | "
                f"{_format_currency(row.get('close'))}"
            )
    else:
        st.write("Top Gainers (latest session): N/A")

    if top_losers:
        st.write("Top Losers (latest session):")
        for idx, row in enumerate(top_losers, start=1):
            st.write(
                f"{idx}. {row.get('ticker', 'N/A')} | "
                f"{_format_signed_percent(row.get('pct_change'))} | "
                f"{_format_currency(row.get('close'))}"
            )
    else:
        st.write("Top Losers (latest session): N/A")

    st.subheader("Latest Scored Headlines")
    if not recent_headlines:
        st.info("No scored headlines available yet. Run news ingestion and sentiment analysis to populate this feed.")
        return

    for row in recent_headlines:
        ticker = str(row.get("ticker") or "N/A")
        label = str(row.get("sentiment_label") or "N/A")
        confidence = _format_ratio(row.get("confidence"))
        headline = str(row.get("headline") or "").strip()
        if not headline:
            continue
        st.write(f"[{ticker}] {label} ({confidence}) | {headline}")


def show_disclaimer():
    st.markdown("---")
    st.caption(
        "**Mandatory Financial Disclaimer**: This application is for educational and informational purposes only. "
        "The content provided here does not constitute financial, investment, tax, or legal advice. "
        "Investing in financial markets involves risks. Always consult with a qualified professional before making any investment decisions."
    )


def show_data_pipeline():
    """Render controls to run ingestion jobs and validate recent news ingestion output."""
    st.header("Data Pipeline")
    st.write("Run ingestion jobs from the UI and validate recent fetched financial news.")

    dal = st.session_state.get("dal")
    if dal is None:
        dal = DataAccessLayer()
        st.session_state.dal = dal

    with st.container():
        st.subheader("Run News Ingestion")
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox(
                "News Provider",
                options=["auto", "rss", "newsapi", "alphavantage", "mock"],
                index=0,
                help="`auto` uses RSS first and API fallbacks when keys are available.",
            )
            days = st.number_input("Lookback Days", min_value=1, max_value=30, value=1, step=1)
        with col2:
            limit = st.number_input("Articles per Ticker", min_value=1, max_value=50, value=5, step=1)
            max_tickers = st.number_input("Tickers per Run (Round-Robin)", min_value=1, max_value=503, value=25, step=1)

        if st.button("Run News Ingestion Now", type="primary", use_container_width=True):
            with st.spinner("Running news ingestion..."):
                start = perf_counter()
                stats = ingest_news_data(
                    dal=dal,
                    days=int(days),
                    provider=provider,
                    limit=int(limit),
                    max_tickers=int(max_tickers),
                )
                elapsed = perf_counter() - start

            st.success(f"News ingestion completed in {elapsed:.1f}s.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tickers Selected", str(stats.get("selected_tickers", 0)))
            m2.metric("Tickers Processed", str(stats.get("processed", 0)))
            m3.metric("Inserted Articles", str(stats.get("records_inserted", 0)))
            m4.metric("Duplicates Skipped", str(stats.get("duplicates", 0)))
            st.caption(
                f"Cursor moved: {stats.get('cursor_start', 0)} -> {stats.get('cursor_end', 0)} | "
                f"Failed tickers: {stats.get('failed', 0)}"
            )

    st.markdown("---")
    with st.expander("Optional: Run Full Pipeline (Stock + News)", expanded=False):
        st.warning("This can take several minutes when many tickers need stock updates.")
        if st.button("Run Full Pipeline", use_container_width=True):
            with st.spinner("Running full pipeline..."):
                start = perf_counter()
                stats = ingest_data(
                    news_days=int(days),
                    news_provider=provider,
                    news_limit=int(limit),
                    news_max_tickers=int(max_tickers),
                )
                elapsed = perf_counter() - start

            stock_stats = stats.get("stock", {}) or {}
            news_stats = stats.get("news", {}) or {}
            st.success(f"Full pipeline completed in {elapsed:.1f}s.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Stocks Processed", str(stock_stats.get("processed", 0)))
            c2.metric("Stock Rows Inserted", str(stock_stats.get("records_inserted", 0)))
            c3.metric("News Rows Inserted", str(news_stats.get("records_inserted", 0)))

    st.markdown("---")
    with st.container():
        st.subheader("Run Sentiment Analysis")
        st.caption("Score unprocessed news headlines with FinBERT and refresh daily ticker aggregates.")

        run_mode = st.selectbox(
            "Sentiment Scope",
            options=["Selected Ticker", "All Unprocessed Headlines", "Specific Date"],
            index=0,
        )

        ticker = st.session_state.get("ticker", "AAPL").upper().strip()
        run_date = datetime.now().date()
        if run_mode == "Selected Ticker":
            st.info(f"Current ticker from sidebar: **{ticker}**")
        elif run_mode == "Specific Date":
            run_date = st.date_input("Date to Process", value=datetime.now().date())

        s1, s2, s3 = st.columns(3)
        sentiment_batch_size = s1.number_input("Sentiment Batch Size", min_value=1, max_value=256, value=32, step=1)
        sentiment_limit = s2.number_input(
            "Sentiment Headline Limit",
            min_value=1,
            max_value=20000,
            value=2000,
            step=100,
            help="Only used for ticker/all-unprocessed modes.",
        )
        sentiment_model_path = s3.text_input("Sentiment Model Path", value="models/finbert")

        if st.button("Run Sentiment Analysis", use_container_width=True):
            with st.spinner("Running sentiment analysis..."):
                start = perf_counter()
                try:
                    if run_mode == "Selected Ticker":
                        stats = _run_sentiment_job(
                            dal=dal,
                            scope="ticker",
                            batch_size=int(sentiment_batch_size),
                            limit=int(sentiment_limit),
                            model_path=(sentiment_model_path or "models/finbert").strip(),
                            ticker=ticker,
                        )
                    elif run_mode == "Specific Date":
                        stats = _run_sentiment_job(
                            dal=dal,
                            scope="date",
                            batch_size=int(sentiment_batch_size),
                            limit=int(sentiment_limit),
                            model_path=(sentiment_model_path or "models/finbert").strip(),
                            date_str=str(run_date),
                        )
                    else:
                        stats = _run_sentiment_job(
                            dal=dal,
                            scope="all",
                            batch_size=int(sentiment_batch_size),
                            limit=int(sentiment_limit),
                            model_path=(sentiment_model_path or "models/finbert").strip(),
                        )
                except Exception as exc:
                    st.error(f"Sentiment analysis failed: {exc}")
                else:
                    elapsed = perf_counter() - start
                    st.success(f"Sentiment analysis completed in {elapsed:.1f}s.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Headlines Processed", str(stats.get("headlines", 0)))
                    m2.metric("Scores Inserted", str(stats.get("scores_inserted", 0)))
                    m3.metric("Aggregate Days Refreshed", str(stats.get("aggregate_days", 0)))

    st.markdown("---")
    st.subheader("Recent News Records")
    ticker_filter = st.text_input("Filter by Ticker (optional)").upper().strip()
    row_limit = st.slider("Rows to Display", min_value=10, max_value=300, value=50, step=10)

    records = dal.get_recent_news(limit=int(row_limit), ticker=ticker_filter or None)
    if not records:
        st.info("No matching news records found yet. Run ingestion above to populate data.")
        return

    preview_df = pd.DataFrame(records)
    columns = [
        col
        for col in ["fetched_at", "ticker", "provider", "source", "headline", "summary", "published_at", "url"]
        if col in preview_df.columns
    ]
    if columns:
        preview_df = preview_df[columns]
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
