import importlib
import sys
import types

import pandas as pd

sys.path.insert(0, ".")


class DummySpinner:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def __enter__(self):
        self.calls.append(("spinner.enter", self.label))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("spinner.exit", self.label))
        return False


class DummyStatus:
    def __init__(self, calls, label, expanded):
        self.calls = calls
        self.label = label
        self.expanded = expanded

    def __enter__(self):
        self.calls.append(("status.enter", self.label, self.expanded))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("status.exit", self.label, self.expanded))
        return False

    def update(self, **kwargs):
        self.calls.append(("status.update", kwargs))


class DummyColumn:
    def __init__(self, calls, index):
        self.calls = calls
        self.index = index

    def metric(self, label, value, delta=None, help=None):
        self.calls.append(("column.metric", self.index, label, value, delta, help))


class DummyTab:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def __enter__(self):
        self.calls.append(("tab.enter", self.label))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("tab.exit", self.label))
        return False


class SessionState:
    def __init__(self):
        self._data = {}

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]


class FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.session_state = SessionState()

    def header(self, text):
        self.calls.append(("header", text))

    def write(self, text):
        self.calls.append(("write", text))

    def info(self, text):
        self.calls.append(("info", text))

    def warning(self, text):
        self.calls.append(("warning", text))

    def subheader(self, text, help=None):
        self.calls.append(("subheader", text, help))

    def plotly_chart(self, fig, use_container_width=False):
        self.calls.append(("plotly_chart", fig, use_container_width))

    def spinner(self, label):
        self.calls.append(("spinner", label))
        return DummySpinner(self.calls, label)

    def status(self, label, expanded=False):
        self.calls.append(("status", label, expanded))
        return DummyStatus(self.calls, label, expanded)

    def columns(self, count):
        self.calls.append(("columns", count))
        return [DummyColumn(self.calls, index) for index in range(count)]

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def error(self, text):
        self.calls.append(("error", text))

    def tabs(self, labels):
        self.calls.append(("tabs", tuple(labels)))
        return [DummyTab(self.calls, label) for label in labels]


def _load_views(fake_streamlit):
    sys.modules["streamlit"] = fake_streamlit
    if "src.ui.views" in sys.modules:
        return importlib.reload(sys.modules["src.ui.views"])
    return importlib.import_module("src.ui.views")


def _mock_viz_data():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    return {
        "history_df": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates),
        "technical_df": pd.DataFrame(
            {
                "Open": [99.0, 100.0, 101.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [98.5, 99.5, 100.5],
                "Close": [100.0, 101.0, 102.0],
                "Volume": [1000, 1200, 1300],
                "RSI": [50.0, 52.0, 54.0],
                "MACD": [0.1, 0.12, 0.15],
                "MACD_Signal": [0.08, 0.1, 0.11],
                "MACD_Hist": [0.02, 0.02, 0.04],
                "BB_Upper": [103.0, 104.0, 105.0],
                "BB_Lower": [97.0, 98.0, 99.0],
                "BB_Middle": [100.0, 101.0, 102.0],
                "Signal_Line": [0.08, 0.1, 0.11],
            },
            index=dates,
        ),
        "backtest_df": pd.DataFrame(
            {"Actual_Close": [101.0, 102.0], "Predicted_Close": [100.8, 101.7]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        "forecast_df": pd.DataFrame(
            {"Predicted_Close": [102.5, 103.1]},
            index=pd.to_datetime(["2024-01-04", "2024-01-05"]),
        ),
        "last_close": 102.0,
        "next_predicted_close": 102.5,
        "rmse": 1.23,
        "mape": 2.34,
        "sequence_length": 60,
        "feature_columns": ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"],
        "scaler_ticker": "AAPL",
        "artifact_paths": {"model_path": "models/lstm_test.keras", "scaler_path": "", "metadata_path": ""},
    }


def _mock_signal_data(model_type: str):
    model_name = "RandomForest_v2" if model_type == "rf" else "XGBoost_v2"
    artifact = "models/random_forest_global.pkl" if model_type == "rf" else "models/xgboost_global.json"
    return {
        "model_type": model_type,
        "model_name": model_name,
        "latest_price_date": "2024-01-05",
        "prediction_date": "2024-01-08",
        "predicted_direction": 1,
        "predicted_label": "UP",
        "prob_up": 0.72,
        "confidence": 0.72,
        "decision_threshold": 0.5,
        "global_test_metrics": {"balanced_accuracy": 0.61, "f1": 0.58},
        "ticker_test_metrics": {"balanced_accuracy": 0.67},
        "recent_eval_metrics": {"accuracy": 0.6},
        "feature_columns": ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal_Line"],
        "artifact_paths": {"model_path": artifact, "metadata_path": ""},
    }


def test_show_stock_analysis_uses_default_ticker(monkeypatch):
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    captured_lstm = {}
    classifier_calls = []

    def fake_generate(**kwargs):
        captured_lstm.update(kwargs)
        return _mock_viz_data()

    def fake_classifier_generate(**kwargs):
        classifier_calls.append(kwargs)
        return _mock_signal_data(kwargs["model_type"])

    class DummyChartGenerator:
        def create_ohlcv_chart(self, df, ticker):
            return "technical_fig"

        def create_lstm_prediction_chart(self, history_df, backtest_df, forecast_df, ticker):
            return "forecast_fig"

    monkeypatch.setattr(views, "generate_lstm_visualization_data", fake_generate)
    monkeypatch.setattr(views, "generate_classification_signal_data", fake_classifier_generate)
    monkeypatch.setattr(views, "ChartGenerator", lambda: DummyChartGenerator())
    monkeypatch.setattr(views, "_render_sentiment_tab", lambda **kwargs: None)

    views.show_stock_analysis()

    assert any(call[0] == "header" and "Stock Analysis" in call[1] for call in fake_st.calls)
    assert ("tabs", ("LSTM", "Random Forest", "XGBoost", "Sentiment")) in fake_st.calls
    assert ("spinner", "Fetching data and model output for AAPL...") in fake_st.calls
    assert captured_lstm["ticker"] == "AAPL"
    assert [c["model_type"] for c in classifier_calls] == ["rf", "xgb"]
    assert ("plotly_chart", "technical_fig", True) in fake_st.calls
    assert ("plotly_chart", "forecast_fig", True) in fake_st.calls
    assert any(call[0] == "column.metric" and call[2] == "Predicted Direction" for call in fake_st.calls)


def test_show_stock_analysis_uses_session_ticker(monkeypatch):
    fake_st = FakeStreamlit()
    fake_st.session_state["ticker"] = "MSFT"
    views = _load_views(fake_st)

    captured_lstm = {}
    classifier_calls = []

    def fake_generate(**kwargs):
        captured_lstm.update(kwargs)
        data = _mock_viz_data()
        data["scaler_ticker"] = "AAPL"
        return data

    def fake_classifier_generate(**kwargs):
        classifier_calls.append(kwargs)
        return _mock_signal_data(kwargs["model_type"])

    class DummyChartGenerator:
        def create_ohlcv_chart(self, df, ticker):
            return "technical_fig"

        def create_lstm_prediction_chart(self, history_df, backtest_df, forecast_df, ticker):
            return "forecast_fig"

    monkeypatch.setattr(views, "generate_lstm_visualization_data", fake_generate)
    monkeypatch.setattr(views, "generate_classification_signal_data", fake_classifier_generate)
    monkeypatch.setattr(views, "ChartGenerator", lambda: DummyChartGenerator())
    monkeypatch.setattr(views, "_render_sentiment_tab", lambda **kwargs: None)

    views.show_stock_analysis()

    assert captured_lstm["ticker"] == "MSFT"
    assert all(call["ticker"] == "MSFT" for call in classifier_calls)
    assert any(call[0] == "write" and "MSFT" in call[1] for call in fake_st.calls)
    assert any(call[0] == "warning" and "fallback" in call[1] for call in fake_st.calls)


def test_show_stock_analysis_handles_generation_error(monkeypatch):
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    def fake_generate(**kwargs):
        raise ValueError("missing artifacts")

    def fake_classifier_generate(**kwargs):
        return _mock_signal_data(kwargs["model_type"])

    monkeypatch.setattr(views, "generate_lstm_visualization_data", fake_generate)
    monkeypatch.setattr(views, "generate_classification_signal_data", fake_classifier_generate)
    monkeypatch.setattr(views, "_render_sentiment_tab", lambda **kwargs: None)

    views.show_stock_analysis()

    assert any(call[0] == "error" and "Unable to load analysis" in call[1] for call in fake_st.calls)
    assert any(call[0] == "info" and "LSTM artifacts" in call[1] for call in fake_st.calls)
    # RF/XGB tabs should still render despite LSTM error.
    assert any(call[0] == "column.metric" and call[2] == "Predicted Direction" for call in fake_st.calls)


def test_show_stock_analysis_rf_error_is_isolated(monkeypatch):
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    def fake_lstm_generate(**kwargs):
        return _mock_viz_data()

    def fake_classifier_generate(**kwargs):
        if kwargs["model_type"] == "rf":
            raise ValueError("rf missing")
        return _mock_signal_data(kwargs["model_type"])

    class DummyChartGenerator:
        def create_ohlcv_chart(self, df, ticker):
            return "technical_fig"

        def create_lstm_prediction_chart(self, history_df, backtest_df, forecast_df, ticker):
            return "forecast_fig"

    monkeypatch.setattr(views, "generate_lstm_visualization_data", fake_lstm_generate)
    monkeypatch.setattr(views, "generate_classification_signal_data", fake_classifier_generate)
    monkeypatch.setattr(views, "ChartGenerator", lambda: DummyChartGenerator())
    monkeypatch.setattr(views, "_render_sentiment_tab", lambda **kwargs: None)

    views.show_stock_analysis()

    assert ("plotly_chart", "technical_fig", True) in fake_st.calls
    assert ("plotly_chart", "forecast_fig", True) in fake_st.calls
    assert any(call[0] == "error" and "Random Forest signal" in call[1] for call in fake_st.calls)
    assert any(
        call[0] == "info" and "random_forest_global.pkl" in call[1]
        for call in fake_st.calls
    )
    # XGBoost still renders its signal cards.
    assert any(call[0] == "caption" and "xgboost_global.json" in call[1] for call in fake_st.calls)


def test_show_chat_interface_with_manager():
    fake_st = FakeStreamlit()
    chat_calls = []

    class DummyChatManager:
        def display_chat(self):
            chat_calls.append("displayed")

    fake_st.session_state["chat_manager"] = DummyChatManager()
    views = _load_views(fake_st)

    views.show_chat_interface()

    assert chat_calls == ["displayed"]
    assert not any(call[0] == "error" for call in fake_st.calls)


def test_show_chat_interface_without_manager():
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    views.show_chat_interface()

    assert ("error", "Chat Manager not initialized. Please check the application setup.") in fake_st.calls


def test_show_dashboard_updates_status_and_metrics(monkeypatch):
    fake_st = FakeStreamlit()
    fake_st.session_state["dal"] = object()
    views = _load_views(fake_st)

    monkeypatch.setattr(
        views,
        "_fetch_market_snapshot",
        lambda dal: {
            "latest_date": "2024-01-05",
            "tracked_tickers": 503,
            "tickers_with_prices": 500,
            "tickers_with_returns": 498,
            "advancers": 290,
            "decliners": 180,
            "unchanged": 28,
            "breadth_ratio": 290 / 498,
            "cross_sectional_volatility": 0.018,
            "average_move": 0.003,
            "median_move": 0.001,
            "top_gainers": [{"ticker": "NVDA", "pct_change": 0.041, "close": 850.0}],
            "top_losers": [{"ticker": "TSLA", "pct_change": -0.033, "close": 190.0}],
        },
    )
    monkeypatch.setattr(
        views,
        "_fetch_global_sentiment_snapshot",
        lambda dal: {
            "date": "2024-01-05",
            "ticker_count": 42,
            "news_count": 130,
            "avg_confidence": 0.74,
            "avg_net_score": 0.08,
            "weighted_net_score": 0.11,
            "label": "Bullish",
        },
    )
    monkeypatch.setattr(
        views,
        "_fetch_recent_scored_headlines",
        lambda dal, limit=5: [
            {
                "ticker": "AAPL",
                "headline": "Apple expands AI features",
                "sentiment_label": "positive",
                "confidence": 0.81,
            }
        ],
    )

    views.show_dashboard()

    assert any(call[0] == "header" and "Market Dashboard" in call[1] for call in fake_st.calls)
    assert ("status", "Loading market data...", True) in fake_st.calls
    assert ("status.update", {"label": "Market Data Loaded", "state": "complete", "expanded": False}) in fake_st.calls
    assert fake_st.calls.count(("columns", 3)) >= 2
    metric_calls = [call for call in fake_st.calls if call[0] == "column.metric"]
    metric_labels = [call[2] for call in metric_calls]
    assert "S&P 500 Breadth" in metric_labels
    assert "Market Volatility" in metric_labels
    assert "Sentiment Score" in metric_labels
    assert "Top Gainer" in metric_labels
    assert "Top Loser" in metric_labels
    assert "Sentiment Confidence" in metric_labels
    assert any(call[0] == "subheader" and "Top Movers" in call[1] for call in fake_st.calls)
    assert any(call[0] == "write" and "Top Gainers (latest session):" in call[1] for call in fake_st.calls)
    assert any(call[0] == "write" and "Top Losers (latest session):" in call[1] for call in fake_st.calls)
    assert any(call[0] == "subheader" and "Latest Scored Headlines" in call[1] for call in fake_st.calls)
    assert any(call[0] == "write" and "[AAPL]" in call[1] for call in fake_st.calls)


def test_show_disclaimer_renders_text():
    fake_st = FakeStreamlit()
    views = _load_views(fake_st)

    views.show_disclaimer()

    assert ("markdown", "---") in fake_st.calls
    assert any(call[0] == "caption" and "Mandatory Financial Disclaimer" in call[1] for call in fake_st.calls)
