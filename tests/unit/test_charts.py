import sys

sys.path.insert(0, ".")

import pandas as pd
import plotly.graph_objects as go

from src.ui.charts import ChartGenerator


def _sample_ohlcv_df():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0],
            "High": [105.0, 106.0, 104.0],
            "Low": [99.0, 101.0, 100.0],
            "Close": [103.0, 104.0, 102.0],
            "Volume": [1000000, 1200000, 1100000],
            "RSI": [45.0, 55.0, 60.0],
            "MACD": [0.2, 0.4, 0.1],
            "MACD_Signal": [0.1, 0.3, 0.2],
            "MACD_Hist": [0.1, 0.1, -0.1],
            "BB_Upper": [110.0, 111.0, 109.5],
            "BB_Lower": [95.0, 96.0, 97.0],
            "BB_Middle": [102.5, 103.0, 103.0],
        },
        index=dates,
    )


def test_create_ohlcv_chart_with_indicators():
    df = _sample_ohlcv_df()
    generator = ChartGenerator(theme="plotly")

    fig = generator.create_ohlcv_chart(df, "AAPL")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Technical Analysis: AAPL"

    trace_names = [trace.name for trace in fig.data]
    assert trace_names == [
        "OHLC",
        "BB Upper",
        "BB Lower",
        "BB Middle",
        "Volume",
        "MACD",
        "Signal",
        "Hist",
        "RSI",
    ]
    assert isinstance(fig.data[0], go.Candlestick)


def test_create_sentiment_chart_axes_and_trace():
    sentiment_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "SentimentScore": [0.1, -0.2, 0.3],
        }
    )
    generator = ChartGenerator(theme="plotly")

    fig = generator.create_sentiment_chart(sentiment_df, "AAPL")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Sentiment Timeline: AAPL"
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Sentiment Score (-1 to 1)"
    assert tuple(fig.layout.yaxis.range) == (-1.1, 1.1)
    assert [trace.name for trace in fig.data] == ["Sentiment"]
