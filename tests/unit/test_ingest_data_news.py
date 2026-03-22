import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import ingest_data


def test_ingest_news_data_uses_fixed_window_and_all_tickers(monkeypatch):
    dal = MagicMock()
    dal.get_all_tickers.return_value = ["AAPL", "MSFT", "NVDA"]
    dal.insert_news_headline.side_effect = [1, 2, 3]
    dal.prune_news_history.return_value = {
        "pruned_headlines": 0,
        "pruned_scores": 0,
        "pruned_daily_sentiment": 0,
    }

    client = MagicMock()
    client.fetch_news.side_effect = lambda ticker, start_date=None, end_date=None, limit=10: [
        {
            "title": f"{ticker} headline",
            "source": "Mock",
            "url": f"https://example.com/{ticker}",
            "published_at": "2024-01-02 00:00:00",
            "summary": "Mock summary",
            "provider": "mock",
        }
    ]
    monkeypatch.setattr(ingest_data, "get_news_client", lambda provider="auto": client)

    stats = ingest_data.ingest_news_data(dal=dal, provider="auto", run_sentiment=False)

    assert stats["selected_tickers"] == 3
    assert stats["processed"] == 3
    assert stats["records_inserted"] == 3
    assert stats["lookback_days"] == ingest_data.DEFAULT_NEWS_LOOKBACK_DAYS
    assert stats["articles_per_ticker"] == ingest_data.DEFAULT_NEWS_ARTICLES_PER_TICKER
    assert stats["retention_days"] == ingest_data.DEFAULT_NEWS_RETENTION_DAYS
    assert stats["sentiment"]["status"] == "disabled"

    assert client.fetch_news.call_count == 3
    for call in client.fetch_news.call_args_list:
        assert call.kwargs["limit"] == ingest_data.DEFAULT_NEWS_ARTICLES_PER_TICKER

    dal.prune_news_history.assert_called_once_with(
        keep_days=ingest_data.DEFAULT_NEWS_RETENTION_DAYS,
        prune_daily_sentiment=True,
    )


def test_ingest_news_data_attaches_auto_sentiment_stats(monkeypatch):
    dal = MagicMock()
    dal.get_all_tickers.return_value = ["AAPL"]
    dal.insert_news_headline.return_value = 1
    dal.prune_news_history.return_value = {
        "pruned_headlines": 2,
        "pruned_scores": 2,
        "pruned_daily_sentiment": 1,
    }

    client = MagicMock()
    client.fetch_news.return_value = [
        {
            "title": "AAPL headline",
            "source": "Mock",
            "url": "https://example.com/AAPL",
            "published_at": "2024-01-02 00:00:00",
            "summary": "Mock summary",
            "provider": "mock",
        }
    ]
    monkeypatch.setattr(ingest_data, "get_news_client", lambda provider="auto": client)
    monkeypatch.setattr(
        ingest_data,
        "_run_automatic_sentiment",
        lambda dal: {"status": "completed", "headlines": 1, "scores_inserted": 1, "aggregate_days": 1},
    )

    stats = ingest_data.ingest_news_data(dal=dal, provider="auto", run_sentiment=True)

    assert stats["records_inserted"] == 1
    assert stats["pruned_headlines"] == 2
    assert stats["pruned_scores"] == 2
    assert stats["sentiment"]["status"] == "completed"
    assert stats["sentiment"]["scores_inserted"] == 1
