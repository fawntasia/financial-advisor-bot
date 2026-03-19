import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.news_client import (
    AlphaVantageNewsClient,
    FallbackNewsClient,
    MockNewsClient,
    NewsAPIClient,
    YahooFinanceRSSClient,
    get_news_client,
)


def _mock_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestNewsAPIClient:
    def test_builds_request_and_parses_articles(self, sample_news):
        client = NewsAPIClient(api_key="test-key")
        articles = []
        for item in sample_news:
            articles.append(
                {
                    "title": item["headline"],
                    "source": {"name": item["source"]},
                    "url": item["url"],
                    "publishedAt": f"{item['date']}T12:00:00Z",
                    "description": f"Summary for {item['headline']}",
                }
            )

        response = _mock_response({"status": "ok", "articles": articles})

        with patch("src.data.news_client.requests.get", return_value=response) as mock_get:
            results = client.fetch_news(
                "AAPL", start_date="2024-01-01", end_date="2024-01-31", limit=3
            )

        assert len(results) == 3
        assert results[0]["title"] == sample_news[0]["headline"]
        assert results[0]["source"] == sample_news[0]["source"]
        assert results[0]["url"] == sample_news[0]["url"]
        assert results[0]["published_at"] == "2024-01-05 12:00:00"
        assert results[0]["summary"] == f"Summary for {sample_news[0]['headline']}"
        assert results[0]["provider"] == "newsapi"

        params = mock_get.call_args.kwargs["params"]
        assert params["q"] == "AAPL"
        assert params["apiKey"] == "test-key"
        assert params["language"] == "en"
        assert params["sortBy"] == "publishedAt"
        assert params["pageSize"] == 3
        assert params["from"] == "2024-01-01"
        assert params["to"] == "2024-01-31"

    def test_returns_empty_on_api_error(self):
        client = NewsAPIClient(api_key="test-key")
        response = _mock_response({"status": "error", "message": "bad"})

        with patch("src.data.news_client.requests.get", return_value=response):
            assert client.fetch_news("AAPL") == []

    def test_returns_empty_on_http_error(self):
        client = NewsAPIClient(api_key="test-key")

        with patch(
            "src.data.news_client.requests.get",
            side_effect=requests.exceptions.RequestException("boom"),
        ):
            assert client.fetch_news("AAPL") == []

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        client = NewsAPIClient(api_key=None)

        with patch("src.data.news_client.requests.get") as mock_get:
            assert client.fetch_news("AAPL") == []
            assert not mock_get.called


class TestAlphaVantageNewsClient:
    def test_builds_request_parses_feed_and_logs_rate_limit(self, sample_news, caplog):
        client = AlphaVantageNewsClient(api_key="test-key")
        feed = []
        for item in sample_news:
            feed.append(
                {
                    "title": item["headline"],
                    "source": item["source"],
                    "url": item["url"],
                    "time_published": f"{item['date'].replace('-', '')}T120000",
                    "summary": f"Summary for {item['headline']}",
                    "overall_sentiment_score": 0.1,
                    "overall_sentiment_label": "neutral",
                }
            )

        response = _mock_response({"Note": "limit", "feed": feed})

        with caplog.at_level("WARNING"):
            with patch("src.data.news_client.requests.get", return_value=response) as mock_get:
                results = client.fetch_news("AAPL", start_date="2024-01-01", limit=3)

        assert len(results) == 3
        assert results[0]["published_at"] == "2024-01-05 12:00:00"
        assert results[0]["av_sentiment_score"] == 0.1
        assert results[0]["av_sentiment_label"] == "neutral"
        assert results[0]["provider"] == "alphavantage"
        assert "limit reached" in caplog.text

        params = mock_get.call_args.kwargs["params"]
        assert params["function"] == "NEWS_SENTIMENT"
        assert params["tickers"] == "AAPL"
        assert params["apikey"] == "test-key"
        assert params["limit"] == 3
        assert params["sort"] == "LATEST"
        assert params["time_from"] == "20240101T0000"

    def test_returns_empty_on_error_message(self):
        client = AlphaVantageNewsClient(api_key="test-key")
        response = _mock_response({"Error Message": "bad"})

        with patch("src.data.news_client.requests.get", return_value=response):
            assert client.fetch_news("AAPL") == []

    def test_returns_empty_on_http_error(self):
        client = AlphaVantageNewsClient(api_key="test-key")

        with patch(
            "src.data.news_client.requests.get",
            side_effect=requests.exceptions.RequestException("boom"),
        ):
            assert client.fetch_news("AAPL") == []

    def test_empty_feed_returns_empty_list(self):
        client = AlphaVantageNewsClient(api_key="test-key")
        response = _mock_response({"feed": []})

        with patch("src.data.news_client.requests.get", return_value=response):
            assert client.fetch_news("AAPL") == []


class TestMockNewsClient:
    def test_returns_requested_limit(self):
        client = MockNewsClient()
        results = client.fetch_news("AAPL", limit=2)

        assert len(results) == 2
        assert results[0]["title"].startswith("Mock News for AAPL")
        assert results[0]["provider"] == "mock"


class TestYahooFinanceRSSClient:
    def test_parses_rss_and_applies_date_filter_and_limit(self):
        client = YahooFinanceRSSClient()
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>First headline</title>
      <link>https://example.com/1</link>
      <pubDate>Fri, 05 Jan 2024 12:00:00 +0000</pubDate>
      <description>First &lt;b&gt;summary&lt;/b&gt;</description>
      <source>Yahoo Finance</source>
    </item>
    <item>
      <title>Second headline</title>
      <link>https://example.com/2</link>
      <pubDate>Thu, 04 Jan 2024 09:00:00 +0000</pubDate>
      <description>Second summary</description>
    </item>
  </channel>
</rss>"""

        response = MagicMock()
        response.content = xml_payload.encode("utf-8")
        response.raise_for_status.return_value = None

        with patch("src.data.news_client.requests.get", return_value=response) as mock_get:
            results = client.fetch_news(
                "AAPL",
                start_date="2024-01-05",
                end_date="2024-01-31",
                limit=1,
            )

        assert len(results) == 1
        assert results[0]["title"] == "First headline"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["published_at"] == "2024-01-05 12:00:00"
        assert results[0]["summary"] == "First <b>summary</b>"
        assert results[0]["provider"] == "rss"
        assert mock_get.call_args.args[0].startswith("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL")


class TestProviderSelection:
    def test_auto_provider_without_api_keys_uses_rss_only(self, monkeypatch):
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        monkeypatch.delenv("ALPHA_VANTAGE_KEY", raising=False)

        client = get_news_client("auto")
        assert isinstance(client, FallbackNewsClient)
        assert len(client.providers) == 1
        assert isinstance(client.providers[0], YahooFinanceRSSClient)

    def test_auto_provider_with_api_keys_builds_fallback_chain(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "news-key")
        monkeypatch.setenv("ALPHA_VANTAGE_KEY", "alpha-key")

        client = get_news_client("auto")
        assert isinstance(client, FallbackNewsClient)
        provider_types = [type(p).__name__ for p in client.providers]
        assert provider_types == ["YahooFinanceRSSClient", "NewsAPIClient", "AlphaVantageNewsClient"]
