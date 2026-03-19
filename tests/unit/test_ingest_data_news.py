import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ingest_data import _get_news_cursor, _select_tickers_round_robin


def test_select_tickers_round_robin_basic():
    selected, next_cursor = _select_tickers_round_robin(["A", "B", "C"], cursor=0, max_tickers=2)
    assert selected == ["A", "B"]
    assert next_cursor == 2


def test_select_tickers_round_robin_wraps():
    selected, next_cursor = _select_tickers_round_robin(["A", "B", "C"], cursor=2, max_tickers=2)
    assert selected == ["C", "A"]
    assert next_cursor == 1


def test_get_news_cursor_defaults_and_sanitizes():
    dal = MagicMock()
    dal.get_user_preference.return_value = None
    assert _get_news_cursor(dal, total_tickers=10) == 0

    dal.get_user_preference.return_value = "12"
    assert _get_news_cursor(dal, total_tickers=5) == 2

    dal.get_user_preference.return_value = "bad-value"
    assert _get_news_cursor(dal, total_tickers=5) == 0
