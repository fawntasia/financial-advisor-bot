import sys
import importlib
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, ".")


class SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def chat_module(monkeypatch):
    mock_streamlit = types.SimpleNamespace()
    mock_streamlit.session_state = SessionState()
    mock_streamlit.spinner = MagicMock(side_effect=lambda *args, **kwargs: DummyContextManager())
    mock_streamlit.chat_message = MagicMock(side_effect=lambda *args, **kwargs: DummyContextManager())
    mock_streamlit.expander = MagicMock(side_effect=lambda *args, **kwargs: DummyContextManager())
    mock_streamlit.markdown = MagicMock()
    mock_streamlit.warning = MagicMock()
    mock_streamlit.error = MagicMock()
    mock_streamlit.write = MagicMock()
    mock_streamlit.subheader = MagicMock()
    mock_streamlit.button = MagicMock(return_value=False)
    mock_streamlit.toast = MagicMock()
    mock_streamlit.caption = MagicMock()
    mock_streamlit.chat_input = MagicMock(return_value=None)
    mock_streamlit.columns = MagicMock(return_value=(MagicMock(), MagicMock()))
    mock_streamlit.rerun = MagicMock()

    monkeypatch.setitem(sys.modules, "streamlit", mock_streamlit)

    import src.ui.chat as chat_module
    importlib.reload(chat_module)

    prompt_manager_mock = MagicMock()
    context_builder_mock = MagicMock()
    llm_mock = MagicMock()

    class DummyPromptManager:
        def get_prompt(self, *args, **kwargs):
            return prompt_manager_mock.get_prompt(*args, **kwargs)

        def get_system_prompt(self):
            return prompt_manager_mock.get_system_prompt()

    class DummyContextBuilder:
        def __init__(self, dal):
            self.dal = dal

        def build_context(self, ticker):
            return context_builder_mock.build_context(ticker)

    class DummyLlamaLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(chat_module, "PromptManager", DummyPromptManager)
    monkeypatch.setattr(chat_module, "ContextBuilder", DummyContextBuilder)
    monkeypatch.setattr(chat_module, "LlamaLoader", DummyLlamaLoader)

    mock_streamlit.session_state.llama_model = llm_mock

    return chat_module, mock_streamlit, prompt_manager_mock, context_builder_mock, llm_mock


def test_extract_tickers_returns_unique_uppercase(chat_module):
    chat_module, _, _, _, _ = chat_module
    manager = chat_module.ChatManager(dal=MagicMock())

    result = manager.extract_tickers("Check $AAPL and TSLA and aapl and msft")

    assert set(result) == {"AAPL", "TSLA"}


def test_handle_response_builds_prompt_and_saves_context(chat_module):
    chat_module, mock_streamlit, prompt_manager_mock, context_builder_mock, llm_mock = chat_module
    prompt_manager_mock.get_system_prompt.return_value = "SYSTEM PROMPT"
    context_builder_mock.build_context.return_value = {
        "price": "123.45",
        "indicators_summary": "RSI 55",
        "sentiment_summary": "neutral",
        "prediction_summary": "steady",
    }
    prompt_manager_mock.get_prompt.return_value = "FULL PROMPT"
    llm_mock.generate.return_value = "AI response"

    manager = chat_module.ChatManager(dal=MagicMock())
    manager._handle_response("Tell me about AAPL")

    context_builder_mock.build_context.assert_called_once_with("AAPL")
    prompt_manager_mock.get_prompt.assert_called_once_with(
        "general",
        ticker="AAPL",
        price="123.45",
        indicators_summary="RSI 55",
        sentiment_summary="neutral",
        prediction_summary="steady",
    )
    llm_mock.generate.assert_called_once()
    composed = llm_mock.generate.call_args.args[0]
    assert "SYSTEM PROMPT" in composed
    assert "Current Task:\nFULL PROMPT" in composed
    mock_streamlit.markdown.assert_any_call("AI response")
    assert mock_streamlit.session_state.messages[-1] == {
        "role": "assistant",
        "content": "AI response",
        "context": context_builder_mock.build_context.return_value,
    }


def test_handle_response_missing_price_warns_and_uses_prompt(chat_module):
    chat_module, mock_streamlit, prompt_manager_mock, context_builder_mock, llm_mock = chat_module
    prompt_manager_mock.get_system_prompt.return_value = "SYSTEM PROMPT"
    context_builder_mock.build_context.return_value = {"price": None}
    llm_mock.generate.return_value = "Fallback response"

    manager = chat_module.ChatManager(dal=MagicMock())
    manager._handle_response("What about AAPL")

    mock_streamlit.warning.assert_called_once()
    prompt_manager_mock.get_prompt.assert_not_called()
    llm_mock.generate.assert_called_once()
    composed = llm_mock.generate.call_args.args[0]
    assert "Current Task:\nWhat about AAPL" in composed
    assert mock_streamlit.session_state.messages[-1]["context"] == {}


def test_handle_response_no_ticker_uses_prompt(chat_module):
    chat_module, mock_streamlit, prompt_manager_mock, context_builder_mock, llm_mock = chat_module
    prompt_manager_mock.get_system_prompt.return_value = "SYSTEM PROMPT"
    llm_mock.generate.return_value = "No ticker response"

    manager = chat_module.ChatManager(dal=MagicMock())
    manager._handle_response("Hello there")

    context_builder_mock.build_context.assert_not_called()
    prompt_manager_mock.get_prompt.assert_not_called()
    llm_mock.generate.assert_called_once()
    composed = llm_mock.generate.call_args.args[0]
    assert "Current Task:\nHello there" in composed
    assert mock_streamlit.session_state.messages[-1]["context"] == {}


def test_handle_response_context_error_reports_error(chat_module):
    chat_module, mock_streamlit, prompt_manager_mock, context_builder_mock, llm_mock = chat_module
    prompt_manager_mock.get_system_prompt.return_value = "SYSTEM PROMPT"
    context_builder_mock.build_context.side_effect = Exception("boom")
    llm_mock.generate.return_value = "Error fallback"

    manager = chat_module.ChatManager(dal=MagicMock())
    manager._handle_response("Tell me about AAPL")

    mock_streamlit.error.assert_called_once()
    prompt_manager_mock.get_prompt.assert_not_called()
    llm_mock.generate.assert_called_once()
    composed = llm_mock.generate.call_args.args[0]
    assert "Current Task:\nTell me about AAPL" in composed
    assert mock_streamlit.session_state.messages[-1]["context"] == {}


def test_allocation_query_plural_investments_uses_grounded_strategy_output(chat_module):
    chat_module, mock_streamlit, prompt_manager_mock, _, llm_mock = chat_module
    prompt_manager_mock.get_system_prompt.return_value = "SYSTEM PROMPT"
    prompt_manager_mock.get_prompt.return_value = "ALLOCATION PROMPT"
    llm_mock.generate.return_value = "Generic investment advice without structure."

    dal = MagicMock()
    dal.get_all_tickers.return_value = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JNJ", "PG", "KO", "PEP", "WMT",
    ]
    dal.get_ticker_universe.return_value = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"},
        {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Information Technology"},
        {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Health Care"},
        {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples"},
        {"ticker": "KO", "name": "Coca-Cola", "sector": "Consumer Staples"},
    ]

    manager = chat_module.ChatManager(dal=dal)
    manager._get_top_prediction_candidates = MagicMock(return_value=[
        {"ticker": "AAPL", "latest_prediction_date": "2026-03-23", "up_votes": 2, "total_votes": 2, "avg_up_confidence": 0.61},
        {"ticker": "MSFT", "latest_prediction_date": "2026-03-23", "up_votes": 2, "total_votes": 2, "avg_up_confidence": 0.58},
        {"ticker": "JNJ", "latest_prediction_date": "2026-03-23", "up_votes": 1, "total_votes": 2, "avg_up_confidence": 0.53},
    ])

    manager._handle_response("I have ten thousand dollars, what investments are good")

    assert prompt_manager_mock.get_prompt.call_args.args[0] == "allocation"
    final_response = mock_streamlit.session_state.messages[-1]["content"]
    assert "Strategy A (Aggressive)" in final_response
    assert "Strategy B (Conservative)" in final_response
    assert "$10,000" in final_response
    assert any(ticker in final_response for ticker in ["AAPL", "MSFT", "JNJ"])
