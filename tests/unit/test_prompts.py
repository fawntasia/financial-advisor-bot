import sys

sys.path.insert(0, ".")

import pytest

from src.llm.prompts import PromptManager


@pytest.fixture
def prompt_manager():
    return PromptManager()


def test_template_registry_has_expected_keys(prompt_manager):
    assert set(prompt_manager.templates.keys()) == {
        "general",
        "technical",
        "sentiment",
        "recommendation",
        "portfolio",
    }


def test_get_prompt_renders_template(prompt_manager):
    rendered = prompt_manager.get_prompt(
        "general",
        ticker="AAPL",
        price="123.45",
        indicators_summary="RSI 55",
        sentiment_summary="neutral",
        prediction_summary="steady",
    )

    assert "AAPL" in rendered
    assert "123.45" in rendered
    assert "RSI 55" in rendered
    assert "neutral" in rendered
    assert "steady" in rendered


def test_get_prompt_invalid_template_raises(prompt_manager):
    with pytest.raises(ValueError, match="Template 'missing' not found"):
        prompt_manager.get_prompt("missing", ticker="AAPL")


def test_system_prompt_includes_disclaimer(prompt_manager):
    system_prompt = prompt_manager.get_system_prompt()
    assert "educational purposes only" in system_prompt.lower()
    assert "not a financial advisor" in system_prompt.lower()
