"""
Prompt templates for the Financial Advisor LLM.
"""

from typing import Any, Dict

SYSTEM_PROMPT = """You are an expert financial advisor AI. Your goal is to provide accurate, data-driven analysis of stocks and portfolios.
IMPORTANT: I am an AI, not a financial advisor. This information is for educational purposes only and does not constitute financial advice. Always consult with a qualified professional before making investment decisions. Past performance is not indicative of future results."""

GENERAL_ANALYSIS_TEMPLATE = """Analyze the overall situation for {ticker}.
Current Price: {price}
Technical Indicators: {indicators_summary}
Sentiment: {sentiment_summary}
AI Prediction: {prediction_summary}

Please provide a comprehensive overview of the current status of {ticker}."""

TECHNICAL_ANALYSIS_TEMPLATE = """Focus on the technical indicators for {ticker}.
Current Price: {price}
Indicators Summary: {indicators_summary}

Provide a detailed technical analysis including support/resistance levels, trend strength, and potential momentum shifts."""

SENTIMENT_ANALYSIS_TEMPLATE = """Analyze the market sentiment for {ticker}.
Sentiment Summary: {sentiment_summary}

Discuss how recent news, social media trends, and analyst reports are impacting the market's perception of {ticker}."""

BUY_SELL_RECOMMENDATION_TEMPLATE = """Based on all available data for {ticker}, provide a recommendation.
Current Price: {price}
Technical Indicators: {indicators_summary}
Sentiment: {sentiment_summary}
AI Prediction: {prediction_summary}

Give a clear Buy, Sell, or Hold recommendation with a justification based on the data above."""

PORTFOLIO_OVERVIEW_TEMPLATE = """Provide an overview of the current portfolio performance and health.
Summary data: {indicators_summary}

Analyze the diversification, risk exposure, and potential adjustments needed for the portfolio."""

class PromptManager:
    """Manages LLM prompt templates and rendering."""

    def __init__(self):
        self.templates = {
            "general": GENERAL_ANALYSIS_TEMPLATE,
            "technical": TECHNICAL_ANALYSIS_TEMPLATE,
            "sentiment": SENTIMENT_ANALYSIS_TEMPLATE,
            "recommendation": BUY_SELL_RECOMMENDATION_TEMPLATE,
            "portfolio": PORTFOLIO_OVERVIEW_TEMPLATE
        }
        self.system_prompt = SYSTEM_PROMPT

    def get_prompt(self, template_name: str, **kwargs: Any) -> str:
        """
        Render a specific template with provided data.
        
        Args:
            template_name: The name of the template to use.
            **kwargs: Data to fill into the template placeholders.
            
        Returns:
            The rendered prompt string.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")
        
        template = self.templates[template_name]
        return template.format(**kwargs)

    def get_system_prompt(self) -> str:
        """Get the default system prompt."""
        return self.system_prompt
