"""
Prompt templates for the Financial Advisor LLM.
"""

from typing import Any

SYSTEM_PROMPT = """You are an expert financial advisor AI. Your goal is to provide accurate, data-driven analysis of stocks and portfolios.
Do not include legal or educational disclaimers unless the user explicitly asks for them."""

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

ALLOCATION_GUIDANCE_TEMPLATE = """A user is asking for stock ideas without specifying a ticker.
User Question: {user_query}
Detected Investment Budget: {investment_amount}
Candidate Signals (from local prediction data): {candidate_signals}

Provide practical educational guidance that is useful, specific, and honest:
1) Give a short answer to the question directly.
2) Suggest exactly 2 sample ways to deploy the detected investment budget, using S&P 500 stocks only:
   - Strategy A (Aggressive): higher-growth and higher-volatility stock mix.
   - Strategy B (Conservative): larger-cap, lower-volatility, and more defensive stock mix.
3) For each strategy, include:
   - 3 to 5 tickers
   - A percentage allocation that totals 100%
   - A dollar split using {investment_amount}
   - A short rationale for why the mix matches that risk style
4) If candidate signals are available, prioritize those tickers and explain why.
5) Mention key risks and uncertainty (model confidence, volatility, concentration risk).
6) Keep it non-personalized.
7) Use {investment_amount} consistently. Do not replace it with another amount.
8) Output sections with exact headings:
   - Strategy A (Aggressive)
   - Strategy B (Conservative)
9) In each strategy, each ticker line must include both percentage and dollar split using this format:
   - TICKER: XX% ({investment_amount}-based dollars)
10) Grounding rule: use Candidate Signals as the primary ticker source. If you include a non-candidate ticker, label it as fallback due to limited local signals.

Do not discuss crypto or blockchains unless explicitly asked."""

STRATEGY_FOLLOWUP_TEMPLATE = """The user is asking a follow-up question about a previously discussed sample strategy.
User Question: {user_query}
Target Strategy: {strategy_name}
Target Tickers: {strategy_tickers}

Provide practical educational guidance that is useful, specific, and honest:
1) Give a short direct answer and continue from the previous strategy context.
2) Focus only on the listed target tickers unless the user asks to expand.
3) For each ticker, provide:
   - what the company does (one line)
   - why it fits this strategy style
   - key risks to monitor
   - 1-2 concrete signals/metrics to watch
4) Keep it concise and non-personalized.
5) Do not pivot to crypto or blockchains unless explicitly asked.

If no target tickers are provided, state that prior strategy details are missing and ask the user to rerun the allocation request."""

class PromptManager:
    """Manages LLM prompt templates and rendering."""

    def __init__(self):
        self.templates = {
            "general": GENERAL_ANALYSIS_TEMPLATE,
            "technical": TECHNICAL_ANALYSIS_TEMPLATE,
            "sentiment": SENTIMENT_ANALYSIS_TEMPLATE,
            "recommendation": BUY_SELL_RECOMMENDATION_TEMPLATE,
            "portfolio": PORTFOLIO_OVERVIEW_TEMPLATE,
            "allocation": ALLOCATION_GUIDANCE_TEMPLATE,
            "strategy_followup": STRATEGY_FOLLOWUP_TEMPLATE,
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
