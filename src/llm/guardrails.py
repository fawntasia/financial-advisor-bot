"""
Response Guardrails and Fact-Checking system for the Financial Advisor LLM.
Ensures safety, relevance, and factual accuracy of LLM interactions.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from src.database.dal import DataAccessLayer

logger = logging.getLogger(__name__)

class Guardrails:
    """
    Validates LLM inputs and outputs to ensure safety and factual grounding.
    """

    def __init__(self, dal: DataAccessLayer):
        self.dal = dal
        # Blocklist for disallowed topics
        self.blocklist = [
            r"cryptocurrency", r"crypto\b", r"bitcoin", r"ethereum", r"dogecoin",
            r"gambling", r"betting", r"casino", r"lottery",
            r"illegal", r"hacking", r"stolen", r"insider trading",
            r"get rich quick", r"guaranteed returns", r"money laundering"
        ]
        
        # Mandatory disclaimers that should be present in the output
        self.mandatory_disclaimers = [
            "not a financial advisor",
            "educational purposes only",
            "consult with a qualified professional"
        ]
        
        # Feedback mechanism URL
        self.report_issue_url = "https://github.com/fawntasia/financial-advisor-bot/issues/new"

    def pre_filter(self, user_query: str) -> Dict[str, Any]:
        """
        Check user query for safety and relevance.
        
        Args:
            user_query: The raw input from the user.
            
        Returns:
            Dict containing safety status and potential rejection message.
        """
        query_lower = user_query.lower()
        
        for pattern in self.blocklist:
            if re.search(pattern, query_lower):
                logger.warning(f"Guardrails: Query blocked due to pattern match: {pattern}")
                return {
                    "is_safe": False,
                    "reason": "disallowed_topic",
                    "message": ("I'm sorry, I cannot discuss topics related to cryptocurrency, gambling, "
                                "or illegal activities. My focus is on providing data-driven analysis "
                                "for S&P 500 stocks.")
                }
        
        return {"is_safe": True}

    def post_filter(self, llm_output: str) -> Dict[str, Any]:
        """
        Scan LLM output for disclaimers and safety violations.
        
        Args:
            llm_output: The generated text from the LLM.
            
        Returns:
            Dict containing safety status and any missing disclaimers.
        """
        output_lower = llm_output.lower()
        
        # Check for safety violations (unrealistic claims)
        unsafe_claims = [
            r"guaranteed returns",
            r"no risk",
            r"sure thing",
            r"100% (profit|gain|success)",
            r"can't lose"
        ]
        
        for pattern in unsafe_claims:
            if re.search(pattern, output_lower):
                logger.error(f"Guardrails: LLM output blocked due to unsafe claim: {pattern}")
                return {
                    "is_safe": False,
                    "reason": "unsafe_claim",
                    "message": "The generated response contained an unrealistic financial claim and was blocked for your safety."
                }
        
        # Check for mandatory disclaimers
        missing_disclaimers = []
        for disclaimer in self.mandatory_disclaimers:
            if disclaimer not in output_lower:
                missing_disclaimers.append(disclaimer)
        
        return {
            "is_safe": True,
            "missing_disclaimers": missing_disclaimers,
            "has_all_disclaimers": len(missing_disclaimers) == 0
        }

    def fact_check(self, text: str) -> Dict[str, Any]:
        """
        Verify ticker symbols and price mentions against the database.
        
        Args:
            text: The text to fact-check.
            
        Returns:
            Dict containing factual consistency report.
        """
        # 1. Verify Tickers
        # Extract potential tickers (uppercase words, 1-5 chars)
        potential_tickers = set(re.findall(r'\b[A-Z]{1,5}\b', text))
        valid_tickers = set(self.dal.get_all_tickers())
        
        mentioned_valid_tickers = potential_tickers.intersection(valid_tickers)
        mentioned_invalid_tickers = [t for t in potential_tickers if t.isalpha() and t not in valid_tickers]
        
        # 2. Verify Price Mentions (Experimental)
        # Find patterns like "$123.45" or "123.45 dollars"
        price_mentions = re.findall(r'\$\s?(\d+\.?\d*)', text)
        price_checks = []
        
        if mentioned_valid_tickers and price_mentions:
            # For simplicity, check if any mentioned price is within 20% of the latest price 
            # for ANY of the mentioned valid tickers. This is a loose check to avoid false positives
            # while catching wildly inaccurate hallucinations.
            for ticker in mentioned_valid_tickers:
                latest_date = self.dal.get_latest_price_date(ticker)
                if latest_date:
                    prices_df = self.dal.get_stock_prices(ticker, latest_date, latest_date)
                    if not prices_df.empty:
                        real_price = prices_df.iloc[0]['close']
                        for mention in price_mentions:
                            try:
                                mentioned_price = float(mention)
                                # If the mentioned price is close to the real price, mark it as verified
                                if abs(mentioned_price - real_price) / real_price < 0.2:
                                    price_checks.append({
                                        "ticker": ticker,
                                        "mentioned": mentioned_price,
                                        "actual": real_price,
                                        "verified": True
                                    })
                            except ValueError:
                                continue

        return {
            "mentioned_tickers": list(mentioned_valid_tickers),
            "invalid_tickers": mentioned_invalid_tickers,
            "price_verifications": price_checks,
            "is_factually_grounded": len(mentioned_invalid_tickers) == 0
        }

    def validate(self, user_query: str, llm_output: str) -> Dict[str, Any]:
        """
        Perform a full validation cycle on the interaction.
        
        Args:
            user_query: The user's input.
            llm_output: The LLM's response.
            
        Returns:
            A consolidated validation object.
        """
        pre = self.pre_filter(user_query)
        if not pre["is_safe"]:
            return {
                "safe_to_display": False,
                "output": pre["message"],
                "reason": pre["reason"],
                "report_issue": self.report_issue_url
            }
        
        post = self.post_filter(llm_output)
        fact = self.fact_check(llm_output)
        
        final_output = llm_output
        
        # If unsafe claim found in output, block it
        if not post["is_safe"]:
            return {
                "safe_to_display": False,
                "output": post["message"],
                "reason": post["reason"],
                "report_issue": self.report_issue_url
            }
        
        # If disclaimers are missing, append them
        if not post["has_all_disclaimers"]:
            disclaimer_block = "\n\n---\n**Disclaimer:** I am an AI, not a financial advisor. This information is for educational purposes only and does not constitute financial advice. Always consult with a qualified professional before making investment decisions."
            # Only append if not already present in some form
            if not any(d in final_output.lower() for d in self.mandatory_disclaimers):
                final_output += disclaimer_block
        
        return {
            "safe_to_display": True,
            "output": final_output,
            "fact_check": fact,
            "report_issue": self.report_issue_url
        }
