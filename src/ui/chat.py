import streamlit as st
import re
import logging
import os
import sys
from typing import List, Dict, Optional, Any
from src.llm.llama_loader import LlamaLoader, DEFAULT_MODEL_PATH
from src.llm.prompts import PromptManager
from src.llm.context_builder import ContextBuilder
from src.database.dal import DataAccessLayer
from src.models.production_config import (
    classifier_model_names_for_type,
    resolve_production_classifier_model_type,
)

logger = logging.getLogger(__name__)

class ChatManager:
    """
    Manages the chat interface, message history, and AI interaction.
    """

    def __init__(self, dal: DataAccessLayer, model_path: Optional[str] = None):
        self.dal = dal
        self.prompt_manager = PromptManager()
        self.context_builder = ContextBuilder(dal)

        # Initialize or refresh LlamaLoader in session state.
        # Refresh happens when an existing mock loader is detected but a real model now exists.
        target_model_path = model_path or DEFAULT_MODEL_PATH
        should_initialize_model = "llama_model" not in st.session_state

        if not should_initialize_model:
            existing_model = st.session_state.llama_model
            existing_mock_mode = getattr(existing_model, "mock_mode", None)
            if isinstance(existing_mock_mode, bool) and existing_mock_mode and os.path.exists(target_model_path):
                should_initialize_model = True

        if should_initialize_model:
            with st.spinner("Initializing AI Engine..."):
                st.session_state.llama_model = LlamaLoader(model_path=target_model_path)
        
        self.llm = st.session_state.llama_model

        # Initialize message history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Store last parsed strategy tickers from allocation responses.
        if "last_strategy_tickers" not in st.session_state:
            st.session_state.last_strategy_tickers = {
                "aggressive": [],
                "conservative": [],
            }
        if "last_allocation_amount" not in st.session_state:
            st.session_state.last_allocation_amount = "$50"

        # Keep memory small to fit local-model context windows.
        self.max_history_messages = 6
        self.max_message_chars = 500
        self._ticker_universe_cache: Optional[List[Dict[str, Any]]] = None
        self._last_allocation_context: Dict[str, Any] = {}

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extracts potential stock tickers from text (e.g., AAPL, $TSLA, MSFT).
        """
        # Matches uppercase words of 1-5 characters, optionally preceded by $
        pattern = r'\b\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        candidates = list(dict.fromkeys(matches))

        # Keep only known universe tickers when available to avoid false positives
        # (e.g., pronoun "I" in normal sentences).
        known_tickers = set()
        try:
            known_tickers = self._get_known_tickers()
        except Exception:
            known_tickers = set()

        if known_tickers:
            filtered = [ticker for ticker in candidates if ticker in known_tickers]
            if filtered:
                return filtered

        # Fallback: resolve company-name queries like "steris" -> "STE".
        name_matches = self._resolve_tickers_from_company_names(text, known_tickers)
        if name_matches:
            return name_matches

        # Only fall back to raw regex candidates when ticker-universe validation
        # is unavailable. This prevents false matches like pronoun "I".
        if not known_tickers:
            return candidates
        return []

    def _is_allocation_query(self, text: str) -> bool:
        """Detect broad investment-allocation questions without explicit ticker targets."""
        query = (text or "").lower()
        patterns = [
            r"\bwhich stocks?\b",
            r"\bwhat stocks?\b",
            r"\brecommend(?:ation)?\b",
            r"\bsuggest(?:ion)?\b",
            r"\binvest(?:ment|ments|ing)?\b",
            r"\bportfolio\b",
            r"\ballocat(?:e|ion)\b",
            r"\bincome\b",
            r"\bdividend(?:s)?\b",
            r"\bwhere should i (?:put|invest|buy)\b",
            r"\bwhat should i (?:buy|invest in)\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _is_allocation_followup_query(self, text: str) -> bool:
        """Detect short follow-ups that should stay in allocation guidance mode."""
        query = (text or "").lower().strip()
        if not query:
            return False

        has_context = bool(self._last_allocation_context.get("investment_amount"))
        if not has_context:
            return False

        patterns = [
            r"\bi am looking for\b",
            r"\blooking for\b",
            r"\bmore conservative\b",
            r"\bmore aggressive\b",
            r"\blower risk\b",
            r"\bhigher risk\b",
            r"\bhigh risk\b",
            r"\bincome\b",
            r"\bgrowth\b",
            r"\bdividend(?:s)?\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _is_recommendation_query(self, text: str) -> bool:
        """Detect explicit recommendation intent for ticker-scoped prompts."""
        query = (text or "").lower()
        patterns = [
            r"\brecommend(?:ation)?\b",
            r"\bbuy\b",
            r"\bsell\b",
            r"\bhold\b",
            r"\brating\b",
            r"\boutlook\b",
            r"\bshould i buy\b",
            r"\bshould i sell\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _is_strategy_followup_query(self, text: str) -> bool:
        """Detect follow-up questions that refer to strategy A/B instead of explicit tickers."""
        query = (text or "").lower()
        patterns = [
            r"\bstrategy\s*a\b",
            r"\bstrategy\s*b\b",
            r"\baggressive strategy\b",
            r"\bconservative strategy\b",
            r"\bthose stocks\b",
            r"\bthese stocks\b",
            r"\bmore about\b.*\bstrategy\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _resolve_followup_strategy_name(self, text: str) -> str:
        """Resolve user text to aggressive/conservative/both strategy labels."""
        query = (text or "").lower()
        if re.search(r"\b(strategy\s*a|aggressive|first strategy)\b", query):
            return "Aggressive (Strategy A)"
        if re.search(r"\b(strategy\s*b|conservative|second strategy)\b", query):
            return "Conservative (Strategy B)"
        return "Both (Strategy A and Strategy B)"

    def _get_known_tickers(self) -> set:
        """Return known universe tickers when available."""
        try:
            return {str(t).upper() for t in self.dal.get_all_tickers()}
        except Exception:
            return set()

    def _get_ticker_universe_cached(self) -> List[Dict[str, Any]]:
        """Cache ticker metadata to avoid repeated DB reads per message."""
        if self._ticker_universe_cache is None:
            try:
                self._ticker_universe_cache = self.dal.get_ticker_universe()
            except Exception:
                self._ticker_universe_cache = []
        return self._ticker_universe_cache

    def _resolve_tickers_from_company_names(self, text: str, known_tickers: set) -> List[str]:
        """Map company-name mentions in user text to ticker symbols."""
        if not text:
            return []

        normalized_query = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        normalized_query = re.sub(r"\s+", " ", normalized_query).strip()
        if not normalized_query:
            return []

        suffixes = {
            "inc", "incorporated", "corp", "corporation", "co", "company",
            "plc", "ltd", "limited", "group", "holdings", "class",
        }

        matches: List[str] = []
        seen = set()
        for row in self._get_ticker_universe_cached():
            ticker = str(row.get("ticker", "")).upper().strip()
            name = str(row.get("name", "")).lower().strip()
            if not ticker or not name:
                continue
            if known_tickers and ticker not in known_tickers:
                continue

            name_norm = re.sub(r"[^a-z0-9 ]+", " ", name)
            tokens = [tok for tok in name_norm.split() if tok and tok not in suffixes]
            if not tokens:
                continue

            aliases = {" ".join(tokens)}
            if len(tokens[0]) >= 4:
                aliases.add(tokens[0])

            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                    if ticker not in seen:
                        matches.append(ticker)
                        seen.add(ticker)
                    break

        return matches

    def _extract_investment_amount(self, text: str, default_amount: float = 50.0) -> float:
        """Extract intended investment budget from user text."""
        if not text:
            return default_amount

        numeric_amount = self._extract_numeric_amount(text)
        if numeric_amount is not None and numeric_amount > 0:
            return numeric_amount

        word_amount = self._extract_word_amount(text)
        if word_amount is not None and word_amount > 0:
            return word_amount

        return default_amount

    def _extract_numeric_amount(self, text: str) -> Optional[float]:
        """Extract numeric amounts like $10,000, 10k, 2.5 million."""
        query = (text or "").lower().replace(",", "")
        if not query:
            return None

        pattern = re.compile(r"(?<![a-z])\$?\s*(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\b")
        multipliers = {
            "k": 1_000,
            "m": 1_000_000,
            "b": 1_000_000_000,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
        }
        relevance_keywords = (
            "invest", "investment", "budget", "capital", "allocate",
            "deploy", "put", "have", "with", "dollar", "usd", "spend",
        )

        best_match: Optional[float] = None
        best_score = -1

        for match in pattern.finditer(query):
            # Skip percentages (e.g., "50%")
            if match.end() < len(query) and query[match.end(): match.end() + 1] == "%":
                continue

            raw_value = float(match.group(1))
            suffix = match.group(2)
            multiplier = multipliers.get(suffix, 1)
            value = raw_value * multiplier

            start, end = match.start(), match.end()
            context = query[max(0, start - 24): min(len(query), end + 24)]
            has_currency_cue = "$" in match.group(0) or ("usd" in context) or ("dollar" in context)

            # Avoid obvious index/year matches unless there is a money cue.
            if not suffix and not has_currency_cue:
                compact_prefix = query[max(0, start - 6):start].replace(" ", "")
                if compact_prefix.endswith("s&p"):
                    continue
                if 1900 <= value <= 2100:
                    continue

            score = sum(1 for kw in relevance_keywords if kw in context)
            if has_currency_cue:
                score += 2
            if suffix:
                score += 1

            if score > best_score:
                best_score = score
                best_match = value

        return best_match

    def _extract_word_amount(self, text: str) -> Optional[float]:
        """Extract word-based amounts like 'ten thousand'."""
        query = (text or "").lower()
        if not query:
            return None

        number_word = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and)"
        scale_pattern = re.compile(rf"\b((?:{number_word}(?:[\s-]+{number_word})*))\s+(thousand|million|billion)\b")
        scale_multipliers = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

        for match in scale_pattern.finditer(query):
            prefix_words = match.group(1)
            scale_word = match.group(2)
            base = self._words_to_int_under_thousand(prefix_words)
            if base is not None and base > 0:
                return float(base * scale_multipliers[scale_word])

        dollars_pattern = re.compile(rf"\b((?:{number_word}(?:[\s-]+{number_word})*))\s+dollars?\b")
        dollars_match = dollars_pattern.search(query)
        if dollars_match:
            value = self._words_to_int_under_thousand(dollars_match.group(1))
            if value is not None and value > 0:
                return float(value)

        return None

    def _words_to_int_under_thousand(self, words: str) -> Optional[int]:
        """Convert words like 'ten' or 'one hundred twenty five' to int (<1000)."""
        if not words:
            return None

        ones = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        }
        tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }

        tokens = [t for t in re.split(r"[\s-]+", words.strip()) if t]
        if not tokens:
            return None

        current = 0
        for token in tokens:
            if token == "and":
                continue
            if token in ones:
                current += ones[token]
                continue
            if token in tens:
                current += tens[token]
                continue
            if token == "hundred":
                current = (current if current > 0 else 1) * 100
                continue
            return None

        return current if current >= 0 else None

    def _format_currency_amount(self, amount: float) -> str:
        """Format numeric amount to currency string for prompts."""
        if amount >= 100:
            return f"${amount:,.0f}"
        if float(amount).is_integer():
            return f"${int(amount)}"
        return f"${amount:,.2f}"

    def _has_usable_ticker_context(self, context_data: Dict[str, Any]) -> bool:
        """Return True when context contains usable ticker data (not placeholder/no-data text)."""
        if not context_data:
            return False

        raw_price = context_data.get("price")
        if raw_price is None:
            return False

        price_summary = str(raw_price).strip().lower()
        if not price_summary:
            return False
        if price_summary in {"none", "n/a", "na"}:
            return False

        unavailable_markers = (
            "no price data available",
            "price data is currently unavailable",
        )
        return not any(marker in price_summary for marker in unavailable_markers)

    def _strip_auto_disclaimer(self, text: str) -> str:
        """Remove auto-appended disclaimer paragraphs from model output."""
        if not text:
            return text

        cleaned = re.sub(r"(?is)(?:^|\n)\s*Disclaimer:\s.*?(?=\n\s*\n|$)", "", text)
        cleaned = re.sub(r"(?im)^\s*Please consult with a qualified financial advisor.*$", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _resolve_production_prediction_model_names(self) -> List[str]:
        """Resolve production model names used by `predictions.model_name`."""
        try:
            production_model_type = resolve_production_classifier_model_type()
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.warning(f"Unable to resolve production model config: {e}")
            return []

        return sorted(classifier_model_names_for_type(production_model_type))

    def _query_top_prediction_candidates(
        self,
        limit: int,
        model_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query top bullish candidates with optional model-name filtering."""
        with self.dal.get_connection() as conn:
            cursor = conn.cursor()
            params: List[Any] = []
            model_filter_sql = ""
            if model_names:
                placeholders = ",".join("?" for _ in model_names)
                model_filter_sql = f"WHERE p.model_name IN ({placeholders})"
                params.extend(model_names)

            sql = f"""
                WITH ranked_predictions AS (
                    SELECT
                        p.ticker,
                        p.model_name,
                        p.date,
                        p.predicted_direction,
                        p.confidence,
                        p.created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY p.ticker, p.model_name
                            ORDER BY p.date DESC, p.created_at DESC
                        ) AS rn
                    FROM predictions p
                    {model_filter_sql}
                ),
                latest_model_predictions AS (
                    SELECT ticker, model_name, date, predicted_direction, confidence
                    FROM ranked_predictions
                    WHERE rn = 1
                ),
                ticker_scores AS (
                    SELECT
                        ticker,
                        MAX(date) AS latest_prediction_date,
                        SUM(CASE WHEN predicted_direction = 1 THEN 1 ELSE 0 END) AS up_votes,
                        COUNT(*) AS total_votes,
                        AVG(
                            CASE
                                WHEN predicted_direction = 1 THEN COALESCE(confidence, 0.0)
                                ELSE NULL
                            END
                        ) AS avg_up_confidence
                    FROM latest_model_predictions
                    GROUP BY ticker
                )
                SELECT
                    ticker,
                    latest_prediction_date,
                    up_votes,
                    total_votes,
                    COALESCE(avg_up_confidence, 0.0) AS avg_up_confidence
                FROM ticker_scores
                WHERE up_votes > 0
                ORDER BY up_votes DESC, avg_up_confidence DESC, ticker ASC
                LIMIT ?
            """
            params.append(int(max(1, limit)))
            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def _get_top_prediction_candidates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top bullish candidates from latest per-model predictions per ticker.
        Returns an empty list if no prediction data is available.
        """
        try:
            production_model_names = self._resolve_production_prediction_model_names()
            if production_model_names:
                filtered = self._query_top_prediction_candidates(
                    limit=limit,
                    model_names=production_model_names,
                )
                if filtered:
                    return filtered
                logger.warning(
                    "No prediction candidates found for production model names "
                    f"{production_model_names}; falling back to all model predictions."
                )

            return self._query_top_prediction_candidates(limit=limit, model_names=None)
        except Exception as e:
            logger.warning(f"Failed to fetch prediction candidates: {e}")
            return []

    def _build_allocation_prompt(self, user_query: str) -> str:
        """Build a targeted prompt for no-ticker investment allocation queries."""
        amount_value = self._extract_investment_amount(user_query, default_amount=50.0)
        investment_amount = self._format_currency_amount(amount_value)
        st.session_state.last_allocation_amount = investment_amount

        candidates = self._get_top_prediction_candidates(limit=5)

        if candidates:
            candidate_lines = []
            for row in candidates:
                ticker = row.get("ticker", "N/A")
                pred_date = row.get("latest_prediction_date", "N/A")
                up_votes = row.get("up_votes", 0)
                total_votes = row.get("total_votes", 0)
                avg_conf = row.get("avg_up_confidence", 0.0)
                candidate_lines.append(
                    f"- {ticker}: bullish votes {up_votes}/{total_votes}, "
                    f"avg bullish confidence {float(avg_conf):.2f}, prediction date {pred_date}"
                )
            candidate_signals = "\n".join(candidate_lines)
        else:
            candidate_signals = (
                "No recent model prediction candidates are available. "
                "Use diversified, risk-aware examples and state that signals are limited."
            )

        allocation_prompt = self.prompt_manager.get_prompt(
            "allocation",
            user_query=user_query,
            investment_amount=investment_amount,
            candidate_signals=candidate_signals,
        )

        self._last_allocation_context = {
            "user_query": user_query,
            "amount_value": float(amount_value),
            "investment_amount": investment_amount,
            "candidates": candidates,
            "candidate_signals": candidate_signals,
        }

        return allocation_prompt

    def _allocation_candidate_tickers(self) -> List[str]:
        """Return candidate ticker symbols from the most recent allocation context."""
        candidates = self._last_allocation_context.get("candidates", [])
        result = []
        for row in candidates:
            ticker = str(row.get("ticker", "")).upper().strip()
            if ticker:
                result.append(ticker)
        return result

    def _is_allocation_response_grounded(self, response: str) -> bool:
        """Basic quality gate for allocation responses before displaying them."""
        if not response:
            return False

        text = response.strip()
        if not re.search(r"strategy\s*a", text, flags=re.IGNORECASE):
            return False
        if not re.search(r"strategy\s*b", text, flags=re.IGNORECASE):
            return False
        if "%" not in text:
            return False

        investment_amount = str(self._last_allocation_context.get("investment_amount", "")).strip()
        if investment_amount and investment_amount not in text:
            return False

        candidate_tickers = self._allocation_candidate_tickers()
        if candidate_tickers:
            if not any(re.search(rf"\b{re.escape(ticker)}\b", text) for ticker in candidate_tickers):
                return False

        return True

    def _pick_unique_tickers(self, primary: List[str], secondary: List[str], desired: int) -> List[str]:
        """Pick a unique ticker list from two ordered pools."""
        picked: List[str] = []
        seen = set()
        for pool in (primary, secondary):
            for ticker in pool:
                if not ticker or ticker in seen:
                    continue
                picked.append(ticker)
                seen.add(ticker)
                if len(picked) >= desired:
                    return picked
        return picked

    def _allocation_weights(self, n: int) -> List[int]:
        """Return simple allocation percentage splits for 3-5 positions."""
        if n <= 3:
            return [40, 35, 25][: max(1, n)]
        if n == 4:
            return [35, 30, 20, 15]
        return [30, 25, 20, 15, 10][:n]

    def _allocation_dollar_split(self, amount_value: float, weights: List[int]) -> List[str]:
        """Convert percentage weights to dollar strings that sum exactly to budget."""
        total_cents = int(round(float(amount_value) * 100))
        raw_cents = [int(round(total_cents * (weight / 100.0))) for weight in weights]
        if raw_cents:
            raw_cents[-1] += total_cents - sum(raw_cents)
        return [f"${cents / 100:,.2f}" for cents in raw_cents]

    def _build_grounded_allocation_fallback(self, user_query: str) -> str:
        """Build deterministic, app-grounded allocation guidance when LLM output drifts."""
        ctx = self._last_allocation_context or {}
        amount_value = float(ctx.get("amount_value", 50.0))
        investment_amount = str(ctx.get("investment_amount") or self._format_currency_amount(amount_value))
        candidates = ctx.get("candidates", []) or []

        sorted_candidates = sorted(
            candidates,
            key=lambda row: (
                -int(row.get("up_votes", 0) or 0),
                -float(row.get("avg_up_confidence", 0.0) or 0.0),
                str(row.get("ticker", "")),
            ),
        )
        candidate_tickers = [str(row.get("ticker", "")).upper().strip() for row in sorted_candidates if row.get("ticker")]
        candidate_map = {str(row.get("ticker", "")).upper().strip(): row for row in sorted_candidates if row.get("ticker")}

        universe_rows = self._get_ticker_universe_cached()
        known_tickers = self._get_known_tickers()
        sector_by_ticker = {
            str(row.get("ticker", "")).upper().strip(): str(row.get("sector", "")).strip()
            for row in universe_rows
            if row.get("ticker")
        }
        name_by_ticker = {
            str(row.get("ticker", "")).upper().strip(): str(row.get("name", "")).strip()
            for row in universe_rows
            if row.get("ticker")
        }

        def _valid(pool: List[str]) -> List[str]:
            if not known_tickers:
                return [t for t in pool if t]
            return [t for t in pool if t in known_tickers]

        fallback_growth = _valid(["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AVGO", "TSLA"])
        fallback_defensive = _valid(["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "XOM", "DUK", "SO", "NEE"])
        defensive_sectors = {"Utilities", "Consumer Staples", "Health Care", "Real Estate"}

        conservative_candidates = [
            ticker for ticker in candidate_tickers
            if sector_by_ticker.get(ticker, "") in defensive_sectors
        ]

        aggressive = self._pick_unique_tickers(
            primary=candidate_tickers,
            secondary=fallback_growth + fallback_defensive,
            desired=4,
        )
        conservative = self._pick_unique_tickers(
            primary=conservative_candidates + candidate_tickers,
            secondary=fallback_defensive + fallback_growth,
            desired=4,
        )

        if len(aggressive) < 3:
            aggressive = self._pick_unique_tickers(aggressive, fallback_growth + fallback_defensive, desired=3)
        if len(conservative) < 3:
            conservative = self._pick_unique_tickers(conservative, fallback_defensive + fallback_growth, desired=3)

        def _strategy_lines(tickers: List[str], style: str) -> List[str]:
            weights = self._allocation_weights(len(tickers))
            dollars = self._allocation_dollar_split(amount_value, weights)
            lines = []
            for idx, ticker in enumerate(tickers):
                company_name = name_by_ticker.get(ticker, "")
                display = f"{ticker} ({company_name})" if company_name else ticker
                row = candidate_map.get(ticker)
                if row:
                    evidence = (
                        f"Local signal {int(row.get('up_votes', 0))}/{int(row.get('total_votes', 0))} bullish votes, "
                        f"avg bullish confidence {float(row.get('avg_up_confidence', 0.0)):.2f}."
                    )
                else:
                    evidence = "Fallback S&P 500 diversification pick because local bullish candidates are limited."
                lines.append(f"- {display}: {weights[idx]}% ({dollars[idx]}). {evidence}")
            return lines

        income_focus = bool(re.search(r"\b(income|dividend)\b", (user_query or "").lower()))
        conservative_rationale = (
            "larger-cap, defensive tilt for steadier cash-flow exposure and lower volatility."
            if income_focus
            else "larger-cap, defensive tilt to reduce volatility and concentration risk."
        )

        candidate_text = str(ctx.get("candidate_signals", "")).strip()
        if not candidate_text:
            candidate_text = "No recent model prediction candidates were available in local data."

        return "\n".join(
            [
                f"Short answer: For {investment_amount}, here are two sample S&P 500 allocations grounded in this app's local prediction data.",
                "",
                "Strategy A (Aggressive): higher-growth and higher-volatility mix.",
                *_strategy_lines(aggressive, style="aggressive"),
                "",
                "Strategy B (Conservative): " + conservative_rationale,
                *_strategy_lines(conservative, style="conservative"),
                "",
                "Grounding (local candidate signals):",
                candidate_text,
                "",
                "Key risks: model confidence can shift quickly, concentrated bets increase drawdown risk, and short-term forecasts are uncertain.",
            ]
        )

    def _parse_strategy_tickers_from_response(self, response: str) -> Dict[str, List[str]]:
        """Parse Strategy A/B ticker mentions from the assistant's allocation response."""
        parsed = {"aggressive": [], "conservative": []}
        if not response:
            return parsed

        normalized = response.replace("\r\n", "\n")
        known_tickers = self._get_known_tickers()

        section_patterns = {
            "aggressive": r"Strategy\s*A\b[\s\S]*?(?=Strategy\s*B\b|Disclaimer\b|$)",
            "conservative": r"Strategy\s*B\b[\s\S]*?(?=Disclaimer\b|$)",
        }

        for key, pattern in section_patterns.items():
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue

            section_text = match.group(0)
            section_body = re.sub(
                r"^\s*Strategy\s*[AB][^\n]*\n?",
                "",
                section_text,
                flags=re.IGNORECASE,
            )
            tickers = []

            # Prefer "(TICKER)" style mentions, then fallback to uppercase tokens.
            for token in re.findall(r"\(([A-Z]{1,5})\)", section_body):
                if not known_tickers or token in known_tickers:
                    tickers.append(token)

            # Common list forms like "1. TSLA - ..." or "- NVDA: ..."
            for token in re.findall(r"(?:^|\n)\s*(?:[-*]|\d+\.)\s*([A-Z]{1,5})\b", section_body):
                if not known_tickers or token in known_tickers:
                    tickers.append(token)

            # Final fallback for plain uppercase mentions.
            for token in re.findall(r"\b[A-Z]{1,5}\b", section_body):
                if token in {"AI", "ETF", "USD", "S", "P", "SP"}:
                    continue
                if not known_tickers or token in known_tickers:
                    tickers.append(token)

            # Preserve order while de-duplicating.
            deduped = []
            seen = set()
            for t in tickers:
                if t not in seen:
                    deduped.append(t)
                    seen.add(t)

            parsed[key] = deduped[:5]

        return parsed

    def _build_strategy_followup_prompt(self, user_query: str) -> str:
        """Build a focused follow-up prompt using previously parsed strategy tickers."""
        strategy_name = self._resolve_followup_strategy_name(user_query)
        stored = st.session_state.get("last_strategy_tickers", {})

        aggressive = stored.get("aggressive", []) if isinstance(stored, dict) else []
        conservative = stored.get("conservative", []) if isinstance(stored, dict) else []

        if "Aggressive" in strategy_name:
            target_tickers = aggressive
        elif "Conservative" in strategy_name:
            target_tickers = conservative
        else:
            target_tickers = aggressive + [t for t in conservative if t not in aggressive]

        strategy_tickers = ", ".join(target_tickers) if target_tickers else "None available"
        return self.prompt_manager.get_prompt(
            "strategy_followup",
            user_query=user_query,
            strategy_name=strategy_name,
            strategy_tickers=strategy_tickers,
        )

    def _build_history_block(self) -> str:
        """Build a compact history block for prompt context."""
        messages = st.session_state.get("messages", [])
        if not messages:
            return "No prior conversation."

        history_lines = []
        for message in messages[-self.max_history_messages:]:
            role = message.get("role", "user")
            role_label = "User" if role == "user" else "Assistant"
            content = str(message.get("content", "")).strip()
            if not content:
                continue

            compact_content = re.sub(r"\s+", " ", content)
            if len(compact_content) > self.max_message_chars:
                compact_content = compact_content[: self.max_message_chars - 3].rstrip() + "..."

            history_lines.append(f"{role_label}: {compact_content}")

        return "\n".join(history_lines) if history_lines else "No prior conversation."

    def _compose_prompt_with_memory(self, task_prompt: str, include_history: bool = True) -> str:
        """Compose final prompt with system instructions and recent conversation context."""
        prompt_parts = [self.prompt_manager.get_system_prompt()]
        if include_history:
            history_block = self._build_history_block()
            prompt_parts.append(f"Conversation Context (most recent turns):\n{history_block}")

        prompt_parts.append(f"Current Task:\n{task_prompt}")
        prompt_parts.append(
            "Respond as the assistant. Prioritize the current task and use context only when relevant."
        )
        return "\n\n".join(prompt_parts)

    def display_chat(self):
        """
        Renders the chat interface.
        """
        active_mock_mode = getattr(self.llm, "mock_mode", None)
        if isinstance(active_mock_mode, bool) and active_mock_mode:
            st.warning(
                "AI engine is running in mock mode. Restart using "
                "`venv\\Scripts\\python.exe -m streamlit run app.py`. "
                f"Current Python: `{sys.executable}`"
            )

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader("AI Financial Advisor", help="Ask questions about stocks, market trends, or technical indicators.")
        with col2:
            if st.button("🚩 Report Issue", help="Report a bug or incorrect AI response."):
                st.toast("Issue reported. Thank you for your feedback!", icon="🚩")
        
        # Display disclaimer
        st.caption("⚠️ **Disclaimer:** This AI advisor provides information for educational purposes only. "
                   "It does not constitute financial advice. Past performance is not indicative of future results.")

        # Display message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "context" in message and message["context"]:
                    with st.expander("View Sources/Context"):
                        for key, value in message["context"].items():
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            st.write(value)

        # Chat input
        if prompt := st.chat_input("Ask me about a stock (e.g., 'What is the outlook for AAPL?')"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process AI response
            self._handle_response(prompt)

    def _handle_response(self, prompt: str):
        """
        Handles generating and displaying the AI response.
        """
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Extract tickers and fetch context
                tickers = self.extract_tickers(prompt)
                context_data = {}
                ticker = None
                include_history = True
                used_allocation_prompt = False
                
                if tickers:
                    # For simplicity, we take the first ticker found for context building
                    ticker = tickers[0]
                    try:
                        context_data = self.context_builder.build_context(ticker)
                        if not self._has_usable_ticker_context(context_data):
                            st.warning(f"Could not find detailed market data for **{ticker}**. Providing a general response.")
                            context_data = {}
                    except Exception as e:
                        logger.error(f"Error fetching context for {ticker}: {e}")
                        st.error(f"Oops! I encountered an error while searching for data on **{ticker}**.")
                        context_data = {}
                
                if ticker and context_data:
                    # Construct prompt with context
                    template_name = "recommendation" if self._is_recommendation_query(prompt) else "general"
                    full_prompt = self.prompt_manager.get_prompt(
                        template_name,
                        ticker=ticker,
                        price=context_data.get("price", "N/A"),
                        indicators_summary=context_data.get("indicators_summary", "N/A"),
                        sentiment_summary=context_data.get("sentiment_summary", "N/A"),
                        prediction_summary=context_data.get("prediction_summary", "N/A")
                    )
                    # Avoid contamination from prior generic replies for explicit ticker analysis.
                    include_history = False
                else:
                    # Generic response if no ticker is found or context failed
                    if self._is_strategy_followup_query(prompt):
                        full_prompt = self._build_strategy_followup_prompt(prompt)
                        include_history = False
                    elif self._is_allocation_query(prompt):
                        full_prompt = self._build_allocation_prompt(prompt)
                        used_allocation_prompt = True
                        include_history = False
                        context_data = {
                            "allocation_budget": self._last_allocation_context.get("investment_amount", "N/A"),
                            "candidate_signals": self._last_allocation_context.get("candidate_signals", "N/A"),
                        }
                    elif self._is_allocation_followup_query(prompt):
                        full_prompt = self._build_allocation_prompt(prompt)
                        used_allocation_prompt = True
                        include_history = False
                        context_data = {
                            "allocation_budget": self._last_allocation_context.get("investment_amount", "N/A"),
                            "candidate_signals": self._last_allocation_context.get("candidate_signals", "N/A"),
                        }
                    else:
                        full_prompt = prompt

                full_prompt = self._compose_prompt_with_memory(
                    full_prompt,
                    include_history=include_history,
                )

                # 2. Generate response
                try:
                    response = self.llm.generate(full_prompt)
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    response = "I'm sorry, I'm having trouble connecting to my AI core right now. Please try again in a moment."
                response = self._strip_auto_disclaimer(response)
                if used_allocation_prompt and not self._is_allocation_response_grounded(response):
                    logger.warning("Allocation response was not grounded enough; using deterministic fallback output.")
                    response = self._build_grounded_allocation_fallback(prompt)
                
                # 3. Display response
                st.markdown(response)
                
                # 4. Display context if available
                if context_data:
                    with st.expander("View Sources/Context"):
                        for key, value in context_data.items():
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            st.write(value)

                # 5. Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "context": context_data
                })

                # 6. Update strategy memory from allocation-style responses.
                if used_allocation_prompt:
                    parsed = self._parse_strategy_tickers_from_response(response)
                    if parsed.get("aggressive") or parsed.get("conservative"):
                        st.session_state.last_strategy_tickers = parsed

    def clear_history(self):
        """Clears the chat history."""
        st.session_state.messages = []
        st.rerun()
