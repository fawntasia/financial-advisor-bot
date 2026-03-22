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

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extracts potential stock tickers from text (e.g., AAPL, $TSLA, MSFT).
        """
        # Matches uppercase words of 1-5 characters, optionally preceded by $
        pattern = r'\b\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        candidates = list(set(matches))

        # Keep only known universe tickers when available to avoid false positives
        # (e.g., pronoun "I" in normal sentences).
        known_tickers = set()
        try:
            known_tickers = self._get_known_tickers()
            filtered = [ticker for ticker in candidates if ticker in known_tickers]
            if filtered:
                return filtered
        except Exception:
            pass

        # Fallback: resolve company-name queries like "steris" -> "STE".
        name_matches = self._resolve_tickers_from_company_names(text, known_tickers)
        if name_matches:
            return name_matches

        return candidates

    def _is_allocation_query(self, text: str) -> bool:
        """Detect broad investment-allocation questions without explicit ticker targets."""
        query = (text or "").lower()
        patterns = [
            r"\bwhich stocks?\b",
            r"\bwhat stocks?\b",
            r"\brecommend(?:ation)?\b",
            r"\bsuggest(?:ion)?\b",
            r"\binvest(?:ment|ing)?\b",
            r"\bportfolio\b",
            r"\ballocat(?:e|ion)\b",
            r"\bwhere should i (?:put|invest|buy)\b",
            r"\bwhat should i (?:buy|invest in)\b",
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

    def _strip_auto_disclaimer(self, text: str) -> str:
        """Remove auto-appended disclaimer paragraphs from model output."""
        if not text:
            return text

        cleaned = re.sub(r"(?is)(?:^|\n)\s*Disclaimer:\s.*?(?=\n\s*\n|$)", "", text)
        cleaned = re.sub(r"(?im)^\s*Please consult with a qualified financial advisor.*$", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _get_top_prediction_candidates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top bullish candidates from latest per-model predictions per ticker.
        Returns an empty list if no prediction data is available.
        """
        try:
            with self.dal.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
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
                    """,
                    (int(max(1, limit)),),
                )
                return [dict(row) for row in cursor.fetchall()]
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

        return allocation_prompt

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
                
                if tickers:
                    # For simplicity, we take the first ticker found for context building
                    ticker = tickers[0]
                    try:
                        context_data = self.context_builder.build_context(ticker)
                        if not context_data or not context_data.get("price"):
                             st.warning(f"Could not find detailed market data for **{ticker}**. Providing a general response.")
                             context_data = {}
                    except Exception as e:
                        logger.error(f"Error fetching context for {ticker}: {e}")
                        st.error(f"Oops! I encountered an error while searching for data on **{ticker}**.")
                        context_data = {}
                
                if ticker and context_data:
                    # Construct prompt with context
                    full_prompt = self.prompt_manager.get_prompt(
                        "general",
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
                    elif self._is_allocation_query(prompt):
                        full_prompt = self._build_allocation_prompt(prompt)
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
                if self._is_allocation_query(prompt):
                    parsed = self._parse_strategy_tickers_from_response(response)
                    if parsed.get("aggressive") or parsed.get("conservative"):
                        st.session_state.last_strategy_tickers = parsed

    def clear_history(self):
        """Clears the chat history."""
        st.session_state.messages = []
        st.rerun()
