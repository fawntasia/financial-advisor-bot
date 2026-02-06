import streamlit as st
import re
import logging
from typing import List, Dict, Optional, Any
from src.llm.llama_loader import LlamaLoader
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
        
        # Initialize LlamaLoader (lazy loading or in session state)
        if "llama_model" not in st.session_state:
            with st.spinner("Initializing AI Engine..."):
                if model_path:
                    st.session_state.llama_model = LlamaLoader(model_path=model_path)
                else:
                    st.session_state.llama_model = LlamaLoader()
        
        self.llm = st.session_state.llama_model

        # Initialize message history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extracts potential stock tickers from text (e.g., AAPL, $TSLA, MSFT).
        """
        # Matches uppercase words of 1-5 characters, optionally preceded by $
        pattern = r'\b\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        return list(set(matches))

    def display_chat(self):
        """
        Renders the chat interface.
        """
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
                else:
                    # Generic response if no ticker is found or context failed
                    full_prompt = prompt

                # 2. Generate response
                try:
                    response = self.llm.generate(full_prompt)
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    response = "I'm sorry, I'm having trouble connecting to my AI core right now. Please try again in a moment."
                
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

    def clear_history(self):
        """Clears the chat history."""
        st.session_state.messages = []
        st.rerun()
