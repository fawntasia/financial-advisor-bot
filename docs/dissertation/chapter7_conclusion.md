# Chapter 7: Conclusion

## 7.1 Summary
This dissertation designed and evaluated a local-first financial advisory system for S&P 500 equities. The project integrated a SQLite-backed data pipeline, technical indicators, FinBERT sentiment features, predictive models, and a Streamlit interface with a local LLM. Emphasis remained on offline runtime, reproducible artifacts, and time-aware evaluation that reflects real-world deployment constraints.

## 7.2 Contributions
- A local-only architecture that consolidates ingestion, feature engineering, modeling, and LLM-grounded explanation for equity analysis.
- A reproducible workflow with walk-forward validation, leakage controls, and standard financial metrics for model comparison.
- An implementation that pairs price-based indicators with FinBERT sentiment to support both regression and classification tasks.
- A user-facing application that surfaces model outputs, indicators, and context in a cohesive interface.

## 7.3 Objectives Revisited
The objectives defined in Chapter 1 were met as follows:

- A local data pipeline and SQLite persistence were implemented to collect and store market and sentiment data.
- Technical indicators and predictive models (LSTM, Random Forest, XGBoost) were built for time-series forecasting and directional signals.
- FinBERT sentiment and a local LLM were integrated to provide grounded, offline advisory responses.
- A Streamlit interface was delivered to present analytics, charts, and model context.
- Evaluation used chronological splits, walk-forward validation, and risk-aware metrics to assess performance.

## 7.4 Limitations and Future Work
Results highlight the need for finalized validation and model-comparison metrics before definitive claims. The local-only constraint improves privacy and reliability but limits real-time updates and model scale. Future work should prioritize expanded regime testing, transaction cost modeling, adaptive model selection, and user-centered evaluation, aligning with the discussion in Chapter 6.

## 7.5 Closing Remarks
Overall, the dissertation demonstrates that a local-only advisory pipeline can deliver transparent, repeatable analysis when paired with rigorous evaluation. The system is positioned as a research-grade foundation for further refinement rather than a production trading tool, with next steps focused on stronger generalization and more realistic backtesting.
