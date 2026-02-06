# Chapter 2: Literature Review

## 2.1 Introduction

The prediction of stock market movements has long been a central pursuit in financial economics and quantitative finance. Traditionally, this field was dominated by econometric models and fundamental analysis. However, the advent of machine learning (ML) and natural language processing (NLP) has catalyzed a paradigm shift, enabling the analysis of vast datasets including non-linear price patterns and unstructured textual data. This chapter reviews the theoretical underpinnings of market predictability, the evolution of algorithmic trading strategies, and the recent emergence of Large Language Models (LLMs) as advisory tools.

## 2.2 Theoretical Foundations: Efficient Market Hypothesis

The Efficient Market Hypothesis (EMH), formalized by Fama (1970), posits that asset prices fully reflect all available information, rendering it impossible to consistently outperform the market on a risk-adjusted basis.

*   **Weak Form Efficiency**: Suggests that past price and volume information is already reflected in current prices. If true, technical analysis would be futile.
*   **Semi-Strong Form Efficiency**: Asserts that all publicly available information, including earnings reports and news, is priced in. This challenges the utility of fundamental analysis.
*   **Strong Form Efficiency**: Claims that even private (insider) information is reflected in prices.

While the EMH provides a rigorous theoretical baseline, empirical anomalies (e.g., momentum, mean reversion) and the success of quantitative hedge funds suggest that markets are not perfectly efficient. This project operates on the **Adaptive Market Hypothesis** (Lo, 2004), which reconciles EMH with behavioral finance, suggesting that profit opportunities exist but evolve over time as market participants adapt.

## 2.3 Machine Learning in Time-Series Forecasting

### 2.3.1 Long Short-Term Memory (LSTM) Networks
Recurrent Neural Networks (RNNs) are designed for sequential data but suffer from the vanishing gradient problem. Hochreiter and Schmidhuber (1997) introduced LSTM networks to overcome this by using gating mechanisms (input, output, and forget gates) to retain long-term dependencies. In financial time-series, LSTMs have demonstrated superior performance over traditional ARIMA models by capturing non-linear temporal dynamics (Fischer & Krauss, 2018).

### 2.3.2 Ensemble Methods: Random Forest and XGBoost
While deep learning offers power, ensemble decision tree methods provide robustness and interpretability.
*   **Random Forest** (Breiman, 2001): Reduces variance by averaging multiple decision trees trained on bootstrapped data. It is particularly effective at handling noisy financial data and preventing overfitting.
*   **XGBoost** (Chen & Guestrin, 2016): A gradient boosting framework that optimizes a differentiable loss function. It has become a standard in Kaggle competitions and industry applications due to its speed and ability to capture complex feature interactions.

This project employs a multi-model approach, comparing LSTM, Random Forest, and XGBoost to leverage the strengths of each—temporal awareness (LSTM) and feature robustness (Ensembles).

## 2.4 Sentiment Analysis in Finance

The correlation between public sentiment and market movements is well-documented. Traditional approaches used "bag-of-words" models (e.g., Loughran-McDonald dictionary), which often failed to capture context.

### 2.4.1 BERT and FinBERT
The introduction of BERT (Bidirectional Encoder Representations from Transformers) by Devlin et al. (2018) revolutionized NLP by enabling context-aware embeddings. However, general-purpose language models often misinterpret financial jargon. Araci (2019) introduced **FinBERT**, a BERT model pre-trained on financial texts (earnings calls, news). FinBERT significantly outperforms generic models in classifying financial sentiment (e.g., "Company X misses earnings" is negative, but "Company X reduces debt" is positive), making it a critical component for modern algorithmic trading systems.

## 2.5 Large Language Models and Financial Advisory

The release of models like Llama 3 (Meta, 2024) and GPT-4 has opened new avenues for automated financial advice. Unlike deterministic trading algorithms, LLMs can synthesize quantitative data (prices) and qualitative data (news) into human-readable narratives.

### 2.5.1 Retrieval-Augmented Generation (RAG)
To address the "hallucination" problem common in LLMs, **Retrieval-Augmented Generation (RAG)** (Lewis et al., 2020) retrieves relevant, up-to-date facts from a knowledge base (in this project, the local SQLite database) and injects them into the model's context window. This ensures that the advice generated is grounded in actual market data rather than the model's training weights.

## 2.6 Conclusion and Gap Analysis

While individual studies have explored LSTMs for prediction or FinBERT for sentiment, few open-source systems integrate these components into a cohesive, **fully local** advisory platform. Existing commercial solutions rely heavily on cloud APIs (privacy risk) or black-box algorithms. This project bridges this gap by implementing an end-to-end, privacy-preserving system that combines ensemble price prediction, domain-specific sentiment analysis, and RAG-based explanation capabilities on consumer-grade hardware.
