# Chapter 1: Introduction

## Background
Financial decision support systems increasingly combine market data, technical indicators, and natural language processing to produce actionable insights. The growth of open-source machine learning tools and local model inference makes it feasible to build such systems without dependence on cloud services or third-party APIs at runtime. This project focuses on the S&P 500 equity universe, where abundant historical data and standardized market structure support reproducible experimentation. The system integrates price-based indicators, sentiment signals from financial news, predictive models, and a local LLM interface to surface analysis in a user-facing application.

## Problem Statement
Individual investors and researchers often lack an integrated, local-first workflow that unifies technical analysis, sentiment scoring, and predictive modeling in a single tool. Existing solutions typically rely on external APIs or fragmented pipelines, which can limit reproducibility, introduce availability risks, and complicate evaluation. The problem addressed in this dissertation is how to design and implement a local, end-to-end financial advisory system for S&P 500 securities that supports data ingestion, modeling, and explanation without runtime dependency on external services.

## Objectives
The dissertation pursues the following objectives:

- Design a local data pipeline to collect and store S&P 500 price data and related news sentiment.
- Implement technical indicator features and predictive models suitable for time-series financial data.
- Integrate a local sentiment model (FinBERT) and a local LLM for grounded advisory responses.
- Provide a Streamlit-based interface that presents analytics and model outputs clearly.
- Evaluate model performance using time-aware validation and standard financial metrics.

## Scope
The scope is limited to U.S. S&P 500 equities and a local execution environment. Runtime inference uses locally stored models, including FinBERT for sentiment and a local LLM for response generation. External APIs, if used, are restricted to offline ingestion workflows and are not required during runtime. The system focuses on short-horizon predictive signals and explanatory outputs rather than automated trading or portfolio optimization.

## Contributions
This work makes the following contributions:

- A local-first architecture that consolidates ingestion, feature engineering, modeling, and LLM-based explanation for S&P 500 analysis.
- A reproducible data and modeling pipeline with time-series validation and standardized financial evaluation metrics.
- A practical integration of FinBERT sentiment features with price-based indicators for downstream prediction tasks.
- A user-facing application that presents analytical outputs and model context in a cohesive interface.
