# Chapter 6: Discussion

## 6.1 Overview
This chapter interprets the results reported in Chapter 5, focusing on how the observed patterns inform the system's strengths, weaknesses, and practical use. It discusses implications for local-only deployment, highlights limitations such as overfitting risk, and outlines future work.

## 6.2 Discussion of Results
The model comparison and walk-forward validation outputs indicate that performance varies by model choice and evaluation window, reinforcing the need for consistent, time-aware validation. The backtesting summary suggests the strategy can reduce downside exposure relative to a benchmark, while potentially lagging in strong bull-market conditions due to more conservative positioning. Taken together, the results support the feasibility of a local-only financial advisor pipeline, while emphasizing that predictive stability and risk management are the primary value rather than maximizing raw returns.

The placeholders in Chapter 5 underscore that final conclusions must be grounded in completed validation and model comparison reports. However, the existing evidence already shows that the evaluation framework is capable of surfacing trade-offs between accuracy and risk-adjusted outcomes, and that the system can be analyzed without external dependencies or cloud services.

## 6.3 Implications
- **Local-Only Deployment**: The system's offline design limits exposure to third-party APIs at runtime, aligning with privacy and reliability goals but constraining model size and update cadence.
- **Risk Management Focus**: The strategy's relative drawdown behavior suggests that a risk-aware advisory stance may be more suitable for conservative users than for maximizing returns.
- **Operational Readiness**: The evaluation pipeline provides a repeatable workflow for comparing models and validating robustness, which is essential before any live advisory use.

## 6.4 Limitations
- **Overfitting Risk**: Walk-forward validation highlights potential gaps between training and validation performance; the system may overfit to historical regimes.
- **Placeholder Results**: Several metrics remain pending insertion, so interpretations must be revisited once final validation and comparison outputs are complete.
- **Market Regime Dependence**: Performance may degrade in regimes not well represented in the training data, especially during extreme volatility shifts.
- **Local-Only Constraints**: Offline runtime limits access to real-time data and advanced cloud-scale models, which can reduce responsiveness and breadth of analysis.

## 6.5 Future Work
- **Expanded Regime Testing**: Evaluate the system across additional market regimes and assets to improve generalizability and reduce overfitting risk.
- **Transaction Cost Modeling**: Integrate more realistic cost and slippage assumptions into backtesting to better reflect deployable performance.
- **Adaptive Model Selection**: Add a lightweight, local model-selection layer that shifts allocation based on recent validation performance.
- **Data Refresh Automation**: Improve local update workflows for market and news data to keep the system current without external runtime dependencies.
- **User-Centered Evaluation**: Conduct qualitative testing with end users to assess interpretability and trust alongside quantitative performance.

## 6.6 Chapter Conclusion
The results indicate that a local-only financial advisor bot can deliver measurable risk-management benefits while maintaining a transparent, reproducible evaluation process. The discussion emphasizes caution around overfitting and incomplete metrics, and it frames future work around stronger generalization, more realistic backtests, and improved operational workflows.
