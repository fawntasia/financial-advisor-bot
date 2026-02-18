import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from src.models.io_utils import ensure_parent_dir

logger = logging.getLogger(__name__)

class ModelComparison:
    """
    Framework to aggregate and compare performance results from different models.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results = {}

    def load_results(self):
        """Load all JSON result files from the results directory."""
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory {self.results_dir} does not exist.")
            return

        for filename in os.listdir(self.results_dir):
            if filename.endswith(".json") and filename.startswith("wf_results_"):
                path = os.path.join(self.results_dir, filename)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # Extract model name from filename: wf_results_{ticker}_{model}.json
                        parts = filename.replace(".json", "").split("_")
                        if len(parts) >= 4:
                            model_name = parts[3]
                            ticker = parts[2]
                            key = f"{ticker}_{model_name}"
                            self.results[key] = data
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

    def aggregate_metrics(self) -> pd.DataFrame:
        """
        Aggregate key metrics across all loaded models.
        Metrics: Directional accuracy, Sharpe ratio, Max drawdown.
        """
        summary = []
        for key, steps in self.results.items():
            ticker, model_name = key.split("_", 1)
            valid_steps = [
                step
                for step in steps
                if step.get("metrics") and not step.get("metadata", {}).get("skipped_reason")
            ]
            if not valid_steps:
                continue
            
            # Aggregate metrics across all walk-forward steps
            test_accuracies = [step['metrics'].get('test_accuracy', 0) for step in valid_steps]
            sharpe_ratios = [step['metrics'].get('sharpe_ratio', 0) for step in valid_steps]
            max_drawdowns = [step['metrics'].get('max_drawdown', 0) for step in valid_steps]
            
            summary.append({
                'ticker': ticker,
                'model': model_name,
                'avg_accuracy': np.mean(test_accuracies),
                'avg_sharpe': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'num_steps': len(valid_steps)
            })
            
        return pd.DataFrame(summary)

    def select_best_model(self, metric: str = 'avg_sharpe') -> Optional[Dict[str, Any]]:
        """
        Select the best model based on a specific metric.
        Default is Sharpe ratio for risk-adjusted returns.
        """
        df = self.aggregate_metrics()
        if df.empty:
            return None
        
        # We want to maximize accuracy and sharpe, but maximize (least negative) max_drawdown
        if metric == 'avg_max_drawdown':
            best_idx = df[metric].idxmax() # Max of negative values is least negative
        else:
            best_idx = df[metric].idxmax()
            
        best_row = df.loc[best_idx]
        return best_row.to_dict()

    def save_production_config(self, best_model: Dict[str, Any], config_path: str = "config/production_model.json"):
        """Save the best model metadata to a production config."""
        ensure_parent_dir(config_path)
        
        config = {
            "production_model": {
                "ticker": best_model['ticker'],
                "model_type": best_model['model'],
                "selection_metric": "avg_sharpe",
                "performance": {
                    "avg_accuracy": best_model['avg_accuracy'],
                    "avg_sharpe": best_model['avg_sharpe'],
                    "avg_max_drawdown": best_model['avg_max_drawdown']
                },
                "updated_at": pd.Timestamp.now().isoformat()
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Production model config saved to {config_path}")

    def generate_report(self) -> str:
        """Generate a markdown report of the comparison."""
        df = self.aggregate_metrics()
        if df.empty:
            return "No results available to compare."
            
        best_model = self.select_best_model()
        
        report = "# Model Comparison and Selection Report\n\n"
        report += f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Selection Criteria\n\n"
        report += "Models are evaluated using walk-forward validation. The following metrics are aggregated across all steps:\n"
        report += "- **Average Accuracy**: Mean directional accuracy (predicting UP vs DOWN).\n"
        report += "- **Average Sharpe Ratio**: Mean risk-adjusted return (using 0% risk-free rate).\n"
        report += "- **Average Max Drawdown**: Mean of the maximum peak-to-trough decline in each step.\n\n"
        report += "The **Production Model** is selected based on the highest **Average Sharpe Ratio**, as it represents the best risk-adjusted performance.\n\n"
        
        report += "## Performance Summary\n\n"
        try:
            report += df.to_markdown(index=False) + "\n\n"
        except ImportError:
            # Fallback when optional dependency `tabulate` is not installed.
            report += "```\n" + df.to_string(index=False) + "\n```\n\n"
        
        report += "## Production Model Selection\n\n"
        report += f"The selected production model is **{best_model['model']}** for **{best_model['ticker']}**.\n"
        report += f"- **Average Sharpe Ratio**: {best_model['avg_sharpe']:.4f}\n"
        report += f"- **Average Accuracy**: {best_model['avg_accuracy']:.4f}\n"
        report += f"- **Average Max Drawdown**: {best_model['avg_max_drawdown']:.4f}\n\n"
        
        report += "## Ensemble Strategy\n\n"
        report += "A simple majority voting ensemble has been implemented to combine predictions from multiple models. "
        report += "This ensemble can be used to improve robustness by requiring agreement between different model architectures.\n"
        
        return report

class EnsembleVoting:
    """Simple voting ensemble (majority vote)."""
    
    @staticmethod
    def vote(predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions from multiple models using majority vote.
        Assumes predictions are 0 or 1.
        """
        if not predictions:
            return np.array([])
            
        # Stack predictions: (num_models, num_samples)
        stacked = np.vstack(predictions)
        # Sum predictions: (num_samples,)
        sums = np.sum(stacked, axis=0)
        # Majority vote: 1 if more than half voted 1
        majority = (sums > (len(predictions) / 2)).astype(int)
        return majority
