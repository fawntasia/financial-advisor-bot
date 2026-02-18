import sys
import os
import argparse
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.comparison import ModelComparison
from src.models.io_utils import ensure_parent_dir

def main():
    parser = argparse.ArgumentParser(description='Compare models and select production model')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing walk-forward results')
    parser.add_argument('--output_report', type=str, default='docs/model_selection.md', help='Path to save markdown report')
    parser.add_argument('--output_csv', type=str, default='results/comparison_summary.csv', help='Path to save summary CSV')
    parser.add_argument('--production_config', type=str, default='config/production_model.json', help='Path to save production config')
    parser.add_argument('--metric', type=str, default='avg_sharpe', help='Metric for model selection')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model comparison...")
    
    comparison = ModelComparison(results_dir=args.results_dir)
    comparison.load_results()
    
    if not comparison.results:
        logger.error(f"No results found in {args.results_dir}. Run walk-forward validation first.")
        return
        
    # Aggregate metrics
    df = comparison.aggregate_metrics()
    logger.info(f"Aggregated metrics for {len(df)} models.")
    
    # Save CSV
    ensure_parent_dir(args.output_csv)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Comparison summary saved to {args.output_csv}")
    
    # Select best model
    best_model = comparison.select_best_model(metric=args.metric)
    if best_model:
        logger.info(f"Best model based on {args.metric}: {best_model['model']} for {best_model['ticker']}")
        comparison.save_production_config(best_model, config_path=args.production_config)
    
    # Generate report
    report = comparison.generate_report()
    ensure_parent_dir(args.output_report)
    with open(args.output_report, 'w') as f:
        f.write(report)
    logger.info(f"Markdown report saved to {args.output_report}")
    
    print("\n=== Comparison Results ===")
    print(df.to_string(index=False))
    
    if best_model:
        print(f"\nRecommended Production Model: {best_model['model']} (Ticker: {best_model['ticker']})")

if __name__ == "__main__":
    main()
