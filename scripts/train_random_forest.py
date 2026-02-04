import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.stock_data import StockDataProcessor
from src.models.random_forest_model import RandomForestModel

def train_rf(ticker, tune=True):
    print(f"Starting Random Forest training for {ticker}...")
    
    # 1. Fetch & Prepare Data
    processor = StockDataProcessor(ticker)
    print("Fetching data...")
    try:
        processor.fetch_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print("Preparing data for classification...")
    # Returns: X_train, y_train, X_test, y_test, feature_cols
    X_train, y_train, X_test, y_test, feature_cols = processor.prepare_for_classification(processor.data)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # 2. Initialize Model
    model = RandomForestModel()
    
    # 3. Train
    print(f"Training model (Tuning: {tune})...")
    model.train(X_train, y_train, X_test, y_test, tune_hyperparameters=tune)
    
    # 4. Save
    date_str = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join("models", f"random_forest_{ticker}_{date_str}.pkl")
    model.save(save_path)
    
    # Feature Importance Display
    if hasattr(model.model, 'feature_importances_'):
        importances = model.model.feature_importances_
        feature_imp = dict(zip(feature_cols, importances))
        print("\n=== Feature Importance Analysis ===")
        for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True):
            print(f"{k:<15}: {v:.4f}")
            
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest Model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    train_rf(args.ticker, tune=not args.no_tune)
