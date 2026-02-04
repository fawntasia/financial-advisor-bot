import os
import sys
import argparse
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.stock_data import StockDataProcessor
from src.models.lstm_model import LSTMModel

def train_lstm(ticker: str = "AAPL", epochs: int = 50, batch_size: int = 32, save_dir: str = "models"):
    """
    Train and save LSTM model for a specific ticker.
    """
    print(f"Starting training for {ticker}...")
    
    # 1. Prepare Data
    processor = StockDataProcessor(ticker)
    
    try:
        print("Fetching data...")
        # Fetch data for the last 5 years
        df = processor.fetch_data(years=5)
        print(f"Fetched {len(df)} records.")
        
        print("Preparing data for LSTM...")
        # Prepare data (sequence_length=60 is default in prepare_for_lstm)
        sequence_length = 60
        X_train, y_train, X_test, y_test, scaler, processed_df = processor.prepare_for_lstm(
            df, sequence_length=sequence_length
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # 2. Initialize Model
    n_features = X_train.shape[2]
    model = LSTMModel(sequence_length=sequence_length, n_features=n_features)
    
    # 3. Train Model
    print("Training model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lstm_{ticker}_{timestamp}.keras"
    save_path = os.path.join(save_dir, model_filename)
    
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        patience=5,
        verbose=1
    )
    
    # 4. Evaluation
    print("Evaluating model...")
    loss = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")
    
    # Make some predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actuals to get real prices
    # y_test is scaled, we need to reshape it
    y_test_reshaped = y_test.reshape(-1, 1)
    
    # We use the target_scaler from processor
    # Note: StockDataProcessor.prepare_for_lstm returns target_scaler as the 5th element
    target_scaler = scaler # This is actually target_scaler based on the return signature in stock_data.py
    
    predicted_prices = target_scaler.inverse_transform(predictions)
    actual_prices = target_scaler.inverse_transform(y_test_reshaped)
    
    # Calculate RMSE in dollars
    rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
    print(f"RMSE (Price): ${rmse:.2f}")
    
    # Save the model (it might have been saved by checkpoint, but ensuring final save)
    # If checkpoint saved best, we leave it. If we want final state:
    # model.save(save_path) 
    # But usually we want the best model which is handled by checkpoint.
    # We print where it was saved.
    print(f"Best model saved to {save_path}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    train_lstm(args.ticker, args.epochs, args.batch_size)
