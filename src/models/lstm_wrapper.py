import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from .base_model import StockPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PyTorchLSTM(nn.Module):
    """Internal PyTorch module for the LSTM."""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :] # Last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class LSTMStockPredictor(StockPredictor):
    """
    LSTM implementation wrapping the PyTorch model.
    """
    
    def __init__(self, input_size=6, hidden_size=50, num_layers=2, dropout=0.2):
        self.model = PyTorchLSTM(input_size, hidden_size, num_layers, dropout).to(DEVICE)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.history = {}

    def get_name(self) -> str:
        return "LSTM (PyTorch)"

    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, **kwargs):
        # Update input size based on data
        if X_train.shape[2] != self.input_size:
            print(f"Adjusting model input size from {self.input_size} to {X_train.shape[2]}")
            self.input_size = X_train.shape[2]
            self.model = PyTorchLSTM(self.input_size, self.hidden_size, self.num_layers, self.dropout).to(DEVICE)
            
        # Convert to Tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
            
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        return self.history

    def predict(self, X_data):
        self.model.eval()
        X_tensor = torch.FloatTensor(X_data).to(DEVICE)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.model.eval()

    def forecast(self, last_sequence, scaler, days_ahead=30, **kwargs):
        """
        Predict future prices for 'days_ahead'.
        Now rewritten to be 100% PURE LSTM with Dynamic Feature Recalculation.
        """
        self.model.eval()
        predictions = []
        
        # Current Sequence used for Model Input (Scaled)
        current_sequence_scaled = last_sequence.copy() # (1, 60, 6)
        
        # We need a Rolling History of Real Prices to calculate RSI/SMA/MACD
        # We extract "Recent Prices" from kwargs (passed from app.py)
        # We need at least 52 days for SMA_50 and MACD (26+9)
        history_prices = list(kwargs.get('recent_prices', [])) 
        
        # If history is too short, we can't properly calc indicators.
        # Fallback: Just assume 50 days of the last known price (not ideal but safe)
        if len(history_prices) < 55:
             history_prices = [history_prices[-1]] * 55 if history_prices else [0]*55
             
        for _ in range(days_ahead):
            # A. Predict Next Step (Scaled)
            seq_tensor = torch.FloatTensor(current_sequence_scaled).to(DEVICE)
            with torch.no_grad():
                # Model outputs Scaled Price
                pred_scaled_price = self.model(seq_tensor).cpu().numpy()[0, 0]
            
            predictions.append(pred_scaled_price)
            
            # B. Inverse Transform Prediction to Real Price
            # We need the TARGET scaler for this, but 'scaler' passed here is likely the FEATURE scaler
            # This is a bit messy API-wise.
            # However, looking at stock_data.py, the columns are: 'Close', 'SMA', ...
            # 'Close' is index 0. Reconstructing placeholder.
            
            # Create a placeholder row with the predicted scaled price at index 0
            placeholder = np.zeros((1, 6)) 
            placeholder[0, 0] = pred_scaled_price
            
            # We assume 'scaler' is the Feature Scaler (MinMax)
            # Use it to inverse transform the whole row and take index 0.
            real_row = scaler.inverse_transform(placeholder)[0]
            pred_real_price = real_row[0]
            
            # C. Update History
            history_prices.append(pred_real_price)
            
            # D. Recalculate Indicators for the NEW DAY
            # We need a mini-dataframe
            calc_df = pd.DataFrame({'Close': history_prices})
            
            # Recalculate usage pandas (fast enough for 30 loops)
            # SMA
            calc_df['SMA_20'] = calc_df['Close'].rolling(window=20).mean()
            calc_df['SMA_50'] = calc_df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = calc_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            calc_df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = calc_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = calc_df['Close'].ewm(span=26, adjust=False).mean()
            calc_df['MACD'] = exp1 - exp2
            calc_df['Signal_Line'] = calc_df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Fill NaNs
            calc_df.fillna(method='bfill', inplace=True)
            calc_df.fillna(method='ffill', inplace=True)
            
            # Extract the LATEST row (all features)
            # features in order: ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
            latest_features = calc_df.iloc[-1][['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].values
            
            # E. Scale the New Features
            features_reshaped = latest_features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)
            
            # F. Update Sequence
            # Roll left
            current_sequence_scaled = np.roll(current_sequence_scaled, -1, axis=1)
            # Insert new scaled features at the end
            current_sequence_scaled[0, -1, :] = features_scaled
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # We need to return REAL values, but we collected SCALED values.
        # We can construct a dummy array for inverse transform again
        final_placeholder = np.zeros((len(predictions), 6))
        final_placeholder[:, 0] = predictions.flatten()
        
        # Use simple inverse transform if possible, but matching dimensions is safer
        unscaled_matrix = scaler.inverse_transform(final_placeholder)
        return unscaled_matrix[:, 0] # Return just the price column
