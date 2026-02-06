import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, accuracy_score
from src.models.base_model import StockPredictor
from src.models.evaluation import calculate_metrics, calculate_strategy_returns
import logging

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Framework for walk-forward validation of time-series models.
    Supports rolling windows for training, validation, and testing.
    """
    
    def __init__(self, 
                 train_years: int = 3, 
                 val_months: int = 3, 
                 test_months: int = 3,
                 step_months: int = 3):
        """
        Initialize the validator with window sizes.
        
        Args:
            train_years: Number of years for training window.
            val_months: Number of months for validation window.
            test_months: Number of months for testing window.
            step_months: Number of months to advance in each walk-forward step.
        """
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months

    def _get_splits(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """
        Generate chronological splits based on the window sizes.
        """
        splits = []
        if 'date' in data.columns:
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
        else:
            raise ValueError("Data must contain a 'date' column.")

        min_date = data['date'].min()
        max_date = data['date'].max()
        
        current_train_start = min_date
        
        while True:
            train_end = current_train_start + relativedelta(years=self.train_years)
            val_end = train_end + relativedelta(months=self.val_months)
            test_end = val_end + relativedelta(months=self.test_months)
            
            if test_end > max_date:
                break
                
            train_mask = (data['date'] >= current_train_start) & (data['date'] < train_end)
            val_mask = (data['date'] >= train_end) & (data['date'] < val_end)
            test_mask = (data['date'] >= val_end) & (data['date'] < test_end)
            
            splits.append({
                'train': data[train_mask],
                'val': data[val_mask],
                'test': data[test_mask],
                'metadata': {
                    'train_start': current_train_start,
                    'train_end': train_end,
                    'val_start': train_end,
                    'val_end': val_end,
                    'test_start': val_end,
                    'test_end': test_end
                }
            })
            
            # Advance start date
            current_train_start = current_train_start + relativedelta(months=self.step_months)
            
        return splits

    def validate(self, 
                 model: StockPredictor, 
                 data: pd.DataFrame, 
                 feature_cols: List[str], 
                 target_col: str,
                 price_col: str = 'close') -> List[Dict[str, Any]]:
        """
        Run walk-forward validation.
        
        Args:
            model: An instance of StockPredictor.
            data: DataFrame containing features and target.
            feature_cols: List of feature column names.
            target_col: Name of the target column.
            price_col: Name of the price column for return calculation.
            
        Returns:
            List of dictionaries containing metrics for each step.
        """
        splits = self._get_splits(data)
        if not splits:
            logger.warning("No splits generated. Check data range and window sizes.")
            return []
            
        results = []
        
        for i, split in enumerate(splits):
            logger.info(f"Processing Step {i+1}/{len(splits)}: {split['metadata']['test_start'].date()} to {split['metadata']['test_end'].date()}")
            
            train_df = split['train']
            val_df = split['val']
            test_df = split['test']
            
            X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
            X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
            X_test, y_test = test_df[feature_cols].values, test_df[target_col].values
            
            # Train model (using val set for early stopping if supported)
            model.train(X_train, y_train, X_val, y_val)
            
            # Predictions
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)
            
            # Accuracy Metrics
            # Directional accuracy (assuming classification)
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            # RMSE (treating predictions as values if applicable, or using probabilities)
            # For pure classification, RMSE on 0/1 might not be great but requested.
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            
            # Overfitting detection: ratio of val accuracy to train accuracy
            overfitting_ratio = val_acc / train_acc if train_acc > 0 else 0
            
            # Financial Metrics (using test set)
            # Calculate strategy returns
            # We need the original prices to calculate returns
            test_prices = test_df[price_col]
            # Convert predictions to signals (assuming 1 for UP, 0 for DOWN -> signals 1, -1)
            # This depends on how the model labels are defined. 
            # If 1 is UP and 0 is DOWN, signals should be 1 and -1.
            test_signals = pd.Series(test_preds, index=test_df.index).map({1: 1, 0: -1})
            
            strategy_returns = calculate_strategy_returns(test_signals, test_prices)
            financial_metrics = calculate_metrics(strategy_returns)
            
            step_result = {
                'step': i + 1,
                'metadata': split['metadata'],
                'metrics': {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'test_rmse': test_rmse,
                    'overfitting_ratio': overfitting_ratio,
                    **financial_metrics
                }
            }
            
            results.append(step_result)
            
        return results
