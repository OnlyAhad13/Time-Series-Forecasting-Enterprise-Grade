from models.base_model import BaseForecaster
import torch
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from evaluation.metrics import ForecastMetrics


class WalkForwardValidator:
    """
    Walk-forward validation for time series
    """

    def __init__(
        self, 
        model: BaseForecaster,
        lookback: int,
        horizon: int,
        step_size: int = 1
    ):
        self.model = model
        self.lookback = lookback
        self.horizon = horizon
        self.step_size = step_size

    @torch.no_grad()
    def validate(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        device: str = 'cuda'
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform walk-forward validation
        
        Args:
            data: Time series data
            target_col: Target column name
            feature_cols: Feature column names
            device: Device to run on
            
        Returns:
            predictions, actuals
        """
        self.model.eval()
        self.model.to(device)
        
        predictions = []
        actuals = []
        
        n = len(data)
        start_idx = self.lookback
        
        while start_idx + self.horizon <= n:
            # Extract window
            window = data.iloc[start_idx - self.lookback:start_idx]
            future = data.iloc[start_idx:start_idx + self.horizon]
            
            # Prepare input
            x = torch.from_numpy(
                window[feature_cols].values.astype(np.float32)
            ).unsqueeze(0).to(device)
            
            # Predict
            pred = self.model(x)
            
            # Handle different output types
            if isinstance(pred, tuple):  # Gaussian
                pred = pred[0]  # Use mean
            
            pred = pred.cpu().numpy()
            
            # Store results
            if pred.ndim == 3:  # Quantile predictions
                pred = pred[:, :, len(pred[0, 0]) // 2]  # Use median
            
            predictions.append(pred[0])
            actuals.append(future[target_col].values)
            
            # Move forward
            start_idx += self.step_size
        
        return predictions, actuals
    
    def evaluate(
        self,
        predictions: List[np.ndarray],
        actuals: List[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate metrics on walk-forward results"""
        # Concatenate all predictions and actuals
        all_preds = np.concatenate(predictions)
        all_actuals = np.concatenate(actuals)
        
        return ForecastMetrics.calculate_all(all_actuals, all_preds)

