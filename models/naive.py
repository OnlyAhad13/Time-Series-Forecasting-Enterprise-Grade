from models.base_model import BaseForecaster
from config.base_config import ModelConfig
import torch

class SeasonalNaive(BaseForecaster):
    """
    Seasonal Naive baseline
    Uses value from seasonal_period steps ago as forecast
    """

    def __init__(self, config: 'ModelConfig', seasonal_period: int = 24):
        super().__init__(config)
        self.seasonal_period = seasonal_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, lookback, input_dim)
            
        Returns:
            (batch_size, horizon)
        """

        batch_size, lookback, _ = x.shape
        horizon = self.config.horizon

        #Use the target column
        target = x[:, :, 0]

        #Repeat seasonal pattern
        predictions = []
        for h in range(horizon):
            #Look back seasonal period steps
            idx = lookback - self.seasonal_period + h%self.seasonal_period
            idx = max(0, min(idx, lookback-1))
            predictions.append(target[:, idx])
        
        return torch.stack(predictions, dim=1)
