from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from config.base_config import ModelConfig

class BaseForecaster(nn.Module, ABC):
    """
    Abstract base class for all forecasting models
    """

    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, lookback, input_dim)
            
        Returns:
            Predictions (batch_size, horizon) or (batch_size, horizon, num_quantiles)
        """
        pass
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
