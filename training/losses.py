import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np

class QuantileLoss(nn.Module):
    """
    Quantile loss(Pinball loss) for probabilistic forecasting
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, horizon, num_quantiles)
            targets: (batch_size, horizon)
        Returns:
            Quantile loss
        """

        #Expand targets to match predicitons
        targets = targets.unsqueeze(-1)

        #Move quantiles to same device as predictions
        quantiles = self.quantiles.to(predictions.device).view(1,1,-1)

        #Calculate errors
        errors = targets - predictions
        
        #Quantile loss
        loss = torch.max(
            quantiles*errors,
            (quantiles-1)*(errors)
        )

        return loss.mean()
    
class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood for probabilistic forecasting
    """
    
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean (batch_size, horizon)
            sigma: Predicted std (batch_size, horizon)
            targets: True values (batch_size, horizon)
            
        Returns:
            Negative log-likelihood
        """
        # Ensure sigma is positive
        sigma = F.softplus(sigma) + 1e-6
        
        # NLL loss
        loss = 0.5 * torch.log(2 * np.pi * sigma ** 2) + \
               0.5 * ((targets - mu) ** 2) / (sigma ** 2)
        
        return loss.mean()

class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error
    """
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, horizon)
            targets: (batch_size, horizon)
        """
        numerator = torch.abs(predictions - targets)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2
        
        # Avoid division by zero
        smape = torch.where(
            denominator == 0,
            torch.zeros_like(numerator),
            numerator / (denominator + 1e-8)
        )
        
        return smape.mean()
    
def get_loss_fn(output_type: str, quantiles: Optional[List[float]] = None) -> nn.Module:
    """
    Get appropriate loss function based on output type
    
    Args:
        output_type: 'point', 'quantile', or 'gaussian'
        quantiles: List of quantiles for quantile loss
        
    Returns:
        Loss function
    """
    if output_type == "point":
        return SMAPELoss()
    elif output_type == "quantile":
        assert quantiles is not None, "Quantiles must be provided for quantile loss"
        return QuantileLoss(quantiles)
    elif output_type == "gaussian":
        return GaussianNLLLoss()
    else:
        raise ValueError(f"Unknown output type: {output_type}")