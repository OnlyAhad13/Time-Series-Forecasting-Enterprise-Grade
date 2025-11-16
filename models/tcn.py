import torch
import torch.nn as nn
from models.base_model import BaseForecaster
from config.base_config import ModelConfig

class Chomp1d(nn.Module):
    """Remove padding from TCN output"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with dilated causal convolutions
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+res)
    
class TCN(BaseForecaster):
    """
    Temporal Convolutional Network for time series forecasting
    """

    def __init__(self, config: 'ModelConfig'):
        super().__init__(config)

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.horizon = config.horizon
        self.output_type = config.output_type

        #TCN parameters
        kernel_size = config.model_params.get("kernel_size", 2)
        num_channels = [self.hidden_dim] * self.num_layers

        #Build TCN layers
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2**i
            in_channels = self.input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation,
                padding=(kernel_size-1)*dilation,
                dropout=config.dropout
            ))

        self.tcn = nn.Sequential(*layers)

        #Output layer
        if self.output_type == "point":
            self.fc_out = nn.Linear(self.hidden_dim, self.horizon)
        elif self.output_type == "quantile":
            num_quantiles = len(config.quantiles)
            self.fc_out = nn.Linear(self.hidden_dim, self.horizon * num_quantiles)
        elif self.output_type == "gaussian":
            self.fc_mu = nn.Linear(self.hidden_dim, self.horizon)
            self.fc_sigma = nn.Linear(self.hidden_dim, self.horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x: (batch_size, lookback, input_dim)
        Returns:
            predictions: (batch_size, horizon) for point
            predictions: (batch_size, horizon, num_quantiles) for quantile
            predictions: (mu, sigma) each (batch_size, horizon) for gaussian
        """

        #TCN expects (batch_size, channels, sequence_length)
        x = x.transpose(1, 2) #(batch_size, input_dim, lookback)

        #Applying TCN
        out = self.tcn(x)

        #Use last time step
        out = out[:, :, -1]

        if self.output_type == "point":
            return self.fc_out(out)  # (batch_size, horizon)
        elif self.output_type == "quantile":
            out = self.fc_out(out)  # (batch_size, horizon * num_quantiles)
            batch_size = out.size(0)
            num_quantiles = len(self.config.quantiles)
            return out.view(batch_size, self.horizon, num_quantiles)
        elif self.output_type == "gaussian":
            mu = self.fc_mu(out)
            sigma = self.fc_sigma(out)
            return mu, sigma

        


