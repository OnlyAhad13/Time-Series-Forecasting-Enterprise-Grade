from models.base_model import BaseForecaster
from config.base_config import ModelConfig
import torch
import torch.nn as nn

class LSTMSeq2Seq(BaseForecaster):
    """
    LSTM/GRU Encoder-Decoder for multi-step forecasting
    """

    def __init__(self, config: 'ModelConfig', cell_type: str = "lstm"):
        super().__init__(config)
        
        self.cell_type = cell_type
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.horizon = config.horizon
        self.output_type = config.output_type
        
        RNNCell = nn.LSTM if cell_type == "lstm" else nn.GRU

        #Encoder
        self.encoder = RNNCell(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=config.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder
        self.decoder = RNNCell(
            input_size=1,  # Previous prediction
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=config.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )

        # Output layer
        if self.output_type == "point":
            self.fc_out = nn.Linear(self.hidden_dim, 1)
        elif self.output_type == "quantile":
            num_quantiles = len(config.quantiles)
            self.fc_out = nn.Linear(self.hidden_dim, num_quantiles)
        elif self.output_type == "gaussian":
            self.fc_mu = nn.Linear(self.hidden_dim, 1)
            self.fc_sigma = nn.Linear(self.hidden_dim, 1)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        Args:
            x: (batch_size, lookback, input_dim)
            teacher_forcing_ratio: Probability of using ground truth during training
            
        Returns:
            (batch_size, horizon) for point
            (batch_size, horizon, num_quantiles) for quantile
            (mu, sigma) each (batch_size, horizon) for gaussian
        """

        batch_size = x.size(0)
        device = x.device

        #Encode
        _, hidden = self.encoder(x)
        #Initialize decoder input with last value - shape (batch_size, 1, 1) for batch_first=True
        decoder_input = x[:, -1:, 0:1]  # Keep sequence dimension
        predictions = []

        for t in range(self.horizon):
            #decode one step
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_output = self.dropout(decoder_output)

            #Generate predictions
            if self.output_type == "point":
                pred = self.fc_out(decoder_output)  # (batch_size, 1, 1)
                predictions.append(pred.squeeze(1))  # (batch_size, 1)
                decoder_input = pred  # (batch_size, 1, 1) - keep sequence dim
            elif self.output_type == "quantile":
                pred = self.fc_out(decoder_output)  # (batch_size, 1, num_quantiles)
                predictions.append(pred.squeeze(1))  # (batch_size, num_quantiles)
                #Use median for next input
                median_idx = len(self.config.quantiles) // 2
                decoder_input = pred[:, :, median_idx:median_idx+1]  # (batch_size, 1, 1)
            elif self.output_type == "gaussian":
                mu = self.fc_mu(decoder_output)  # (batch_size, 1, 1)
                sigma = self.fc_sigma(decoder_output)  # (batch_size, 1, 1)
                predictions.append((mu.squeeze(1), sigma.squeeze(1)))  # Both (batch_size, 1)
                decoder_input = mu  # (batch_size, 1, 1) - keep sequence dim
        
        if self.output_type == "gaussian":
            mus = torch.stack([p[0] for p in predictions], dim=1)
            sigmas = torch.stack([p[1] for p in predictions], dim=1)
            return mus, sigmas
        
        else:
            return torch.stack(predictions, dim=1)
        
