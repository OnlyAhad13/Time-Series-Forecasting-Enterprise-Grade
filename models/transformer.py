import torch
import torch.nn as nn
import numpy as np
from models.base_model import BaseForecaster
from config.base_config import ModelConfig

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerForecaster(BaseForecaster):
    """
    Transformer-based time series forecaster
    """
    
    def __init__(self, config: 'ModelConfig'):
        super().__init__(config)
        
        self.input_dim = config.input_dim
        self.d_model = config.hidden_dim
        self.nhead = config.model_params.get('nhead', 8)
        self.num_encoder_layers = config.num_layers
        self.num_decoder_layers = config.model_params.get('num_decoder_layers', config.num_layers)
        self.dim_feedforward = config.model_params.get('dim_feedforward', self.d_model * 4)
        self.horizon = config.horizon
        self.output_type = config.output_type
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=config.dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=config.dropout,
            batch_first=False  # (seq, batch, feature)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        # Output layers
        if self.output_type == "point":
            self.fc_out = nn.Linear(self.d_model, 1)
        elif self.output_type == "quantile":
            num_quantiles = len(config.quantiles)
            self.fc_out = nn.Linear(self.d_model, num_quantiles)
        elif self.output_type == "gaussian":
            self.fc_mu = nn.Linear(self.d_model, 1)
            self.fc_sigma = nn.Linear(self.d_model, 1)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, lookback, input_dim)
            
        Returns:
            Predictions
        """
        batch_size, lookback, _ = x.shape
        device = x.device
        
        # Project input
        src = self.input_projection(x)  # (batch_size, lookback, d_model)
        src = src.permute(1, 0, 2)  # (lookback, batch_size, d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Create decoder input (learnable embeddings for forecast horizon)
        tgt = torch.zeros(self.horizon, batch_size, self.d_model, device=device)
        tgt = self.pos_encoder(tgt)
        
        # Generate masks
        tgt_mask = self.generate_square_subsequent_mask(self.horizon).to(device)
        
        # Transformer forward
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
        )  # (horizon, batch_size, d_model)
        
        output = self.norm(output)
        output = output.permute(1, 0, 2)  # (batch_size, horizon, d_model)
        
        # Generate predictions
        if self.output_type == "point":
            pred = self.fc_out(output).squeeze(-1)  # (batch_size, horizon)
            return pred
        elif self.output_type == "quantile":
            pred = self.fc_out(output)  # (batch_size, horizon, num_quantiles)
            return pred
        elif self.output_type == "gaussian":
            mu = self.fc_mu(output).squeeze(-1)  # (batch_size, horizon)
            sigma = self.fc_sigma(output).squeeze(-1)  # (batch_size, horizon)
            return mu, sigma
