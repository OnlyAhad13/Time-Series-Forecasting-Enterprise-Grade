import pytest
from config.base_config import ModelConfig
from models.lstm_seq2seq import LSTMSeq2Seq
import torch
from models.tcn import TCN
from models.transformer import TransformerForecaster

class TestModels:
    """Test model implementations"""

    @pytest.fixture
    def sample_config(self):
        config = ModelConfig(
            model_type="lstm",
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            lookback=24,
            horizon=12,
            dropout=0.2
        )
        return config

    def test_lstm_forward(self, sample_config):
        """Test LSTM forward pass"""
    
        model = LSTMSeq2Seq(sample_config)
        x = torch.randn(4, 24, 10)
        output = model(x)

        assert output.shape == (4, 12)

    def test_tcn_forward(self, sample_config):
        """Test TCN forward pass"""
        model = TCN(sample_config)
        x = torch.randn(4, 24, 10)
        
        output = model(x)
        
        assert output.shape == (4, 12)
    
    def test_transformer_forward(self, sample_config):
        """Test Transformer forward pass"""
        sample_config.model_params = {'nhead': 4}
        model = TransformerForecaster(sample_config)
        x = torch.randn(4, 24, 10)
        
        output = model(x)
        
        assert output.shape == (4, 12)
    
    def test_quantile_output(self, sample_config):
        """Test quantile output"""
        sample_config.output_type = 'quantile'
        sample_config.quantiles = [0.1, 0.5, 0.9]
        
        model = LSTMSeq2Seq(sample_config)
        x = torch.randn(4, 24, 10)
        
        output = model(x)
        
        assert output.shape == (4, 12, 3)
    