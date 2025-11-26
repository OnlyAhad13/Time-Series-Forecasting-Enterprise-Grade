import pytest
import torch
from config.base_config import ModelConfig
from models.lstm_seq2seq import LSTMSeq2Seq

class TestLSTMTeacherForcing:
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            model_type="lstm",
            input_dim=10,
            hidden_dim=64,
            num_layers=1,
            lookback=24,
            horizon=12,
            dropout=0.0
        )
    
    def test_teacher_forcing_shape(self, config):
        """Test that forward pass with teacher forcing returns correct shape"""
        model = LSTMSeq2Seq(config)
        x = torch.randn(4, 24, 10)
        y = torch.randn(4, 12)
        
        # With teacher forcing
        output = model(x, y=y, teacher_forcing_ratio=1.0)
        assert output.shape == (4, 12)
        
        # Without teacher forcing
        output = model(x, y=y, teacher_forcing_ratio=0.0)
        assert output.shape == (4, 12)
        
    def test_teacher_forcing_effect(self, config):
        """
        Test that teacher forcing actually changes the output.
        We can't easily check internal state, but we can check that 
        passing y with ratio=1.0 vs ratio=0.0 produces different results 
        (because decoder input changes).
        """
        # Set seeds for reproducibility
        torch.manual_seed(42)
        model = LSTMSeq2Seq(config)
        
        x = torch.randn(4, 24, 10)
        y = torch.randn(4, 12) # Random targets
        
        # Run with ratio=0.0 (autoregressive)
        torch.manual_seed(42)
        out_no_tf = model(x, y=y, teacher_forcing_ratio=0.0)
        
        # Run with ratio=1.0 (teacher forcing)
        torch.manual_seed(42)
        out_tf = model(x, y=y, teacher_forcing_ratio=1.0)
        
        # Results should be different because decoder inputs are different
        # (unless the model learns to ignore inputs, which is unlikely initialized)
        assert not torch.allclose(out_no_tf, out_tf), "Teacher forcing should produce different results"

    def test_teacher_forcing_ratio_zero_ignores_y(self, config):
        """Test that ratio=0.0 ignores y (same result as y=None)"""
        torch.manual_seed(42)
        model = LSTMSeq2Seq(config)
        x = torch.randn(4, 24, 10)
        y = torch.randn(4, 12)
        
        torch.manual_seed(42)
        out_y_none = model(x, y=None, teacher_forcing_ratio=0.0)
        
        torch.manual_seed(42)
        out_y_provided = model(x, y=y, teacher_forcing_ratio=0.0)
        
        assert torch.allclose(out_y_none, out_y_provided)
