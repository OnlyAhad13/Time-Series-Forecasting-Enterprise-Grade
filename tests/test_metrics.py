import numpy as np
import pytest
from evaluation.metrics import ForecastMetrics

class TestMetrics:
    """Test evaluation metrics"""
    
    def test_mae(self):
        """Test MAE calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        mae = ForecastMetrics.mae(y_true, y_pred)
        
        assert mae == pytest.approx(0.14, abs=0.01)
    
    def test_rmse(self):
        """Test RMSE calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        rmse = ForecastMetrics.rmse(y_true, y_pred)
        
        assert rmse == pytest.approx(0.0)
    
    def test_quantile_loss(self):
        """Test quantile loss"""
        y_true = np.random.randn(10, 12)
        y_pred = np.random.randn(10, 12, 3)
        quantiles = [0.1, 0.5, 0.9]
        
        losses = ForecastMetrics.quantile_loss(y_true, y_pred, quantiles)
        
        assert 'quantile_0.5' in losses
        assert 'quantile_mean' in losses