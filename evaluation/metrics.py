from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from typing import List, Dict, Optional

class ForecastMetrics:
    """
    Calculate various forecasting metrics
    """

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Mean absolute percentage error"""
        mask = np.abs(y_true) > epsilon
        return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred:np.ndarray, epsilon: float = 1e-10) -> float:
        """Symmetric MAPE"""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred))/2

        mask = denominator > epsilon
        return np.mean(numerator[mask]/denominator[mask])*100

    @staticmethod
    def quantile_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float],
    ) -> Dict[str, float]:
        """
        Quantile loss for probabilistic forecasts
        
        Args:
            y_true: True values (n_samples, horizon)
            y_pred: Predicted quantiles (n_samples, horizon, n_quantiles)
            quantiles: List of quantile levels
            
        Returns:
            Dictionary of quantile losses
        """

        losses = {}

        for i, q in enumerate(quantiles):
            pred_q = y_pred[:, : i]
            errors = y_true - pred_q
            loss = np.maximum(q*errors, (q-1)*errors)
            losses[f"quantile_{q}"] = np.mean(loss)
        
        losses['quantile_mean'] = np.mean(list(losses.values()))

        return losses
    
    @staticmethod
    def coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> float:
        """
        Calculate coverage of prediction intervals
        
        Args:
            y_true: True values
            lower: Lower bound of prediction interval
            upper: Upper bound of prediction interval
            
        Returns:
            Coverage percentage
        """

        within_interval = (y_true >= lower) & (y_true <= upper)
        return np.mean(within_interval)*100

    
    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile_preds: Optional[np.ndarray] = None,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate all metrics"""
        metrics = {
            'mae': ForecastMetrics.mae(y_true, y_pred),
            'rmse': ForecastMetrics.rmse(y_true, y_pred),
            'mape': ForecastMetrics.mape(y_true, y_pred),
            'smape': ForecastMetrics.smape(y_true, y_pred)
        }

        # Add quantile metrics if available
        if quantile_preds is not None and quantiles is not None:
            q_metrics = ForecastMetrics.quantile_loss(y_true, quantile_preds, quantiles)
            metrics.update(q_metrics)
            
            # Coverage for 80% prediction interval (0.1, 0.9 quantiles)
            if 0.1 in quantiles and 0.9 in quantiles:
                idx_lower = quantiles.index(0.1)
                idx_upper = quantiles.index(0.9)
                coverage = ForecastMetrics.coverage(
                    y_true,
                    quantile_preds[:, :, idx_lower],
                    quantile_preds[:, :, idx_upper]
                )
                metrics['coverage_80'] = coverage
        
        return metrics