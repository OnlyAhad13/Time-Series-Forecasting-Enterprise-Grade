import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, Optional, Dict, List
import pandas as pd

class ForecastVisualizer:
    """
    Visualization utilities for forecasts
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 6)):
        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_forecast(
        self,
        timestamps: np.ndarray,
        actuals: np.ndarray,
        predictions: np.ndarray,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        title: str = "Forecast vs Actual",
        save_path: Optional[str] = None
    ):
        """
        Plot forecast vs actual values with optional confidence intervals
        
        Args:
            timestamps: Time points
            actuals: Actual values
            predictions: Predicted values
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            title: Plot title
            save_path: Path to save figure
        """

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot actuals and predictions
        ax.plot(timestamps, actuals, label='Actual', color='black', linewidth=2)
        ax.plot(timestamps, predictions, label='Predicted', color='red', linewidth=2, linestyle='--')

        # Plot confidence interval
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(
                timestamps,
                lower_bound,
                upper_bound,
                alpha=0.3,
                color='red',
                label='80% Prediction Interval'
            )
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_residuals(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        title: str = "Residual Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot residual analysis
        
        Args:
            actuals: Actual values
            predictions: Predicted values
            title: Plot title
            save_path: Path to save figure
        """
        residuals = actuals - predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, color='blue', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Residual')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals vs predictions
        axes[1, 1].scatter(predictions, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals vs Predictions')
        axes[1, 1].set_xlabel('Predictions')
        axes[1, 1].set_ylabel('Residuals')
        
        plt.suptitle(title, fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of metrics across models
        
        Args:
            metrics_dict: Dict of {model_name: {metric_name: value}}
            title: Plot title
            save_path: Path to save figure
        """
        # Prepare data
        df_metrics = pd.DataFrame(metrics_dict).T
        
        # Select main metrics
        main_metrics = ['mae', 'rmse', 'mape', 'smape']
        df_plot = df_metrics[[m for m in main_metrics if m in df_metrics.columns]]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.legend(title='Metrics', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training History",
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation losses
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

