import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import TimeSeriesDataset
from config.base_config import Config

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: 'Config',
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        config: Configuration object
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Determine feature columns
    feature_cols = config.data.numeric_covariates
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_df,
        target_col=config.data.target_col,
        feature_cols=feature_cols,
        lookback=config.model.lookback,
        horizon=config.model.horizon,
        timestamp_col=config.data.timestamp_col,
        series_id_col=config.data.series_id_col,
        stride=1
    )
    
    val_dataset = TimeSeriesDataset(
        data=val_df,
        target_col=config.data.target_col,
        feature_cols=feature_cols,
        lookback=config.model.lookback,
        horizon=config.model.horizon,
        timestamp_col=config.data.timestamp_col,
        series_id_col=config.data.series_id_col,
        stride=config.model.horizon  # Non-overlapping for validation
    )
    
    test_dataset = TimeSeriesDataset(
        data=test_df,
        target_col=config.data.target_col,
        feature_cols=feature_cols,
        lookback=config.model.lookback,
        horizon=config.model.horizon,
        timestamp_col=config.data.timestamp_col,
        series_id_col=config.data.series_id_col,
        stride=config.model.horizon  # Non-overlapping for testing
    )
    
    # Check dataset sizes and warn if empty
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty. Need at least {config.model.lookback + config.model.horizon} "
            f"data points per series, but training data may be too short."
        )
    if len(val_dataset) == 0:
        import warnings
        warnings.warn(
            f"Validation dataset is empty. Need at least {config.model.lookback + config.model.horizon} "
            f"data points per series. Consider reducing validation stride or increasing validation data size.",
            UserWarning
        )
    if len(test_dataset) == 0:
        import warnings
        warnings.warn(
            f"Test dataset is empty. Need at least {config.model.lookback + config.model.horizon} "
            f"data points per series.",
            UserWarning
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader