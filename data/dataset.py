import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

class TimeSeriesDataset(Dataset):
    """
    Pytorch dataset for both point and probabilistic forecasting
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        lookback: int,
        horizon: int,
        timestamp_col: str = "timestamp",
        series_id_col: Optional[str] = None,
        stride: int = 1
    ):
        """
        Args:
            data: DataFrame with time series data
            target_col: Target column name
            feature_cols: List of feature column names
            lookback: Number of historical steps
            horizon: Number of steps to forecast
            timestamp_col: Timestamp column name
            series_id_col: Series ID column for panel data
            stride: Stride for creating sequences
        """
        self.data = data.sort_values(timestamp_col).reset_index(drop=True)
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.horizon = horizon
        self.series_id_col = series_id_col
        self.stride = stride
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Tuple[int, int, Optional[str]]]:
        """
        Create valid sequences with no leakage
        
        Returns:
            List of (start_idx, end_idx, series_id) tuples
        """
        sequences = []
        
        if self.series_id_col is None:
            # Single series
            for i in range(0, len(self.data) - self.lookback - self.horizon + 1, self.stride):
                sequences.append((i, i + self.lookback + self.horizon, None))
        else:
            # Panel data - create sequences per series
            for series_id in self.data[self.series_id_col].unique():
                series_mask = self.data[self.series_id_col] == series_id
                series_data = self.data[series_mask].reset_index(drop=True)
                series_len = len(series_data)
                
                for i in range(0, series_len - self.lookback - self.horizon + 1, self.stride):
                    # Store the series_id and relative position within the series
                    sequences.append((i, i + self.lookback + self.horizon, series_id))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence
        
        Returns:
            Dictionary with:
                - x: Input features (lookback, num_features)
                - y: Target values (horizon,)
        """
        start_idx, end_idx, series_id = self.sequences[idx]
        
        # Extract sequence
        if self.series_id_col is None:
            # Single series - use direct indexing
            sequence = self.data.iloc[start_idx:end_idx]
        else:
            # Panel data - filter by series_id first, then use relative indexing
            series_mask = self.data[self.series_id_col] == series_id
            series_data = self.data[series_mask].reset_index(drop=True)
            sequence = series_data.iloc[start_idx:end_idx]
        
        # Input features (historical)
        x = sequence.iloc[:self.lookback][self.feature_cols].values.astype(np.float32)
        
        # Target (future)
        y = sequence.iloc[self.lookback:][self.target_col].values.astype(np.float32)
        
        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y)
        }