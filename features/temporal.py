import pandas as pd
import numpy as np
from typing import List

class TemporalFeatureEngine:
    """Generate time-based features"""

    def __init__(self, timestamp_col: str = "timestamp", cyclical: bool = True):
        self.timestamp_col = timestamp_col
        self.cyclical = cyclical
    
    def add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic datetime features"""
        df = df.copy()
        dt = df[self.timestamp_col].dt
        
        # Basic features
        df['year'] = dt.year
        df['month'] = dt.month
        df['day'] = dt.day
        df['dayofweek'] = dt.dayofweek
        df['dayofyear'] = dt.dayofyear
        df['week'] = dt.isocalendar().week
        df['hour'] = dt.hour
        df['minute'] = dt.minute
        
        # Binary features
        df['is_weekend'] = (dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = dt.is_month_start.astype(int)
        df['is_month_end'] = dt.is_month_end.astype(int)
        df['is_quarter_start'] = dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = dt.is_quarter_end.astype(int)
        
        return df
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical (sin/cos) encodings for temporal features"""
        if not self.cyclical:
            return df
        
        df = df.copy()
        dt = df[self.timestamp_col].dt
        
        # Hour of day (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        
        # Day of week (0-6)
        df['dayofweek_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        
        # Day of month (1-31)
        df['day_sin'] = np.sin(2 * np.pi * dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * dt.day / 31)
        
        # Month of year (1-12)
        df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        
        # Week of year (1-52)
        df['week_sin'] = np.sin(2 * np.pi * dt.isocalendar().week / 52)
        df['week_cos'] = np.cos(2 * np.pi * dt.isocalendar().week / 52)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all temporal features"""
        df = self.add_datetime_features(df)
        df = self.add_cyclical_features(df)
        return df
    

    
    
    