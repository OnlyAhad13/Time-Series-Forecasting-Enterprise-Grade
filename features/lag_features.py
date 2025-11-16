from typing import Optional, List
import pandas as pd

class LagFeatureEngine:
    """Generate lag and rolling window features"""

    def __init__(self, target_col: str = "value", series_id_col: Optional[str] = None):
        self.target_col = target_col
        self.series_id_col = series_id_col
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lagged values"""
        df = df.copy()
        if self.series_id_col is None:
            #Single series
            for lag in lags:
                df[f"{self.target_col}_lag_{lag}"] = df[self.target_col].shift(lag)
        else:
            #Panel data
            for lag in lags:
                df[f"{self.target_col}_lag_{lag}"] = df.groupby(self.series_id_col)[self.target_col].shift(lag)
        
        return df
    
    def add_rolling_features(
        self, df: pd.DataFrame,
        windows: List[int],
        stats: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Add rolling window statistics"""
        df = df.copy()

        for window in windows:
            if self.series_id_col is None:
                #Single series
                rolling = df[self.target_col].rolling(window=window, min_periods=1)
            else:
                #Panel data
                rolling = df.groupby(self.series_id_col)[self.target_col].rolling(window=window, min_periods=1)

            for stat in stats:
                col_name = f"{self.target_col}_rolling_{window}_{stat}"
                if self.series_id_col is None:
                    df[col_name] = getattr(rolling, stat)()
                else:
                    # For panel data, call the function first, then reset_index
                    df[col_name] = getattr(rolling, stat)().reset_index(0, drop=True)
        
        return df

    def add_diff_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """Add differenced features"""
        df = df.copy()
        
        for period in periods:
            if self.series_id_col is None:
                df[f'{self.target_col}_diff_{period}'] = df[self.target_col].diff(period)
            else:
                df[f'{self.target_col}_diff_{period}'] = df.groupby(self.series_id_col)[
                    self.target_col
                ].diff(period)
        
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        lags: List[int],
        windows: List[int],
        stats: List[str] = ['mean', 'std'],
        periods: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """Apply all lag-based features"""
        df = self.add_lag_features(df, lags)
        df = self.add_rolling_features(df, windows, stats)
        df = self.add_diff_features(df, periods)
        return df
