from typing import Optional, Tuple
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class TimeSeriesScaler:
    """
    Wrapper for sklearn scalers with time series specific handling
    Stores scaler per series for panel data
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scalers = {}
        self.global_scaler = self._create_scaler()
    
    def _create_scaler(self):
        """Create scaler instance"""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def fit(self, data: pd.DataFrame, series_col: Optional[str] = None):
        """
        Fit scaler on training data

        Args:
            data: DataFrame with values to scale
            series_col: Column name for series ID (panel data)
        """

        if series_col is None:
            #Single series
            self.global_scaler.fit(data)
        else:
            #Panel data - fit per-series scalers and also fit global scaler as fallback
            for series_id in data[series_col].unique():
                series_data = data[data[series_col] == series_id].drop(columns=[series_col])
                scaler = self._create_scaler()
                scaler.fit(series_data)
                self.scalers[series_id] = scaler
            
            # Fit global scaler on all data (without series_col) as fallback for unseen series
            all_data = data.drop(columns=[series_col])
            self.global_scaler.fit(all_data)
            
        return self
    
    def transform(self, data: pd.DataFrame, series_col: Optional[str] = None) -> pd.DataFrame:
        """Transforms data"""
        results = data.copy()

        if series_col is None:
            #Single series
            cols_to_scale = [c for c in data.columns if c not in [series_col]]
            results[cols_to_scale] = self.global_scaler.transform(data[cols_to_scale])
        else:
            #Panel data
            for series_id in data[series_col].unique():
                mask = data[series_col] == series_id
                series_data = data.loc[mask].drop(columns=[series_col])

                if series_id in self.scalers:
                    scaled = self.scalers[series_id].transform(series_data)
                else:
                    #Fallback to global scaler
                    scaled = self.global_scaler.transform(series_data)
                
                # Update results for this series
                results.loc[mask, series_data.columns] = scaled
        
        return results
    
    def inverse_transform(self, data: pd.DataFrame, series_col: Optional[str] = None) -> pd.DataFrame:
        """Inverse transform data"""
        result = data.copy()

        if series_col is None:
            cols_to_scale = [c for c in data.columns if c not in [series_col]]
            result[cols_to_scale] = self.global_scaler.inverse_transform(data[cols_to_scale])
        else:
            for series_id in data[series_col].unique():
                mask = data[series_col] == series_id
                series_data = data.loc[mask].drop(columns=[series_col])

                if series_id in self.scalers:
                    unscaled = self.scalers[series_id].inverse_transform(series_data)
                else:
                    unscaled = self.global_scaler.inverse_transform(series_data)
                
                result.loc[mask, series_data.columns] = unscaled


    

                
