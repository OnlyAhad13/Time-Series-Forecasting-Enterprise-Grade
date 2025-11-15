import pandas as pd
import numpy as np 
from typing import Optional, List, Tuple, Dict
import holidays
import pytz
from sklearn.preprocessing import LabelEncoder

class TimeSeriesPreprocessor:
    """
    Time series preprocessor
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        target_col: str = "value",
        series_id_col: Optional[str] = None,
        freq: str = "H",
        timezone: str = "UTC"
    ):
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.series_id_col = series_id_col
        self.freq = freq
        self.timezone = pytz.timezone(timezone)

        self.label_encoders = {}
    
    def normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps to specified timezones"""

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        #Convert to timezone
        if df[self.timestamp_col].dt.tz is None:
            df[self.timestamp_col] = df[self.timestamp_col].dt.tz_localize(self.timezone)
        else:
            df[self.timestamp_col] = df[self.timestamp_col].dt.tz_convert(self.timezone)
        
        return df
    
    def detect_and_fill_missing(self, df: pd.DataFrame, method: str = "forward") -> pd.DataFrame:
        """
        Detect missing timestamps and fill gaps

        Args:
            df: Input DataFrame
            method: Fill method - 'forward', 'interpolate', 'zero'
        """
        df = df.copy()
        
        if self.series_id_col is not None:
            #Single series
            df = self._fill_single_series(df, method)
        else:
            #Panel data - process each series
            processed_series = []
            for series_id in df[self.series_id_col].unique():
                series_df = df[df[self.series_id_col] == series_id].copy()
                series_df = self._fill_single_series(series_df, method)
                processed_series.append(series_df)

            df = pd.concat(processed_series, ignore_index=True)
        
        return df
    
    def _fill_single_series(self, df: pd.DataFrame, method: str = "forward") -> pd.DataFrame:
        """Fill missing timestamps for single series"""

        df = df.sort_values(self.timestamp_col)
       
        #Create complete time range
        full_range = pd.date_range(
            start=df[self.timestamp_col].min(),
            end=df[self.timestamp_col].max(),
            freq=self.freq
        )

        #Reindex to include all timestamps
        df = df.set_index(self.timestamp_col)
        df = df.reindex(full_range)
        df.index.name = self.timestamp_col
        df = df.reset_index()

        #Fill missing values
        if method == "forward":
            df = df.fillna(method="ffill")
        elif method == "interpolate":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        elif method == "zero":
            df = df.fillna(0)
        else:
            raise ValueError(f"Invalid fill method: {method}")
        
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: Input DataFrame
            method: Detection method - 'iqr', 'zscore', 'none'
            threshold: Threshold for outlier detection
            columns: Columns to check (default: target_col only)
        """
        if method == "none":
            return df
        
        df = df.copy()
        columns = columns or [self.target_col]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower, upper=upper)
                
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
                
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Encode categorical variables"""
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Transform with existing encoder
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def add_holiday_features(
        self,
        df: pd.DataFrame,
        country: str = "US"
    ) -> pd.DataFrame:
        """Add holiday indicators"""
        df = df.copy()

        country_holidays = holidays.country_holidays(country)
        df['is_holiday'] = df[self.timestamp_col].dt.date.apply(
            lambda x: 1 if x in country_holidays else 0
        )

        return df
    
    def chronological_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically with no leakage
        
        Args:
            df: Input DataFrame (must be sorted by timestamp)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            train_df, val_df, test_df
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        df = df.sort_values(self.timestamp_col)
        n = len(df)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df

print("Data Preprocessing Module created")

