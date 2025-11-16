from unittest import result
import pytest
import pandas as pd
import numpy as np
from data.preprocessing import TimeSeriesPreprocessor
from features.temporal import TemporalFeatureEngine
from features.lag_features import LagFeatureEngine

class TestProcessing:
    """Test data preprocessing pipeline"""
    
    def test_timezone_normalization(self):
        """Test timezone normalization"""

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'value': np.random.randn(100)
        })

        preprocessor = TimeSeriesPreprocessor(timezone='America/New_York')
        result = preprocessor.normalize_timezone(df)
        assert result['timestamp'].dt.tz is not None
        assert str(result['timestamp'].dt.tz) == "America/New_York"
    
    def test_missing_timestamp_filling(self):
        """Test missing timestamp detection and filling"""
        #Create data with missing timestamps
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(100)
        })

        #Remove some timestamps
        preprocessor = TimeSeriesPreprocessor(freq='H')
        result = preprocessor.detect_and_fill_missing(df, method='forward')

        expected_length = 100
        assert len(result) == expected_length
    
    def test_outlier_handling(self):
        """Test outlier detection and handling"""

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'value': np.random.randn(100)
        })

        #Add outliers
        df.loc[10, 'value'] = 100
        df.loc[20, 'value'] = -100

        preprocessor = TimeSeriesPreprocessor()
        results = preprocessor.handle_outliers(df, method='iqr', threshold=1.5)

        #Outliers should be capped
        assert results['value'].max() < 100
        assert results['value'].min() > -100

class TestFeatureEngineering:
    """Test feature engineering"""

    def test_temporal_features(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'value': np.random.randn(100)
        })

        engine = TemporalFeatureEngine()
        result = engine.transform(df)

        #Check features exist
        assert 'hour' in result.columns
        assert 'dayofweek' in result.columns
        assert 'is_weekend' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns

    def test_lag_features(self):
        """Test lag features"""

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'value': np.random.randn(100)
        })

        engine = LagFeatureEngine()
        result = engine.add_lag_features(df, lags=[1,2,7])

        assert 'value_lag_1' in result.columns
        assert 'value_lag_2' in result.columns
        assert 'value_lag_7' in result.columns
