"""
Compare multiple models
Usage: python compare_models.py --data data.csv --models lstm gru tcn transformer
"""

"""
Compare multiple models
Usage: python compare_models.py --data data.csv --models lstm gru tcn transformer
"""

import argparse
import os
from config.base_config import Config
from utils.logger import setup_logger
import pandas as pd
import torch
from data.preprocessing import TimeSeriesPreprocessor
from features.temporal import TemporalFeatureEngine
from features.lag_features import LagFeatureEngine
from data.dataloader import create_dataloaders
from models.create_model import create_model
from training.trainer import Trainer
from evaluation.metrics import calculate_metrics
from evaluation.visualization import ForecastVisualizer
from evaluation.backtesting import WalkForwardValidator
from utils.scalers import TimeSeriesScaler


def compare_models():
    parser = argparse.ArgumentParser(description='Compare multiple forecasting models')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru', 'tcn', 'transformer'],
                       help='Models to compare')
    parser.add_argument('--config', type=str, default='config.yaml', help='Base config file')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output', type=str, default='outputs/comparison', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('comparison', log_file=os.path.join(args.output, 'comparison.log'))
    logger.info("Starting model comparison")
    logger.info(f"Models: {args.models}")
    
    # Load base config
    if os.path.exists(args.config):
        base_config = Config.from_yaml(args.config)
    else:
        base_config = Config()
    
    base_config.training.epochs = args.epochs
    base_config.training.use_wandb = False  # Disable for comparison
    
    # Load and preprocess data once
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv(args.data)
    
    preprocessor = TimeSeriesPreprocessor(
        timestamp_col=base_config.data.timestamp_col,
        target_col=base_config.data.target_col,
        series_id_col=base_config.data.series_id_col,
        freq=base_config.data.freq
    )
    
    df = preprocessor.normalize_timezone(df)
    df = preprocessor.detect_and_fill_missing(df, method=base_config.data.fill_method)
    df = preprocessor.handle_outliers(df, method=base_config.data.outlier_method)
    
    # Feature engineering
    temporal_engine = TemporalFeatureEngine(
        timestamp_col=base_config.data.timestamp_col,
        cyclical=base_config.data.cyclical_encoding
    )
    df = temporal_engine.transform(df)
    
    if base_config.data.use_temporal_features:
        lag_engine = LagFeatureEngine(
            target_col=base_config.data.target_col,
            series_id_col=base_config.data.series_id_col
        )
        df = lag_engine.transform(
            df,
            lags=base_config.data.lags,
            windows=base_config.data.rolling_windows,
            stats=base_config.data.rolling_stats
        )
    
    df = preprocessor.add_holiday_features(df)
    df = df.dropna()
    
    # Split
    train_df, val_df, test_df = preprocessor.chronological_split(df)
    
    # Scale
    scaler = TimeSeriesScaler(scaler_type=base_config.data.scaler_type)
    cols_to_scale = [base_config.data.target_col] + base_config.data.numeric_covariates
    
    if cols_to_scale:
        scaler.fit(train_df[cols_to_scale])
        train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
        val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
        test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    
    # Results storage
    results = {}
    training_times = {}
    
    # Train each model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Create config for this model
        config = base_config
        config.model.model_type = model_name
        config.experiment_name = f'{model_name}_comparison'
        
        # Update input dim
        feature_cols = [config.data.target_col] + config.data.numeric_covariates
        config.model.input_dim = len(feature_cols)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, config
        )
        
        # Create model
        model = create_model(config.model)
        logger.info(f"Parameters: {model.get_num_parameters():,}")
        
        # Train
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        import time
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        training_times[model_name] = training_time
        logger.info(f"Training time: {training_time:.2f}s")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        validator = WalkForwardValidator(
            model=model,
            lookback=config.model.lookback,
            horizon=config.model.horizon
        )
        
        predictions, actuals = validator.validate(
            test_df,
            target_col=config.data.target_col,
            feature_cols=feature_cols,
            device=device
        )
        
        metrics = validator.evaluate(predictions, actuals)
        results[model_name] = metrics
        
        logger.info(f"Test MAE: {metrics['mae']:.4f}")
        logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
        
        # Save model
        checkpoint_path = os.path.join(args.output, f'{model_name}_best.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'metrics': metrics
        }, checkpoint_path)

    # Create comparison visualizations
    logger.info("\nCreating comparison visualizations.....")
    
    visualizer = ForecastVisualizer()
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(
        results,
        title="Model Performance Comparison",
        save_path=os.path.join(args.output, 'metrics_comparison.png')
    )
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df['training_time'] = pd.Series(training_times)
    results_df.to_csv(os.path.join(args.output, 'comparison_results.csv'))
    
    logger.info("\n" + "="*60)
    logger.info("FINAL COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(results_df.to_string())
    logger.info("="*60)
    
    # Find best model
    best_model = results_df['mae'].idxmin()
    logger.info(f"\nBest model (lowest MAE): {best_model.upper()}")
    logger.info(f"MAE: {results_df.loc[best_model, 'mae']:.4f}")
    logger.info(f"RMSE: {results_df.loc[best_model, 'rmse']:.4f}")
    
    logger.info(f"\nAll results saved to {args.output}")
    logger.info("Comparison complete!")


if __name__ == '__main__':
    compare_models()