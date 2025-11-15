"""
Main training script
Usage: python train.py --config config.yaml --model lstm
"""

import argparse
import os
from config.base_config import Config
from utils.reproducibility import set_seed
from utils.logger import init_wandb, setup_logger
import torch
import wandb
import pandas as pd
from data.preprocessing import TimeSeriesPreprocessor
from features.temporal import TemporalFeatureEngine
from features.lag_features import LagFeatureEngine
from utils.scalers import TimeSeriesScaler
from training.trainer import Trainer
from evaluation.visualization import ForecastVisualizer
from data.dataloader import create_dataloaders
from models.create_model import create_model


def main():
    parser = argparse.ArgumentParser(description='Train time series forecasting model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='lstm', choices=['naive', 'lstm', 'gru', 'tcn', 'transformer'])
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--experiment', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')

    args = parser.parse_args()

    # Load or create config
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command line args
    config.model.model_type = args.model
    config.data.data_path = args.data
    config.experiment_name = args.experiment
    config.training.seed = args.seed
    config.training.use_wandb = not args.no_wandb
    
    # Set seed for reproducibility
    set_seed(config.training.seed, config.training.deterministic)   

    #Setup logging
    logger = setup_logger(
        'main',
        log_file=os.path.join(config.log_dir, f'{config.experiment_name}.log')
    )

    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Model: {config.model.model_type}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Initialize W&B
    if config.training.use_wandb:
        init_wandb(
            config.__dict__,
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            name=config.experiment_name
        )
    
    #Load and preprocess data
    logger.info("Loading data.....")
    df = pd.read_csv(config.data.data_path)

    preprocessor = TimeSeriesPreprocessor(
        timestamp_col=config.data.timestamp_col,
        target_col=config.data.target_col,
        series_id_col=config.data.series_id_col,
        freq=config.data.freq
    )

    df = preprocessor.normalize_timezone(df)
    df = preprocessor.detect_and_fill_missing(df, method=config.data.fill_method)
    df = preprocessor.handle_outliers(
        df,
        method=config.data.outlier_method,
        threshold=config.data.outlier_threshold,
        columns=[config.data.target_col] + config.data.numeric_covariates
    )

    # Add features
    logger.info("Engineering features...")
    temporal_engine = TemporalFeatureEngine(
        timestamp_col=config.data.timestamp_col,
        cyclical=config.data.cyclical_encoding
    )
    df = temporal_engine.transform(df)
    
    if config.data.use_temporal_features:
        lag_engine = LagFeatureEngine(
            target_col=config.data.target_col,
            series_id_col=config.data.series_id_col
        )
        df = lag_engine.transform(
            df,
            lags=config.data.lags,
            windows=config.data.rolling_windows,
            stats=config.data.rolling_stats
        )
    
    #Add holiday features
    df = preprocessor.add_holiday_features(df)

    #Drop NaN from lag features
    df = df.dropna()

    #Split data
    logger.info("Splitting data.....")
    train_df, val_df, test_df = preprocessor.chronological_split(
        df,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )

    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    #Scale data
    logger.info("Scaling data.....")
    scaler = TimeSeriesScaler(scaler_type=config.data.scaler_type)

    # Fit on training data only
    cols_to_scale = [config.data.target_col] + config.data.numeric_covariates
    if cols_to_scale:
        scaler.fit(train_df[cols_to_scale], series_col=config.data.series_id_col)
        train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale], series_col=config.data.series_id_col)
        val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale], series_col=config.data.series_id_col)
        test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale], series_col=config.data.series_id_col)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )
    
    # Update config with input dimension
    feature_cols = [config.data.target_col] + config.data.numeric_covariates
    config.model.input_dim = len(feature_cols)
    
    # Create model
    logger.info(f"Creating {config.model.model_type} model...")
    model = create_model(config.model)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, final_path)
    
    logger.info(f"Model saved to {final_path}")
    
    # Plot training history
    visualizer = ForecastVisualizer()
    visualizer.plot_training_history(
        trainer.train_losses,
        trainer.val_losses,
        title=f"Training History - {config.experiment_name}",
        save_path=os.path.join(config.output_dir, f'{config.experiment_name}_training_history.png')
    )
    
    if config.training.use_wandb:
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()


