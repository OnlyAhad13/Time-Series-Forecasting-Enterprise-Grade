import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml

@dataclass
class DataConfig:
    """Data Configuration"""
    data_path: str = "data/raw/timeseries.csv"
    timestamp_col: str = "timestamp"
    target_col: str = "target"
    series_id_col: Optional[str] = None

    #Covariates
    numeric_covariates: List[str] = field(default_factory=list)
    categorical_covariates: List[str] = field(default_factory=list)

    #Preprocessing
    freq: str = "W-FRI"
    fill_method: str = "forward"
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0

    #Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    #Feature Engineering
    lags: List[int] = field(default_factory=lambda: [1,2,3,7,14,28])
    rolling_windows: List[int] = field(default_factory=lambda: [7,14,28])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])

    #Temporal Features
    use_temporal_features: bool = True
    cyclical_encoding: bool = True

    #Scaling
    scaler_type: str = "standard"

@dataclass
class ModelConfig:
    """Model Configuration"""
    model_type: str = "lstm"

    #Architecture
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    #Forecasting
    lookback: int = 168
    horizon: int = 24

    #Output type
    output_type: str = "point"
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    #Model-specific params
    model_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Training Configuration"""

    #Optimization
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    #Scheduler
    scheduler: str = "onecycle"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    #Training settings
    mixed_precision: bool = True
    accumulation_steps: int = 1

    #Early Stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    #Checkpointing
    save_best: bool = True
    save_frequency: int = 5

    #Reproducibility
    seed: int = 42
    deterministic: bool = True

    #Logging
    log_frequency: int = 10
    use_wandb: bool = True
    wandb_project: str = "ts-forecasting"
    wandb_entity: Optional[str] = None

@dataclass
class Config:
    """Master Configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    #Paths
    root_dir: str = "."
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    #Experiment
    experiment_name: str = "baseline"

    def __post_init__(self):
        """Create directories"""
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(os.path.join(self.root_dir, dir_name), exist_ok=True)
        
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            **{k:v for k, v in config_dict.items() if k not in ['data', 'model', 'training']}
        )
    
    def to_yaml(self, path: str):
        """Save config to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'root_dir': self.root_dir,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'experiment_name': self.experiment_name
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

print("Configuration module created")
print("Project structure defined")

