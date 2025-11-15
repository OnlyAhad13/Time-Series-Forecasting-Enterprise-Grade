from tqdm import tqdm
import torch
from models.base_model import BaseForecaster
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from config.base_config import Config
from training.losses import get_loss_fn
from training.callbacks import EarlyStopping, ModelCheckpoint
from utils.logger import setup_logger
import os

class Trainer:
    """My training manager """

    def __init__(
        self,
        model: BaseForecaster,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: 'Config',
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        #Loss function
        self.criterion = get_loss_fn(
            config.model.output_type,
            config.model.quantiles
        )

        #Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        #Learning rate scheduler
        self.scheduler = self._create_scheduler()

        #Mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None

        #Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,   
        ) if config.training.early_stopping else None

        self.checkpoint = ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, f'{config.experiment_name}_best.pt'),
            save_best_only=config.training.save_best
        )

        #Logging
        self.logger = setup_logger(
            name='trainer',
            log_file=os.path.join(config.log_dir, f'{config.experiment_name}.log')
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        config = self.config.training

        if config.scheduler == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.epochs,
                steps_per_epoch=len(self.train_loader),
                **config.scheduler_params
            )
        elif config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                **config.scheduler_params
            )
        elif config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                **config.scheduler_params
            )
        else:
            return None

