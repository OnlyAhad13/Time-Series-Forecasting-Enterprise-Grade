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
import wandb

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
            # Default values for ReduceLROnPlateau
            default_params = {
                'mode': 'min',
                'factor': 0.5,
                'patience': 5,
                'verbose': False
            }
            # Merge with config params (config overrides defaults)
            scheduler_params = {**default_params, **config.scheduler_params}
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_params
            )
        elif config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                **config.scheduler_params
            )
        else:
            return None
            
    def _compute_loss(self, predictions, y):
        """Compute loss based on output type"""
        if self.config.model.output_type == "gaussian":
            mu, sigma = predictions
            loss = self.criterion(mu, sigma, y)
        else:
            loss = self.criterion(predictions, y)
        return loss
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")

        for batch_idx, batch in enumerate(pbar):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            #Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    # Pass y for teacher forcing if model supports it
                    if isinstance(self.model, BaseForecaster) and hasattr(self.model, 'cell_type') and self.model.cell_type in ["LSTM", "GRU"]:
                         predictions = self.model(x, y=y, teacher_forcing_ratio=self.config.training.teacher_forcing_ratio)
                    else:
                        predictions = self.model(x)

                    loss = self._compute_loss(predictions, y)
                
                #Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                if isinstance(self.model, BaseForecaster) and hasattr(self.model, 'cell_type') and self.model.cell_type in ["LSTM", "GRU"]:
                     predictions = self.model(x, y=y, teacher_forcing_ratio=self.config.training.teacher_forcing_ratio)
                else:
                    predictions = self.model(x)

                loss = self._compute_loss(predictions, y)
                
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip
                    )
                
                self.optimizer.step()

            # Update scheduler (for OneCycleLR)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            # Log to wandb
            if self.config.training.use_wandb and batch_idx % self.config.training.log_frequency == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

        return epoch_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            
            predictions = self.model(x)
            loss = self._compute_loss(predictions, y)
            
            val_loss += loss.item()
            num_batches += 1
        
        if num_batches == 0:
            self.logger.warning("Validation dataloader is empty. Returning 0.0 as validation loss.")
            return 0.0
        
        return val_loss/num_batches
    
    def train(self):
        """Full training loop"""
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        self.logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        
        for epoch in range(self.config.training.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler (for ReduceLROnPlateau)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            
            # Log
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            
            if self.config.training.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            self.checkpoint(self.model, self.optimizer, epoch, metrics)
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_loss, epoch):
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best epoch was {self.early_stopping.best_epoch+1}"
                    )
                    break
        
        self.logger.info("Training completed")
    
    

