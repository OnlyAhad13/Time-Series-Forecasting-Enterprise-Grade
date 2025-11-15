import torch
import torch.nn as nn
from typing import Dict
import os

class EarlyStopping:
    """Early stopping to stop training when validation loss stops increasing"""

    def __init__(
        self, 
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Args:
            score: Validation metric
            epoch: Current epoch
        Returns:
            True if it should stop
        """

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        #Check if score improved
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False

class ModelCheckpoint:
    """Save model checkpoints"""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
    
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save checkpoint if conditions are met"""

        score = metrics.get(self.monitor)

        if score is None:
            return
        
        save = False

        if not self.save_best_only:
            save = True
        elif self.best_score is None:
            save = True
            self.best_score = score
        else:
            if self.mode == "min":
                if score < self.best_score:
                    save = True
                    self.best_score = score
            else:
                if score > self.best_score:
                    save = True
                    self.best_score = score
                
        if save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        torch.save(checkpoint, self.filepath)
    