"""
Generic trainer for PyTorch models with validation and early stopping.
Saves best model based on validation AUROC.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Optional, Callable, Dict, Any
import logging

from .utils import set_seed

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config: Dict[str, Any],
                 experiment_dir: str = './results/models',
                 use_amp: bool = False):
        """
        Args:
            model: PyTorch model to train.
            device: device to use.
            config: dictionary containing training hyperparameters.
            experiment_dir: directory to save model checkpoints.
            use_amp: whether to use automatic mixed precision.
        """
        self.model = model
        self.device = device
        self.config = config
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        self.use_amp = use_amp

        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        self.criterion = nn.BCELoss()

        # Learning rate scheduler (optional)
        self.scheduler = None
        if config.get('use_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )

        # Early stopping parameters
        self.patience = config.get('early_stopping_patience', 10)
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_model_path = os.path.join(experiment_dir, 'best_model.pth')

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        for batch_idx, (x, mask, y, _) in enumerate(train_loader):
            x = x.to(self.device)
            mask = mask.to(self.device) if mask is not None else None
            y = y.to(self.device).unsqueeze(1)  # (batch, 1)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(x, mask)
                    loss = self.criterion(output, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(x, mask)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate and return metrics: loss, AUROC."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, mask, y, _ in val_loader:
                x = x.to(self.device)
                mask = mask.to(self.device) if mask is not None else None
                y = y.to(self.device).unsqueeze(1)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(x, mask)
                        loss = self.criterion(output, y)
                else:
                    output = self.model(x, mask)
                    loss = self.criterion(output, y)

                total_loss += loss.item() * x.size(0)
                all_preds.append(output.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        auroc = roc_auc_score(all_labels, all_preds)

        return {'loss': avg_loss, 'auroc': auroc}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, fold: int = 0):
        """Full training loop with early stopping."""
        logger.info(f"Starting training for fold {fold}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_auroc = val_metrics['auroc']

            logger.info(
                f"Fold {fold} | Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f}"
            )

            if self.scheduler is not None:
                self.scheduler.step(val_auroc)

            # Check for improvement
            if val_auroc > self.best_metric:
                self.best_metric = val_auroc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                # Save best model
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with AUROC {val_auroc:.4f}")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        logger.info(f"Training finished. Best AUROC: {self.best_metric:.4f} at epoch {self.best_epoch}")
        return self.best_metric

    def load_best_model(self):
        """Load the best model weights."""
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        logger.info("Best model loaded")