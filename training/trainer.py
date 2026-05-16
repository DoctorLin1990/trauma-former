"""
Generic training loop for PyTorch deep-learning models.
Implements AdamW optimisation, binary cross-entropy loss,
and AUROC-based early stopping as described in Section 2.7 and
Supplementary S2.2.2.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from training.utils import get_device, setup_logger

logger = setup_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Single training epoch. Returns mean batch loss."""
    model.train()
    running_loss = 0.0
    for batch in loader:
        # Unpack: window (B, T, N), mask (B, T, N), label (B,), patient_id
        x, mask, y, _ = batch
        x, y = x.to(device), y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        out  = model(x)          # (B, 1)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader.
    Returns (auroc, y_true, y_score).
    """
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int]   = []

    for batch in loader:
        x, mask, y, _ = batch
        x = x.to(device)
        scores = model(x).squeeze(1).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(y.numpy().tolist())

    y_true  = np.array(all_labels,  dtype=np.int32)
    y_score = np.array(all_scores,  dtype=np.float32)
    auroc   = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    return auroc, y_true, y_score


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, list[float]]:
    """
    Train a model with AdamW and AUROC-based early stopping.

    Args:
        model:          Untrained PyTorch model.
        train_loader:   DataLoader for training windows.
        val_loader:     DataLoader for validation windows.
        learning_rate:  AdamW learning rate (1e-4 per paper).
        weight_decay:   AdamW weight decay (1e-4 per Supplementary S2.2.2).
        max_epochs:     Maximum epochs (200 per Supplementary S2.2.2).
        patience:       Early-stopping patience in epochs (10 per paper).
        device:         Target device (auto-detected if None).

    Returns:
        best_model:   Model state-dict restored to best validation epoch.
        val_aurocs:   List of per-epoch validation AUROC values.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)
    criterion = nn.BCELoss()

    best_auroc   = -1.0
    patience_cnt = 0
    best_state   = copy.deepcopy(model.state_dict())
    val_aurocs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auroc, _, _ = evaluate(model, val_loader, device)
        val_aurocs.append(val_auroc)

        if val_auroc > best_auroc + 1e-6:
            best_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or patience_cnt == 0:
            logger.debug(
                f"Epoch {epoch:3d}/{max_epochs}  loss={train_loss:.4f}  "
                f"val_auroc={val_auroc:.4f}  best={best_auroc:.4f}"
            )

        if patience_cnt >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience}).")
            break

    model.load_state_dict(best_state)
    logger.info(f"Best validation AUROC: {best_auroc:.4f}")
    return model, val_aurocs
