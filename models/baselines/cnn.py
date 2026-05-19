"""
1D Convolutional Neural Network (1D-CNN) baseline for TIC prediction.
Matches Supplementary Table S2.3 exactly.

Architecture:
    Conv1d(4 → 32, kernel=3, ReLU) → MaxPool(2)
    Conv1d(32 → 64, kernel=3, ReLU) → MaxPool(2)
    Flatten → FC(128, ReLU) → Dropout(0.2) → FC(1, Sigmoid)

Input : (batch, T=60, N=4) — permuted internally to (batch, N=4, T=60)
Output: (batch, 1) sigmoid TIC probability.
"""
import torch
import torch.nn as nn
from typing import Optional


class CNNModel(nn.Module):
    """
    Temporal 1D-CNN for multivariate vital-sign classification.
    Treats the 4 vital signs as channels and applies 1D convolutions
    over the time axis (T=60 seconds).
    """

    def __init__(
        self,
        input_dim: int = 4,
        window_size: int = 60,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim   = input_dim
        self.window_size = window_size

        # Two convolutional + pooling blocks
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, conv1_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )

        # Compute flattened size after two MaxPool(2) operations
        #   T → T//2 → T//4   (with padding='same' equivalent via padding=k//2)
        t_after_pool = window_size // (pool_size ** 2)
        flat_dim = conv2_filters * t_after_pool

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, T, N) input.
            mask: unused (kept for API compatibility).
        Returns:
            (batch, 1) TIC probability.
        """
        # Conv1d expects (batch, channels, length) → permute
        x = x.permute(0, 2, 1)        # (batch, N=4, T=60)
        x = self.conv_block(x)         # (batch, 64, T//4)
        x = x.flatten(start_dim=1)     # (batch, flat_dim)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> "np.ndarray":
        import numpy as np
        with torch.no_grad():
            return self.forward(x).cpu().numpy()

    @staticmethod
    def get_default_config() -> dict:
        return {
            "conv1_filters": 32,
            "conv2_filters": 64,
            "kernel_size": 3,
            "pool_size": 2,
            "fc_hidden": 128,
            "dropout": 0.2,
        }
