"""
Bidirectional GRU baseline for TIC prediction.
Architecture: 2-layer BiGRU (64 hidden units/direction) + FC head.
Matches Supplementary S2.3.3 exactly.
"""
import torch
import torch.nn as nn
from typing import Optional


class GRUModel(nn.Module):
    """
    Bidirectional GRU with two layers and 64 hidden units per direction.
    Input:  (batch_size, T, N) — T=60 s window, N=4 vital signs.
    Output: (batch_size, 1)   — sigmoid TIC probability.

    Architecture (Supplementary Table S2.4):
        BiGRU layer 1: 64 hidden / direction, dropout between layers
        BiGRU layer 2: 64 hidden / direction
        FC layer:      64 units, ReLU
        Output layer:  1 unit, Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_size * directions  # final-step concat

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, T, N) input tensor.
            mask: unused (GRU does not use attention masking); kept for API compatibility.
        Returns:
            (batch, 1) sigmoid TIC probability.
        """
        _, h_n = self.gru(x)
        # h_n shape: (num_layers * directions, batch, hidden)
        # Extract the last layer's forward and backward hidden states.
        if self.bidirectional:
            # Last layer: indices -2 (forward) and -1 (backward)
            h_fwd = h_n[-2]  # (batch, hidden)
            h_bwd = h_n[-1]  # (batch, hidden)
            h_last = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, 2*hidden)
        else:
            h_last = h_n[-1]

        return self.classifier(h_last)

    def predict_proba(self, x: torch.Tensor) -> "np.ndarray":
        import numpy as np
        with torch.no_grad():
            return self.forward(x).cpu().numpy()

    @staticmethod
    def get_default_config() -> dict:
        return {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": True,
            "fc_hidden": 64,
        }
