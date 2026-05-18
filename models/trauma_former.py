"""
Trauma-Former: iTransformer architecture for real-time TIC prediction.
Implements Algorithm 1 from the paper.
"""
import torch
import torch.nn as nn
import math

class TraumaFormer(nn.Module):
    """
    Inverted Transformer (iTransformer) for multivariate time series classification.
    Input: (batch_size, T, N) where T=60, N=4 (HR, SBP, DBP, SpO2).
    Output: (batch_size, 1) sigmoid probability of TIC.
    """
    def __init__(self,
                 input_dim: int = 4,
                 window_size: int = 60,
                 d_model: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 512,
                 dropout: float = 0.2,
                 classifier_hidden: int = 128,
                 activation: str = 'gelu'):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = d_model

        # Linear projection: each variable's 60-length sequence -> d_model
        self.input_proj = nn.Linear(window_size, d_model)

        # Transformer encoder (cross-variable attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * input_dim, classifier_hidden),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, T, N) input tensor.
            mask: (batch_size, N, T) or None; if provided, used in attention.
                  True = valid, False = masked (padded). iTransformer expects mask of shape (batch, N, N)
                  but we simplify: we will apply token-level masking by zeroing out invalid tokens.
                  For simplicity, we assume mask is (batch, N) indicating if a variable token is entirely valid.
                  However, the paper masks padded positions within the token sequence (i.e., time steps).
                  To support per-time-step masking, we would need to modify the encoder.
                  We implement a simplified version: if mask is provided (batch, N), we zero out embeddings
                  of completely missing variable tokens. This aligns with the paper's handling of prolonged
                  signal loss (e.g., whole variable missing for 30s). For partial missing, we rely on
                  interpolation in preprocessing.
        """
        # Transpose: (batch, T, N) -> (batch, N, T)
        x = x.transpose(1, 2)  # (batch, N, T)

        # Project each variable's time series to token embedding
        tokens = self.input_proj(x)  # (batch, N, d_model)

        # Apply token masking if mask provided (mask: batch, N)
        if mask is not None:
            # mask: True = valid, False = invalid (whole variable missing)
            # Expand to (batch, N, d_model) and set invalid tokens to zero
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, N, 1)
            tokens = tokens * mask_expanded

        # Pass through encoder (self-attention across variables)
        encoded = self.encoder(tokens)  # (batch, N, d_model)

        # Flatten and classify
        flat = encoded.reshape(encoded.size(0), -1)  # (batch, N*d_model)
        out = self.classifier(flat)  # (batch, 1)
        return out

    def predict_proba(self, x, mask=None):
        """Return probability of TIC (between 0 and 1)."""
        with torch.no_grad():
            return self.forward(x, mask).cpu().numpy()

    @staticmethod
    def get_default_config():
        """Return default hyperparameters from paper."""
        return {
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
            'dropout': 0.2,
            'classifier_hidden': 128,
            'activation': 'gelu'
        }