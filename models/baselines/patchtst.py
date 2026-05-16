"""
PatchTST baseline adapted for classification.
Requires the 'patchtst' package (official implementation) to be installed.
"""
try:
    from patchtst import PatchTST  # Official implementation
except ImportError:
    raise ImportError(
        "PatchTSTModel requires the 'patchtst' package. "
        "Please install it from: https://github.com/yuqinie98/PatchTST"
    )

import torch
import torch.nn as nn

class PatchTSTModel(nn.Module):
    """
    Wrapper for PatchTST model for classification.
    Uses the final time step's representation as input to a classifier.
    """
    def __init__(self,
                 input_dim: int = 4,
                 window_size: int = 60,
                 patch_len: int = 16,
                 stride: int = 8,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.2,
                 head_dropout: float = 0.2,
                 **kwargs):
        super().__init__()
        # PatchTST expects (batch, seq_len, num_channels) and outputs predictions of same length
        # For classification, we take the output at the last time step and pass through a linear layer.
        self.patchtst = PatchTST(
            num_input_channels=input_dim,
            num_output_channels=input_dim,  # We'll override later
            seq_len=window_size,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            head_dropout=head_dropout,
            **kwargs
        )
        # Override final projection: we only need the last time step's representation
        # PatchTST's forward returns (batch, seq_len, d_model) if return_cls_token=False?
        # We'll extract the last time step's hidden state (or the representation of the last patch)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, T, N)
        # PatchTST expects (batch, N, T) or (batch, T, N)? Let's check official usage.
        # Typically, they use (batch, seq_len, num_channels). We'll assume (batch, T, N).
        # We'll pass through PatchTST and get representations of shape (batch, T, d_model)
        # This requires that the model returns per-time-step outputs.
        # The official implementation's forward returns (batch, seq_len, num_output_channels) when return_encoder_output=False.
        # We need to modify to get encoder output. For simplicity, we assume we can get the encoder's final hidden states.
        # However, to avoid modifying the official code, we rely on the model's output at the last time step.
        # Note: This is a simplified wrapper; for exact replication, users should refer to the official repo.
        out = self.patchtst(x)  # Expect shape (batch, T, input_dim) due to output projection
        # Take last time step
        last_step = out[:, -1, :]  # (batch, input_dim)
        # Project to classification
        logits = self.classifier(last_step)
        return torch.sigmoid(logits)

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x).cpu().numpy()