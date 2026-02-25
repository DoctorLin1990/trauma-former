"""
Informer baseline adapted for classification.
Requires the 'informer' package (official implementation) to be installed.
"""
try:
    from informer import Informer  # Official implementation
except ImportError:
    raise ImportError(
        "InformerModel requires the 'informer' package. "
        "Please install it from: https://github.com/zhouhaoyi/Informer2020"
    )

import torch
import torch.nn as nn

class InformerModel(nn.Module):
    """
    Wrapper for Informer model for classification.
    Uses the final time step's representation as input to a classifier.
    """
    def __init__(self,
                 input_dim: int = 4,
                 window_size: int = 60,
                 enc_in: int = 4,
                 dec_in: int = 4,
                 c_out: int = 4,
                 d_model: int = 128,
                 n_heads: int = 4,
                 e_layers: int = 3,
                 d_layers: int = 2,
                 d_ff: int = 512,
                 dropout: float = 0.2,
                 factor: int = 5,
                 **kwargs):
        super().__init__()
        self.informer = Informer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=window_size,
            label_len=window_size//2,
            out_len=1,  # We only need one output time step
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn='prob',
            factor=factor,
            **kwargs
        )
        # The Informer's forward returns predictions for each decoder output (out_len, batch, c_out)
        # We'll take the first (and only) output and project to classification
        self.classifier = nn.Linear(c_out, 1)

    def forward(self, x):
        # x: (batch, T, N)
        # Informer expects (batch, seq_len, features) for encoder and decoder.
        # We'll use x as encoder input, and use zeros as decoder start token.
        batch_size, T, N = x.shape
        dec_inp = torch.zeros(batch_size, 1, N).to(x.device)  # start token
        out = self.informer(x, dec_inp)  # (batch, out_len, c_out)
        last = out[:, -1, :]  # (batch, c_out)
        logits = self.classifier(last)
        return torch.sigmoid(logits)

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x).cpu().numpy()