"""
Bidirectional LSTM baseline for TIC prediction.
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with 2 layers, hidden size 64, followed by a dense classifier.
    Input: (batch_size, T, N) where T=60, N=4.
    Output: (batch_size, 1) sigmoid probability.
    """
    def __init__(self,
                 input_dim: int = 4,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, T, N)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use final hidden state from both directions (concatenated)
        if self.lstm.bidirectional:
            h_n = h_n[-2:, :, :]  # last two layers (forward+backward)
            h_n = h_n.permute(1, 0, 2).contiguous().view(h_n.size(1), -1)
        else:
            h_n = h_n[-1]  # last layer
        out = self.classifier(h_n)
        return out

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x).cpu().numpy()