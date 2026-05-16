"""
PyTorch Dataset for sliding windows over patient episodes.
Supports patient-level splitting (all windows from one patient stay together).
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List, Optional
from .preprocessing import interpolate_and_mask

class TICDataset(Dataset):
    """
    Dataset that returns sliding windows of 60 seconds from each patient episode.
    Each item: (window, mask, label, patient_id)
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 window_size: int = 60, stride: int = 1,
                 apply_preprocessing: bool = True,
                 normalizer: Optional['ZScoreNormalizer'] = None):
        """
        Args:
            data: array of shape (n_patients, n_timesteps, n_features) – raw episodes.
            labels: array of shape (n_patients,) – 1 for TIC, 0 for control.
            window_size: length of sliding window in seconds.
            stride: stride of sliding window.
            apply_preprocessing: if True, apply interpolation/masking.
            normalizer: fitted ZScoreNormalizer (if None, data must already be normalized).
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.apply_preprocessing = apply_preprocessing
        self.normalizer = normalizer

        # Build index mapping: (patient_idx, start_time)
        self.indices: List[Tuple[int, int]] = []
        for p_idx in range(len(data)):
            n_timesteps = data.shape[1]
            for start in range(0, n_timesteps - window_size + 1, stride):
                self.indices.append((p_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        p_idx, start = self.indices[idx]
        # Extract raw window
        window = self.data[p_idx, start:start+self.window_size, :].copy()

        # Apply preprocessing (interpolation, masking)
        mask = None
        if self.apply_preprocessing:
            # Introduce artificial missingness? The paper uses masking for network loss,
            # but here we assume data is complete; missingness is simulated at the network layer.
            # We still apply interpolation for ≤5s gaps if any NaNs exist.
            window, mask = interpolate_and_mask(window, max_gap=5)
        else:
            # If no preprocessing, assume no NaNs and mask all ones
            mask = np.ones_like(window, dtype=bool)

        # Normalize (if normalizer provided)
        if self.normalizer is not None:
            # Normalizer expects (n_samples, n_features) but we have (window_size, n_features)
            # Reshape temporarily
            original_shape = window.shape
            window = window.reshape(-1, original_shape[-1])
            window = self.normalizer.transform(window)
            window = window.reshape(original_shape)

        # Convert to torch tensors
        window_t = torch.tensor(window, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        label_t = torch.tensor(self.labels[p_idx], dtype=torch.float32)

        return window_t, mask_t, label_t, p_idx