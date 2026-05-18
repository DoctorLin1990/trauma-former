"""
Data preprocessing utilities: Z-score normalization, interpolation, masking.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class ZScoreNormalizer:
    """
    Fit Z-score normalization on training data and transform any data.
    Each variable (HR, SBP, DBP, SpO2) is normalized independently.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, data: np.ndarray):
        """
        data: array of shape (n_samples, n_timesteps, n_features)
        Compute mean and std per feature across all samples and time steps.
        """
        n_samples, n_timesteps, n_features = data.shape
        # Reshape to (n_samples * n_timesteps, n_features)
        flat = data.reshape(-1, n_features)
        self.scaler.fit(flat)
        self.fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted first.")
        shape = data.shape
        flat = data.reshape(-1, shape[-1])
        normalized = self.scaler.transform(flat)
        return normalized.reshape(shape)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(data)
        return self.transform(data)

def interpolate_and_mask(window: np.ndarray, max_gap: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values in a 60-second window.
    - For gaps <= max_gap seconds: linear interpolation.
    - For longer gaps: leave as NaN, which will be zero-padded and masked.

    Args:
        window: array of shape (T, 4) with possible NaNs.
        max_gap: maximum gap length to interpolate.

    Returns:
        filled_window: array with same shape, NaNs replaced by 0 (for masking).
        mask: boolean array of same shape, True where value is valid (not NaN originally or after interpolation).
    """
    T, C = window.shape
    filled = window.copy()
    mask = np.ones((T, C), dtype=bool)

    # Process each channel independently
    for c in range(C):
        channel = window[:, c]
        # Find NaN positions
        nan_idx = np.where(np.isnan(channel))[0]
        if len(nan_idx) == 0:
            continue

        # Group consecutive NaNs
        gaps = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)
        for gap in gaps:
            if len(gap) <= max_gap:
                # Interpolate using surrounding non-NaN values
                left = gap[0] - 1
                right = gap[-1] + 1
                # If gap touches the edge, use nearest valid
                if left < 0 and right >= T:
                    # Entire window NaN? Should not happen; fill with 0 but mask False
                    filled[gap, c] = 0
                    mask[gap, c] = False
                elif left < 0:
                    filled[gap, c] = channel[right]
                elif right >= T:
                    filled[gap, c] = channel[left]
                else:
                    # Linear interpolation
                    x_left, x_right = left, right
                    y_left, y_right = channel[left], channel[right]
                    slope = (y_right - y_left) / (x_right - x_left)
                    for i, t in enumerate(gap):
                        filled[t, c] = y_left + slope * (t - x_left)
            else:
                # Long gap: leave as 0 and mask out
                filled[gap, c] = 0
                mask[gap, c] = False

    return filled, mask