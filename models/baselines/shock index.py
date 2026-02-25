"""
Shock index baseline: HR/SBP at the final time step of the window.
Threshold >1.0 used for classification.
"""
import numpy as np

class ShockIndex:
    """
    Simple shock index calculator.
    Not a trainable model; provides predict_proba returning the shock index value
    and predict returning binary class based on threshold.
    """
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Nothing to fit."""
        return self

    def predict_proba(self, X):
        """
        X: numpy array of shape (n_windows, T, 4) where last dimension is [HR, SBP, DBP, SpO2].
        Returns shock index values (HR/SBP) for the last time step of each window.
        """
        # Extract last time step
        last_step = X[:, -1, :]  # (n_windows, 4)
        hr = last_step[:, 0]
        sbp = last_step[:, 1]
        # Avoid division by zero
        si = np.divide(hr, sbp, out=np.full_like(hr, np.nan), where=sbp!=0)
        # Replace NaN with large value (will be > threshold)
        si = np.nan_to_num(si, nan=1e6)
        return si

    def predict(self, X):
        si = self.predict_proba(X)
        return (si > self.threshold).astype(int)