"""
Shock index baseline: HR / SBP at the final time step of the window.
Threshold > 1.0 used for binary classification.

BUG-FIX v3: Renamed class from `ShockIndex` → `ShockIndexModel` to match
the import in training/train_cv.py:
    from models.baselines.shock_index import ShockIndexModel   ← was failing
"""
import numpy as np


class ShockIndexModel:
    """
    Shock Index = HR / SBP at the last time step of a 60-second window.
    Not a trainable model. `predict_proba` returns the raw shock-index value
    (can be used as a score for AUROC); `predict` applies a fixed threshold.

    Per paper Section 2.7: threshold > 1.0.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X=None, y=None):
        """Nothing to fit."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_windows, T, 4) — columns [HR, SBP, DBP, SpO2].
        Returns: (n_windows,) shock-index values at last time step.
        """
        last_step = X[:, -1, :]          # (n_windows, 4)
        hr  = last_step[:, 0].astype(np.float64)
        sbp = last_step[:, 1].astype(np.float64)
        # Avoid division by zero
        si = np.where(sbp > 0, hr / sbp, np.full_like(hr, 1e6))
        return si

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows,) shock-index score (alias for API compatibility)."""
        return self.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > self.threshold).astype(int)

    @staticmethod
    def get_default_config() -> dict:
        return {"threshold": 1.0}
