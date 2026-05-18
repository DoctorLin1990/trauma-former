"""
Cubic spline imputation for the S3 non-linear stress-test experiment.
Handles 30% MCAR missing data as described in Supplementary S3.3.

⚠ KNOWN LIMITATION: CubicSpline may introduce Runge-phenomenon artefacts
  over long gaps, potentially inflating model AUROC (Supplementary S3.3).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def introduce_mcar(
    X: np.ndarray,
    missing_rate: float = 0.30,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Randomly set values to NaN (MCAR) at the specified rate.

    Args:
        X:            Array of shape (n, T, C).
        missing_rate: Fraction of time steps to mark as missing per channel.
        random_seed:  Reproducibility seed.
    Returns:
        X_miss: Copy of X with NaNs introduced.
    """
    rng   = np.random.default_rng(random_seed)
    X_out = X.astype(np.float32).copy()
    n, T, C = X_out.shape
    n_miss   = max(1, int(T * missing_rate))

    for i in range(n):
        for c in range(C):
            miss_idx = rng.choice(T, size=n_miss, replace=False)
            X_out[i, miss_idx, c] = np.nan

    return X_out


def spline_impute(X_miss: np.ndarray) -> np.ndarray:
    """
    Impute NaNs in X_miss using cubic spline interpolation per channel.

    Args:
        X_miss: (n, T, C) float32 with NaNs.
    Returns:
        X_imp:  (n, T, C) float32 with NaNs replaced by spline estimates.
    """
    X_imp = X_miss.copy()
    n, T, C = X_imp.shape
    t_all = np.arange(T, dtype=np.float64)

    for i in range(n):
        for c in range(C):
            y     = X_imp[i, :, c].astype(np.float64)
            valid = ~np.isnan(y)

            if valid.sum() < 2:
                # Insufficient data: fill with global mean
                X_imp[i, :, c] = float(np.nanmean(y)) if valid.any() else 0.0
                continue

            if valid.all():
                continue  # nothing to impute

            t_valid = t_all[valid]
            y_valid = y[valid]

            try:
                cs    = CubicSpline(t_valid, y_valid, bc_type="not-a-knot", extrapolate=True)
                y_imp = cs(t_all)
            except Exception:
                # Fallback: linear interpolation
                y_imp = np.interp(t_all, t_valid, y_valid)

            X_imp[i, :, c] = y_imp.astype(np.float32)

    # Final forward-fill / back-fill for any residual NaNs (edge effects)
    for i in range(n):
        for c in range(C):
            y = X_imp[i, :, c]
            if np.any(np.isnan(y)):
                mask      = ~np.isnan(y)
                if mask.any():
                    X_imp[i, :, c] = np.interp(t_all, t_all[mask], y[mask])
                else:
                    X_imp[i, :, c] = 0.0

    return X_imp
