"""
Evaluation metrics used in the paper.

Covers: AUROC, AUPRC, Brier score, sensitivity/specificity/PPV/NPV/F1,
calibration curve with slope/intercept, multivariate Hellinger distance,
and Monte Carlo Standard Error (MCSE).

All metrics referenced in Tables 2 and 3 of the main manuscript, and
in Supplementary S2 (MCSE formula, Koehler et al.).

IMPORTANT: The sklearn `calibration_curve` function is imported under an
alias to avoid name conflict with the local `calibration_stats` function.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve as _sklearn_calibration_curve


# ---------------------------------------------------------------------------
# Primary metrics
# ---------------------------------------------------------------------------

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve."""
    return float(average_precision_score(y_true, y_score))


def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score (lower is better)."""
    return float(brier_score_loss(y_true, y_score))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Sensitivity, specificity, PPV, NPV, F1 at a fixed probability threshold.
    Returns dict with keys: sensitivity, specificity, ppv, npv, f1, tp, tn, fp, fn.
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return dict(
        sensitivity=sensitivity, specificity=specificity,
        ppv=ppv, npv=npv, f1=f1,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all primary and secondary metrics reported in Tables 2 and 3.
    Returns a flat dict with keys: auroc, auprc, brier, sensitivity,
    specificity, ppv, npv, f1, tp, tn, fp, fn, calib_slope, calib_intercept.
    """
    metrics = {
        'auroc': compute_auroc(y_true, y_score),
        'auprc': compute_auprc(y_true, y_score),
        'brier': compute_brier(y_true, y_score),
    }
    clf = compute_classification_metrics(y_true, y_score, threshold)
    metrics.update(clf)

    _, _, slope, intercept = calibration_stats(y_true, y_score)
    metrics['calib_slope']     = slope
    metrics['calib_intercept'] = intercept

    return metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_stats(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute calibration curve and fit a linear model.
    Returns (prob_true, prob_pred, slope, intercept).
    Ideal calibration: slope=1.0, intercept=0.0.
    """
    prob_true, prob_pred = _sklearn_calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy='uniform'
    )
    if len(prob_pred) < 2:
        return prob_true, prob_pred, 1.0, 0.0

    coeffs = np.polyfit(prob_pred, prob_true, deg=1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return prob_true, prob_pred, slope, intercept


# ---------------------------------------------------------------------------
# Hellinger distance  (synthetic-data fidelity, Section 2.2)
# ---------------------------------------------------------------------------

def multivariate_hellinger_distance(
    X_empirical: np.ndarray,
    X_theoretical: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Estimate the multivariate Hellinger distance D(P, Q).

    D(P, Q) = 1 - integral sqrt(p(x) q(x)) dx

    Approximated as the mean of per-variable 1-D Hellinger distances.
    Values < 0.05 indicate acceptable fidelity (Supplementary S1.5.1).
    """
    n_features = X_empirical.shape[1]
    h_per_var  = []
    all_data   = np.concatenate([X_empirical, X_theoretical], axis=0)

    for j in range(n_features):
        lo   = all_data[:, j].min()
        hi   = all_data[:, j].max()
        bins = np.linspace(lo, hi, n_bins + 1)

        p, _ = np.histogram(X_empirical[:, j],   bins=bins, density=True)
        q, _ = np.histogram(X_theoretical[:, j], bins=bins, density=True)

        bin_width = bins[1] - bins[0]
        overlap   = float(np.clip(np.sum(np.sqrt(p * q)) * bin_width, 0.0, 1.0))
        h_per_var.append(1.0 - overlap)

    return float(np.mean(h_per_var))


# ---------------------------------------------------------------------------
# Monte Carlo Standard Error  (Section 2.8 / Koehler et al.)
# ---------------------------------------------------------------------------

def monte_carlo_standard_error(
    fold_metric_values: List[float],
    n_folds: Optional[int] = None,
) -> float:
    """
    MCSE = sqrt(Var(metric) / K)

    where K = number of cross-validation folds.
    Reference: Koehler et al. (2009). Am Stat. DOI:10.1198/tast.2009.0015.
    """
    vals  = np.array(fold_metric_values, dtype=np.float64)
    K     = n_folds if n_folds is not None else len(vals)
    return float(np.sqrt(np.var(vals, ddof=1) / K))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true:    np.ndarray,
    y_score:   np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = compute_auroc,
    n_iter:    int = 1000,
    alpha:     float = 0.05,
    seed:      int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval for any scalar metric function.

    Args:
        y_true:     Ground-truth binary labels.
        y_score:    Predicted probabilities.
        metric_fn:  Callable (y_true, y_score) -> scalar.
        n_iter:     Bootstrap iterations (default 1000, per paper Section 2.8).
        alpha:      Significance level (default 0.05 -> 95% CI).
        seed:       Random seed.

    Returns:
        (lower, upper) CI bounds.
    """
    rng       = np.random.default_rng(seed)
    n         = len(y_true)
    boot_vals: List[float] = []

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        if len(np.unique(y_t)) < 2:
            continue
        try:
            boot_vals.append(metric_fn(y_t, y_s))
        except Exception:
            continue

    if not boot_vals:
        return float('nan'), float('nan')

    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return lo, hi
