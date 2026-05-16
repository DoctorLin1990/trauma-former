"""
Evaluation metrics used in the paper.

Covers: AUROC, AUPRC, Brier score, sensitivity/specificity/PPV/NPV/F1,
calibration curve with slope/intercept, multivariate Hellinger distance,
and Monte Carlo Standard Error (MCSE).

IMPORTANT: The sklearn `calibration_curve` function is imported under an
alias to avoid name conflict with the local `calibration_stats` function.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve as _sklearn_calibration_curve


# ─────────────────────────────────────────────────────────────────────
# Primary metrics
# ─────────────────────────────────────────────────────────────────────

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

    return dict(sensitivity=sensitivity, specificity=specificity,
                ppv=ppv, npv=npv, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────

def calibration_stats(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute calibration curve and fit a linear model.

    Returns:
        prob_true:  empirical event rate per bin
        prob_pred:  mean predicted probability per bin
        slope:      linear-fit slope (ideal = 1.0)
        intercept:  linear-fit intercept (ideal = 0.0)
    """
    prob_true, prob_pred = _sklearn_calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )
    if len(prob_pred) < 2:
        return prob_true, prob_pred, 1.0, 0.0

    coeffs = np.polyfit(prob_pred, prob_true, deg=1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return prob_true, prob_pred, slope, intercept


# ─────────────────────────────────────────────────────────────────────
# Hellinger distance (synthetic-data fidelity, Section 2.2)
# ─────────────────────────────────────────────────────────────────────

def hellinger_distance(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Approximate multivariate Hellinger distance D(P, Q) via histogram binning.

    D(P, Q) = sqrt(1 - sum(sqrt(p_i * q_i)) * bin_volume)

    where p_i, q_i are densities in each histogram bin.
    Threshold for acceptable synthetic-data fidelity: < 0.05 (Section 3.1).
    """
    if p_samples.ndim == 1:
        p_samples = p_samples[:, np.newaxis]
        q_samples = q_samples[:, np.newaxis]

    n_dims = p_samples.shape[1]
    bins = [
        np.linspace(
            min(p_samples[:, i].min(), q_samples[:, i].min()),
            max(p_samples[:, i].max(), q_samples[:, i].max()),
            n_bins + 1,
        )
        for i in range(n_dims)
    ]

    p_hist, _ = np.histogramdd(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogramdd(q_samples, bins=bins, density=True)

    p_flat = np.clip(p_hist.flatten(), 1e-12, None)
    q_flat = np.clip(q_hist.flatten(), 1e-12, None)

    bin_volume = np.prod([b[1] - b[0] for b in bins])
    h = float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(p_flat * q_flat))) * bin_volume)))
    return h


# ─────────────────────────────────────────────────────────────────────
# Monte Carlo Standard Error (Section 2.8)
# ─────────────────────────────────────────────────────────────────────

def monte_carlo_standard_error(metric_values: List[float]) -> float:
    """
    MCSE = std(metric) / sqrt(n_repetitions).
    Here, repetitions are the CV folds.
    """
    n = len(metric_values)
    if n < 2:
        return 0.0
    return float(np.std(metric_values, ddof=1) / np.sqrt(n))


# ─────────────────────────────────────────────────────────────────────
# Bootstrap confidence interval
# ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn=compute_auroc,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Patient-level bootstrap 95 % CI for any scalar metric.
    Returns (lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = [
        metric_fn(y_true[idx := rng.integers(0, n, n)], y_score[idx])
        for _ in range(n_iter)
    ]
    return float(np.percentile(boot, 100 * alpha / 2)), \
           float(np.percentile(boot, 100 * (1 - alpha / 2)))


# ─────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_bins_cal: int = 10,
) -> dict:
    """
    Compute the full set of metrics reported in Tables 2 and 3.
    Returns a flat dictionary.
    """
    metrics: dict = {}
    metrics["auroc"] = compute_auroc(y_true, y_score)
    metrics["auprc"] = compute_auprc(y_true, y_score)
    metrics["brier"] = compute_brier(y_true, y_score)
    metrics.update(compute_classification_metrics(y_true, y_score, threshold))

    _, _, slope, intercept = calibration_stats(y_true, y_score, n_bins=n_bins_cal)
    metrics["calibration_slope"]     = slope
    metrics["calibration_intercept"] = intercept

    return metrics
