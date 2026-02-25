"""
Computation of all evaluation metrics used in the paper:
- AUROC, AUPRC, Brier score
- Calibration curve (slope, intercept)
- Multivariate Hellinger distance (for synthetic data validation)
- Monte Carlo Standard Error (MCSE)
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import wasserstein_distance
from typing import Tuple, Optional, List
import warnings

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic curve."""
    return roc_auc_score(y_true, y_score)

def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the Precision-Recall curve."""
    return average_precision_score(y_true, y_score)

def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score (mean squared error between prediction and true label)."""
    return brier_score_loss(y_true, y_score)

def calibration_curve(y_true: np.ndarray, y_score: np.ndarray,
                      n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute calibration curve and its slope/intercept.
    Returns:
        prob_true: proportion of positives in each bin.
        prob_pred: mean predicted probability in each bin.
        slope: slope of linear fit to calibration curve (ideal = 1).
        intercept: intercept of linear fit (ideal = 0).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='uniform')
    # Fit line: prob_true = slope * prob_pred + intercept
    coeffs = np.polyfit(prob_pred, prob_true, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    return prob_true, prob_pred, slope, intercept

def hellinger_distance(p_samples: np.ndarray, q_samples: np.ndarray,
                       n_bins: int = 50) -> float:
    """
    Approximate Hellinger distance between two multivariate distributions using binning.
    For continuous data, we discretize each dimension independently and compute
    the Hellinger distance of the joint histogram.
    This is a simplified approximation; for exact formula see paper Eq. (1).
    """
    # Handle 1D and multi-dimensional cases
    if p_samples.ndim == 1:
        p_samples = p_samples.reshape(-1, 1)
        q_samples = q_samples.reshape(-1, 1)

    n_dims = p_samples.shape[1]
    # Compute joint histogram for both distributions
    # We'll use np.histogramdd with fixed bins per dimension
    bins = [np.linspace(
        min(p_samples[:, i].min(), q_samples[:, i].min()),
        max(p_samples[:, i].max(), q_samples[:, i].max()),
        n_bins + 1
    ) for i in range(n_dims)]

    p_hist, _ = np.histogramdd(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogramdd(q_samples, bins=bins, density=True)

    # Flatten histograms
    p_flat = p_hist.flatten()
    q_flat = q_hist.flatten()

    # Avoid zero bins causing issues; add small epsilon
    p_flat = np.clip(p_flat, 1e-12, None)
    q_flat = np.clip(q_flat, 1e-12, None)

    # Hellinger distance = sqrt(1 - sum(sqrt(p_i * q_i)) * bin_volume)
    # The bin volume is the product of bin widths
    bin_widths = [bins[i][1] - bins[i][0] for i in range(n_dims)]
    bin_volume = np.prod(bin_widths)

    hellinger = np.sqrt(1 - np.sum(np.sqrt(p_flat * q_flat)) * bin_volume)
    return hellinger

def monte_carlo_standard_error(metric_values: List[float]) -> float:
    """
    Compute Monte Carlo Standard Error (MCSE) as standard deviation / sqrt(n_repetitions).
    In cross-validation, repetitions are the folds.
    """
    return np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))

def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Convenience function to compute all classification metrics."""
    metrics = {}
    metrics['auroc'] = compute_auroc(y_true, y_score)
    metrics['auprc'] = compute_auprc(y_true, y_score)
    metrics['brier'] = compute_brier(y_true, y_score)
    # For sensitivity/specificity at default threshold 0.5
    y_pred_bin = (y_score > 0.5).astype(int)
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Calibration
    prob_true, prob_pred, slope, intercept = calibration_curve(y_true, y_score)
    metrics['calibration_slope'] = slope
    metrics['calibration_intercept'] = intercept
    return metrics