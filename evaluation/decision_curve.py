"""
Decision Curve Analysis (DCA) as per Vickers & Elkin (2006).
Computes net benefit across threshold probabilities.
"""
import numpy as np
from typing import Tuple

def net_benefit(y_true: np.ndarray, y_score: np.ndarray,
                threshold: float) -> float:
    """
    Net benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
    where pt is the threshold probability.
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    n = len(y_true)
    nb = tp / n - (fp / n) * (threshold / (1 - threshold))
    return nb

def decision_curve_analysis(y_true: np.ndarray, y_score: np.ndarray,
                            thresholds: np.ndarray = None) -> dict:
    """
    Perform decision curve analysis over a range of thresholds.

    Args:
        y_true: binary labels.
        y_score: predicted probabilities.
        thresholds: array of threshold probabilities to evaluate.
                   If None, uses np.linspace(0.01, 0.99, 99).

    Returns:
        dict with keys:
            - thresholds: thresholds used.
            - net_benefit: net benefit of the model at each threshold.
            - treat_all_nb: net benefit of treating all patients.
            - treat_none_nb: net benefit of treating none (always 0).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    nb_model = np.array([net_benefit(y_true, y_score, pt) for pt in thresholds])

    # Treat all: assume everyone is positive
    nb_treat_all = []
    for pt in thresholds:
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        n = len(y_true)
        nb_all = tp_all / n - (fp_all / n) * (pt / (1 - pt))
        nb_treat_all.append(nb_all)
    nb_treat_all = np.array(nb_treat_all)

    # Treat none: always 0
    nb_treat_none = np.zeros_like(thresholds)

    return {
        'thresholds': thresholds,
        'net_benefit': nb_model,
        'treat_all': nb_treat_all,
        'treat_none': nb_treat_none
    }