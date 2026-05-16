"""
Alert rule implementation: threshold + persistence requirement.
Computes early warning time (EWT) and false positive rate.
"""
import numpy as np
from typing import Tuple, List

def compute_early_warning_time(prob_series: np.ndarray,
                                threshold: float = 0.8,
                                persistence: int = 3,
                                arrival_time: int = 30) -> Tuple[float, bool, int]:
    """
    Compute early warning time for a single patient episode.

    Args:
        prob_series: array of predicted probabilities over time (length T, e.g., 1800 for 30 min at 1 Hz).
        threshold: probability threshold for alert.
        persistence: number of consecutive minutes (at 1 Hz, this is seconds) required.
        arrival_time: time of hospital arrival in minutes (default 30).

    Returns:
        ewt_minutes: early warning time (minutes before arrival), or NaN if no alert.
        alerted: True if alert triggered.
        first_alert_idx: index (in seconds) of first alert.
    """
    T = len(prob_series)
    # persistence minutes -> seconds (data sampled at 1 Hz)
    # e.g. persistence=3 min -> 180 consecutive samples above threshold
    persistence_sec = persistence * 60
    # Check for contiguous segment above threshold
    above = prob_series >= threshold
    # Find runs
    runs = []
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i-1))
            start = None
    if start is not None:
        runs.append((start, len(above)-1))

    # Find first run of length >= persistence_sec
    first_alert_idx = None
    for s, e in runs:
        if e - s + 1 >= persistence_sec:
            first_alert_idx = s
            break

    if first_alert_idx is not None:
        # Convert seconds to minutes before arrival (arrival at T seconds)
        ewt_minutes = (T - first_alert_idx) / 60.0
        return ewt_minutes, True, first_alert_idx
    else:
        return np.nan, False, -1

def optimize_alert_rule(prob_series_list: List[np.ndarray],
                        labels: List[int],
                        thresholds: List[float] = [0.7, 0.8, 0.9],
                        persistences: List[int] = [1, 2, 3, 4, 5]) -> dict:
    """
    Sweep over thresholds and persistence to find best trade-off.
    Returns a dictionary with results for each combination.
    """
    results = []
    for thr in thresholds:
        for per in persistences:
            ewts = []
            alerts = []
            for prob, label in zip(prob_series_list, labels):
                ewt, alerted, _ = compute_early_warning_time(prob, thr, per)
                ewts.append(ewt)
                alerts.append(alerted)
            # Compute metrics
            tp = np.sum([(label == 1 and alerted) for label, alerted in zip(labels, alerts)])
            fn = np.sum([(label == 1 and not alerted) for label, alerted in zip(labels, alerts)])
            fp = np.sum([(label == 0 and alerted) for label, alerted in zip(labels, alerts)])
            tn = np.sum([(label == 0 and not alerted) for label, alerted in zip(labels, alerts)])
            sensitivity = tp / (tp + fn) if (tp+fn)>0 else 0
            specificity = tn / (tn + fp) if (tn+fp)>0 else 0
            fpr = fp / (fp + tn) if (fp+tn)>0 else 0
            ppv = tp / (tp + fp) if (tp+fp)>0 else 0
            # Early warning time: median of those that alerted in TIC group
            tic_ewts = [ewt for ewt, label in zip(ewts, labels) if label == 1 and not np.isnan(ewt)]
            median_ewt = np.median(tic_ewts) if tic_ewts else np.nan

            results.append({
                'threshold': thr,
                'persistence': per,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'fpr': fpr,
                'ppv': ppv,
                'median_ewt': median_ewt,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
    return results