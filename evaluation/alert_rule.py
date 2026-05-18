"""
Alert rule implementation: threshold + persistence requirement.
Computes early warning time (EWT) and false-positive rate.

BUG-FIX v3 (vs original repo):
  The original `compute_early_warning_time` converted `persistence` (minutes)
  to seconds by multiplying by 60, then searched for that many *consecutive
  samples* in prob_series. However, `run_alert_analysis.py` calls this with a
  prob_series collected at *stride = 60 s* (one sample per minute).
  With persistence = 3 min, the original code searched for 180 consecutive
  samples in a ~30-sample series — so it almost never fired.

  Corrected behaviour:
    The function now accepts a `samples_per_minute` argument (default 1 for
    minute-level series from run_alert_analysis.py, or 60 for second-level).
    The persistence check counts *consecutive qualifying samples*, where each
    sample represents (60 / samples_per_minute) seconds. This makes the
    function correct for both stride = 1 s and stride = 60 s callers.

    For run_alert_analysis.py (stride=60 s → samples_per_minute=1):
        persistence = 3 min → need 3 consecutive qualifying samples ✓

    For hypothetical second-level usage (samples_per_minute=60):
        persistence = 3 min → need 180 consecutive samples ✓
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def compute_early_warning_time(
    prob_series: np.ndarray,
    threshold: float = 0.8,
    persistence: int = 3,           # minutes
    arrival_time: int = 30,         # minutes (episode length)
    samples_per_minute: int = 1,    # 1 for minute-stride series; 60 for 1-Hz series
) -> Tuple[float, bool, int]:
    """
    Compute early warning time (EWT) for a single patient episode.

    Args:
        prob_series:        Array of predicted probabilities over time.
                            Length = n_time_steps (minute- or second-level).
        threshold:          Probability threshold to trigger alert (default 0.8).
        persistence:        Required sustained duration in MINUTES (default 3).
        arrival_time:       Episode length in minutes (default 30).
        samples_per_minute: How many samples correspond to one minute.
                            • 1   if prob_series has one entry per minute
                              (stride=60 s in the dataset).
                            • 60  if prob_series has one entry per second
                              (stride=1 s).

    Returns:
        ewt_minutes:      Minutes before hospital arrival of the first alert,
                          or NaN if no alert triggered.
        alerted:          True if alert was triggered.
        first_alert_idx:  Index (in prob_series) of the first alert sample,
                          or -1 if no alert.
    """
    # How many consecutive samples are needed?
    persistence_samples = persistence * samples_per_minute  # BUG-FIX

    above = prob_series >= threshold

    # Find runs of consecutive True values
    runs: List[Tuple[int, int]] = []
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))

    # First run satisfying the persistence requirement
    first_alert_idx = None
    for s, e in runs:
        if e - s + 1 >= persistence_samples:
            first_alert_idx = s
            break

    if first_alert_idx is not None:
        # Convert sample index to minutes; then compute minutes before arrival
        alert_time_min = first_alert_idx / samples_per_minute
        ewt_minutes    = arrival_time - alert_time_min
        return float(ewt_minutes), True, first_alert_idx
    else:
        return float("nan"), False, -1


def optimize_alert_rule(
    prob_series_list: List[np.ndarray],
    labels: List[int],
    thresholds: List[float] = None,
    persistences: List[int] = None,
    samples_per_minute: int = 1,
) -> List[dict]:
    """
    Sweep over (threshold × persistence) grid and compute metrics.

    Args:
        prob_series_list:   List of per-patient probability series.
        labels:             Corresponding binary labels (1=TIC, 0=control).
        thresholds:         Probability thresholds to evaluate.
        persistences:       Persistence durations in MINUTES to evaluate.
        samples_per_minute: See `compute_early_warning_time`.

    Returns:
        List of dicts with keys:
            threshold, persistence, sensitivity, specificity, fpr, ppv,
            median_ewt, tp, fp, fn, tn.
    """
    if thresholds is None:
        thresholds = [0.7, 0.8, 0.9]
    if persistences is None:
        persistences = [1, 2, 3, 4, 5]

    results = []
    for thr in thresholds:
        for per in persistences:
            ewts:    List[float] = []
            alerted: List[bool]  = []

            for prob, label in zip(prob_series_list, labels):
                ewt, alert, _ = compute_early_warning_time(
                    prob, thr, per,
                    samples_per_minute=samples_per_minute,
                )
                ewts.append(ewt)
                alerted.append(alert)

            paired = list(zip(labels, alerted, ewts))
            tp = sum(1 for l, a, _ in paired if l == 1 and     a)
            fn = sum(1 for l, a, _ in paired if l == 1 and not a)
            fp = sum(1 for l, a, _ in paired if l == 0 and     a)
            tn = sum(1 for l, a, _ in paired if l == 0 and not a)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            tic_ewts = [e for l, a, e in paired if l == 1 and a and not np.isnan(e)]
            median_ewt = float(np.median(tic_ewts)) if tic_ewts else float("nan")

            results.append({
                "threshold":   thr,
                "persistence": per,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "fpr":         fpr,
                "ppv":         ppv,
                "median_ewt":  median_ewt,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

    return results
