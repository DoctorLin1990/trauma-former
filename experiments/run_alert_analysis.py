#!/usr/bin/env python3
"""
Alert rule sensitivity analysis (Section 3.4 / Supplementary Figure S4).

Sweeps threshold [0.7, 0.8, 0.9] × persistence [1, 2, 3, 4, 5] minutes
and reports sensitivity, false-positive rate, PPV, and median EWT per
combination.

Key result (paper Section 3.4):
  3-min persistence at threshold 0.80:
    → median EWT = 18.1 min (IQR 13.4–22.3)
    → detection rate = 91.5% (567/620 TIC episodes)
    → false-positive rate = 8.3% (51/620 control episodes)

BUG-FIX v3 vs original repo:
  The original code collected prob_series with stride=60 s (one prob per
  minute) but then passed those minute-level series to compute_early_warning_time
  without telling it they are minute-level. The old function multiplied
  `persistence` × 60 to convert minutes → seconds, so persistence=3 required
  180 consecutive samples in a ~30-sample series — essentially never fired.

  Fix: pass `samples_per_minute=1` to `compute_early_warning_time` so that
  persistence=3 min correctly requires 3 consecutive minute-level samples.

Usage:
    python experiments/run_alert_analysis.py \\
        --model_path results/models/trauma_former_best.pt \\
        --dev_data   data/development_set.npz
"""
from __future__ import annotations

import os, sys, argparse, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trauma_former import TraumaFormer
from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from evaluation.alert_rule import compute_early_warning_time
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)

WINDOW = 60   # seconds
STRIDE = 60   # one prediction per minute (→ samples_per_minute = 1)
BATCH  = 64
SAMPLES_PER_MIN = STRIDE // 60   # = 1  (BUG-FIX: explicit)


def collect_episode_prob_series(
    model, data, labels, norm, device
) -> tuple[list[np.ndarray], list[int]]:
    """
    For each patient episode collect the full time-series of predicted TIC
    probabilities, one value per minute (stride = 60 s).
    Returns (series_list, label_list).
    """
    model.eval()
    patient_probs:  dict[int, list[float]] = {}
    patient_labels: dict[int, int]         = {}

    ds = TICDataset(data, labels, window_size=WINDOW, stride=STRIDE, normalizer=norm)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    with torch.no_grad():
        for x_b, _, y_b, pid_b in dl:
            out = model(x_b.to(device)).squeeze(1).cpu().numpy()
            for i, pid in enumerate(pid_b.numpy()):
                patient_probs.setdefault(int(pid), []).append(float(out[i]))
                patient_labels[int(pid)] = int(y_b[i].item())

    pids   = sorted(patient_probs)
    series = [np.array(patient_probs[p]) for p in pids]
    lbls   = [patient_labels[p] for p in pids]
    return series, lbls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="results/models/trauma_former_best.pt")
    ap.add_argument("--dev_data",   default="data/development_set.npz")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output",     default="results/alert_analysis.csv")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    dev  = np.load(args.dev_data)
    data, labels = dev["data"], dev["labels"]

    norm = ZScoreNormalizer()
    norm.fit(data)

    model = TraumaFormer(**TraumaFormer.get_default_config()).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    logger.info("Collecting episode probability series (stride=60 s, minute-level) …")
    prob_series, lbls = collect_episode_prob_series(model, data, labels, norm, device)

    thresholds   = [0.7, 0.8, 0.9]
    persistences = [1, 2, 3, 4, 5]   # minutes

    rows = []
    for thr in thresholds:
        for per in persistences:
            ewts, alerted = [], []
            for prob, lbl in zip(prob_series, lbls):
                # BUG-FIX: pass samples_per_minute=1 for minute-level series
                ewt, alert, _ = compute_early_warning_time(
                    prob_series=prob,
                    threshold=thr,
                    persistence=per,
                    arrival_time=30,
                    samples_per_minute=SAMPLES_PER_MIN,   # ← FIX
                )
                ewts.append(ewt)
                alerted.append((alert, lbl))

            tp = sum(1 for a, l in alerted if a and l == 1)
            fn = sum(1 for a, l in alerted if not a and l == 1)
            fp = sum(1 for a, l in alerted if a and l == 0)
            tn = sum(1 for a, l in alerted if not a and l == 0)

            sens = tp / (tp + fn) if (tp + fn) else 0.0
            fpr  = fp / (fp + tn) if (fp + tn) else 0.0
            ppv  = tp / (tp + fp) if (tp + fp) else 0.0

            tic_ewts = [e for e, (a, l) in zip(ewts, alerted) if l == 1 and a and not np.isnan(e)]
            med_ewt  = float(np.median(tic_ewts)) if tic_ewts else float("nan")
            iqr_lo   = float(np.percentile(tic_ewts, 25)) if tic_ewts else float("nan")
            iqr_hi   = float(np.percentile(tic_ewts, 75)) if tic_ewts else float("nan")

            row = dict(
                threshold=thr, persistence_min=per,
                sensitivity=round(sens, 4), fpr=round(fpr, 4), ppv=round(ppv, 4),
                median_ewt=round(med_ewt, 2),
                iqr_lo=round(iqr_lo, 2), iqr_hi=round(iqr_hi, 2),
                tp=tp, fp=fp, fn=fn, tn=tn,
            )
            rows.append(row)

            marker = " ← paper (Section 3.4)" if abs(thr - 0.80) < 1e-9 and per == 3 else ""
            logger.info(
                f"Thr={thr:.1f}  Per={per}min: "
                f"Sens={sens:.3f}  FPR={fpr:.3f}  EWT={med_ewt:.1f}min{marker}"
            )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
