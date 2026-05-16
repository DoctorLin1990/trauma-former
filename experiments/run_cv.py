#!/usr/bin/env python3
"""
Run patient-level 5-fold cross-validation for any model.
Outputs Table 1 (baseline characteristics) and Table 2 (performance metrics).

Usage:
    # From the repository root:
    python experiments/run_cv.py --config configs/trauma_former.yaml --model trauma-former
    python experiments/run_cv.py --config configs/gru.yaml            --model gru
    python experiments/run_cv.py --config configs/cnn.yaml            --model cnn
    python experiments/run_cv.py --config configs/lr_trend.yaml       --model lr-trend
    python experiments/run_cv.py --config configs/xgboost.yaml        --model xgboost
    python experiments/run_cv.py --config configs/lstm.yaml           --model lstm
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_cv import run_cv
from training.utils import set_seed, setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Table 1 — baseline characteristics
# ─────────────────────────────────────────────────────────────────────

def compute_baseline_stats(data_path: str) -> pd.DataFrame:
    """
    Compute Table 1: mean ± SD of the first 60-second window per patient,
    stratified by TIC / control, with two-sample t-test p-values.
    """
    loaded = np.load(data_path)
    data   = loaded["data"]    # (N, 1800, 4)
    labels = loaded["labels"]  # (N,)

    baseline = data[:, :60, :]   # first 60-s window: (N, 60, 4)
    # Patient-level mean for the window
    pat_mean = baseline.mean(axis=1)  # (N, 4)

    ctrl_data = pat_mean[labels == 0]
    tic_data  = pat_mean[labels == 1]

    rows = []
    var_labels = ["HR (bpm)", "SBP (mmHg)", "DBP (mmHg)", "SpO2 (%)"]

    for i, vname in enumerate(var_labels):
        c, t = ctrl_data[:, i], tic_data[:, i]
        _, pval = stats.ttest_ind(c, t, equal_var=False)
        rows.append({
            "Variable": vname,
            f"Control (n={len(ctrl_data)})": f"{c.mean():.1f} ± {c.std():.1f}",
            f"TIC (n={len(tic_data)})":      f"{t.mean():.1f} ± {t.std():.1f}",
            "p-value": f"{'<0.001' if pval < 0.001 else f'{pval:.3f}'}",
        })

    # Pulse pressure (SBP - DBP)
    ctrl_pp = ctrl_data[:, 1] - ctrl_data[:, 2]
    tic_pp  = tic_data[:, 1]  - tic_data[:, 2]
    _, pval_pp = stats.ttest_ind(ctrl_pp, tic_pp, equal_var=False)
    rows.append({
        "Variable": "Pulse pressure (mmHg)",
        f"Control (n={len(ctrl_data)})": f"{ctrl_pp.mean():.1f} ± {ctrl_pp.std():.1f}",
        f"TIC (n={len(tic_data)})":      f"{tic_pp.mean():.1f} ± {tic_pp.std():.1f}",
        "p-value": f"{'<0.001' if pval_pp < 0.001 else f'{pval_pp:.3f}'}",
    })

    # Shock index (HR / SBP)
    ctrl_si = ctrl_data[:, 0] / ctrl_data[:, 1]
    tic_si  = tic_data[:, 0]  / tic_data[:, 1]
    _, pval_si = stats.ttest_ind(ctrl_si, tic_si, equal_var=False)
    rows.append({
        "Variable": "Shock index",
        f"Control (n={len(ctrl_data)})": f"{ctrl_si.mean():.2f} ± {ctrl_si.std():.2f}",
        f"TIC (n={len(tic_data)})":      f"{tic_si.mean():.2f} ± {tic_si.std():.2f}",
        "p-value": f"{'<0.001' if pval_si < 0.001 else f'{pval_si:.3f}'}",
    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="5-fold patient-level CV")
    ap.add_argument("--config", required=True, help="Path to model YAML config")
    ap.add_argument("--model",  required=True, help="Model name (e.g. trauma-former, lstm …)")
    ap.add_argument("--data",   default="data/development_set.npz")
    ap.add_argument("--folds",  type=int, default=5)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--output", default="results/cv_results.csv")
    ap.add_argument("--table1_output", default="results/table1_baseline.csv")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)

    # Table 1
    logger.info("Computing baseline characteristics (Table 1) …")
    t1 = compute_baseline_stats(args.data)
    t1.to_csv(args.table1_output, index=False)
    logger.info(f"Table 1 saved → {args.table1_output}")
    print("\nTable 1: Baseline characteristics\n" + t1.to_string(index=False))

    # CV
    logger.info(f"\nRunning {args.folds}-fold CV for model '{args.model}' …")
    cv_results = run_cv(
        config_path=args.config,
        model_name=args.model,
        data_path=args.data,
        n_folds=args.folds,
        seed=args.seed,
    )

    df = pd.DataFrame([cv_results])
    df.to_csv(args.output, index=False)
    logger.info(f"CV results saved → {args.output}")


if __name__ == "__main__":
    main()
