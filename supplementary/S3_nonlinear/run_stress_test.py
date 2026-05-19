#!/usr/bin/env python3
"""
Supplementary S3 Non-linear Stress Test.

Reproduces Supplementary Table S3.2 and Supplementary Figure S3_nonlinear.

Pipeline (Supplementary S3.4):
  1. Generate 1 000 non-linear episodes (500 TIC, 500 control; seed 42).
  2. Introduce 30% MCAR missing data; impute with cubic spline.
  3. Train and evaluate Trauma-Former, 1D-CNN, GRU on 80/20 train/val split.
  4. Report AUROC, sensitivity, specificity, Brier score (Table S3.2).

Usage:
    python supplementary/S3_nonlinear/run_stress_test.py \
        --seed 42 --missing_rate 0.30 --n_episodes 1000

Output: results/S3_nonlinear_results.csv
"""
from __future__ import annotations

import os, sys, argparse, csv, time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ── path setup ────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from supplementary.S3_nonlinear.nonlinear_generator import generate_nonlinear_batch
from supplementary.S3_nonlinear.spline_imputer import introduce_mcar, spline_impute
from models.trauma_former import TraumaFormer
from models.baselines.gru import GRUModel
from models.baselines.cnn import CNNModel
from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from evaluation.metrics import compute_all_metrics
from training.trainer import train_model
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)


def run_model(
    model_name: str,
    train_data: np.ndarray, train_labels: np.ndarray,
    val_data:   np.ndarray, val_labels:   np.ndarray,
    device: torch.device,
    seed: int = 42,
    batch_size: int = 32,
    window_size: int = 60,
) -> dict:
    """Train one model and return metrics dict."""

    set_seed(seed)
    norm = ZScoreNormalizer()
    norm.fit(train_data)

    tr_ds  = TICDataset(train_data, train_labels, window_size=window_size, stride=30, normalizer=norm)
    val_ds = TICDataset(val_data,   val_labels,   window_size=window_size, stride=30, normalizer=norm)
    tr_ld  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if model_name == "trauma-former":
        model = TraumaFormer()
    elif model_name == "gru":
        model = GRUModel()
    elif model_name == "cnn":
        model = CNNModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model, _ = train_model(
        model, tr_ld, val_ld,
        learning_rate=1e-4, weight_decay=1e-4,
        max_epochs=200, patience=15, device=device,
    )

    model.eval()
    patient_scores, patient_labels = {}, {}
    with torch.no_grad():
        for x_b, _, y_b, pid_b in val_ld:
            x_b = x_b.to(device)
            out = model(x_b).squeeze(1).cpu().numpy()
            for i, pid in enumerate(pid_b.numpy()):
                patient_scores[pid] = max(patient_scores.get(pid, 0.0), float(out[i]))
                patient_labels[pid] = int(y_b[i].item())

    pids   = sorted(patient_scores)
    y_true = np.array([patient_labels[p] for p in pids])
    y_sc   = np.array([patient_scores[p] for p in pids])
    return compute_all_metrics(y_true, y_sc)


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 Non-linear stress test")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--missing_rate", type=float, default=0.30)
    ap.add_argument("--n_episodes",   type=int,   default=1000)
    ap.add_argument("--output",       default="results/S3_nonlinear_results.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = get_device()
    set_seed(args.seed)

    # ── 1. Generate non-linear data ──────────────────────────────────
    logger.info("Generating non-linear episodes …")
    data, labels = generate_nonlinear_batch(
        n_episodes=args.n_episodes, tic_ratio=0.5,
        duration_min=30, random_seed=args.seed,
    )

    # ── 2. Introduce MCAR missingness and impute ──────────────────────
    logger.info(f"Introducing {args.missing_rate:.0%} MCAR missingness + spline imputation …")
    data_miss = introduce_mcar(data, missing_rate=args.missing_rate, random_seed=args.seed)
    data_imp  = spline_impute(data_miss)

    # ── 3. 80/20 train-val split (stratified) ─────────────────────────
    idx_tr, idx_val = train_test_split(
        np.arange(args.n_episodes), test_size=0.20,
        stratify=labels, random_state=args.seed,
    )
    tr_data, tr_lbl = data_imp[idx_tr], labels[idx_tr]
    va_data, va_lbl = data_imp[idx_val], labels[idx_val]
    logger.info(f"Train: {len(idx_tr)}, Val: {len(idx_val)}")

    # ── 4. Evaluate models ────────────────────────────────────────────
    results = []
    for model_name in ["trauma-former", "cnn", "gru"]:
        logger.info(f"Training {model_name} …")
        t0 = time.perf_counter()
        m  = run_model(model_name, tr_data, tr_lbl, va_data, va_lbl,
                       device=device, seed=args.seed)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"  {model_name:15s}: AUROC={m['auroc']:.4f}  "
            f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
            f"Brier={m['brier']:.4f}  [{elapsed:.0f} s]"
        )
        results.append({"model": model_name, **m})

    # ── 5. Save ───────────────────────────────────────────────────────
    fieldnames = ["model", "auroc", "auprc", "sensitivity", "specificity",
                  "ppv", "npv", "f1", "brier"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    logger.info(f"\nResults saved → {args.output}")

    # ── Print comparison table (vs primary linear results) ────────────
    PRIMARY = {"trauma-former": 0.939, "cnn": 0.868, "gru": 0.854}
    print("\n── S3 Stress Test: Linear vs Non-linear ──────────────────")
    print(f"{'Model':<20} {'Linear':>8} {'Non-linear':>12} {'Δ AUROC':>9}")
    print("-" * 55)
    for r in results:
        lin = PRIMARY.get(r["model"], float("nan"))
        print(f"{r['model']:<20} {lin:>8.3f} {r['auroc']:>12.3f} {r['auroc']-lin:>9.3f}")


if __name__ == "__main__":
    main()
