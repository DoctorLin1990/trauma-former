#!/usr/bin/env python3
"""
Evaluate the trained Trauma-Former on the independent test set with
25% TIC prevalence. Reproduces Table 3 and Figure 3B (PPV collapse).

Usage (from repository root):
    python experiments/run_test_set.py \
        --model_path results/models/trauma_former_best.pt \
        --test_data  data/test_set.npz \
        --dev_data   data/development_set.npz
"""
from __future__ import annotations

import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trauma_former import TraumaFormer
from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from evaluation.metrics import compute_all_metrics, bootstrap_ci, compute_auroc
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)

WINDOW = 60
BATCH  = 64


def evaluate_on_test_set(
    model: TraumaFormer,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    normalizer: ZScoreNormalizer,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Collect patient-level TIC probability on the test set."""
    ds = TICDataset(test_data, test_labels, window_size=WINDOW, stride=30,
                    normalizer=normalizer)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model.eval()
    patient_scores: dict[int, float] = {}
    patient_labels: dict[int, int]   = {}

    with torch.no_grad():
        for x_b, _, y_b, pid_b in dl:
            x_b = x_b.to(device)
            out = model(x_b).squeeze(1).cpu().numpy()
            for i, pid in enumerate(pid_b.numpy()):
                patient_scores[pid] = max(patient_scores.get(pid, 0.0), float(out[i]))
                patient_labels[pid] = int(y_b[i].item())

    pids   = sorted(patient_scores)
    y_true = np.array([patient_labels[p] for p in pids])
    y_sc   = np.array([patient_scores[p] for p in pids])

    metrics = compute_all_metrics(y_true, y_sc)
    ci_lo, ci_hi = bootstrap_ci(y_true, y_sc, metric_fn=compute_auroc,
                                 n_iter=1000, seed=seed)
    metrics["auroc_ci_lo"] = ci_lo
    metrics["auroc_ci_hi"] = ci_hi
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Trauma-Former on 25% prevalence test set")
    ap.add_argument("--model_path", default="results/models/trauma_former_best.pt")
    ap.add_argument("--test_data",  default="data/test_set.npz")
    ap.add_argument("--dev_data",   default="data/development_set.npz")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output",     default="results/table3_test_set.json")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    # Normaliser fitted on full development set (all patients)
    dev    = np.load(args.dev_data)
    norm   = ZScoreNormalizer()
    norm.fit(dev["data"])

    # Load model
    model = TraumaFormer(**TraumaFormer.get_default_config())
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    logger.info(f"Model loaded from {args.model_path}")

    # Evaluate
    test  = np.load(args.test_data)
    metrics = evaluate_on_test_set(
        model, test["data"], test["labels"], norm, device, seed=args.seed
    )

    # Print Table 3
    print("\n── Table 3: Trauma-Former — independent test set (25% TIC prevalence) ──")
    print(f"  AUROC       : {metrics['auroc']:.3f}  "
          f"(95% CI {metrics['auroc_ci_lo']:.3f}–{metrics['auroc_ci_hi']:.3f})")
    print(f"  AUPRC       : {metrics['auprc']:.3f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.3f}")
    print(f"  Specificity : {metrics['specificity']:.3f}")
    print(f"  PPV         : {metrics['ppv']:.3f}  ← key: collapsed from 0.89 (50%) to here (25%)")
    print(f"  NPV         : {metrics['npv']:.3f}")
    print(f"  F1          : {metrics['f1']:.3f}")
    print(f"  Brier       : {metrics['brier']:.3f}")

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
