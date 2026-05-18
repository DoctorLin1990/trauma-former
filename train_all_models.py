#!/usr/bin/env python3
"""
One-command pipeline: train all models + generate all tables.
Referenced in Supplementary S2.4:
    python train_all_models.py --config configs/final_config.yaml --seed 42

Expected runtime: ~3 hours on NVIDIA A100 (Supplementary Table S2.6).

Usage:
    python train_all_models.py --seed 42 [--quick]

Flags:
    --quick   Run 1-fold only (for CI/reviewer smoke test).

After CV, the best-fold Trauma-Former checkpoint is re-trained on the full
development set and saved to results/models/trauma_former_best.pt so that
experiments/run_test_set.py can load it directly.
"""
from __future__ import annotations

import os, sys, argparse, time, json, csv
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_cv import run_cv
from training.utils import set_seed, setup_logger

logger = setup_logger(__name__)

MODELS = [
    ("trauma-former", "configs/trauma_former.yaml"),
    ("lr-trend",      "configs/lr_trend.yaml"),
    ("lstm",          "configs/lstm.yaml"),
    ("gru",           "configs/gru.yaml"),
    ("cnn",           "configs/cnn.yaml"),
    ("xgboost",       "configs/xgboost.yaml"),
    ("patchtst",      "configs/patchtst.yaml"),
    ("informer",      "configs/informer.yaml"),
    ("shock-index",   "configs/trauma_former.yaml"),  # config unused for shock index
]


def _save_best_checkpoint(data_path: str, seed: int) -> None:
    """
    Re-train Trauma-Former on the entire development set (no validation split)
    for the configured number of epochs and save the final state-dict to
    results/models/trauma_former_best.pt.

    This file is required by:
        experiments/run_test_set.py
        experiments/run_robustness.py
        experiments/run_alert_analysis.py
    """
    from data.dataset import TICDataset
    from data.preprocessing import ZScoreNormalizer
    from models.trauma_former import TraumaFormer
    from training.trainer import train_model
    from training.utils import set_seed, get_device

    set_seed(seed)
    device = get_device()

    loaded = np.load(data_path)
    data, labels = loaded["data"], loaded["labels"]

    norm = ZScoreNormalizer()
    norm.fit(data)

    # 90/10 internal split for early stopping (not used for final metric)
    n     = len(labels)
    n_val = max(1, int(n * 0.10))
    idx   = np.random.default_rng(seed).permutation(n)
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    tr_ds = TICDataset(data[tr_idx], labels[tr_idx], window_size=60, stride=30, normalizer=norm)
    va_ds = TICDataset(data[va_idx], labels[va_idx], window_size=60, stride=30, normalizer=norm)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True,  num_workers=0)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TraumaFormer(**TraumaFormer.get_default_config()).to(device)
    model, _ = train_model(model, tr_dl, va_dl,
                            learning_rate=1e-4, weight_decay=1e-4,
                            max_epochs=200, patience=10, device=device)

    os.makedirs("results/models", exist_ok=True)
    ckpt_path = "results/models/trauma_former_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Trauma-Former checkpoint saved → {ckpt_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train all models and generate Table 2")
    ap.add_argument("--data",    default="data/development_set.npz")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--folds",   type=int, default=5)
    ap.add_argument("--output",  default="results/table2_all_models.csv")
    ap.add_argument("--quick",   action="store_true",
                    help="1-fold only (reviewer smoke test, ~10 min)")
    args = ap.parse_args()

    n_folds = 1 if args.quick else args.folds
    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)

    if not os.path.exists(args.data):
        logger.info("Dataset not found. Generating …")
        import subprocess
        subprocess.run([sys.executable, "data/generate_datasets.py"], check=True)

    all_results = []
    t_total = time.perf_counter()

    for model_name, cfg_path in MODELS:
        logger.info(f"\n{'='*60}\nTraining {model_name} …\n{'='*60}")
        t0 = time.perf_counter()
        try:
            result = run_cv(
                config_path=cfg_path,
                model_name=model_name,
                data_path=args.data,
                n_folds=n_folds,
                seed=args.seed,
            )
            result["elapsed_sec"] = round(time.perf_counter() - t0, 1)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  {model_name} failed: {e}")

    elapsed = time.perf_counter() - t_total
    logger.info(f"\nTotal wall-clock time: {elapsed/60:.1f} min")

    # Save Table 2
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    logger.info(f"\nTable 2 saved → {args.output}")

    # ── Re-train Trauma-Former on the FULL development set and save checkpoint ──
    # Required by experiments/run_test_set.py (BUG-7 fix).
    logger.info("\nRe-training Trauma-Former on full development set for checkpoint …")
    _save_best_checkpoint(args.data, args.seed)

    # Print summary
    print("\n── Table 2 Summary (development set, 5-fold CV) ──────────────")
    print(f"{'Model':<20} {'AUROC':>8} {'95% CI':>16} {'MCSE':>7} {'AUPRC':>7} {'PPV':>7} {'Brier':>7}")
    print("-" * 76)
    for r in all_results:
        ci = f"{r.get('auroc_ci_lo', 0):.3f}–{r.get('auroc_ci_hi', 0):.3f}"
        print(f"{r['model']:<20} {r['auroc_mean']:>8.3f} {ci:>16} "
              f"{r.get('auroc_mcse', 0):>7.4f} {r.get('auprc_mean', 0):>7.3f} "
              f"{r.get('ppv_mean', 0):>7.3f} {r.get('brier_mean', 0):>7.3f}")


if __name__ == "__main__":
    main()
