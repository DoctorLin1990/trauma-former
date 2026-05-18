#!/usr/bin/env python3
"""
Ablation studies for Trauma-Former (Section 3.7 / Supplementary Table S2.7).

Tests:
  1. Standard time-step tokenization (no variable inversion)  → expected AUROC ~0.893
  2. MLP-only classifier (no self-attention)                  → expected AUROC ~0.837
  3. Single encoder layer (L=1)                               → expected AUROC ~0.918
  4. Four encoder layers (L=4)                                → expected AUROC ~0.931
  5. Window length T=30 s                                     → expected AUROC ~0.904
  6. Window length T=120 s                                    → expected AUROC ~0.932

Usage:
    python experiments/run_ablation.py --data data/development_set.npz --seed 42
"""
from __future__ import annotations

import os, sys, argparse, csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trauma_former import TraumaFormer
from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from evaluation.metrics import compute_auroc, monte_carlo_standard_error
from training.trainer import train_model
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)

N_FOLDS    = 5
MAX_EPOCHS = 200
PATIENCE   = 10
BATCH      = 64
LR         = 1e-4
WD         = 1e-4


# ── Ablation model variants ───────────────────────────────────────────

class StandardTokenTraumaFormer(nn.Module):
    """
    Ablation: standard time-step tokenization (no variable inversion).
    Attention applied across T time steps rather than N variables.
    """
    def __init__(self, input_dim=4, window_size=60, d_model=256,
                 n_heads=4, n_layers=2, d_ff=512, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)   # project N → d_model at each t
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * window_size, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # x: (B, T, N) → project to (B, T, d_model) → attend over T
        h   = self.input_proj(x)
        enc = self.encoder(h)
        return self.classifier(enc.flatten(start_dim=1))


class MLPOnlyTraumaFormer(nn.Module):
    """
    Ablation: MLP-only classifier (no self-attention).
    """
    def __init__(self, input_dim=4, window_size=60, d_model=256,
                 n_heads=4, n_layers=2, d_ff=512, dropout=0.2):
        super().__init__()
        flat = input_dim * window_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),    nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        return self.classifier(x)


def _build_ablation(name: str, window_size: int = 60) -> nn.Module:
    cfg = TraumaFormer.get_default_config()
    if name == "full":
        return TraumaFormer(window_size=window_size, **cfg)
    if name == "std_token":
        return StandardTokenTraumaFormer(window_size=window_size)
    if name == "mlp_only":
        return MLPOnlyTraumaFormer(window_size=window_size)
    if name == "L1":
        return TraumaFormer(window_size=window_size, n_layers=1, **{k:v for k,v in cfg.items() if k!="n_layers"})
    if name == "L4":
        return TraumaFormer(window_size=window_size, n_layers=4, **{k:v for k,v in cfg.items() if k!="n_layers"})
    raise ValueError(f"Unknown ablation: {name}")


def run_ablation_cv(
    ablation_name: str,
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 60,
    seed: int = 42,
) -> dict:
    device = get_device()
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aurocs = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        set_seed(seed + fold_i)
        norm = ZScoreNormalizer()
        norm.fit(data[tr_idx])

        tr_ds  = TICDataset(data[tr_idx], labels[tr_idx], window_size=window_size, stride=30, normalizer=norm)
        val_ds = TICDataset(data[val_idx], labels[val_idx], window_size=window_size, stride=30, normalizer=norm)
        tr_ld  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True,  num_workers=0)
        val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

        model = _build_ablation(ablation_name, window_size=window_size)
        model, _ = train_model(model, tr_ld, val_ld, learning_rate=LR,
                                weight_decay=WD, max_epochs=MAX_EPOCHS,
                                patience=PATIENCE, device=device)

        model.eval()
        scores, ys = [], []
        with torch.no_grad():
            for x_b, _, y_b, _ in val_ld:
                scores.extend(model(x_b.to(device)).squeeze(1).cpu().tolist())
                ys.extend(y_b.tolist())
        aurocs.append(compute_auroc(np.array(ys, dtype=np.int32),
                                    np.array(scores, dtype=np.float32)))

    return {
        "ablation":    ablation_name,
        "window_size": window_size,
        "auroc_mean":  float(np.mean(aurocs)),
        "auroc_mcse":  monte_carlo_standard_error(aurocs),
        "delta_from_full": float(np.mean(aurocs)) - 0.939,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/development_set.npz")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--output", default="results/ablation_results.csv")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    loaded = np.load(args.data)
    data, labels = loaded["data"], loaded["labels"]

    ablation_configs = [
        ("full",       60),    # baseline
        ("std_token",  60),    # no inversion
        ("mlp_only",   60),    # no attention
        ("L1",         60),    # 1 encoder layer
        ("L4",         60),    # 4 encoder layers
        ("full",       30),    # 30-s window
        ("full",       120),   # 120-s window
    ]

    results = []
    for name, win in ablation_configs:
        label = name if win == 60 else f"{name}_T{win}"
        logger.info(f"Running ablation: {label} (window={win}s) …")
        r = run_ablation_cv(name, data, labels, window_size=win, seed=args.seed)
        r["label"] = label
        results.append(r)
        logger.info(f"  AUROC={r['auroc_mean']:.4f}  Δ={r['delta_from_full']:+.4f}")

    # Print table
    print("\n── Ablation Results (Supplementary Table S2.7) ──")
    print(f"{'Configuration':<30} {'AUROC':>8} {'Δ AUROC':>10}")
    print("-" * 52)
    for r in results:
        print(f"{r['label']:<30} {r['auroc_mean']:>8.3f} {r['delta_from_full']:>+10.3f}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
