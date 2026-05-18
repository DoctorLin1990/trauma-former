#!/usr/bin/env python3
"""
Binary Missingness Indicator Sensitivity Analysis (Section 2.6 / Figure S2).

Reproduces the experiment described in paper Section 2.6 and Section 3.3:
  "Adding binary missingness indicators yielded an AUROC of 0.926 (95% CI
   0.91–0.94) and a PPV of 0.50, compared with 0.931 and 0.48 under standard
   masking alone."

Design:
  Under the 25% prevalence test set with 30% MCAR random missingness,
  binary missingness indicators (1 if missing, 0 if observed) are appended
  as additional features to the input tensor before the masking step,
  allowing the model to exploit signal-absence as a prognostic cue.

This module was MISSING from the original repository (BUG-8).

Usage:
    python experiments/missingness_indicator_analysis.py \\
        --model_path results/models/trauma_former_best.pt \\
        --test_data  data/test_set.npz \\
        --dev_data   data/development_set.npz \\
        --missing_rate 0.30 \\
        --seed 42
"""
from __future__ import annotations

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import ZScoreNormalizer
from evaluation.metrics import compute_all_metrics, bootstrap_ci, compute_auroc
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)

WINDOW = 60
BATCH  = 64


# ─────────────────────────────────────────────────────────────────────────────
# Extended TraumaFormer accepting 8-channel input (4 vitals + 4 indicators)
# ─────────────────────────────────────────────────────────────────────────────

class TraumaFormerWithIndicators(nn.Module):
    """
    iTransformer that accepts N=8 channels:
        channels 0-3  : [HR, SBP, DBP, SpO2]  (zero-padded for missing)
        channels 4-7  : binary missingness indicators (1=missing, 0=observed)

    The architecture is identical to TraumaFormer except input_dim=8
    and window_size is the same 60 s.
    """

    def __init__(
        self,
        input_dim: int   = 8,   # 4 vitals + 4 indicators
        window_size: int = 60,
        d_model: int     = 256,
        n_heads: int     = 4,
        n_layers: int    = 2,
        d_ff: int        = 512,
        dropout: float   = 0.2,
        classifier_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(window_size, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * input_dim, classifier_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
            nn.Sigmoid(),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: (B, T, N=8) → transpose → (B, N=8, T=60)
        x   = x.transpose(1, 2)
        tok = self.input_proj(x)          # (B, N, d_model)
        enc = self.encoder(tok)           # (B, N, d_model)
        return self.classifier(enc.flatten(start_dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with missingness indicators
# ─────────────────────────────────────────────────────────────────────────────

class TICDatasetWithIndicators(Dataset):
    """
    Sliding-window dataset that:
      1. Introduces MCAR missingness.
      2. Creates binary indicators (1 = missing, 0 = observed).
      3. Zero-pads missing values.
      4. Returns concatenated tensor [zero-padded vitals | indicators].
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int = 60,
        stride: int = 30,
        normalizer: ZScoreNormalizer = None,
        missing_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.data         = data
        self.labels       = labels
        self.window_size  = window_size
        self.stride       = stride
        self.normalizer   = normalizer
        self.missing_rate = missing_rate
        self.rng          = np.random.default_rng(seed)

        self.indices = [
            (p, s)
            for p in range(len(data))
            for s in range(0, data.shape[1] - window_size + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        p_idx, start = self.indices[idx]
        window = self.data[p_idx, start:start + self.window_size, :].copy()

        # --- Introduce MCAR missingness ---
        if self.missing_rate > 0:
            miss_mask = self.rng.random(window.shape) < self.missing_rate
        else:
            miss_mask = np.zeros(window.shape, dtype=bool)

        # Binary indicators: 1 = missing
        indicators = miss_mask.astype(np.float32)

        # Zero-pad missing positions
        window_padded = window.copy()
        window_padded[miss_mask] = 0.0

        # Normalise (using training-set statistics)
        if self.normalizer is not None:
            orig_shape = window_padded.shape
            window_padded = self.normalizer.transform(
                window_padded.reshape(-1, orig_shape[-1])
            ).reshape(orig_shape)
            # Keep indicators un-normalised (already 0/1)

        # Concatenate along feature axis: (T, 8)
        combined = np.concatenate([window_padded, indicators], axis=-1)

        x = torch.tensor(combined,                dtype=torch.float32)
        y = torch.tensor(self.labels[p_idx],      dtype=torch.float32)
        return x, y, p_idx


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_model_with_indicators(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-4,
    wd: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 10,
) -> nn.Module:
    import copy
    from sklearn.metrics import roc_auc_score

    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.BCELoss()

    best_auroc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_b, y_b, _ in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device).unsqueeze(1)
            opt.zero_grad()
            crit(model(x_b), y_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        all_s, all_y = [], []
        with torch.no_grad():
            for x_b, y_b, _ in val_loader:
                all_s.extend(model(x_b.to(device)).squeeze(1).cpu().tolist())
                all_y.extend(y_b.tolist())
        auroc = roc_auc_score(np.array(all_y), np.array(all_s))

        if auroc > best_auroc + 1e-6:
            best_auroc = auroc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best AUROC={best_auroc:.4f})")
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_patient_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores_dict: dict[int, float] = {}
    labels_dict: dict[int, int]   = {}
    for x_b, y_b, pid_b in loader:
        out = model(x_b.to(device)).squeeze(1).cpu().numpy()
        for i, pid in enumerate(pid_b.numpy()):
            scores_dict[int(pid)] = max(scores_dict.get(int(pid), 0.0), float(out[i]))
            labels_dict[int(pid)] = int(y_b[i].item())
    pids   = sorted(scores_dict)
    y_true = np.array([labels_dict[p] for p in pids])
    y_sc   = np.array([scores_dict[p] for p in pids])
    return y_true, y_sc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Section 2.6 missingness indicator sensitivity analysis"
    )
    ap.add_argument("--model_path",   default="results/models/trauma_former_best.pt")
    ap.add_argument("--test_data",    default="data/test_set.npz")
    ap.add_argument("--dev_data",     default="data/development_set.npz")
    ap.add_argument("--missing_rate", type=float, default=0.30)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--output",       default="results/missingness_indicator_results.json")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    # --- Load data ---
    dev  = np.load(args.dev_data)
    test = np.load(args.test_data)

    norm = ZScoreNormalizer()
    norm.fit(dev["data"])

    # --- Experiment 1: standard masking (baseline, from run_test_set.py) ---
    # This just loads the already-trained Trauma-Former results as reference;
    # we re-evaluate on the test set with MCAR missingness (standard masking).
    from models.trauma_former import TraumaFormer
    from data.dataset import TICDataset

    tf_base = TraumaFormer(**TraumaFormer.get_default_config()).to(device)
    tf_base.load_state_dict(torch.load(args.model_path, map_location=device))
    tf_base.eval()

    # Apply MCAR missing (zero-pad) to test set, evaluate standard masking
    rng = np.random.default_rng(args.seed)
    test_data_miss = test["data"].copy().astype(np.float32)
    miss_mask_all  = rng.random(test_data_miss.shape) < args.missing_rate
    test_data_miss[miss_mask_all] = 0.0

    ds_std = TICDataset(test_data_miss, test["labels"],
                        window_size=WINDOW, stride=30, normalizer=norm)
    dl_std = DataLoader(ds_std, batch_size=BATCH, shuffle=False, num_workers=0)

    patient_scores_std: dict[int, float] = {}
    patient_labels_std: dict[int, int]   = {}
    with torch.no_grad():
        for x_b, _, y_b, pid_b in dl_std:
            out = tf_base(x_b.to(device)).squeeze(1).cpu().numpy()
            for i, pid in enumerate(pid_b.numpy()):
                patient_scores_std[int(pid)] = max(
                    patient_scores_std.get(int(pid), 0.0), float(out[i])
                )
                patient_labels_std[int(pid)] = int(y_b[i].item())
    pids = sorted(patient_scores_std)
    yt_std = np.array([patient_labels_std[p] for p in pids])
    ys_std = np.array([patient_scores_std[p] for p in pids])
    m_std  = compute_all_metrics(yt_std, ys_std)
    ci_lo_std, ci_hi_std = bootstrap_ci(yt_std, ys_std, n_iter=1000, seed=args.seed)
    logger.info(
        f"Standard masking:         AUROC={m_std['auroc']:.3f} "
        f"(95% CI {ci_lo_std:.3f}–{ci_hi_std:.3f})  PPV={m_std['ppv']:.3f}"
    )

    # --- Experiment 2: with binary missingness indicators ---
    # Re-train from scratch on dev set + indicators, then evaluate on test set + indicators.
    logger.info("\nTraining Trauma-Former WITH missingness indicators (8-channel input) …")

    ds_tr = TICDatasetWithIndicators(
        dev["data"], dev["labels"],
        window_size=WINDOW, stride=30, normalizer=norm,
        missing_rate=args.missing_rate, seed=args.seed,
    )
    ds_te = TICDatasetWithIndicators(
        test["data"], test["labels"],
        window_size=WINDOW, stride=30, normalizer=norm,
        missing_rate=args.missing_rate, seed=args.seed + 1,
    )
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=0)

    tf_ind = TraumaFormerWithIndicators(input_dim=8, window_size=WINDOW)
    tf_ind = train_model_with_indicators(tf_ind, dl_tr, dl_tr, device)  # train on dev

    yt_ind, ys_ind = evaluate_patient_level(tf_ind, dl_te, device)
    m_ind  = compute_all_metrics(yt_ind, ys_ind)
    ci_lo_ind, ci_hi_ind = bootstrap_ci(yt_ind, ys_ind, n_iter=1000, seed=args.seed)
    logger.info(
        f"With missingness indicators: AUROC={m_ind['auroc']:.3f} "
        f"(95% CI {ci_lo_ind:.3f}–{ci_hi_ind:.3f})  PPV={m_ind['ppv']:.3f}"
    )

    # --- Summary (Table in Section 3.3 / Supplementary Figure S2) ---
    print("\n── Section 2.6 Missingness Indicator Sensitivity Analysis ──")
    print(f"{'Condition':<30} {'AUROC':>8} {'95% CI':>16} {'PPV':>7}")
    print("-" * 65)
    print(f"{'Standard masking':<30} {m_std['auroc']:>8.3f} "
          f"{ci_lo_std:.3f}–{ci_hi_std:.3f}    {m_std['ppv']:>7.3f}")
    print(f"{'With missingness indicators':<30} {m_ind['auroc']:>8.3f} "
          f"{ci_lo_ind:.3f}–{ci_hi_ind:.3f}    {m_ind['ppv']:>7.3f}")
    ppv_delta = m_ind['ppv'] - m_std['ppv']
    print(f"\n  Marginal PPV improvement: {ppv_delta:+.3f} "
          f"(paper reports +0.02)")

    results = {
        "standard_masking": {**m_std, "auroc_ci_lo": ci_lo_std, "auroc_ci_hi": ci_hi_std},
        "with_indicators":  {**m_ind, "auroc_ci_lo": ci_lo_ind, "auroc_ci_hi": ci_hi_ind},
        "ppv_delta": float(ppv_delta),
        "missing_rate": args.missing_rate,
        "seed": args.seed,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
