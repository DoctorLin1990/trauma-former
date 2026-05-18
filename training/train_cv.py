"""
Patient-level 5-fold cross-validation (Section 2.8).

All windows from the same synthetic patient stay in the same fold to
prevent temporal leakage. Performance metrics are averaged across folds;
bootstrap 95 % CI and MCSE are computed over fold-level AUROC estimates.

Supported model names:
    trauma-former | lstm | gru | cnn | xgboost | lr-trend
    patchtst | informer | shock-index

Usage:
    python experiments/run_cv.py --config configs/trauma_former.yaml --model trauma-former
"""
from __future__ import annotations

import os
import sys
import yaml
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# ── project root on path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from evaluation.metrics import compute_all_metrics, monte_carlo_standard_error
from training.trainer import train_model
from training.utils import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────

def _build_model(model_name: str, cfg: dict):
    """Instantiate the requested model from its config dict."""
    name = model_name.lower().replace("_", "-")

    if name == "trauma-former":
        from models.trauma_former import TraumaFormer
        return TraumaFormer(**{k: v for k, v in cfg.get("model", {}).items()
                               if k not in ("name", "architecture",
                                            "input_variables", "total_parameters")})
    if name == "lstm":
        from models.baselines.lstm import LSTMModel
        return LSTMModel()
    if name == "gru":
        from models.baselines.gru import GRUModel
        return GRUModel()
    if name == "cnn":
        from models.baselines.cnn import CNNModel
        return CNNModel()
    if name in ("lr-trend", "lr_trend"):
        from models.baselines.lr_trend import LRTrendModel
        return LRTrendModel()
    if name == "xgboost":
        from models.baselines.xgboost_model import XGBoostModel
        return XGBoostModel()
    if name == "patchtst":
        from models.baselines.patchtst import PatchTSTModel
        return PatchTSTModel()
    if name == "informer":
        from models.baselines.informer import InformerModel
        return InformerModel()
    if name in ("shock-index", "shock_index"):
        from models.baselines.shock_index import ShockIndexModel
        return ShockIndexModel()
    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────
# Sklearn-compatible models (LR-trend, XGBoost, Shock-Index)
# ─────────────────────────────────────────────────────────────────────

_SKLEARN_MODELS = {"lr-trend", "lr_trend", "xgboost", "shock-index", "shock_index"}


def _is_sklearn(name: str) -> bool:
    return name.lower().replace("_", "-") in _SKLEARN_MODELS


# ─────────────────────────────────────────────────────────────────────
# One-fold evaluation
# ─────────────────────────────────────────────────────────────────────

def _run_fold(
    fold_idx: int,
    model_name: str,
    cfg: dict,
    train_data: np.ndarray, train_labels: np.ndarray,
    val_data: np.ndarray,   val_labels: np.ndarray,
    window_size: int = 60,
    seed: int = 42,
) -> dict:
    """Train and evaluate one CV fold. Returns dict of metric values."""

    fold_seed = seed + fold_idx
    name = model_name.lower().replace("_", "-")

    # ---- sklearn-style models ----------------------------------------
    if _is_sklearn(name):
        model = _build_model(model_name, cfg)

        # Fit on last window per patient (patient-level label)
        # For windowed models: use the final 60-s window of each episode
        T = train_data.shape[1]
        X_train = train_data[:, T - window_size:T, :]  # (n, 60, 4)
        X_val   = val_data[:,   T - window_size:T, :]

        model.fit(X_train, train_labels)
        if hasattr(model, "predict_proba_positive"):
            scores = model.predict_proba_positive(X_val)
        else:
            scores = model.predict_proba(X_val)[:, 1]

        y_true = val_labels
        metrics = compute_all_metrics(y_true, scores)
        return metrics

    # ---- PyTorch models ----------------------------------------------
    import torch
    from training.utils import get_device, set_seed
    set_seed(fold_seed)
    device = get_device()

    train_cfg = cfg.get("training", {})
    lr     = float(train_cfg.get("learning_rate", 1e-4))
    wd     = float(train_cfg.get("weight_decay",  1e-4))
    epochs = int(train_cfg.get("max_epochs",      200))
    pat    = int(train_cfg.get("early_stopping_patience", 10))
    batch  = int(train_cfg.get("batch_size",      64))

    # Normaliser fitted on training data only
    normalizer = ZScoreNormalizer()
    normalizer.fit(train_data)

    train_ds = TICDataset(train_data, train_labels, window_size=window_size,
                          stride=30, normalizer=normalizer)
    val_ds   = TICDataset(val_data,   val_labels,   window_size=window_size,
                          stride=30, normalizer=normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    model = _build_model(model_name, cfg)
    model, _ = train_model(model, train_loader, val_loader,
                            learning_rate=lr, weight_decay=wd,
                            max_epochs=epochs, patience=pat, device=device)

    # Collect patient-level predictions: take the max probability per patient
    from evaluation.metrics import compute_all_metrics
    model.eval()
    patient_scores = {}
    patient_labels = {}

    with torch.no_grad():
        for x_b, mask_b, y_b, pid_b in val_loader:
            x_b = x_b.to(device)
            out = model(x_b).squeeze(1).cpu().numpy()
            for i, pid in enumerate(pid_b.numpy()):
                # Track max over all windows for this patient
                patient_scores[pid] = max(patient_scores.get(pid, 0.0), float(out[i]))
                patient_labels[pid] = int(y_b[i].item())

    pids   = sorted(patient_scores)
    y_true = np.array([patient_labels[p] for p in pids])
    y_sc   = np.array([patient_scores[p] for p in pids])
    return compute_all_metrics(y_true, y_sc)


# ─────────────────────────────────────────────────────────────────────
# Main CV runner
# ─────────────────────────────────────────────────────────────────────

def run_cv(
    config_path: str,
    model_name:  str,
    data_path:   str,
    n_folds:     int = 5,
    seed:        int = 42,
    window_size: int = 60,
) -> dict:
    """
    Run patient-level n-fold CV and return aggregated metrics.

    Returns a dict with keys: auroc_mean, auroc_ci_lo, auroc_ci_hi, auroc_mcse,
    auprc_mean, brier_mean, sensitivity_mean, specificity_mean, ppv_mean,
    npv_mean, f1_mean, calibration_slope_mean, calibration_intercept_mean.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    loaded      = np.load(data_path)
    data        = loaded["data"]    # (N, 1800, 4)
    labels      = loaded["labels"]  # (N,)
    patient_ids = np.arange(len(labels))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        logger.info(f"── Fold {fold_idx + 1}/{n_folds} — "
                    f"train={len(tr_idx)}, val={len(val_idx)} ──")
        metrics = _run_fold(
            fold_idx,
            model_name,
            cfg,
            data[tr_idx],   labels[tr_idx],
            data[val_idx],  labels[val_idx],
            window_size=window_size,
            seed=seed,
        )
        fold_metrics.append(metrics)
        logger.info(f"  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
                    f"PPV={metrics['ppv']:.3f}  Brier={metrics['brier']:.4f}")

    # ── Aggregate ──
    def _agg(key: str) -> tuple[float, float, float]:
        vals = [m[key] for m in fold_metrics if key in m]
        return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    auroc_vals = [m["auroc"] for m in fold_metrics]

    # Bootstrap 95 % CI on mean AUROC (1 000 iterations, patient-level)
    rng = np.random.default_rng(seed)
    boot_aurocs = [
        float(np.mean(rng.choice(auroc_vals, size=n_folds, replace=True)))
        for _ in range(1000)
    ]
    ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])

    result = {
        "model":                   model_name,
        "auroc_mean":              float(np.mean(auroc_vals)),
        "auroc_ci_lo":             float(ci_lo),
        "auroc_ci_hi":             float(ci_hi),
        "auroc_mcse":              monte_carlo_standard_error(auroc_vals),
        **{f"{k}_mean": _agg(k)[0] for k in
           ["auprc", "brier", "sensitivity", "specificity",
            "ppv", "npv", "f1",
            "calibration_slope", "calibration_intercept"]},
    }

    logger.info(
        f"\n{'='*60}\nCV Results ({model_name}) — {n_folds} folds\n"
        f"  AUROC : {result['auroc_mean']:.4f} "
        f"(95% CI {result['auroc_ci_lo']:.3f}–{result['auroc_ci_hi']:.3f}, "
        f"MCSE={result['auroc_mcse']:.4f})\n"
        f"  AUPRC : {result['auprc_mean']:.4f}\n"
        f"  PPV   : {result['ppv_mean']:.4f}\n"
        f"  Brier : {result['brier_mean']:.4f}\n"
        f"{'='*60}"
    )
    return result
