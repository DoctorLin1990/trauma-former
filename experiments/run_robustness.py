#!/usr/bin/env python3
"""
Robustness tests for Trauma-Former (Section 3.5 / Figure 4).

Tests:
  B. AUROC vs SNR (Gaussian noise: 20 → 5 dB)
  C. AUROC vs missing rate (MCAR: 0% → 50%)
  D. AUROC vs HR-sensor dropout duration (0 → 30 s)

Each condition repeated 5 times with different random seeds.

Usage:
    python experiments/run_robustness.py \
        --model_path results/models/trauma_former_best.pt \
        --dev_data   data/development_set.npz \
        --seed 42
"""
from __future__ import annotations

import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trauma_former import TraumaFormer
from models.baselines.lstm import LSTMModel
from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from evaluation.metrics import compute_auroc
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)

WINDOW  = 60
BATCH   = 128
N_REPS  = 5   # repetitions per condition


def _add_gaussian_noise(data: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at specified SNR (dB) to all channels."""
    noisy = data.copy()
    signal_power = np.mean(data ** 2, axis=(1, 2), keepdims=True)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise_std    = np.sqrt(noise_power)
    noisy += np.random.randn(*data.shape).astype(np.float32) * noise_std
    return noisy


def _apply_mcar(data: np.ndarray, missing_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-out random time steps (MCAR) and return masked copy."""
    masked = data.copy()
    n, T, C = masked.shape
    n_miss = max(1, int(T * missing_rate))
    for i in range(n):
        for c in range(C):
            idx = rng.choice(T, size=n_miss, replace=False)
            masked[i, idx, c] = 0.0
    return masked


def _hr_sensor_dropout(data: np.ndarray, dropout_sec: int) -> np.ndarray:
    """Zero out last `dropout_sec` seconds of HR channel (column 0)."""
    dropped = data.copy()
    if dropout_sec > 0:
        dropped[:, -dropout_sec:, 0] = 0.0
    return dropped


@torch.no_grad()
def _get_patient_auroc(model, data, labels, norm, device):
    ds = TICDataset(data, labels, window_size=WINDOW, stride=30, normalizer=norm)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    model.eval()
    patient_scores: dict[int, float] = {}
    patient_labels: dict[int, int]   = {}
    for x_b, _, y_b, pid_b in dl:
        out = model(x_b.to(device)).squeeze(1).cpu().numpy()
        for i, pid in enumerate(pid_b.numpy()):
            patient_scores[pid] = max(patient_scores.get(pid, 0.0), float(out[i]))
            patient_labels[pid] = int(y_b[i].item())
    pids   = sorted(patient_scores)
    y_true = np.array([patient_labels[p] for p in pids])
    y_sc   = np.array([patient_scores[p] for p in pids])
    return compute_auroc(y_true, y_sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="results/models/trauma_former_best.pt")
    ap.add_argument("--dev_data",   default="data/development_set.npz")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output",     default="results/robustness_results.json")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    dev  = np.load(args.dev_data)
    data, labels = dev["data"], dev["labels"]

    norm = ZScoreNormalizer()
    norm.fit(data)

    # Load Trauma-Former
    tf = TraumaFormer(**TraumaFormer.get_default_config()).to(device)
    tf.load_state_dict(torch.load(args.model_path, map_location=device))

    # Load LSTM for comparison
    lstm_path = args.model_path.replace("trauma_former", "lstm")
    lstm = LSTMModel().to(device)
    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    else:
        logger.warning(f"LSTM model not found at {lstm_path}; skipping LSTM curves.")
        lstm = None

    results = {"snr": {}, "missing": {}, "hr_dropout": {}}

    # ── B: Gaussian noise ────────────────────────────────────────────
    snr_levels = [20, 15, 10, 7, 5]
    for snr in snr_levels:
        aurocs_tf, aurocs_lstm = [], []
        for rep in range(N_REPS):
            np.random.seed(args.seed + rep)
            d_noisy = _add_gaussian_noise(data, snr)
            aurocs_tf.append(_get_patient_auroc(tf, d_noisy, labels, norm, device))
            if lstm:
                aurocs_lstm.append(_get_patient_auroc(lstm, d_noisy, labels, norm, device))
        results["snr"][snr] = {
            "tf_mean": float(np.mean(aurocs_tf)), "tf_std": float(np.std(aurocs_tf)),
            "lstm_mean": float(np.mean(aurocs_lstm)) if lstm else None,
        }
        logger.info(f"SNR={snr:>2} dB: TF={results['snr'][snr]['tf_mean']:.4f}")

    # ── C: Random missing data (MCAR) ────────────────────────────────
    miss_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    for rate in miss_rates:
        aurocs_tf = []
        for rep in range(N_REPS):
            rng    = np.random.default_rng(args.seed + rep)
            d_miss = _apply_mcar(data, rate, rng)
            aurocs_tf.append(_get_patient_auroc(tf, d_miss, labels, norm, device))
        results["missing"][rate] = {
            "tf_mean": float(np.mean(aurocs_tf)), "tf_std": float(np.std(aurocs_tf)),
        }
        logger.info(f"Missing={rate:.0%}: TF={results['missing'][rate]['tf_mean']:.4f}")

    # ── D: HR sensor dropout ──────────────────────────────────────────
    dropout_secs = [0, 5, 10, 15, 20, 30]
    for sec in dropout_secs:
        d_drop = _hr_sensor_dropout(data, sec)
        auroc  = _get_patient_auroc(tf, d_drop, labels, norm, device)
        results["hr_dropout"][sec] = {"tf_auroc": float(auroc)}
        logger.info(f"HR dropout={sec:>2} s: TF={auroc:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
