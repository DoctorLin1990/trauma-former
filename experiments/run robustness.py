#!/usr/bin/env python3
"""
Run robustness tests:
- Gaussian noise at various SNR levels.
- Random missing data (MCAR) at various rates.
- Transient sensor failure (HR dropout).
- Network latency simulation (4G vs 5G).
Generates data for Figure 4.
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from models import TraumaFormer, LSTMModel, XGBoostModel, ShockIndex
from training.train_cv import get_model
from training.utils import setup_logger, set_seed
from evaluation.robustness_tests import test_gaussian_noise, test_random_missing, test_sensor_failure
from evaluation.network_simulation import simulate_network_latency, NETWORK_PROFILES, apply_network_to_batch
from sklearn.metrics import roc_auc_score

logger = setup_logger(__name__)

def evaluate_on_corrupted_data(model, data_loader, device):
    """Helper to compute AUROC on a data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, mask, y, _ in data_loader:
            x = x.to(device)
            mask = mask.to(device) if mask is not None else None
            output = model(x, mask)
            all_preds.append(output.cpu().numpy())
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    return roc_auc_score(all_labels, all_preds)

def main():
    parser = argparse.ArgumentParser(description='Run robustness tests')
    parser.add_argument('--config', type=str, required=True, help='Model config')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--data', type=str, default='data/development_set.npz', help='Data path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/robustness', help='Output directory')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and data
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    loaded = np.load(args.data)
    data = loaded['data']
    labels = loaded['labels']

    # Fit normalizer on all data (for simplicity, use all)
    full_dataset = TICDataset(
        data=data,
        labels=labels,
        window_size=config['model'].get('window_length', 60),
        stride=1,
        apply_preprocessing=True,
        normalizer=None
    )
    all_windows = np.array([full_dataset[i][0].numpy() for i in range(len(full_dataset))])
    normalizer = ZScoreNormalizer()
    normalizer.fit(all_windows)
    full_dataset.normalizer = normalizer

    # Train model on full dataset (or load pre-trained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, config, device)
    if args.model.lower() in ['xgboost', 'shock-index']:
        # Non-PyTorch models: handle separately
        # For simplicity, we'll only run PyTorch models for robustness.
        logger.warning("Robustness tests for XGBoost/ShockIndex not implemented; skipping.")
        return
    else:
        # Train on full dataset (quickly, maybe fewer epochs)
        from training.trainer import Trainer
        train_loader = DataLoader(full_dataset, batch_size=config['training']['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        trainer = Trainer(
            model=model,
            device=device,
            config=config['training'],
            experiment_dir='./results/models/robustness_temp',
            use_amp=False
        )
        logger.info("Training model for robustness tests...")
        trainer.fit(
            train_loader=train_loader,
            val_loader=train_loader,
            epochs=config['training']['max_epochs'] // 2,  # fewer for speed
            fold=0
        )
        trainer.load_best_model()

    # Create a baseline data loader (clean data)
    clean_loader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=4)
    baseline_auroc = evaluate_on_corrupted_data(model, clean_loader, device)
    logger.info(f"Baseline AUROC (clean): {baseline_auroc:.4f}")

    # 1. Gaussian noise test
    snr_levels = [20, 15, 10, 5]
    noise_results = test_gaussian_noise(model, all_windows, labels, snr_levels, n_repeats=3, seed=args.seed)
    df_noise = pd.DataFrame(noise_results)
    df_noise.to_csv(os.path.join(args.output_dir, 'gaussian_noise.csv'), index=False)

    # 2. Random missing test
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    missing_results = test_random_missing(model, all_windows, labels, missing_rates, n_repeats=3, seed=args.seed)
    df_missing = pd.DataFrame(missing_results)
    df_missing.to_csv(os.path.join(args.output_dir, 'random_missing.csv'), index=False)

    # 3. Sensor failure (HR dropout)
    failure_durations = [5, 10, 15, 20, 30]  # seconds
    sensor_results = test_sensor_failure(model, all_windows, labels, sensor_idx=0,
                                         failure_durations=failure_durations,
                                         n_repeats=3, seed=args.seed)
    df_sensor = pd.DataFrame(sensor_results)
    df_sensor.to_csv(os.path.join(args.output_dir, 'sensor_failure.csv'), index=False)

    # 4. Network latency simulation
    # Create a copy of data and apply network impairments
    # For each network profile, compute AUROC after applying packet loss.
    # We'll use a subset of patients for speed.
    n_patients = min(100, data.shape[0])  # use first 100 patients
    data_subset = data[:n_patients]
    labels_subset = labels[:n_patients]

    # Create dataset with normalizer
    subset_dataset = TICDataset(
        data=data_subset,
        labels=labels_subset,
        window_size=config['model'].get('window_length', 60),
        stride=1,
        apply_preprocessing=True,
        normalizer=normalizer
    )

    network_results = []
    for profile_name, profile in NETWORK_PROFILES.items():
        logger.info(f"Simulating {profile_name}...")
        # Apply network impairment to data (add NaNs for lost packets)
        corrupted_data, mask = apply_network_to_batch(data_subset, profile, random_seed=args.seed)
        # Create new dataset with corrupted data
        corrupted_dataset = TICDataset(
            data=corrupted_data,
            labels=labels_subset,
            window_size=config['model'].get('window_length', 60),
            stride=1,
            apply_preprocessing=True,
            normalizer=normalizer
        )
        corrupted_loader = DataLoader(corrupted_dataset, batch_size=128, shuffle=False, num_workers=4)
        auroc = evaluate_on_corrupted_data(model, corrupted_loader, device)
        network_results.append({'profile': profile_name, 'auroc': auroc})
        logger.info(f"{profile_name} AUROC: {auroc:.4f}")

    df_network = pd.DataFrame(network_results)
    df_network.to_csv(os.path.join(args.output_dir, 'network_latency.csv'), index=False)

    logger.info(f"All robustness results saved to {args.output_dir}")

if __name__ == '__main__':
    main()