#!/usr/bin/env python3
"""
Run patient-level 5-fold cross-validation on the development set.
Generates results for Table 1 (baseline characteristics) and Table 2 (model performance).
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from training.train_cv import run_cv
from training.utils import setup_logger, set_seed

logger = setup_logger(__name__)

def compute_baseline_stats(data_path: str):
    """
    Compute baseline characteristics (first 60 seconds) for Table 1.
    Returns a dictionary of means ± std for control and TIC groups.
    """
    loaded = np.load(data_path)
    data = loaded['data']  # (n_patients, 1800, 4)
    labels = loaded['labels']

    # Extract first 60 seconds (index 0:60)
    baseline = data[:, :60, :]  # (n_patients, 60, 4)

    # Separate by label
    ctrl_mask = labels == 0
    tic_mask = labels == 1

    ctrl_data = baseline[ctrl_mask].reshape(-1, 4)  # all time steps from control patients
    tic_data = baseline[tic_mask].reshape(-1, 4)

    stats = {}
    var_names = ['HR', 'SBP', 'DBP', 'SpO2']
    for i, name in enumerate(var_names):
        ctrl_mean = np.mean(ctrl_data[:, i])
        ctrl_std = np.std(ctrl_data[:, i])
        tic_mean = np.mean(tic_data[:, i])
        tic_std = np.std(tic_data[:, i])
        stats[f'{name}_ctrl'] = f"{ctrl_mean:.1f}±{ctrl_std:.1f}"
        stats[f'{name}_tic'] = f"{tic_mean:.1f}±{tic_std:.1f}"

    # Pulse pressure (SBP - DBP)
    ctrl_pp = ctrl_data[:, 1] - ctrl_data[:, 2]
    tic_pp = tic_data[:, 1] - tic_data[:, 2]
    stats['PP_ctrl'] = f"{np.mean(ctrl_pp):.1f}±{np.std(ctrl_pp):.1f}"
    stats['PP_tic'] = f"{np.mean(tic_pp):.1f}±{np.std(tic_pp):.1f}"

    # Shock index (HR/SBP)
    ctrl_si = ctrl_data[:, 0] / ctrl_data[:, 1]
    tic_si = tic_data[:, 0] / tic_data[:, 1]
    stats['SI_ctrl'] = f"{np.mean(ctrl_si):.2f}±{np.std(ctrl_si):.2f}"
    stats['SI_tic'] = f"{np.mean(tic_si):.2f}±{np.std(tic_si):.2f}"

    return stats

def main():
    parser = argparse.ArgumentParser(description='Run 5-fold cross-validation')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., trauma-former, lstm, xgboost, etc.)')
    parser.add_argument('--data', type=str, default='data/development_set.npz', help='Path to development dataset')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/cv_results.csv', help='Output CSV file for metrics')
    args = parser.parse_args()

    set_seed(args.seed)

    # Compute baseline characteristics (Table 1)
    logger.info("Computing baseline characteristics for Table 1...")
    baseline_stats = compute_baseline_stats(args.data)
    for key, val in baseline_stats.items():
        logger.info(f"{key}: {val}")
    # Optionally save to file
    pd.Series(baseline_stats).to_csv('results/table1_baseline.csv')

    # Run cross-validation
    logger.info(f"Running {args.folds}-fold CV for model {args.model}...")
    cv_results = run_cv(
        config_path=args.config,
        model_name=args.model,
        data_path=args.data,
        n_folds=args.folds,
        seed=args.seed
    )

    # Convert to DataFrame and save
    df = pd.DataFrame([cv_results])
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()