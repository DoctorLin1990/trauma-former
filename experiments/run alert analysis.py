#!/usr/bin/env python3
"""
Analyze alert rule: threshold and persistence requirements.
Generates early warning time and false positive rate for various settings.
Produces data for Figure S2.
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
from models import TraumaFormer
from training.train_cv import get_model
from training.utils import setup_logger, set_seed
from evaluation.alert_rule import compute_early_warning_time, optimize_alert_rule

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Alert rule analysis')
    parser.add_argument('--config', type=str, default='configs/trauma_former.yaml', help='Model config')
    parser.add_argument('--model', type=str, default='trauma-former', help='Model name (only Trauma-Former supported)')
    parser.add_argument('--data', type=str, default='data/development_set.npz', help='Data path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/alert_analysis.csv', help='Output CSV')
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config and data
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    loaded = np.load(args.data)
    data = loaded['data']
    labels = loaded['labels']

    # Fit normalizer
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

    # Train model (or load pre-trained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, config, device)
    # For alert analysis, we need probability over time for each patient episode.
    # We'll generate predictions for each time window (sliding window) to form a probability series per patient.
    # This requires processing each patient separately.
    # For simplicity, we'll train a model on the full dataset (or load a checkpoint) and then generate probabilities.

    # Train quickly (or load pre-trained). Here we assume we have a pre-trained model saved.
    # For demonstration, we'll just use a placeholder: load best model from previous CV run.
    # In practice, you would train on full dataset and save.
    # We'll assume a checkpoint exists at ./results/models/full_train/best_model.pth
    checkpoint_path = './results/models/full_train/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Loaded pre-trained model.")
    else:
        logger.warning("No pre-trained model found; training from scratch (may take time).")
        from training.trainer import Trainer
        train_loader = DataLoader(full_dataset, batch_size=config['training']['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        trainer = Trainer(
            model=model,
            device=device,
            config=config['training'],
            experiment_dir='./results/models/full_train',
            use_amp=False
        )
        trainer.fit(train_loader, train_loader, config['training']['max_epochs'], fold=0)
        trainer.load_best_model()

    # Generate probability series for each patient
    # For each patient, we need the 30-minute series of probabilities (one per second).
    # Since model uses 60-second windows, we can slide window and get probability at each second (using last window's output).
    # For simplicity, we'll compute for each patient all windows and align with time.
    n_patients = len(data)
    prob_series_list = []
    labels_list = []
    window_size = config['model'].get('window_length', 60)

    model.eval()
    with torch.no_grad():
        for p_idx in range(n_patients):
            patient_data = data[p_idx]  # (1800, 4)
            # Create windows
            n_windows = 1800 - window_size + 1
            probs = []
            for start in range(n_windows):
                window = patient_data[start:start+window_size, :]
                # Normalize
                window_norm = normalizer.transform(window.reshape(1, -1, 4)).reshape(1, window_size, 4)
                window_t = torch.tensor(window_norm, dtype=torch.float32).to(device)
                # Predict
                output = model(window_t)
                prob = output.item()
                probs.append(prob)
            # Pad to length 1800 by repeating first value? Actually first window gives probability at time window_size-1.
            # We'll create a series of length 1800 by assigning probability at the end of each window.
            prob_series = np.full(1800, np.nan)
            for t in range(n_windows):
                prob_series[t + window_size - 1] = probs[t]
            # Forward fill for initial times
            prob_series[:window_size-1] = prob_series[window_size-1]
            prob_series_list.append(prob_series)
            labels_list.append(labels[p_idx])

    # Optimize alert rule over thresholds and persistences
    thresholds = [0.7, 0.8, 0.9]
    persistences = [1, 2, 3, 4, 5]  # minutes
    results = optimize_alert_rule(prob_series_list, labels_list, thresholds, persistences)

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"Alert analysis saved to {args.output}")

if __name__ == '__main__':
    main()