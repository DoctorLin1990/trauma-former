#!/usr/bin/env python3
"""
Evaluate the best model (trained on full development set) on the independent test set
with 25% TIC prevalence. Generates Table 3.
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
from models import TraumaFormer, LSTMModel, XGBoostModel, PatchTSTModel, InformerModel, ShockIndex
from training.train_cv import get_model
from training.utils import setup_logger, set_seed
from evaluation.metrics import compute_all_metrics

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate on 25% prevalence test set')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--train_data', type=str, default='data/development_set.npz', help='Training data (to fit normalizer)')
    parser.add_argument('--test_data', type=str, default='data/test_set.npz', help='Test data (25% prevalence)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/table3_test_results.csv', help='Output CSV file')
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load training data to fit normalizer
    train_loaded = np.load(args.train_data)
    train_data = train_loaded['data']
    train_labels = train_loaded['labels']

    # Fit normalizer on all training windows
    full_train_dataset = TICDataset(
        data=train_data,
        labels=train_labels,
        window_size=config['model'].get('window_length', 60),
        stride=1,
        apply_preprocessing=True,
        normalizer=None
    )
    # Collect all windows to fit normalizer
    all_train_windows = np.array([full_train_dataset[i][0].numpy() for i in range(len(full_train_dataset))])
    normalizer = ZScoreNormalizer()
    normalizer.fit(all_train_windows)

    # Load test data
    test_loaded = np.load(args.test_data)
    test_data = test_loaded['data']
    test_labels = test_loaded['labels']

    test_dataset = TICDataset(
        data=test_data,
        labels=test_labels,
        window_size=config['model'].get('window_length', 60),
        stride=1,
        apply_preprocessing=True,
        normalizer=normalizer
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Train model on full training data (if needed) or load pre-trained weights
    # For simplicity, we assume we will train from scratch on the full training set.
    # In practice, you might want to load a pre-trained checkpoint.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Special handling for non-PyTorch models
    if args.model.lower() == 'xgboost':
        # Collect features and labels from training set
        X_train = np.array([full_train_dataset[i][0].numpy() for i in range(len(full_train_dataset))])
        y_train = np.array([full_train_dataset[i][2].item() for i in range(len(full_train_dataset))])
        X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
        y_test = test_labels  # patient-level labels? Actually each window has same label.
        # For XGBoost, we use window-level labels (same as patient label).
        y_train_windows = np.repeat(train_labels, 1800-60+1)  # approximate, but dataset returns per-window labels
        # Simpler: use patient labels directly if dataset returns per-patient label; but TICDataset returns per-window label.
        # We'll just use the labels from the dataset (which are per-window and correct).
        from models.baselines.xgboost_model import XGBoostModel
        model = XGBoostModel(
            n_estimators=config['model'].get('n_estimators', 200),
            max_depth=config['model'].get('max_depth', 6),
            learning_rate=config['model'].get('learning_rate', 0.1),
            random_state=args.seed
        )
        logger.info("Training XGBoost on full training set...")
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, preds)

    elif args.model.lower() == 'shock-index':
        # Collect all test windows
        X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
        y_test = np.array([test_dataset[i][2].item() for i in range(len(test_dataset))])
        model = ShockIndex(threshold=1.0)
        si_values = model.predict_proba(X_test)
        # Convert to pseudo-probability for metrics
        pseudo_proba = np.clip((si_values - 0.3) / (2.0 - 0.3), 0, 1)
        metrics = compute_all_metrics(y_test, pseudo_proba)

    else:
        # PyTorch model
        model = get_model(args.model, config, device)
        # Train on full training set
        from training.trainer import Trainer
        # Prepare training loader
        train_loader = DataLoader(full_train_dataset, batch_size=config['training']['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        trainer = Trainer(
            model=model,
            device=device,
            config=config['training'],
            experiment_dir='./results/models/full_train',
            use_amp=False
        )
        logger.info("Training on full development set...")
        trainer.fit(
            train_loader=train_loader,
            val_loader=train_loader,  # dummy, not used for early stopping
            epochs=config['training']['max_epochs'],
            fold=0
        )
        # Load best model (saved during training)
        trainer.load_best_model()

        # Evaluate on test set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, mask, y, _ in test_loader:
                x = x.to(device)
                mask = mask.to(device) if mask is not None else None
                output = model(x, mask)
                all_preds.append(output.cpu().numpy())
                all_labels.append(y.numpy())
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        metrics = compute_all_metrics(all_labels, all_preds)

    # Save metrics
    df = pd.DataFrame([metrics])
    df.to_csv(args.output, index=False)
    logger.info(f"Test set results saved to {args.output}")
    for key, val in metrics.items():
        logger.info(f"{key}: {val:.4f}")

if __name__ == '__main__':
    main()