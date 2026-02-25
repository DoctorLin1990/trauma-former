"""
Optuna-based hyperparameter search for Trauma-Former.
Searches over d_model, n_layers, dropout, learning rate.
"""
import os
import sys
import yaml
import argparse
import numpy as np
import optuna
from optuna.trial import Trial
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from models.trauma_former import TraumaFormer
from training.trainer import Trainer
from training.utils import set_seed, setup_logger

logger = setup_logger(__name__)

def objective(trial: Trial, data_path: str, n_folds: int = 3, seed: int = 42) -> float:
    """
    Optuna objective: returns mean validation AUROC across folds.
    Uses a subset of folds to speed up search.
    """
    set_seed(seed)

    # Hyperparameter search space
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    n_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    # Fixed parameters
    window_size = 60
    n_heads = 4
    d_ff = d_model * 2
    classifier_hidden = 128

    # Load data
    loaded = np.load(data_path)
    data = loaded['data']
    labels = loaded['labels']

    full_dataset = TICDataset(
        data=data,
        labels=labels,
        window_size=window_size,
        stride=1,
        apply_preprocessing=True,
        normalizer=None
    )

    patient_ids = np.arange(len(data))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_aurocs = []

    for fold, (train_patients, val_patients) in enumerate(kf.split(patient_ids)):
        train_indices = [i for i, (_, _, _, pid) in enumerate(full_dataset) if pid in train_patients]
        val_indices = [i for i, (_, _, _, pid) in enumerate(full_dataset) if pid in val_patients]

        # Fit normalizer on training windows
        train_windows = np.array([full_dataset[i][0].numpy() for i in train_indices])
        normalizer = ZScoreNormalizer()
        normalizer.fit(train_windows)
        full_dataset.normalizer = normalizer

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TraumaFormer(
            input_dim=4,
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            classifier_hidden=classifier_hidden,
            activation='gelu'
        ).to(device)

        trainer = Trainer(
            model=model,
            device=device,
            config={
                'learning_rate': lr,
                'weight_decay': 0.01,
                'early_stopping_patience': 5,  # smaller patience for speed
                'use_scheduler': False
            },
            experiment_dir=f'./results/hparam_search/trial_{trial.number}_fold{fold}',
            use_amp=False
        )

        best_auroc = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,  # fewer epochs for speed
            fold=fold
        )
        fold_aurocs.append(best_auroc)

        # Clean up to save disk space
        import shutil
        shutil.rmtree(f'./results/hparam_search/trial_{trial.number}_fold{fold}', ignore_errors=True)

    mean_auroc = np.mean(fold_aurocs)
    return mean_auroc

def run_hyperparameter_search(data_path: str, n_trials: int = 50, n_folds: int = 3, seed: int = 42):
    """Run Optuna hyperparameter search."""
    set_seed(seed)
    study = optuna.create_study(direction='maximize', study_name='trauma_former_hparam')
    study.optimize(
        lambda trial: objective(trial, data_path, n_folds, seed),
        n_trials=n_trials,
        show_progress_bar=True
    )

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (AUROC): {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save best params to YAML
    best_config = {
        'model': {
            'embedding_dim': trial.params['d_model'],
            'num_encoder_layers': trial.params['n_layers'],
            'dropout': trial.params['dropout'],
            'num_attention_heads': 4,  # fixed
            'feedforward_dim': trial.params['d_model'] * 2,
            'classifier_hidden_dim': 128,
            'activation': 'gelu'
        },
        'training': {
            'learning_rate': trial.params['learning_rate'],
            'weight_decay': 0.01,
            'batch_size': 64,
            'max_epochs': 100,
            'early_stopping_patience': 10
        }
    }

    os.makedirs('configs', exist_ok=True)
    with open('configs/trauma_former_optimized.yaml', 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    logger.info("Best configuration saved to configs/trauma_former_optimized.yaml")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/development_set.npz', help='Path to data .npz')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--folds', type=int, default=3, help='Number of folds for inner CV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    run_hyperparameter_search(args.data, args.trials, args.folds, args.seed)