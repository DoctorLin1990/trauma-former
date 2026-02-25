"""
Patient-level 5-fold cross-validation script.
Ensures that all windows from the same synthetic patient are confined to a single fold.
Computes average metrics across folds with Monte Carlo standard errors.
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import logging
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TICDataset
from models import (
    TraumaFormer, LSTMModel, XGBoostModel,
    PatchTSTModel, InformerModel, ShockIndex
)
from .trainer import Trainer
from .utils import set_seed, setup_logger

logger = setup_logger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all relevant metrics for a set of predictions."""
    metrics = {}
    metrics['auroc'] = roc_auc_score(y_true, y_pred)
    metrics['auprc'] = average_precision_score(y_true, y_pred)
    # Brier score: mean squared error between prediction and true label
    metrics['brier'] = brier_score_loss(y_true, y_pred)
    # For sensitivity/specificity, we need a threshold (default 0.5)
    y_pred_bin = (y_pred > 0.5).astype(int)
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return metrics

def get_model(model_name: str, config: dict, device: torch.device):
    """Instantiate model based on name and config."""
    if model_name.lower() == 'trauma-former':
        model = TraumaFormer(
            input_dim=4,
            window_size=config['model'].get('window_length', 60),
            d_model=config['model'].get('embedding_dim', 256),
            n_heads=config['model'].get('num_attention_heads', 4),
            n_layers=config['model'].get('num_encoder_layers', 2),
            d_ff=config['model'].get('feedforward_dim', 512),
            dropout=config['model'].get('dropout', 0.2),
            classifier_hidden=config['model'].get('classifier_hidden_dim', 128),
            activation=config['model'].get('activation', 'gelu')
        ).to(device)
    elif model_name.lower() == 'lstm':
        model = LSTMModel(
            input_dim=4,
            hidden_size=config['model'].get('hidden_size', 64),
            num_layers=config['model'].get('num_layers', 2),
            dropout=config['model'].get('dropout', 0.2),
            bidirectional=config['model'].get('bidirectional', True)
        ).to(device)
    elif model_name.lower() == 'xgboost':
        # XGBoost is not a PyTorch model; we handle it separately in evaluation
        return None
    elif model_name.lower() == 'patchtst':
        model = PatchTSTModel(
            input_dim=4,
            window_size=config['model'].get('window_length', 60),
            patch_len=config['model'].get('patch_len', 16),
            stride=config['model'].get('stride', 8),
            n_layers=config['model'].get('n_layers', 3),
            d_model=config['model'].get('d_model', 128),
            n_heads=config['model'].get('n_heads', 4),
            dropout=config['model'].get('dropout', 0.2),
            head_dropout=config['model'].get('head_dropout', 0.2)
        ).to(device)
    elif model_name.lower() == 'informer':
        model = InformerModel(
            enc_in=config['model'].get('enc_in', 4),
            dec_in=config['model'].get('dec_in', 4),
            c_out=config['model'].get('c_out', 4),
            seq_len=config['model'].get('window_length', 60),
            label_len=config['model'].get('label_len', 30),
            out_len=config['model'].get('out_len', 1),
            d_model=config['model'].get('d_model', 128),
            n_heads=config['model'].get('n_heads', 4),
            e_layers=config['model'].get('e_layers', 3),
            d_layers=config['model'].get('d_layers', 2),
            d_ff=config['model'].get('d_ff', 512),
            dropout=config['model'].get('dropout', 0.2),
            factor=config['model'].get('factor', 5)
        ).to(device)
    elif model_name.lower() == 'shock-index':
        return ShockIndex(threshold=1.0)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def run_cv(config_path: str, model_name: str, data_path: str = 'data/development_set.npz',
           n_folds: int = 5, seed: int = 42):
    """
    Run patient-level k-fold cross-validation.

    Args:
        config_path: path to YAML config file.
        model_name: name of model to evaluate.
        data_path: path to .npz file with 'data' and 'labels'.
        n_folds: number of folds.
        seed: random seed for reproducibility.
    """
    set_seed(seed)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    loaded = np.load(data_path)
    data = loaded['data']  # (n_patients, T, 4)
    labels = loaded['labels']  # (n_patients,)

    # Create dataset (entire episodes, will be windowed in TICDataset)
    full_dataset = TICDataset(
        data=data,
        labels=labels,
        window_size=config['model'].get('window_length', 60),
        stride=1,
        apply_preprocessing=True,
        normalizer=None  # We'll fit normalizer on each fold
    )

    # Patient-level indices (each patient appears once)
    patient_ids = np.arange(len(data))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    all_preds = []
    all_labels = []

    for fold, (train_patients, val_patients) in enumerate(kf.split(patient_ids)):
        logger.info(f"\n===== Fold {fold+1}/{n_folds} =====")

        # Build train/val subsets based on patient IDs
        train_indices = [i for i, (_, _, _, pid) in enumerate(full_dataset) if pid in train_patients]
        val_indices = [i for i, (_, _, _, pid) in enumerate(full_dataset) if pid in val_patients]

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Fit normalizer on training data only
        # We need to collect all windows from training patients to compute mean/std
        train_windows = np.array([full_dataset[i][0].numpy() for i in train_indices])
        from data.preprocessing import ZScoreNormalizer
        normalizer = ZScoreNormalizer()
        normalizer.fit(train_windows)
        # Update dataset with fitted normalizer (it will transform on the fly)
        full_dataset.normalizer = normalizer

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'] * 2,
                                shuffle=False, num_workers=4, pin_memory=True)

        # Special handling for XGBoost and ShockIndex (non-PyTorch)
        if model_name.lower() == 'xgboost':
            # Extract features and labels from train/val subsets
            X_train = np.array([full_dataset[i][0].numpy() for i in train_indices])
            y_train = np.array([full_dataset[i][2].item() for i in train_indices])
            X_val = np.array([full_dataset[i][0].numpy() for i in val_indices])
            y_val = np.array([full_dataset[i][2].item() for i in val_indices])

            from models.baselines.xgboost_model import XGBoostModel
            model = XGBoostModel(
                n_estimators=config['model'].get('n_estimators', 200),
                max_depth=config['model'].get('max_depth', 6),
                learning_rate=config['model'].get('learning_rate', 0.1),
                random_state=seed
            )
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            metrics = compute_metrics(y_val, preds)
            fold_results.append(metrics)
            all_preds.extend(preds)
            all_labels.extend(y_val)
            continue

        elif model_name.lower() == 'shock-index':
            # Collect all windows from val subset
            X_val = np.array([full_dataset[i][0].numpy() for i in val_indices])
            y_val = np.array([full_dataset[i][2].item() for i in val_indices])
            model = ShockIndex(threshold=1.0)
            si_values = model.predict_proba(X_val)  # returns shock index values
            # Convert to binary predictions using threshold
            preds_bin = (si_values > 1.0).astype(int)
            # For metrics requiring probabilities, we treat shock index as a pseudo-probability scaled to [0,1]
            # Not ideal but for comparison we'll use the raw index (normalized)
            # Use min-max scaling to [0,1] based on reasonable range (0.3-2.0)
            pseudo_proba = np.clip((si_values - 0.3) / (2.0 - 0.3), 0, 1)
            metrics = compute_metrics(y_val, pseudo_proba)
            fold_results.append(metrics)
            all_preds.extend(pseudo_proba)
            all_labels.extend(y_val)
            continue

        # PyTorch models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(model_name, config, device)

        trainer = Trainer(
            model=model,
            device=device,
            config=config['training'],
            experiment_dir=f'./results/models/{model_name}_fold{fold}',
            use_amp=False  # set to True if GPU supports and want faster training
        )

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['max_epochs'],
            fold=fold
        )

        # Load best model and evaluate on validation set
        trainer.load_best_model()
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for x, mask, y, _ in val_loader:
                x = x.to(device)
                mask = mask.to(device) if mask is not None else None
                output = model(x, mask)
                all_val_preds.append(output.cpu().numpy())
                all_val_labels.append(y.numpy())
        all_val_preds = np.concatenate(all_val_preds).ravel()
        all_val_labels = np.concatenate(all_val_labels).ravel()

        metrics = compute_metrics(all_val_labels, all_val_preds)
        fold_results.append(metrics)
        all_preds.extend(all_val_preds)
        all_labels.extend(all_val_labels)

    # Aggregate results across folds
    avg_metrics = {}
    for key in fold_results[0].keys():
        values = [f[key] for f in fold_results]
        avg_metrics[f'avg_{key}'] = np.mean(values)
        avg_metrics[f'std_{key}'] = np.std(values, ddof=1)
        # Monte Carlo Standard Error (MCSE) = std / sqrt(n_folds)
        avg_metrics[f'mcse_{key}'] = np.std(values, ddof=1) / np.sqrt(n_folds)

    # Also compute overall metrics on concatenated predictions (if desired)
    overall_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    avg_metrics['overall_auroc'] = overall_metrics['auroc']
    avg_metrics['overall_auprc'] = overall_metrics['auprc']

    # Print results
    logger.info("\n===== Cross-Validation Results =====")
    for key, value in avg_metrics.items():
        logger.info(f"{key}: {value:.4f}")

    return avg_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--data', type=str, default='data/development_set.npz', help='Path to data .npz')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    run_cv(args.config, args.model, args.data, args.folds, args.seed)