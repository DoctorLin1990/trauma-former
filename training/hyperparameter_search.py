"""
Bayesian hyperparameter optimisation for Trauma-Former using Optuna.
Implements the 50-trial search over {d_model, n_layers, n_heads, d_ff,
dropout, learning_rate, batch_size} described in Supplementary S2.2.1.

Usage:
    python training/hyperparameter_search.py \
        --data data/development_set.npz \
        --n_trials 50 --seed 42 \
        --output results/optuna_study.pkl
"""
from __future__ import annotations

import os, sys, argparse, pickle
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TICDataset
from data.preprocessing import ZScoreNormalizer
from models.trauma_former import TraumaFormer
from training.trainer import train_model
from training.utils import set_seed, get_device, setup_logger

logger = setup_logger(__name__)

# Fixed training settings (not searched)
N_FOLDS    = 5
MAX_EPOCHS = 200
PATIENCE   = 10
WINDOW     = 60


def objective(trial: optuna.Trial, data: np.ndarray, labels: np.ndarray,
              seed: int) -> float:
    """Optuna objective: mean 5-fold CV AUROC."""
    d_model = trial.suggest_categorical("d_model",    [128, 256, 512])
    n_layers = trial.suggest_categorical("n_layers",  [1, 2, 3, 4])
    # n_heads must divide d_model evenly
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads  = trial.suggest_categorical("n_heads",   valid_heads)
    # d_ff >= d_model
    valid_ff = [f for f in [256, 512, 1024] if f >= d_model]
    d_ff     = trial.suggest_categorical("d_ff",      valid_ff)
    dropout  = trial.suggest_float("dropout", 0.1, 0.5, log=False)
    lr       = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch    = trial.suggest_categorical("batch_size", [32, 64, 128])

    device = get_device()
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aurocs: list[float] = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        set_seed(seed + fold_i + trial.number * 10)

        norm = ZScoreNormalizer()
        norm.fit(data[tr_idx])

        tr_ds  = TICDataset(data[tr_idx], labels[tr_idx], window_size=WINDOW, stride=30, normalizer=norm)
        val_ds = TICDataset(data[val_idx], labels[val_idx], window_size=WINDOW, stride=30, normalizer=norm)

        tr_ld  = DataLoader(tr_ds,  batch_size=batch, shuffle=True,  num_workers=0)
        val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

        model = TraumaFormer(
            input_dim=4, window_size=WINDOW,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout,
        )
        model, _ = train_model(model, tr_ld, val_ld,
                                learning_rate=lr, weight_decay=1e-4,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE, device=device)

        from evaluation.metrics import compute_auroc
        import torch
        model.eval()
        scores, ys = [], []
        with torch.no_grad():
            for x_b, _, y_b, _ in val_ld:
                scores.extend(model(x_b.to(device)).squeeze(1).cpu().tolist())
                ys.extend(y_b.tolist())
        import numpy as np
        aurocs.append(compute_auroc(np.array(ys, dtype=np.int32),
                                    np.array(scores, dtype=np.float32)))

    return float(np.mean(aurocs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian HP search for Trauma-Former")
    ap.add_argument("--data",      default="data/development_set.npz")
    ap.add_argument("--n_trials",  type=int, default=50)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--output",    default="results/optuna_study.pkl")
    ap.add_argument("--study_name", default="trauma_former_hp_search")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    set_seed(args.seed)

    loaded = np.load(args.data)
    data, labels = loaded["data"], loaded["labels"]

    sampler = TPESampler(seed=args.seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler,
                                   study_name=args.study_name)

    study.optimize(
        lambda trial: objective(trial, data, labels, args.seed),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    logger.info(f"\nBest AUROC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    with open(args.output, "wb") as f:
        pickle.dump(study, f)
    logger.info(f"Study saved → {args.output}")

    # Print top-10 trials
    trials_df = study.trials_dataframe(attrs=("number", "value", "params"))
    trials_df = trials_df.sort_values("value", ascending=False).head(10)
    print("\nTop-10 trials:\n", trials_df.to_string(index=False))


if __name__ == "__main__":
    main()
