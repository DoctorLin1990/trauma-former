#!/usr/bin/env python3
"""
Generate the two synthetic datasets used in the paper.

  Development set : 1,240 episodes, 50% TIC prevalence  (seed 42)
  Test set        : 1,000 episodes, 25% TIC prevalence  (seed 43)

Datasets are saved as compressed NumPy archives (.npz) in data/.
Run from the repository root:

    python data/generate_datasets.py

Expected run-time: ≈ 2 minutes on a modern CPU.
"""
import sys
import os
import time
import numpy as np

# Allow running from both the repo root and the data/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import generate_batch  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEV_N        = 1240
DEV_PREV     = 0.50
DEV_SEED     = 42

TEST_N       = 1000
TEST_PREV    = 0.25
TEST_SEED    = 43      # independent seed ensures no distributional overlap

DURATION_MIN = 30

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def _print_summary(name: str, data: np.ndarray, labels: np.ndarray) -> None:
    n_tic  = int(labels.sum())
    n_ctrl = int((1 - labels).sum())
    shape  = data.shape
    print(
        f"  {name}: {shape}  |  TIC={n_tic}  Control={n_ctrl}  "
        f"Prevalence={n_tic/len(labels):.1%}"
    )


def main() -> None:
    # -----------------------------------------------------------------------
    # Development cohort (Section 2.2, paper)
    # -----------------------------------------------------------------------
    print("Generating development set …")
    t0 = time.perf_counter()
    dev_data, dev_labels = generate_batch(
        n_episodes=DEV_N,
        tic_ratio=DEV_PREV,
        duration_min=DURATION_MIN,
        random_seed=DEV_SEED,
    )
    dev_path = os.path.join(OUT_DIR, "development_set.npz")
    np.savez_compressed(dev_path, data=dev_data, labels=dev_labels)
    _print_summary("development_set.npz", dev_data, dev_labels)
    print(f"  Saved → {dev_path}  [{time.perf_counter()-t0:.1f} s]")

    # -----------------------------------------------------------------------
    # Independent test cohort – realistic prevalence (Section 2.2, paper)
    # -----------------------------------------------------------------------
    print("Generating test set …")
    t0 = time.perf_counter()
    test_data, test_labels = generate_batch(
        n_episodes=TEST_N,
        tic_ratio=TEST_PREV,
        duration_min=DURATION_MIN,
        random_seed=TEST_SEED,
    )
    test_path = os.path.join(OUT_DIR, "test_set.npz")
    np.savez_compressed(test_path, data=test_data, labels=test_labels)
    _print_summary("test_set.npz", test_data, test_labels)
    print(f"  Saved → {test_path}  [{time.perf_counter()-t0:.1f} s]")

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
