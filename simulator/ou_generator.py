#!/usr/bin/env python3
"""
CLI wrapper for the OU synthetic data generator.

Referenced in Supplementary S1.7:
    python simulator/ou_generator.py --n_episodes 1240 --prevalence 0.5 \
        --seed 42 --output data/dev_cohort.pkl

This file is a thin CLI shim around data/synthetic_generator.py.
The actual generator implementation is in data/synthetic_generator.py.
"""
import argparse
import os
import sys
import pickle
import numpy as np

# Allow running from repo root or simulator/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import generate_batch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ornstein-Uhlenbeck synthetic vital-sign generator (Supplementary S1)"
    )
    ap.add_argument("--n_episodes",  type=int,   default=1240,
                    help="Total number of episodes to generate (default: 1240)")
    ap.add_argument("--prevalence",  type=float, default=0.5,
                    help="Fraction of TIC-positive episodes (default: 0.50)")
    ap.add_argument("--seed",        type=int,   default=42,
                    help="Global random seed (default: 42)")
    ap.add_argument("--duration_min", type=int,  default=30,
                    help="Episode duration in minutes (default: 30)")
    ap.add_argument("--output",      type=str,   default="data/dev_cohort.pkl",
                    help="Output file path (.pkl or .npz)")
    args = ap.parse_args()

    print(f"Generating {args.n_episodes} episodes "
          f"(TIC prevalence={args.prevalence:.0%}, seed={args.seed}) …")

    data, labels = generate_batch(
        n_episodes=args.n_episodes,
        tic_ratio=args.prevalence,
        duration_min=args.duration_min,
        random_seed=args.seed,
    )

    n_tic  = int(labels.sum())
    n_ctrl = int((1 - labels).sum())
    print(f"Generated: shape={data.shape}  TIC={n_tic}  Control={n_ctrl}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.output.endswith(".pkl"):
        with open(args.output, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
    else:
        np.savez_compressed(args.output, data=data, labels=labels)

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
