"""
Generate the synthetic datasets used in the paper:
- Development set: 1240 episodes, 50% TIC prevalence
- Test set: 1000 episodes, 25% TIC prevalence
Saves them as compressed numpy files.
"""
import numpy as np
from synthetic_generator import generate_batch

# Set random seed for reproducibility
RANDOM_SEED = 42

def main():
    # Development set (balanced)
    print("Generating development set (1240 episodes, 50% TIC)...")
    dev_data, dev_labels = generate_batch(
        n_episodes=1240,
        tic_ratio=0.5,
        duration_min=30,
        random_seed=RANDOM_SEED
    )
    np.savez_compressed('data/development_set.npz', data=dev_data, labels=dev_labels)
    print(f"Saved: development_set.npz, shape {dev_data.shape}")

    # Test set (25% TIC prevalence)
    print("Generating test set (1000 episodes, 25% TIC)...")
    test_data, test_labels = generate_batch(
        n_episodes=1000,
        tic_ratio=0.25,
        duration_min=30,
        random_seed=RANDOM_SEED + 1  # different seed to ensure independence
    )
    np.savez_compressed('data/test_set.npz', data=test_data, labels=test_labels)
    print(f"Saved: test_set.npz, shape {test_data.shape}")

if __name__ == '__main__':
    main()