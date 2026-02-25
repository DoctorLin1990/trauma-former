"""
Ornstein-Uhlenbeck (OU) process-based synthetic physiological data generator.
Produces 1 Hz waveforms for HR, SBP, DBP, and SpO2 for control and TIC patients.
"""
import numpy as np
from typing import Optional, Tuple

# Default parameters from Supplementary Table S1
DEFAULT_PARAMS = {
    'HR': {'mu': 75.0, 'theta': 0.8, 'sigma': 8.0},
    'SBP': {'mu': 120.0, 'theta': 0.6, 'sigma': 10.0},
    'DBP': {'mu': 75.0, 'theta': 0.6, 'sigma': 7.0},
    'SpO2': {'mu': 98.0, 'theta': 0.5, 'sigma': 1.0},
}

# Cross-correlation matrix (lower triangular, Cholesky factor will be computed)
CORRELATIONS = np.array([
    [1.0, -0.2, -0.1, -0.05],   # HR with HR, SBP, DBP, SpO2
    [-0.2, 1.0, 0.7, -0.02],    # SBP with HR, SBP, DBP, SpO2
    [-0.1, 0.7, 1.0, -0.02],    # DBP with HR, SBP, DBP, SpO2
    [-0.05, -0.02, -0.02, 1.0], # SpO2 with HR, SBP, DBP, SpO2
])

# Physiologically plausible clipping ranges
CLIP_RANGES = {
    'HR': (40, 180),
    'SBP': (50, 200),
    'DBP': (30, 130),
    'SpO2': (70, 100)
}

# TIC drift rates (per minute)
TIC_DRIFT = {
    'HR': +0.5,      # bpm per minute
    'SBP': -1.0,     # mmHg per minute
    'SpO2': -0.02,   # % per minute
}
DBP_SBP_FACTOR = 0.5      # DBP = factor * SBP + noise
DBP_NOISE_STD = 3.0       # mmHg

def _ensure_positive_definite(corr: np.ndarray) -> np.ndarray:
    """Ensure correlation matrix is positive definite (add small epsilon)."""
    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < 1e-6:
        corr += 1e-6 * np.eye(corr.shape[0])
    return corr

def _compute_cholesky(corr: np.ndarray) -> np.ndarray:
    """Compute Cholesky factor of the correlation matrix."""
    corr = _ensure_positive_definite(corr)
    return np.linalg.cholesky(corr).astype(np.float32)

# Precompute Cholesky factor for efficiency
CHOLESKY = _compute_cholesky(CORRELATIONS)

class OUSimulator:
    """
    Multivariate Ornstein-Uhlenbeck process simulator for vital signs.
    Generates 1 Hz time series for a single patient episode.
    """
    def __init__(self, dt: float = 1.0, random_seed: Optional[int] = None):
        """
        Args:
            dt: time step in seconds (must be 1.0 to match 1 Hz)
            random_seed: optional seed for reproducibility
        """
        self.dt = dt
        if random_seed is not None:
            np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)

    def generate_episode(self, is_tic: bool, duration_min: int = 30,
                         add_motion_artifacts: bool = True,
                         add_sensor_noise: bool = True) -> np.ndarray:
        """
        Generate a 30-minute episode of vital signs.

        Args:
            is_tic: True for TIC patient (with drift), False for control.
            duration_min: length of episode in minutes (default 30).
            add_motion_artifacts: if True, add brief spikes to 5% of episodes.
            add_sensor_noise: if True, add Gaussian sensor noise.

        Returns:
            Array of shape (duration_min*60, 4) with columns:
            HR, SBP, DBP, SpO2.
        """
        n_steps = duration_min * 60
        # Extract parameters
        mu_base = np.array([DEFAULT_PARAMS[v]['mu'] for v in ['HR', 'SBP', 'DBP', 'SpO2']])
        theta = np.array([DEFAULT_PARAMS[v]['theta'] for v in ['HR', 'SBP', 'DBP', 'SpO2']])
        sigma = np.array([DEFAULT_PARAMS[v]['sigma'] for v in ['HR', 'SBP', 'DBP', 'SpO2']])

        # Initialize trajectory
        X = np.zeros((n_steps, 4))
        X[0] = mu_base + self.rng.normal(0, sigma)  # start near baseline

        # Precompute OU coefficients
        A = 1 - theta * self.dt
        B = sigma * np.sqrt(self.dt)

        # Generate Wiener increments (correlated)
        dW = self.rng.normal(0, 1, size=(n_steps, 4)) @ CHOLESKY.T

        # Time steps
        for t in range(1, n_steps):
            # Mean-reverting drift
            drift = theta * (mu_base - X[t-1]) * self.dt
            # Diffusion term
            diffusion = B * dW[t]
            X[t] = X[t-1] + drift + diffusion

        # Apply TIC-specific linear drift after 5 minutes
        if is_tic:
            drift_start_idx = 5 * 60  # 5 minutes in seconds
            # HR drift
            hr_drift = TIC_DRIFT['HR'] / 60  # per second
            sbp_drift = TIC_DRIFT['SBP'] / 60
            spo2_drift = TIC_DRIFT['SpO2'] / 60
            for t in range(drift_start_idx, n_steps):
                time_since_start = (t - drift_start_idx) / 60  # in minutes
                X[t, 0] += hr_drift * time_since_start * 60
                X[t, 1] += sbp_drift * time_since_start * 60
                X[t, 3] += spo2_drift * time_since_start * 60
                # Recompute DBP from SBP
                X[t, 2] = DBP_SBP_FACTOR * X[t, 1] + self.rng.normal(0, DBP_NOISE_STD)

        # Clip to plausible ranges
        for i, (low, high) in enumerate(CLIP_RANGES.values()):
            X[:, i] = np.clip(X[:, i], low, high)

        # Add motion artifacts (random spikes) to some episodes
        if add_motion_artifacts and self.rng.random() < 0.05:
            artifact_start = self.rng.randint(0, n_steps - 10)
            artifact_duration = self.rng.randint(5, 10)  # seconds
            artifact_magnitude = self.rng.uniform(1.2, 1.5)  # multiplicative factor
            # Apply to all vital signs briefly
            for i in range(artifact_start, min(artifact_start + artifact_duration, n_steps)):
                X[i] *= artifact_magnitude
                # Re-clip
                for j, (low, high) in enumerate(CLIP_RANGES.values()):
                    X[i, j] = np.clip(X[i, j], low, high)

        # Add Gaussian sensor noise (1% of signal range)
        if add_sensor_noise:
            noise_std = np.array([0.01 * (high - low) for (low, high) in CLIP_RANGES.values()])
            noise = self.rng.normal(0, noise_std, size=X.shape)
            X += noise
            # Re-clip
            for i, (low, high) in enumerate(CLIP_RANGES.values()):
                X[:, i] = np.clip(X[:, i], low, high)

        return X

# Utility function for batch generation
def generate_batch(n_episodes: int, tic_ratio: float, duration_min: int = 30,
                   random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of episodes.

    Returns:
        data: array of shape (n_episodes, duration_min*60, 4)
        labels: array of shape (n_episodes,) with 1 for TIC, 0 for control.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    simulator = OUSimulator(random_seed=random_seed)
    n_tic = int(n_episodes * tic_ratio)
    n_ctrl = n_episodes - n_tic

    data_list = []
    labels_list = []
    for _ in range(n_tic):
        data_list.append(simulator.generate_episode(is_tic=True, duration_min=duration_min))
        labels_list.append(1)
    for _ in range(n_ctrl):
        data_list.append(simulator.generate_episode(is_tic=False, duration_min=duration_min))
        labels_list.append(0)

    # Shuffle
    indices = np.random.permutation(n_episodes)
    data = np.stack(data_list)[indices]
    labels = np.array(labels_list)[indices]
    return data, labels