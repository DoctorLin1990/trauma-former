"""
Non-linear physiological data generator for the stress-test experiment
described in Supplementary Material S3.

Differences from the primary linear generator (data/synthetic_generator.py):
  1. Power-law drift: mu_i(t) = mu_base + beta_i * max((t - T_decomp)/60, 0)^gamma
     with gamma=1.1 (superlinear) instead of gamma=1.0 (linear).
  2. Randomised decompensation onset: T_decomp ~ Uniform(300, 1200) s per episode.
  3. Attenuated drift amplitudes: beta_HR = +0.001, beta_SBP = -0.0015 (per s^gamma).
  4. Amplified baseline OU noise: higher sigma and lower theta (Table S3.1).

⚠ CAUTION: Parameters were NOT calibrated against real clinical data.
Results on this dataset are indicative only (Supplementary S3.6).
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

# -----------------------------------------------------------------------
# Non-linear OU parameters (Supplementary Table S3.1)
# -----------------------------------------------------------------------
NL_PARAMS = {
    'HR':   {'mu': 80.0,  'theta': 0.05, 'sigma': 4.0,  'sensor_noise': 1.5},
    'SBP':  {'mu': 120.0, 'theta': 0.05, 'sigma': 5.0,  'sensor_noise': 2.0},
    'DBP':  {'mu': 75.0,  'theta': 0.05, 'sigma': 3.0,  'sensor_noise': 1.5},
    'SpO2': {'mu': 98.0,  'theta': 0.10, 'sigma': 1.0,  'sensor_noise': 0.3},
}
VITAL_ORDER = ['HR', 'SBP', 'DBP', 'SpO2']

# Power-law drift amplitudes (per s^gamma, Supplementary S3.2.2)
BETA_HR  = +0.001   # bpm / s^gamma
BETA_SBP = -0.0015  # mmHg / s^gamma
GAMMA    = 1.1      # power-law exponent (superlinear)

# Onset time: randomised per episode
T_DECOMP_LO = 300   # seconds
T_DECOMP_HI = 1200  # seconds

PP_OFFSET    = 45.0   # pulse-pressure offset (same as primary)
DBP_NOISE    = 2.0    # mmHg

CLIP_RANGES = {
    'HR':   (35,  200),
    'SBP':  (60,  220),
    'DBP':  (30,  130),
    'SpO2': (70,  100),
}

_CORR = np.array([
    [1.00, -0.30, -0.20, -0.05],
    [-0.30,  1.00,  0.70, -0.02],
    [-0.20,  0.70,  1.00, -0.02],
    [-0.05, -0.02, -0.02,  1.00],
], dtype=np.float64)
_CHOL = np.linalg.cholesky(
    _CORR + 1e-8 * np.eye(4)
).astype(np.float32)


class NonlinearOUSimulator:
    """Non-linear OU simulator for the S3 stress-test experiment."""

    def __init__(self, dt: float = 1.0, random_seed: Optional[int] = None) -> None:
        self.dt  = dt
        self.rng = np.random.default_rng(random_seed)

    def generate_episode(
        self,
        is_tic: bool,
        duration_min: int = 30,
    ) -> np.ndarray:
        n_steps = duration_min * 60
        mu    = np.array([NL_PARAMS[v]['mu']    for v in VITAL_ORDER], dtype=np.float64)
        theta = np.array([NL_PARAMS[v]['theta'] for v in VITAL_ORDER], dtype=np.float64)
        sigma = np.array([NL_PARAMS[v]['sigma'] for v in VITAL_ORDER], dtype=np.float64)

        # Correlated Wiener increments
        dW = (self.rng.standard_normal((n_steps, 4)).astype(np.float32) @ _CHOL.T).astype(np.float64)

        X = np.empty((n_steps, 4), dtype=np.float64)
        init_std = sigma / np.sqrt(2.0 * theta)
        X[0]     = mu + self.rng.standard_normal(4) * init_std
        sqrt_dt  = np.sqrt(self.dt)

        for t in range(1, n_steps):
            X[t] = X[t-1] + theta * (mu - X[t-1]) * self.dt + sigma * sqrt_dt * dW[t]

        # DBP derived from SBP
        eta      = self.rng.standard_normal(n_steps) * DBP_NOISE
        X[:, 2]  = X[:, 1] - PP_OFFSET + eta

        # Power-law drift (Supplementary S3.2.2)
        if is_tic:
            t_decomp = int(self.rng.integers(T_DECOMP_LO, T_DECOMP_HI + 1))
            for t in range(t_decomp, n_steps):
                elapsed = (t - t_decomp) / 60.0  # minutes
                X[t, 0] += BETA_HR  * (max(elapsed, 0.0) ** GAMMA)
                X[t, 1] += BETA_SBP * (max(elapsed, 0.0) ** GAMMA)
                X[t, 2]  = X[t, 1] - PP_OFFSET + self.rng.standard_normal() * DBP_NOISE

        # Clip
        for col, vname in enumerate(VITAL_ORDER):
            lo, hi = CLIP_RANGES[vname]
            X[:, col] = np.clip(X[:, col], lo, hi)

        # Sensor noise
        for col, vname in enumerate(VITAL_ORDER):
            X[:, col] += self.rng.standard_normal(n_steps) * NL_PARAMS[vname]['sensor_noise']
            lo, hi = CLIP_RANGES[vname]
            X[:, col] = np.clip(X[:, col], lo, hi)

        return X.astype(np.float32)


def generate_nonlinear_batch(
    n_episodes: int,
    tic_ratio: float = 0.5,
    duration_min: int = 30,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng_meta = np.random.default_rng(random_seed)
    n_tic  = round(n_episodes * tic_ratio)
    n_ctrl = n_episodes - n_tic
    seeds  = rng_meta.integers(0, 2**31, size=n_episodes)

    data, labels = [], []
    for i in range(n_tic):
        sim = NonlinearOUSimulator(random_seed=int(seeds[i]))
        data.append(sim.generate_episode(is_tic=True,  duration_min=duration_min))
        labels.append(1)
    for i in range(n_tic, n_episodes):
        sim = NonlinearOUSimulator(random_seed=int(seeds[i]))
        data.append(sim.generate_episode(is_tic=False, duration_min=duration_min))
        labels.append(0)

    perm   = rng_meta.permutation(n_episodes)
    return np.stack(data)[perm].astype(np.float32), np.array(labels, dtype=np.int32)[perm]
