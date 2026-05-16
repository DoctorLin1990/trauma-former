"""
Ornstein-Uhlenbeck (OU) process-based synthetic physiological data generator.
Produces 1 Hz waveforms for HR, SBP, DBP, and SpO2 for control and TIC patients.

Parameters are taken directly from Supplementary Material S1, Tables S1.1 and S1.2.

Euler-Maruyama discretization scheme (dt = 1 s):
    X_i(t+1) = X_i(t) + theta_i * (mu_i(t) - X_i(t)) * dt + sigma_i * sqrt(dt) * eps_i(t)

TIC drift (after T_onset = 300 s):
    mu_i(t) = mu_i_baseline + alpha_i * max(t - T_onset, 0) / 60
    (alpha_i in units per minute; converted to per-second below)

DBP modeled as: DBP(t) = SBP(t) - PP_target(t) + eta(t), eta ~ N(0, sigma_eta^2)
PP_target(t) = PP_INITIAL - k_PP * max((t - T_onset)/60, 0)  [Supplementary Eq. 4]
This ensures pulse pressure narrows progressively in TIC episodes.

Cross-variable noise correlation is introduced via a Cholesky-decomposed
correlation matrix consistent with known HR-BP physiology.
"""
import numpy as np
from typing import Optional, Tuple

# -----------------------------------------------------------------------
# OU process parameters (Supplementary Table S1.1)
# -----------------------------------------------------------------------
OU_PARAMS = {
    'HR':   {'mu': 80.0,  'theta': 0.15, 'sigma': 3.5,  'sensor_noise': 1.5},
    'SBP':  {'mu': 120.0, 'theta': 0.10, 'sigma': 4.5,  'sensor_noise': 2.0},
    'DBP':  {'mu': 75.0,  'theta': 0.10, 'sigma': 3.5,  'sensor_noise': 1.5},
    'SpO2': {'mu': 98.0,  'theta': 0.20, 'sigma': 0.8,  'sensor_noise': 0.3},
}
VITAL_ORDER = ['HR', 'SBP', 'DBP', 'SpO2']

# -----------------------------------------------------------------------
# TIC linear drift rates (Supplementary Table S1.2)
# Units: bpm/min for HR, mmHg/min for SBP, %/min for SpO2
# -----------------------------------------------------------------------
TIC_DRIFT_PER_MIN = {
    'HR':   +0.50,   # bpm / min  → positive drift (tachycardia)
    'SBP':  -0.50,   # mmHg / min → negative drift (hypotension)
    'SpO2': 0.0,     # minimal drift during early compensated shock
}
TIC_ONSET_SEC = 300  # T_onset = 5 minutes

# DBP derived from SBP (maintains pulse-pressure relationship)
# PP_target(t) = PP_initial - k_PP * max((t - T_onset)/60, 0)
# Paper Supplementary S1.2.3: PP_initial=45 mmHg, k_PP=0.5 mmHg/min
PP_INITIAL = 45.0    # mmHg: initial target pulse pressure (SBP - DBP)
PP_SLOPE   = 0.5     # mmHg / min: rate of PP narrowing after TIC onset (k_PP)
DBP_NOISE_STD = 2.0  # mmHg: additional OU noise on DBP (sigma_eta)

# -----------------------------------------------------------------------
# Physiological clipping bounds (Supplementary S1.4)
# -----------------------------------------------------------------------
CLIP_RANGES = {
    'HR':   (35,  200),
    'SBP':  (60,  220),
    'DBP':  (30,  130),
    'SpO2': (70,  100),
}

# -----------------------------------------------------------------------
# Cross-variable noise correlation (Cholesky factor)
# Reflects known baroreflex-mediated HR-BP coupling
# -----------------------------------------------------------------------
_CORR = np.array([
    [1.00, -0.30, -0.20, -0.05],
    [-0.30,  1.00,  0.70, -0.02],
    [-0.20,  0.70,  1.00, -0.02],
    [-0.05, -0.02, -0.02,  1.00],
], dtype=np.float64)

def _ensure_posdef(C: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-8:
        C = C + (1e-8 - eigvals.min()) * np.eye(C.shape[0])
    return C

_CHOL = np.linalg.cholesky(_ensure_posdef(_CORR)).astype(np.float32)


class OUSimulator:
    """
    Multivariate Ornstein-Uhlenbeck simulator for prehospital vital-sign episodes.

    Generates a single 30-minute (1800-sample) episode of HR, SBP, DBP, SpO2
    at 1 Hz, with optional TIC-specific linear drift, motion artifacts, and
    Gaussian sensor noise, exactly as specified in Supplementary Material S1.
    """

    def __init__(self, dt: float = 1.0, random_seed: Optional[int] = None):
        """
        Args:
            dt:          Sampling interval in seconds. Must be 1.0 (1 Hz).
            random_seed: NumPy seed for reproducibility.
        """
        assert dt == 1.0, "Simulator must run at 1 Hz (dt=1.0 s)."
        self.dt = dt
        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_episode(
        self,
        is_tic: bool,
        duration_min: int = 30,
        add_motion_artifacts: bool = True,
        add_sensor_noise: bool = True,
    ) -> np.ndarray:
        """
        Generate one patient transport episode.

        Args:
            is_tic:               True → TIC trajectory (linear drift after T_onset).
            duration_min:         Episode length in minutes (default 30).
            add_motion_artifacts: Add brief HR spikes (Table S1.3).
            add_sensor_noise:     Add independent Gaussian measurement noise.

        Returns:
            Array of shape (duration_min * 60, 4) with columns [HR, SBP, DBP, SpO2].
        """
        n_steps = duration_min * 60
        mu    = np.array([OU_PARAMS[v]['mu']    for v in VITAL_ORDER], dtype=np.float64)
        theta = np.array([OU_PARAMS[v]['theta'] for v in VITAL_ORDER], dtype=np.float64)
        sigma = np.array([OU_PARAMS[v]['sigma'] for v in VITAL_ORDER], dtype=np.float64)

        # ----------------------------------------------------------------
        # 1. Correlated Wiener increments
        # ----------------------------------------------------------------
        raw_noise = self.rng.standard_normal((n_steps, 4)).astype(np.float32)
        dW = (raw_noise @ _CHOL.T).astype(np.float64)  # shape (n_steps, 4)

        # ----------------------------------------------------------------
        # 2. Euler-Maruyama integration
        # ----------------------------------------------------------------
        X = np.empty((n_steps, 4), dtype=np.float64)
        # Initialise from OU stationary distribution N(mu, sigma^2 / (2*theta))
        init_std = sigma / np.sqrt(2.0 * theta)
        X[0] = mu + self.rng.standard_normal(4) * init_std

        sqrt_dt = np.sqrt(self.dt)
        for t in range(1, n_steps):
            drift     = theta * (mu - X[t - 1]) * self.dt
            diffusion = sigma * sqrt_dt * dW[t]
            X[t]      = X[t - 1] + drift + diffusion

        # ----------------------------------------------------------------
        # 3. TIC linear drift for HR, SBP, SpO2 (Supplementary S1.2.3)
        #    Applied BEFORE DBP derivation so DBP can use the drifted SBP.
        # ----------------------------------------------------------------
        if is_tic:
            for t in range(TIC_ONSET_SEC, n_steps):
                elapsed_min = (t - TIC_ONSET_SEC) / 60.0
                X[t, 0] += TIC_DRIFT_PER_MIN['HR']   * elapsed_min   # HR
                X[t, 1] += TIC_DRIFT_PER_MIN['SBP']  * elapsed_min   # SBP
                X[t, 3] += TIC_DRIFT_PER_MIN['SpO2'] * elapsed_min   # SpO2

        # ----------------------------------------------------------------
        # 4. DBP derived from SBP with time-varying pulse-pressure target
        #    PP_target(t) = PP_INITIAL - k_PP * max((t - T_onset)/60, 0)
        #    (Supplementary Eq. 4 / Table S1.2: PP narrows 45→32.5 mmHg
        #     over 25 min post-onset; constant for control episodes)
        #    DBP(t) = SBP(t) - PP_target(t) + eta,  eta ~ N(0, DBP_NOISE_STD²)
        # ----------------------------------------------------------------
        t_vec = np.arange(n_steps, dtype=np.float64)
        if is_tic:
            pp_target = PP_INITIAL - PP_SLOPE * np.maximum(
                (t_vec - TIC_ONSET_SEC) / 60.0, 0.0
            )
        else:
            pp_target = np.full(n_steps, PP_INITIAL, dtype=np.float64)

        eta = self.rng.standard_normal(n_steps) * DBP_NOISE_STD
        X[:, 2] = X[:, 1] - pp_target + eta   # column 2 = DBP, column 1 = SBP

        # ----------------------------------------------------------------
        # 5. Physiological clipping (Supplementary S1.4)
        # ----------------------------------------------------------------
        for col, vname in enumerate(VITAL_ORDER):
            lo, hi = CLIP_RANGES[vname]
            X[:, col] = np.clip(X[:, col], lo, hi)

        # ----------------------------------------------------------------
        # 6. Motion artifacts (Supplementary Table S1.3)
        #    Probability: 0.15/episode; amplitude: U(5,20) bpm added to HR;
        #    duration: U(2,10) s.
        # ----------------------------------------------------------------
        if add_motion_artifacts and self.rng.random() < 0.15:
            dur   = int(self.rng.integers(2, 11))   # U(2,10) s inclusive
            start = int(self.rng.integers(0, n_steps - dur))
            amp   = self.rng.uniform(5.0, 20.0)     # bpm additive spike
            X[start:start + dur, 0] += amp           # affect HR only
            X[start:start + dur, 0]  = np.clip(
                X[start:start + dur, 0], *CLIP_RANGES['HR']
            )

        # ----------------------------------------------------------------
        # 7. Gaussian sensor noise (Table S1.1)
        # ----------------------------------------------------------------
        if add_sensor_noise:
            for col, vname in enumerate(VITAL_ORDER):
                noise = self.rng.standard_normal(n_steps) * OU_PARAMS[vname]['sensor_noise']
                X[:, col] += noise
                lo, hi = CLIP_RANGES[vname]
                X[:, col] = np.clip(X[:, col], lo, hi)

        return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch generation helper
# ---------------------------------------------------------------------------

def generate_batch(
    n_episodes: int,
    tic_ratio: float,
    duration_min: int = 30,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a shuffled batch of patient episodes.

    Args:
        n_episodes:   Total number of episodes.
        tic_ratio:    Fraction of TIC-positive episodes (e.g. 0.5 or 0.25).
        duration_min: Episode length (default 30 min).
        random_seed:  Global seed for reproducibility.

    Returns:
        data:   float32 array of shape (n_episodes, duration_min*60, 4)
        labels: int32   array of shape (n_episodes,) — 1=TIC, 0=control
    """
    rng_meta = np.random.default_rng(random_seed)
    n_tic  = round(n_episodes * tic_ratio)
    n_ctrl = n_episodes - n_tic

    data_list:   list = []
    labels_list: list = []

    # Episode-level seeds are derived from the global seed to guarantee
    # independence while remaining exactly reproducible (Supplementary S1.4).
    episode_seeds = rng_meta.integers(0, 2**31, size=n_episodes)
    ep_idx = 0

    for _ in range(n_tic):
        sim = OUSimulator(random_seed=int(episode_seeds[ep_idx]))
        data_list.append(sim.generate_episode(is_tic=True,  duration_min=duration_min))
        labels_list.append(1)
        ep_idx += 1

    for _ in range(n_ctrl):
        sim = OUSimulator(random_seed=int(episode_seeds[ep_idx]))
        data_list.append(sim.generate_episode(is_tic=False, duration_min=duration_min))
        labels_list.append(0)
        ep_idx += 1

    # Shuffle
    perm = rng_meta.permutation(n_episodes)
    data   = np.stack(data_list, axis=0)[perm]
    labels = np.array(labels_list, dtype=np.int32)[perm]

    return data, labels
