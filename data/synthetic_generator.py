"""
Ornstein-Uhlenbeck (OU) process-based synthetic physiological data generator.
Produces 1 Hz waveforms for HR, SBP, DBP, and SpO2 for control and TIC patients.

Parameters taken directly from Supplementary Material S1, Tables S1.1 and S1.2.

BUG-FIX v3 (vs original repo):
  The original code added TIC drift *after* the OU integration loop via
  post-hoc addition:
      X[t, 0] += alpha_HR * elapsed_min        # WRONG
  This violates Eq. 3 of S1.2.3, which defines the drift as a modification of
  the TIME-VARYING MEAN mu_i(t) inside the Euler-Maruyama step:
      X_i(t+1) = X_i(t) + theta_i * [mu_i(t) - X_i(t)] * dt + sigma_i * sqrt(dt) * eps(t)
  The corrected implementation passes the drifted mean to each EM step so the
  mean-reversion force is applied relative to the correct (drifted) target.

Euler-Maruyama discretisation (dt = 1 s, Supplementary Eq. 2):
    X_i(t+1) = X_i(t) + theta_i * (mu_i(t) - X_i(t)) * dt + sigma_i * sqrt(dt) * eps_i(t)

TIC drift after T_onset = 300 s (Supplementary Eq. 3):
    mu_i(t) = mu_i_baseline + alpha_i * max(t - T_onset, 0) / 60
    (alpha_i in units-per-minute, converted to per-60-step below)

DBP modelled as (Supplementary Eq. 4):
    DBP(t) = SBP(t) - PP_target(t) + eta(t),  eta ~ N(0, sigma_eta^2)
    PP_target(t) = PP_INITIAL - k_PP * max((t - T_onset)/60, 0)
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
# TIC drift rates (Supplementary Table S1.2)
# Units: bpm/min for HR, mmHg/min for SBP, %/min for SpO2
# -----------------------------------------------------------------------
TIC_DRIFT_PER_MIN = {
    'HR':   +0.50,   # bpm / min  → compensatory tachycardia
    'SBP':  -0.50,   # mmHg / min → progressive SBP fall
    'SpO2':  0.0,    # minimal drift (early compensated shock)
}
TIC_ONSET_SEC = 300  # T_onset = 5 minutes

# DBP derived from SBP via time-varying pulse-pressure target (S1.2.3)
PP_INITIAL    = 45.0   # mmHg: initial target pulse pressure
PP_SLOPE      = 0.5    # mmHg / min: PP narrowing rate after onset (k_PP)
DBP_NOISE_STD = 2.0    # mmHg: additional OU noise on DBP (sigma_eta)

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
# Reflects baroreflex-mediated HR-BP coupling.
# -----------------------------------------------------------------------
_CORR = np.array([
    [ 1.00, -0.30, -0.20, -0.05],
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
    Multivariate OU simulator for prehospital vital-sign episodes.

    Generates a single 30-minute (1800-sample) episode of HR, SBP, DBP, SpO2
    at 1 Hz exactly as specified in Supplementary Material S1.

    Key correction vs. original repo (BUG-1):
      TIC drift is incorporated into the OU mean function mu_i(t) *inside*
      the Euler-Maruyama loop, not added post-hoc. This ensures the
      mean-reversion force during the OU step correctly targets the drifted
      physiological mean, consistent with Supplementary Eq. 3.
    """

    def __init__(self, dt: float = 1.0, random_seed: Optional[int] = None):
        assert dt == 1.0, "Simulator must run at 1 Hz (dt = 1.0 s)."
        self.dt  = dt
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
            add_sensor_noise:     Add Gaussian measurement noise (Table S1.1).

        Returns:
            Array of shape (duration_min * 60, 4): columns [HR, SBP, DBP, SpO2].
        """
        n_steps = duration_min * 60
        mu_base = np.array([OU_PARAMS[v]['mu']    for v in VITAL_ORDER], dtype=np.float64)
        theta   = np.array([OU_PARAMS[v]['theta'] for v in VITAL_ORDER], dtype=np.float64)
        sigma   = np.array([OU_PARAMS[v]['sigma'] for v in VITAL_ORDER], dtype=np.float64)

        # ----------------------------------------------------------------
        # 1. Correlated Wiener increments  (shape: n_steps × 4)
        # ----------------------------------------------------------------
        raw_noise = self.rng.standard_normal((n_steps, 4)).astype(np.float32)
        dW = (raw_noise @ _CHOL.T).astype(np.float64)

        # ----------------------------------------------------------------
        # 2. Euler-Maruyama integration  — BUG-FIX: drift enters mu(t)
        #    inside the loop, not as a post-hoc correction.
        #
        #    For HR, SBP, SpO2 (index 0, 1, 3):
        #      mu_i(t) = mu_i_baseline + alpha_i * max(t - T_onset, 0) / 60
        #    For DBP (index 2):
        #      Derived from SBP mean via PP_target; skipped in EM loop and
        #      overwritten entirely in step 3 below.
        # ----------------------------------------------------------------
        X = np.empty((n_steps, 4), dtype=np.float64)
        init_std = sigma / np.sqrt(2.0 * theta)
        X[0] = mu_base + self.rng.standard_normal(4) * init_std

        sqrt_dt = np.sqrt(self.dt)

        # Pre-compute time-varying PP_target for TIC (vectorised for speed)
        t_vec = np.arange(n_steps, dtype=np.float64)
        if is_tic:
            pp_target = PP_INITIAL - PP_SLOPE * np.maximum(
                (t_vec - TIC_ONSET_SEC) / 60.0, 0.0
            )
        else:
            pp_target = np.full(n_steps, PP_INITIAL, dtype=np.float64)

        for t in range(1, n_steps):
            # ---- Compute time-varying mean (FIXED) ----
            mu_t = mu_base.copy()
            if is_tic and t >= TIC_ONSET_SEC:
                elapsed_min = (t - TIC_ONSET_SEC) / 60.0
                mu_t[0] += TIC_DRIFT_PER_MIN['HR']   * elapsed_min  # HR
                mu_t[1] += TIC_DRIFT_PER_MIN['SBP']  * elapsed_min  # SBP
                mu_t[3] += TIC_DRIFT_PER_MIN['SpO2'] * elapsed_min  # SpO2
                # DBP mean is derived from SBP mean below; skip index 2 here.

            drift     = theta * (mu_t - X[t - 1]) * self.dt
            diffusion = sigma * sqrt_dt * dW[t]
            X[t]      = X[t - 1] + drift + diffusion

        # ----------------------------------------------------------------
        # 3. Overwrite DBP column using time-varying PP_target (Eq. 4)
        #    DBP(t) = SBP(t) - PP_target(t) + eta,  eta ~ N(0, DBP_NOISE_STD²)
        # ----------------------------------------------------------------
        eta      = self.rng.standard_normal(n_steps) * DBP_NOISE_STD
        X[:, 2]  = X[:, 1] - pp_target + eta

        # ----------------------------------------------------------------
        # 4. Physiological clipping (S1.4)
        # ----------------------------------------------------------------
        for col, vname in enumerate(VITAL_ORDER):
            lo, hi = CLIP_RANGES[vname]
            X[:, col] = np.clip(X[:, col], lo, hi)

        # ----------------------------------------------------------------
        # 5. Motion artifacts (Table S1.3)
        #    Prob 0.15/episode; amplitude U(5,20) bpm; duration U(2,10) s.
        # ----------------------------------------------------------------
        if add_motion_artifacts and self.rng.random() < 0.15:
            dur   = int(self.rng.integers(2, 11))
            start = int(self.rng.integers(0, max(1, n_steps - dur)))
            amp   = self.rng.uniform(5.0, 20.0)
            X[start:start + dur, 0] += amp
            X[start:start + dur, 0]  = np.clip(
                X[start:start + dur, 0], *CLIP_RANGES['HR']
            )

        # ----------------------------------------------------------------
        # 6. Gaussian sensor noise (Table S1.1)
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

    # Episode-level seeds derived from global seed (Supplementary S1.4)
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

    perm   = rng_meta.permutation(n_episodes)
    data   = np.stack(data_list, axis=0)[perm]
    labels = np.array(labels_list, dtype=np.int32)[perm]

    return data, labels
