"""
Robustness tests: Gaussian noise, random missing data, sensor failure.
These functions apply distortions to input data and evaluate model performance.
"""
import numpy as np
from typing import Callable, Tuple, Optional
import warnings

def add_gaussian_noise(data: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise to achieve specified signal-to-noise ratio (dB).
    SNR_dB = 10 * log10(signal_power / noise_power)
    """
    # Compute signal power (mean of squared values)
    signal_power = np.mean(data ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise

def test_gaussian_noise(model: Callable, data: np.ndarray, labels: np.ndarray,
                        snr_levels: list = [20, 15, 10, 5],
                        n_repeats: int = 5, seed: int = 42) -> dict:
    """
    Test model performance under additive Gaussian noise at various SNR levels.
    Returns dictionary with keys 'snr' and 'auroc' (mean and std over repeats).
    """
    np.random.seed(seed)
    results = {'snr': [], 'auroc_mean': [], 'auroc_std': []}
    from sklearn.metrics import roc_auc_score

    for snr in snr_levels:
        aurocs = []
        for rep in range(n_repeats):
            data_noisy = add_gaussian_noise(data, snr)
            # Clip to plausible ranges? Optional.
            y_score = model.predict_proba(data_noisy)
            auroc = roc_auc_score(labels, y_score)
            aurocs.append(auroc)
        results['snr'].append(snr)
        results['auroc_mean'].append(np.mean(aurocs))
        results['auroc_std'].append(np.std(aurocs, ddof=1))
    return results

def test_random_missing(model: Callable, data: np.ndarray, labels: np.ndarray,
                        missing_rates: list = [0.1, 0.2, 0.3, 0.4, 0.5],
                        n_repeats: int = 5, seed: int = 42) -> dict:
    """
    Test model performance with random missing data (MCAR).
    Missing values are set to NaN and then handled by model's preprocessing.
    """
    np.random.seed(seed)
    results = {'missing_rate': [], 'auroc_mean': [], 'auroc_std': []}
    from sklearn.metrics import roc_auc_score

    for rate in missing_rates:
        aurocs = []
        for rep in range(n_repeats):
            # Create mask for missing values
            mask = np.random.rand(*data.shape) < rate
            data_missing = data.copy()
            data_missing[mask] = np.nan
            # Model must handle NaNs (via preprocessing)
            y_score = model.predict_proba(data_missing)
            # Ensure no NaNs in predictions
            if np.any(np.isnan(y_score)):
                warnings.warn(f"NaNs in predictions at missing rate {rate}")
                y_score = np.nan_to_num(y_score, nan=0.5)
            auroc = roc_auc_score(labels, y_score)
            aurocs.append(auroc)
        results['missing_rate'].append(rate)
        results['auroc_mean'].append(np.mean(aurocs))
        results['auroc_std'].append(np.std(aurocs, ddof=1))
    return results

def test_sensor_failure(model: Callable, data: np.ndarray, labels: np.ndarray,
                        sensor_idx: int = 0,  # 0=HR
                        failure_durations: list = [5, 10, 15, 20, 30],
                        n_repeats: int = 5, seed: int = 42) -> dict:
    """
    Simulate complete failure of a sensor (e.g., HR) for a contiguous duration.
    The affected time steps are set to NaN.
    """
    np.random.seed(seed)
    n_samples, T, N = data.shape
    results = {'duration': [], 'auroc_mean': [], 'auroc_std': []}
    from sklearn.metrics import roc_auc_score

    for dur in failure_durations:
        aurocs = []
        for rep in range(n_repeats):
            data_failure = data.copy()
            # Randomly choose start time for each sample independently
            for i in range(n_samples):
                start = np.random.randint(0, T - dur + 1)
                data_failure[i, start:start+dur, sensor_idx] = np.nan
            y_score = model.predict_proba(data_failure)
            if np.any(np.isnan(y_score)):
                y_score = np.nan_to_num(y_score, nan=0.5)
            auroc = roc_auc_score(labels, y_score)
            aurocs.append(auroc)
        results['duration'].append(dur)
        results['auroc_mean'].append(np.mean(aurocs))
        results['auroc_std'].append(np.std(aurocs, ddof=1))
    return results