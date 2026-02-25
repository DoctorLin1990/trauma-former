"""
Unit tests for the synthetic data generator.
Tests shape, distributional properties, and preprocessing utilities.
"""
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.synthetic_generator import OUSimulator, generate_batch, DEFAULT_PARAMS, CLIP_RANGES
from data.preprocessing import ZScoreNormalizer, interpolate_and_mask
from data.dataset import TICDataset

class TestOUSimulator:
    def test_generate_episode_shape(self):
        """Test that generated episode has correct shape."""
        sim = OUSimulator(random_seed=42)
        ep = sim.generate_episode(is_tic=False, duration_min=30)
        assert ep.shape == (1800, 4)  # 30 min * 60 sec

    def test_generate_episode_clipping(self):
        """Test that values are clipped to physiological ranges."""
        sim = OUSimulator(random_seed=42)
        ep = sim.generate_episode(is_tic=True, duration_min=30)
        for i, (low, high) in enumerate(CLIP_RANGES.values()):
            assert np.all(ep[:, i] >= low)
            assert np.all(ep[:, i] <= high)

    def test_tic_drift_present(self):
        """Test that TIC episodes exhibit drift in HR and SBP."""
        sim = OUSimulator(random_seed=42)
        ctrl = sim.generate_episode(is_tic=False, duration_min=30)
        tic = sim.generate_episode(is_tic=True, duration_min=30)

        # Compare last 5 minutes vs first 5 minutes
        # HR should increase in TIC
        hr_ctrl_start = np.mean(ctrl[:300, 0])
        hr_ctrl_end = np.mean(ctrl[-300:, 0])
        hr_tic_start = np.mean(tic[:300, 0])
        hr_tic_end = np.mean(tic[-300:, 0])

        assert (hr_tic_end - hr_tic_start) > (hr_ctrl_end - hr_ctrl_start) + 5  # at least 5 bpm more drift

    def test_generate_batch(self):
        """Test batch generation function."""
        data, labels = generate_batch(n_episodes=10, tic_ratio=0.5, random_seed=42)
        assert data.shape == (10, 1800, 4)
        assert labels.shape == (10,)
        assert np.sum(labels) == 5  # exactly half TIC

class TestPreprocessing:
    def test_zscore_normalizer(self):
        """Test Z-score normalizer fitting and transformation."""
        data = np.random.randn(100, 60, 4) * 10 + 50  # synthetic data
        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(data)
        # After fit_transform, mean should be ~0, std ~1 across all samples/timesteps
        assert np.abs(np.mean(normalized)) < 1e-6
        assert np.abs(np.std(normalized) - 1.0) < 0.1

    def test_interpolate_and_mask_short_gap(self):
        """Test interpolation for short gaps (≤5)."""
        window = np.random.randn(60, 4)
        # Create a gap of 3 consecutive NaNs in first channel
        window[10:13, 0] = np.nan
        filled, mask = interpolate_and_mask(window, max_gap=5)
        # Check that gap was filled (no NaNs)
        assert not np.any(np.isnan(filled[:, 0]))
        # Check that mask is True for those positions
        assert np.all(mask[10:13, 0])

    def test_interpolate_and_mask_long_gap(self):
        """Test that long gaps (>5) are zeroed and masked."""
        window = np.random.randn(60, 4)
        # Create a gap of 10 consecutive NaNs
        window[20:30, 1] = np.nan
        filled, mask = interpolate_and_mask(window, max_gap=5)
        # Gap should be zeroed
        assert np.all(filled[20:30, 1] == 0)
        # Mask should be False for those positions
        assert not np.any(mask[20:30, 1])

class TestDataset:
    def test_ticdataset_length(self):
        """Test dataset length calculation."""
        data = np.random.randn(5, 1800, 4)
        labels = np.array([0,1,0,1,0])
        dataset = TICDataset(data, labels, window_size=60, stride=1)
        expected_windows = 5 * (1800 - 60 + 1)
        assert len(dataset) == expected_windows

    def test_ticdataset_item_shape(self):
        """Test that dataset returns correct shapes."""
        data = np.random.randn(2, 1800, 4)
        labels = np.array([0,1])
        dataset = TICDataset(data, labels, window_size=60, stride=1)
        x, mask, y, pid = dataset[0]
        assert x.shape == (60, 4)
        assert mask.shape == (60, 4)
        assert mask.dtype == bool
        assert isinstance(y, float)
        assert isinstance(pid, int)