"""
Unit tests for evaluation metrics.
Compares with scikit-learn where applicable, tests edge cases.
"""
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve as sk_calibration_curve
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.metrics import (
    compute_auroc, compute_auprc, compute_brier, calibration_curve,
    hellinger_distance, monte_carlo_standard_error, compute_all_metrics
)

class TestBinaryMetrics:
    def test_auroc_vs_sklearn(self):
        y_true = np.array([0,1,0,1,0,1])
        y_score = np.array([0.1,0.9,0.2,0.8,0.3,0.7])
        our = compute_auroc(y_true, y_score)
        sk = roc_auc_score(y_true, y_score)
        assert np.isclose(our, sk)

    def test_auprc_vs_sklearn(self):
        y_true = np.array([0,1,0,1,0,1])
        y_score = np.array([0.1,0.9,0.2,0.8,0.3,0.7])
        our = compute_auprc(y_true, y_score)
        sk = average_precision_score(y_true, y_score)
        assert np.isclose(our, sk)

    def test_brier_vs_sklearn(self):
        y_true = np.array([0,1,0,1,0,1])
        y_score = np.array([0.1,0.9,0.2,0.8,0.3,0.7])
        our = compute_brier(y_true, y_score)
        sk = brier_score_loss(y_true, y_score)
        assert np.isclose(our, sk)

    def test_calibration_curve(self):
        y_true = np.array([0,1,0,1,0,1,0,1,0,1])
        y_score = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6,0.5,0.5])
        prob_true, prob_pred, slope, intercept = calibration_curve(y_true, y_score, n_bins=5)
        sk_true, sk_pred = sk_calibration_curve(y_true, y_score, n_bins=5)
        assert np.allclose(prob_true, sk_true)
        assert np.allclose(prob_pred, sk_pred)
        # slope and intercept are not computed by sklearn, but we can verify they make sense
        assert isinstance(slope, float)
        assert isinstance(intercept, float)

    def test_compute_all_metrics(self):
        y_true = np.array([0,1,0,1,0,1])
        y_score = np.array([0.1,0.9,0.2,0.8,0.3,0.7])
        metrics = compute_all_metrics(y_true, y_score)
        expected_keys = ['auroc', 'auprc', 'brier', 'sensitivity', 'specificity',
                         'ppv', 'npv', 'f1', 'calibration_slope', 'calibration_intercept']
        for key in expected_keys:
            assert key in metrics
        # Check that sensitivity/specificity are within [0,1]
        assert 0 <= metrics['sensitivity'] <= 1
        assert 0 <= metrics['specificity'] <= 1

class TestHellinger:
    def test_hellinger_same_distribution(self):
        # Two identical samples should give distance 0
        np.random.seed(42)
        a = np.random.randn(1000, 2)
        b = a.copy()
        dist = hellinger_distance(a, b, n_bins=10)
        assert dist < 0.1  # approximation may not be exact zero due to binning, but close

    def test_hellinger_different_distributions(self):
        # Very different distributions should give distance near 1
        a = np.random.randn(1000, 1)  # N(0,1)
        b = np.random.randn(1000, 1) + 10  # N(10,1)
        dist = hellinger_distance(a, b, n_bins=20)
        assert dist > 0.9

    def test_hellinger_1d(self):
        a = np.random.randn(1000)
        b = np.random.randn(1000) + 2
        dist = hellinger_distance(a, b, n_bins=20)
        assert 0 <= dist <= 1

class TestMCSE:
    def test_mcse_constant(self):
        # If all values are identical, MCSE = 0
        values = [0.9] * 10
        mcse = monte_carlo_standard_error(values)
        assert mcse == 0.0

    def test_mcse_vs_std(self):
        values = [0.8, 0.82, 0.79, 0.81]
        mcse = monte_carlo_standard_error(values)
        std = np.std(values, ddof=1)
        expected = std / np.sqrt(len(values))
        assert np.isclose(mcse, expected)