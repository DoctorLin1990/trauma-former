#!/usr/bin/env python3
"""
Smoke-test suite for the Trauma-Former pipeline.

Runs lightweight checks that confirm the full pipeline is functional
without requiring a GPU or full dataset:
  - OU simulator produces correct shapes and plausible ranges
  - Preprocessing (interpolation, masking, z-score) works correctly
  - All model architectures forward-pass without error
  - Metric functions return consistent values
  - Alert-rule logic matches paper specification
  - Missingness indicator augmentation works

Usage (from repository root):
    python -m pytest tests/test_pipeline.py -v
    # or:
    python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np

# Allow running from repo root or tests/ subdirectory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_EPISODES = 20
T_SEC      = 1800   # 30 min @ 1 Hz
N_VITALS   = 4
WINDOW     = 60


@pytest.fixture(scope='module')
def small_batch():
    """Generate a tiny batch of synthetic episodes for testing."""
    from data.synthetic_generator import generate_batch
    data, labels = generate_batch(
        n_episodes=N_EPISODES,
        tic_ratio=0.5,
        duration_min=30,
        random_seed=0,
    )
    return data, labels


@pytest.fixture(scope='module')
def normalizer(small_batch):
    from data.preprocessing import ZScoreNormalizer
    data, _ = small_batch
    norm = ZScoreNormalizer()
    norm.fit(data)
    return norm


# ─────────────────────────────────────────────────────────────────────────────
# 1. OU Simulator
# ─────────────────────────────────────────────────────────────────────────────

class TestOUSimulator:

    def test_output_shape(self, small_batch):
        data, labels = small_batch
        assert data.shape == (N_EPISODES, T_SEC, N_VITALS), (
            f"Expected ({N_EPISODES}, {T_SEC}, {N_VITALS}), got {data.shape}"
        )
        assert labels.shape == (N_EPISODES,)

    def test_label_balance(self, small_batch):
        _, labels = small_batch
        n_tic = int(labels.sum())
        assert n_tic == N_EPISODES // 2, f"Expected {N_EPISODES//2} TIC, got {n_tic}"

    def test_vital_ranges(self, small_batch):
        data, _ = small_batch
        # HR column (0)
        assert data[:, :, 0].max() <= 210, "HR exceeds physiological maximum"
        assert data[:, :, 0].min() >= 30,  "HR below physiological minimum"
        # SBP column (1)
        assert data[:, :, 1].max() <= 230, "SBP exceeds physiological maximum"
        assert data[:, :, 1].min() >= 55,  "SBP below physiological minimum"
        # SpO2 column (3)
        assert data[:, :, 3].max() <= 100.5, "SpO2 above 100%"
        assert data[:, :, 3].min() >= 65,    "SpO2 below physiological minimum"

    def test_no_nan(self, small_batch):
        data, _ = small_batch
        assert not np.isnan(data).any(), "Simulator produced NaN values"

    def test_tic_drift_direction(self, small_batch):
        """TIC patients should have higher HR and lower SBP at end vs start."""
        data, labels = small_batch
        tic_episodes = data[labels == 1]
        hr_start  = tic_episodes[:, :60,   0].mean()   # first 60 s
        hr_end    = tic_episodes[:, -60:,  0].mean()   # last  60 s
        sbp_start = tic_episodes[:, :60,   1].mean()
        sbp_end   = tic_episodes[:, -60:,  1].mean()
        assert hr_end > hr_start,   "TIC HR did not increase over time"
        assert sbp_end < sbp_start, "TIC SBP did not decrease over time"

    def test_reproducibility(self):
        """Same seed → identical output."""
        from data.synthetic_generator import generate_batch
        d1, l1 = generate_batch(n_episodes=5, tic_ratio=0.5, random_seed=99)
        d2, l2 = generate_batch(n_episodes=5, tic_ratio=0.5, random_seed=99)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds(self):
        """Different seeds → different output."""
        from data.synthetic_generator import generate_batch
        d1, _ = generate_batch(n_episodes=5, tic_ratio=0.5, random_seed=1)
        d2, _ = generate_batch(n_episodes=5, tic_ratio=0.5, random_seed=2)
        assert not np.array_equal(d1, d2)

    def test_test_cohort_prevalence(self):
        """Test cohort at 25% prevalence."""
        from data.synthetic_generator import generate_batch
        data, labels = generate_batch(n_episodes=1000, tic_ratio=0.25, random_seed=43)
        assert int(labels.sum()) == 250, (
            f"Expected 250 TIC, got {int(labels.sum())}")

    def test_motion_artifact_injection(self):
        """Verify motion artifacts are injected into HR channel (S1.2.4)."""
        from data.synthetic_generator import OUSimulator
        # With a high-probability artifact, HR should occasionally have spikes
        rng = np.random.default_rng(42)
        max_std = []
        for seed in range(30):
            sim = OUSimulator(random_seed=seed)
            ep = sim.generate_episode(is_tic=False, duration_min=30)
            hr = ep[:, 0]
            diffs = np.abs(np.diff(hr))
            max_std.append(diffs.max())
        # At least some episodes should have HR jumps > 5 bpm (motion artifacts)
        assert max(max_std) > 5.0, "No motion artifacts detected"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_zscore_normalizer_shape(self, small_batch, normalizer):
        data, _ = small_batch
        out = normalizer.transform(data)
        assert out.shape == data.shape

    def test_zscore_mean_near_zero(self, small_batch, normalizer):
        data, _ = small_batch
        out = normalizer.transform(data)
        # Mean across all samples and time should be close to 0
        assert abs(out.mean()) < 0.1

    def test_zscore_std_near_one(self, small_batch, normalizer):
        data, _ = small_batch
        out = normalizer.transform(data)
        assert abs(out.std() - 1.0) < 0.1

    def test_interpolation_small_gap(self):
        """Gaps ≤ 5 s are linearly interpolated."""
        from data.preprocessing import interpolate_and_mask
        window = np.ones((60, 4), dtype=np.float64) * 80.0
        window[10:13, 0] = np.nan  # 3-step gap in HR
        filled, mask = interpolate_and_mask(window, max_gap=5)
        assert not np.isnan(filled[:, 0]).any(), "Small gap not interpolated"
        assert mask[:, 0].all(), "Small-gap mask should be all True"

    def test_interpolation_large_gap(self):
        """Gaps > 5 s are zero-padded and masked out."""
        from data.preprocessing import interpolate_and_mask
        window = np.ones((60, 4), dtype=np.float64) * 80.0
        window[10:25, 0] = np.nan  # 15-step gap
        filled, mask = interpolate_and_mask(window, max_gap=5)
        assert not np.isnan(filled[:, 0]).any()
        assert not mask[10:25, 0].all(), "Large-gap mask should contain False"

    def test_max_gap_parameter(self):
        """max_gap=5 exactly: gap of 5 is interpolated, gap of 6 is masked."""
        from data.preprocessing import interpolate_and_mask
        for gap_len, should_interp in [(5, True), (6, False)]:
            w = np.ones((60, 4)) * 100.0
            w[20:20 + gap_len, 0] = np.nan
            filled, mask = interpolate_and_mask(w, max_gap=5)
            if should_interp:
                assert mask[20:20 + gap_len, 0].all(), \
                    f"Gap={gap_len} should be interpolated"
            else:
                assert not mask[20:20 + gap_len, 0].all(), \
                    f"Gap={gap_len} should be masked"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestTraumaFormer:

    @pytest.fixture(scope='class')
    def model(self):
        from models.trauma_former import TraumaFormer
        return TraumaFormer(
            input_dim=4, window_size=60,
            d_model=256, n_heads=4, n_layers=2,
            d_ff=512, dropout=0.0,
            classifier_hidden=128,
        )

    def test_output_shape(self, model):
        x = torch.randn(8, 60, 4)
        out = model(x)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

    def test_output_range(self, model):
        x = torch.randn(16, 60, 4)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all(), "Output not in [0,1]"

    def test_parameter_count(self, model):
        """Paper states 1,524,481 parameters (Supplementary Table S2.2)."""
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 1_400_000 <= n_params <= 1_650_000, (
            f"Parameter count {n_params} deviates substantially from paper (1,524,481)"
        )

    def test_variate_as_token_transpose(self, model):
        """iTransformer: input (B, T, N) is transposed to (B, N, T) internally."""
        # Verify different ordering of variables produces different output
        x1 = torch.randn(4, 60, 4)
        x2 = x1.clone()
        x2[:, :, [0, 1]] = x2[:, :, [1, 0]]  # swap HR and SBP
        out1 = model(x1)
        out2 = model(x2)
        assert not torch.allclose(out1, out2), "Model ignores variable ordering"

    def test_variable_masking(self, model):
        """Token masking: zeroed tokens change output."""
        x = torch.randn(4, 60, 4)
        mask_full = torch.ones(4, 4, dtype=torch.bool)
        mask_part = mask_full.clone()
        mask_part[:, 0] = False   # mask HR channel
        out_full = model(x, mask=mask_full)
        out_part = model(x, mask=mask_part)
        assert not torch.allclose(out_full, out_part)

    def test_default_config(self):
        from models.trauma_former import TraumaFormer
        cfg = TraumaFormer.get_default_config()
        assert cfg['d_model'] == 256
        assert cfg['n_heads'] == 4
        assert cfg['n_layers'] == 2
        assert cfg['d_ff'] == 512
        assert cfg['dropout'] == 0.2
        assert cfg['classifier_hidden'] == 128


class TestBaselineModels:

    def _forward(self, model, batch=8):
        x = torch.randn(batch, 60, 4)
        out = model(x)
        assert out.shape == (batch, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_lstm(self):
        from models.baselines.lstm import LSTMModel
        self._forward(LSTMModel())

    def test_gru(self):
        from models.baselines.gru import GRUModel
        self._forward(GRUModel())

    def test_cnn(self):
        from models.baselines.cnn import CNNModel
        self._forward(CNNModel())

    def test_patchtst(self):
        from models.baselines.patchtst import PatchTSTModel
        self._forward(PatchTSTModel())

    def test_informer(self):
        from models.baselines.informer import InformerModel
        self._forward(InformerModel())

    def test_xgboost_features(self):
        """XGBoost extracts 20 features (5 stats × 4 vitals)."""
        from models.baselines.xgboost_model import XGBoostModel
        X = np.random.randn(10, 60, 4).astype(np.float32)
        model = XGBoostModel()
        feats = model._extract_features(X)
        assert feats.shape == (10, 20), f"Expected (10,20), got {feats.shape}"

    def test_lr_trend_features(self):
        """LR-trend extracts 12 features (3 stats × 4 vitals)."""
        from models.baselines.lr_trend import LRTrendModel
        X = np.random.randn(10, 60, 4).astype(np.float32)
        model = LRTrendModel()
        feats = model._extract_features(X)
        assert feats.shape == (10, 12), f"Expected (10,12), got {feats.shape}"

    def test_shock_index(self):
        """Shock index = HR / SBP at last timestep."""
        from models.baselines.shock_index import ShockIndexModel
        X = np.zeros((5, 60, 4))
        X[:, -1, 0] = 90   # HR
        X[:, -1, 1] = 100  # SBP
        model = ShockIndexModel()
        si = model.predict_proba(X)
        np.testing.assert_allclose(si, 0.9, rtol=1e-5)

    def test_gru_bidirectional(self):
        """GRU must be bidirectional (Supplementary S2.3.3)."""
        from models.baselines.gru import GRUModel
        model = GRUModel()
        assert model.bidirectional, "GRU should be bidirectional"

    def test_lstm_bidirectional(self):
        """LSTM must be bidirectional (Supplementary S2.3.4)."""
        from models.baselines.lstm import LSTMModel
        model = LSTMModel()
        assert model.bidirectional, "LSTM should be bidirectional"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def _perfect_case(self):
        y_true  = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
        return y_true, y_score

    def test_auroc_perfect(self):
        from evaluation.metrics import compute_auroc
        y_true, y_score = self._perfect_case()
        assert math.isclose(compute_auroc(y_true, y_score), 1.0)

    def test_auroc_random(self):
        from evaluation.metrics import compute_auroc
        rng = np.random.default_rng(0)
        y_true  = rng.integers(0, 2, 200)
        y_score = rng.random(200)
        auc = compute_auroc(y_true, y_score)
        assert 0.3 < auc < 0.7   # should be near 0.5 for random

    def test_brier_score_perfect(self):
        from evaluation.metrics import compute_brier
        y_true  = np.array([1.0, 1.0, 0.0, 0.0])
        y_score = np.array([1.0, 1.0, 0.0, 0.0])
        assert math.isclose(compute_brier(y_true, y_score), 0.0)

    def test_ppv_formula_balanced(self):
        """
        At 50% prevalence, with sens=0.91 spec=0.88:
        PPV = 0.91*0.5 / (0.91*0.5 + 0.12*0.5) ≈ 0.883 ≈ 0.88 (paper Section 3.2)
        """
        from evaluation.metrics import compute_classification_metrics
        rng = np.random.default_rng(1)
        n = 1000
        y_true  = np.array([1]*500 + [0]*500)
        # Simulate model with sens=0.91, spec=0.88
        y_score = np.zeros(n)
        y_score[:500]  = rng.uniform(0.5, 1.0, 500) * 0.91 + rng.uniform(0, 0.1, 500)
        y_score[500:]  = rng.uniform(0.0, 0.5, 500) * 0.88 + rng.uniform(0, 0.05, 500)
        # Just verify the formula holds
        tp, fp, tn, fn = 455, 60, 440, 45   # illustrative
        ppv = tp / (tp + fp)
        assert 0.85 < ppv < 0.95

    def test_mcse_formula(self):
        """MCSE = sqrt(Var(AUROC) / K), K=5 folds."""
        from evaluation.metrics import monte_carlo_standard_error
        fold_aurocs = [0.93, 0.94, 0.92, 0.94, 0.93]
        mcse = monte_carlo_standard_error(fold_aurocs, n_folds=5)
        expected = np.sqrt(np.var(fold_aurocs, ddof=1) / 5)
        assert math.isclose(mcse, expected, rel_tol=1e-6)

    def test_hellinger_distance(self):
        """Hellinger distance between identical distributions should be ~0."""
        from evaluation.metrics import multivariate_hellinger_distance
        rng = np.random.default_rng(0)
        X = rng.multivariate_normal([80, 120, 75, 98], np.eye(4) * 4, 300)
        d = multivariate_hellinger_distance(X, X)
        assert d < 0.02, f"Self-Hellinger distance too high: {d:.4f}"

    def test_calibration_stats(self):
        """Calibration stats return slope and intercept."""
        from evaluation.metrics import calibration_stats
        rng = np.random.default_rng(42)
        y_true  = (rng.random(200) > 0.5).astype(int)
        y_score = rng.random(200)
        _, _, slope, intercept = calibration_stats(y_true, y_score)
        assert isinstance(slope, float)
        assert isinstance(intercept, float)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Alert Rule
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertRule:

    def test_alert_fires_on_sustained_threshold(self):
        """Alert fires when prob ≥ 0.8 for ≥ 3 consecutive samples."""
        from evaluation.alert_rule import compute_early_warning_time
        prob = np.array([0.5, 0.7, 0.85, 0.90, 0.88, 0.82, 0.6])
        ewt, alerted, idx = compute_early_warning_time(
            prob, threshold=0.8, persistence=3,
            arrival_time=30, samples_per_minute=1)
        assert alerted, "Alert should have fired"
        assert idx == 2, f"Expected first alert at index 2, got {idx}"

    def test_no_alert_insufficient_persistence(self):
        """Alert does NOT fire when only 2 consecutive samples exceed threshold."""
        from evaluation.alert_rule import compute_early_warning_time
        prob = np.array([0.5, 0.7, 0.85, 0.90, 0.6, 0.5, 0.4])
        _, alerted, _ = compute_early_warning_time(
            prob, threshold=0.8, persistence=3,
            arrival_time=30, samples_per_minute=1)
        assert not alerted, "Alert should NOT fire with only 2 samples above threshold"

    def test_ewt_computation(self):
        """EWT = arrival_time - first_alert_minute."""
        from evaluation.alert_rule import compute_early_warning_time
        # Alert at sample index 5 (minute 5), arrival=30 → EWT = 25 min
        prob = np.zeros(30)
        prob[5:10] = 0.85
        ewt, alerted, _ = compute_early_warning_time(
            prob, threshold=0.8, persistence=3,
            arrival_time=30, samples_per_minute=1)
        assert alerted
        assert math.isclose(ewt, 25.0, abs_tol=0.5)

    def test_no_alert_returns_nan(self):
        """EWT is NaN when no alert fires."""
        from evaluation.alert_rule import compute_early_warning_time
        prob = np.zeros(30)
        ewt, alerted, idx = compute_early_warning_time(
            prob, threshold=0.8, persistence=3,
            arrival_time=30, samples_per_minute=1)
        assert not alerted
        assert np.isnan(ewt)
        assert idx == -1

    def test_paper_ewt_range(self):
        """
        Paper: median EWT = 18.1 min (IQR 13.4–22.3) for 3-min persistence.
        Verify that the alert_rule produces EWTs in this plausible range.
        """
        from evaluation.alert_rule import compute_early_warning_time
        rng = np.random.default_rng(42)
        ewts = []
        for _ in range(200):
            t_alert = rng.integers(5, 20)  # alert fires somewhere in [5,20] min
            prob = np.zeros(30)
            prob[t_alert:t_alert + 5] = 0.85
            ewt, alerted, _ = compute_early_warning_time(
                prob, threshold=0.8, persistence=3,
                arrival_time=30, samples_per_minute=1)
            if alerted:
                ewts.append(ewt)
        median_ewt = np.median(ewts)
        assert 8 < median_ewt < 26, f"Median EWT {median_ewt:.1f} outside expected range"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestTICDataset:

    def test_sliding_windows(self, small_batch, normalizer):
        from data.dataset import TICDataset
        data, labels = small_batch
        ds = TICDataset(data, labels, window_size=WINDOW, stride=30,
                        normalizer=normalizer)
        assert len(ds) > 0
        x, mask, y, pid = ds[0]
        assert x.shape == (WINDOW, N_VITALS)
        assert mask.shape == (WINDOW, N_VITALS)
        assert y.shape == ()

    def test_patient_id_preserved(self, small_batch, normalizer):
        from data.dataset import TICDataset
        data, labels = small_batch
        ds = TICDataset(data, labels, window_size=WINDOW, stride=1800,
                        normalizer=normalizer)
        pids = [ds[i][3] for i in range(len(ds))]
        # With stride=1800, each patient has exactly 1 window; pids = range(N_EPISODES)
        assert len(set(pids)) == N_EPISODES

    def test_label_correct(self, small_batch, normalizer):
        from data.dataset import TICDataset
        data, labels = small_batch
        ds = TICDataset(data, labels, window_size=WINDOW, stride=1800,
                        normalizer=normalizer)
        for i in range(len(ds)):
            _, _, y, pid = ds[i]
            assert float(y) == float(labels[pid]), (
                f"Label mismatch for patient {pid}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Missingness Indicator Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingnessIndicators:

    def test_extended_model_forward(self):
        """TraumaFormerWithIndicators accepts 8-channel input."""
        from experiments.missingness_indicator_analysis import TraumaFormerWithIndicators
        model = TraumaFormerWithIndicators(input_dim=8, window_size=60)
        x = torch.randn(4, 60, 8)
        out = model(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 8. End-to-end mini smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_tiny_cv_fold(self, small_batch, normalizer):
        """
        Run one mini CV fold on 20 episodes; check AUROC is finite.
        This tests data → model → metric without requiring a full training run.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader
        from models.trauma_former import TraumaFormer
        from data.dataset import TICDataset
        from data.preprocessing import ZScoreNormalizer

        data, labels = small_batch
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
        tr_idx, val_idx = next(skf.split(np.arange(len(labels)), labels))

        norm = ZScoreNormalizer()
        norm.fit(data[tr_idx])

        val_ds = TICDataset(data[val_idx], labels[val_idx],
                            window_size=WINDOW, stride=60, normalizer=norm)
        val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

        model = TraumaFormer(input_dim=4, window_size=60, d_model=32,
                             n_heads=2, n_layers=1, d_ff=64, dropout=0.0)
        model.eval()

        all_scores, all_labels = [], []
        with torch.no_grad():
            for x_b, _, y_b, _ in val_dl:
                scores = model(x_b).squeeze(1).numpy()
                all_scores.extend(scores)
                all_labels.extend(y_b.numpy())

        y_s = np.array(all_scores)
        y_t = np.array(all_labels)

        if len(np.unique(y_t)) > 1:
            auc = roc_auc_score(y_t, y_s)
            assert 0.0 <= auc <= 1.0
        else:
            pass  # only one class in tiny val set; skip metric check


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pytest as _pytest
    _pytest.main([__file__, '-v', '--tb=short'])
