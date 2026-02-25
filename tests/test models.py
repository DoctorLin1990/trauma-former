"""
Unit tests for model architectures.
Tests forward pass, output shape, and absence of NaN gradients.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.trauma_former import TraumaFormer
from models.baselines.lstm import LSTMModel
from models.baselines.xgboost_model import XGBoostModel
from models.baselines.shock_index import ShockIndex
# Skip PatchTST and Informer if not installed
try:
    from models.baselines.patchtst import PatchTSTModel
    PATCHTST_AVAILABLE = True
except ImportError:
    PATCHTST_AVAILABLE = False
try:
    from models.baselines.informer import InformerModel
    INFORMER_AVAILABLE = True
except ImportError:
    INFORMER_AVAILABLE = False

class TestTraumaFormer:
    def test_forward_shape(self):
        model = TraumaFormer(input_dim=4, window_size=60, d_model=256, n_layers=2)
        batch_size = 8
        x = torch.randn(batch_size, 60, 4)
        out = model(x)
        assert out.shape == (batch_size, 1)
        assert torch.all((out >= 0) & (out <= 1))  # sigmoid output

    def test_forward_no_nan(self):
        model = TraumaFormer()
        x = torch.randn(4, 60, 4)
        out = model(x)
        assert not torch.any(torch.isnan(out))

    def test_backward(self):
        model = TraumaFormer()
        x = torch.randn(2, 60, 4)
        y = torch.rand(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward()
        # Check that gradients are computed (not None)
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

class TestLSTM:
    def test_forward_shape(self):
        model = LSTMModel(input_dim=4, hidden_size=64, num_layers=2, bidirectional=True)
        x = torch.randn(6, 60, 4)
        out = model(x)
        assert out.shape == (6, 1)

    def test_forward_no_nan(self):
        model = LSTMModel()
        x = torch.randn(3, 60, 4)
        out = model(x)
        assert not torch.any(torch.isnan(out))

class TestXGBoost:
    def test_fit_predict_shape(self):
        model = XGBoostModel(n_estimators=10, max_depth=3)  # small for testing
        X = np.random.randn(20, 60, 4)
        y = np.random.randint(0, 2, size=20)
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (20,)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_feature_extraction(self):
        # Test internal feature extraction
        model = XGBoostModel()
        X = np.random.randn(5, 60, 4)
        feat = model._extract_features(X)
        assert feat.shape == (5, 20)  # 4 variables * 5 features

class TestShockIndex:
    def test_predict_proba(self):
        model = ShockIndex(threshold=1.0)
        X = np.array([[[70, 120, 80, 98]]])  # one window, last step (but shape must be (n,T,4))
        # Create proper shape: n=1, T=60, N=4
        X = np.random.randn(1, 60, 4)
        # Override last step HR/SBP
        X[0, -1, 0] = 100  # HR
        X[0, -1, 1] = 110  # SBP
        si = model.predict_proba(X)
        expected_si = 100/110
        assert np.abs(si[0] - expected_si) < 1e-6

    def test_predict(self):
        model = ShockIndex(threshold=1.0)
        X = np.random.randn(3, 60, 4)
        X[:, -1, 0] = [90, 110, 80]   # HR
        X[:, -1, 1] = [100, 90, 85]   # SBP
        preds = model.predict(X)
        expected = (np.array([90/100, 110/90, 80/85]) > 1.0).astype(int)
        assert np.array_equal(preds, expected)

@pytest.mark.skipif(not PATCHTST_AVAILABLE, reason="PatchTST not installed")
class TestPatchTST:
    def test_forward_shape(self):
        model = PatchTSTModel(input_dim=4, window_size=60)
        x = torch.randn(4, 60, 4)
        out = model(x)
        assert out.shape == (4, 1)

@pytest.mark.skipif(not INFORMER_AVAILABLE, reason="Informer not installed")
class TestInformer:
    def test_forward_shape(self):
        model = InformerModel(enc_in=4, dec_in=4, c_out=4, seq_len=60, out_len=1)
        x = torch.randn(4, 60, 4)
        out = model(x)
        assert out.shape == (4, 1)