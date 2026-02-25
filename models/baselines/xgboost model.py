"""
XGBoost baseline with feature engineering from 60-second windows.
Extracts 20 summary features per window (mean, slope, min, max, variance for each of 4 vital signs).
"""
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Optional

class XGBoostModel(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier using handcrafted features from each window.
    Features per window (20 total):
        For each of HR, SBP, DBP, SpO2:
            - mean
            - slope (linear trend over 60 seconds)
            - minimum
            - maximum
            - variance
    """
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

    def _extract_features(self, X):
        """
        X: numpy array of shape (n_windows, T, N) with T=60, N=4.
        Returns feature matrix of shape (n_windows, 20).
        """
        n_windows, T, N = X.shape
        features = []
        for i in range(n_windows):
            win = X[i]  # (60, 4)
            feat = []
            for j in range(N):
                series = win[:, j]
                feat.append(np.mean(series))
                # slope: linear fit over time (0..59)
                t = np.arange(T)
                slope = np.polyfit(t, series, 1)[0] if not np.all(np.isnan(series)) else 0.0
                feat.append(slope)
                feat.append(np.min(series))
                feat.append(np.max(series))
                feat.append(np.var(series))
            features.append(feat)
        return np.array(features)

    def fit(self, X, y):
        """X: numpy array of shape (n_windows, T, N)"""
        X_feat = self._extract_features(X)
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
        self.model.fit(X_feat, y)
        return self

    def predict_proba(self, X):
        """Return probability of TIC (positive class index 1)."""
        check_is_fitted(self)
        X_feat = self._extract_features(X)
        proba = self.model.predict_proba(X_feat)
        # Return probability for class 1
        return proba[:, 1] if proba.shape[1] > 1 else proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)