"""
Linear Trend Logistic Regression (LR-trend) baseline.

For each 60-second window and each of the 4 vital signs, fit an
ordinary least-squares linear regression and extract three features:
    slope (β₁), intercept (β₀), coefficient of determination (R²)

This yields 4 × 3 = 12 features per window, which are fed into an
L2-regularised logistic regression (C = 1.0, solver = 'lbfgs').

This is the KEY diagnostic baseline in the paper: it directly quantifies
how much of the prediction task reduces to detecting monotonic vital-sign
trends. An AUROC of 0.917 (vs Trauma-Former's 0.939) shows the gap is
only 0.022 AUROC points.

Reference: Supplementary S2.3.1.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class LRTrendModel(BaseEstimator, ClassifierMixin):
    """
    Logistic regression on linear-regression features extracted per window.

    Features per window (12 total):
        For each of HR, SBP, DBP, SpO2:
            slope β₁  — linear trend coefficient (most discriminative for TIC)
            intercept β₀ — level at t=0
            R²           — goodness of fit of the linear trend

    No hyperparameter search was conducted for LR-trend; C = 1.0 is the
    standard scikit-learn default (Supplementary S2.3.1).
    """

    N_FEATURES_PER_VAR = 3   # slope, intercept, R²
    VAR_NAMES = ["HR", "SBP", "DBP", "SpO2"]

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 solver: str = "lbfgs", random_state: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self._lr: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_windows, T, N) — e.g. (n, 60, 4)
        Returns feature matrix of shape (n_windows, 12).
        """
        n_windows, T, N = X.shape
        t_vec = np.arange(T, dtype=np.float64)
        t_mean = t_vec.mean()
        t_var  = np.sum((t_vec - t_mean) ** 2)

        features = np.empty((n_windows, N * self.N_FEATURES_PER_VAR), dtype=np.float64)

        for i in range(n_windows):
            for j, _ in enumerate(self.VAR_NAMES):
                y = X[i, :, j].astype(np.float64)
                # Guard against all-NaN or zero-variance windows
                if np.all(np.isnan(y)) or t_var == 0:
                    slope, intercept, r2 = 0.0, float(np.nanmean(y)), 0.0
                else:
                    y_mean  = np.nanmean(y)
                    slope   = np.nansum((t_vec - t_mean) * (y - y_mean)) / t_var
                    intercept = y_mean - slope * t_mean
                    ss_res  = np.nansum((y - (slope * t_vec + intercept)) ** 2)
                    ss_tot  = np.nansum((y - y_mean) ** 2)
                    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    r2      = float(np.clip(r2, 0.0, 1.0))

                base = j * self.N_FEATURES_PER_VAR
                features[i, base + 0] = slope
                features[i, base + 1] = intercept
                features[i, base + 2] = r2

        return features

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LRTrendModel":
        """X: (n_windows, T, N);  y: (n_windows,) binary labels."""
        X_feat = self._extract_features(X)
        self._lr = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._lr.fit(X_feat, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows, 2) probability matrix (positive class = index 1)."""
        check_is_fitted(self, "_lr")
        X_feat = self._extract_features(X)
        return self._lr.predict_proba(X_feat)

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows,) probability of TIC (positive class)."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba_positive(X) >= 0.5).astype(int)

    def get_feature_names(self) -> list:
        names = []
        for vname in self.VAR_NAMES:
            names += [f"{vname}_slope", f"{vname}_intercept", f"{vname}_R2"]
        return names

    @staticmethod
    def get_default_config() -> dict:
        return {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
