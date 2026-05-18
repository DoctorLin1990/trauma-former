"""
XGBoost baseline with hand-crafted feature engineering.
Extracts 20 summary features per 60-second window (mean, slope, min, max,
variance for each of 4 vital signs).

Hyperparameters from Supplementary Table S2.5:
    n_estimators   = 500
    max_depth      = 6
    learning_rate  = 0.05
    subsample      = 0.8
    colsample_bytree = 0.8

Note: XGBoost receives only 20 features vs the full 240-point waveform
available to deep-learning models. The AUROC gap (0.821 vs 0.939) reflects
both architectural differences AND reduced feature richness (Supplementary S2.3.5).
"""
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class XGBoostModel(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier using 20 hand-crafted summary features per window.

    Features (5 per vital sign × 4 vital signs = 20):
        mean, slope (linear trend), minimum, maximum, variance
    """

    VITAL_ORDER = ["HR", "SBP", "DBP", "SpO2"]
    STAT_NAMES  = ["mean", "slope", "min", "max", "var"]

    def __init__(
        self,
        n_estimators:     int   = 500,
        max_depth:        int   = 6,
        learning_rate:    float = 0.05,
        subsample:        float = 0.8,
        colsample_bytree: float = 0.8,
        random_state:     int   = 42,
    ) -> None:
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state     = random_state
        self._model: xgb.XGBClassifier | None = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_windows, T, N)  → feature matrix (n_windows, 20).
        """
        n_windows, T, N = X.shape
        t_vec   = np.arange(T, dtype=np.float64)
        t_mean  = t_vec.mean()
        t_var   = float(np.sum((t_vec - t_mean) ** 2))

        features = np.empty((n_windows, N * len(self.STAT_NAMES)), dtype=np.float64)

        for i in range(n_windows):
            for j in range(N):
                y      = X[i, :, j].astype(np.float64)
                y_mean = float(np.nanmean(y))

                if t_var > 0:
                    slope = float(np.nansum((t_vec - t_mean) * (y - y_mean)) / t_var)
                else:
                    slope = 0.0

                base = j * len(self.STAT_NAMES)
                features[i, base + 0] = y_mean
                features[i, base + 1] = slope
                features[i, base + 2] = float(np.nanmin(y))
                features[i, base + 3] = float(np.nanmax(y))
                features[i, base + 4] = float(np.nanvar(y))

        return features

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """
        X: (n_windows, T, N)  or  (n_patients, T, N) with patient-level labels y.
        """
        X_feat = self._extract_features(X)
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="auc",
            use_label_encoder=False,
        )
        self._model.fit(X_feat, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows, 2) probability matrix."""
        check_is_fitted(self, "_model")
        X_feat = self._extract_features(X)
        return self._model.predict_proba(X_feat)

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows,) probability of TIC (positive class)."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba_positive(X) >= 0.5).astype(int)

    def get_feature_names(self) -> list:
        return [
            f"{v}_{s}" for v in self.VITAL_ORDER for s in self.STAT_NAMES
        ]

    @staticmethod
    def get_default_config() -> dict:
        return {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
