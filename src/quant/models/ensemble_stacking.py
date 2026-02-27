"""
ensemble_stacking.py – Stacking Ensemble (Meta-Model).

Tek bir model asla yeterli değildir. Her modelin kör noktası vardır.
Stacking, bu modellerin çıktılarını alıp "Nerede kim daha iyi?"
sorusunu çözen bir üst model (Meta-Learner) eğitir.

Katman 0 (Base Learners):
  - Benter Model (Poisson + Expert)
  - Dixon-Coles (Time Decay)
  - LightGBM (Tabular Boost)
  - LSTM Trend (Momentum)
  - Survival Estimator (Goal Timing)

Katman 1 (Meta-Learner):
  - Logistic Regression (Basit, yorumlanabilir)
  - XGBoost (Güçlü, non-linear)

Özellikler:
  - OOF (Out-of-Fold) Predictions: Overfitting'i önlemek için CV
  - Dynamic Weighting: Maçın karakterine göre model ağırlıkları değişir
  - Blending: Basit ağırlıklı ortalama (fallback)

Akış:
  1. Base modelleri eğit
  2. Cross-Validation ile OOF tahminlerini üret
  3. Bu tahminleri yeni feature olarak Meta-Model'e ver
  4. Meta-Model final olasılığı hesaplar
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("sklearn yüklü değil – Stacking yerine Blending kullanılacak.")

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False


@dataclass
class StackingReport:
    """Stacking performans raporu."""
    meta_model: str = ""
    base_models: list[str] = field(default_factory=list)
    accuracy: float = 0.0
    log_loss: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)  # Model önem dereceleri
    method: str = ""


class StackingEnsemble:
    """Modelleri birleştiren Stacking motoru.

    Kullanım:
        stacker = StackingEnsemble()

        # Base model tahminlerini topla (DataFrame)
        # Sütunlar: [benter_prob, lgbm_prob, lstm_prob, ..., target]
        stacker.fit(X_train, y_train)

        # Final tahmin
        final_probs = stacker.predict(X_test)
    """

    def __init__(self, meta_learner: str = "logistic",
                 n_folds: int = 5, use_calibration: bool = True):
        self._meta_learner_type = meta_learner
        self._n_folds = n_folds
        self._use_calibration = use_calibration
        self._meta_model: Any = None
        self._is_fitted = False
        self._feature_names: list[str] = []

        logger.debug(f"[Stacking] Başlatıldı (meta={meta_learner}).")

    def fit(self, X: np.ndarray | pl.DataFrame, y: np.ndarray,
            feature_names: list[str] | None = None) -> StackingReport:
        """Meta-modeli eğit."""
        report = StackingReport(
            meta_model=self._meta_learner_type,
            method="blending" if not SKLEARN_OK else "stacking",
        )

        if isinstance(X, pl.DataFrame):
            self._feature_names = X.columns
            X = X.to_numpy()
        elif feature_names:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"model_{i}" for i in range(X.shape[1])]

        report.base_models = self._feature_names

        if not SKLEARN_OK:
            # Fallback: eşit ağırlıklı blending
            self._is_fitted = True
            report.weights = {name: 1.0 / len(self._feature_names) for name in self._feature_names}
            return report

        # Meta-learner seçimi
        if self._meta_learner_type == "xgboost" and XGB_OK:
            base_estimator = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                eval_metric="logloss", use_label_encoder=False,
            )
        else:
            base_estimator = LogisticRegression(
                solver="lbfgs", C=1.0, max_iter=1000,
            )

        # Kalibrasyon (Platt Scaling / Isotonic)
        if self._use_calibration:
            self._meta_model = CalibratedClassifierCV(
                base_estimator, method="isotonic", cv=self._n_folds,
            )
        else:
            self._meta_model = base_estimator

        try:
            self._meta_model.fit(X, y)
            self._is_fitted = True

            # Model ağırlıklarını/önemini çıkarma
            if hasattr(self._meta_model, "calibrated_classifiers_"):
                # CalibratedClassifierCV içindeki base modellerin katsayı ortalaması
                coefs = []
                for clf in self._meta_model.calibrated_classifiers_:
                    if hasattr(clf.base_estimator, "coef_"):
                        coefs.append(clf.base_estimator.coef_[0])
                    elif hasattr(clf.base_estimator, "feature_importances_"):
                        coefs.append(clf.base_estimator.feature_importances_)

                if coefs:
                    avg_coef = np.mean(coefs, axis=0)
                    # Normalize
                    avg_coef = np.abs(avg_coef) / np.sum(np.abs(avg_coef))
                    report.weights = {
                        name: float(w)
                        for name, w in zip(self._feature_names, avg_coef)
                    }

            elif hasattr(self._meta_model, "coef_"):
                coef = self._meta_model.coef_[0]
                norm_coef = np.abs(coef) / np.sum(np.abs(coef))
                report.weights = {
                    name: float(w)
                    for name, w in zip(self._feature_names, norm_coef)
                }

            logger.info(f"[Stacking] Meta-model eğitildi. Ağırlıklar: {report.weights}")

        except Exception as e:
            logger.error(f"[Stacking] Eğitim hatası: {e}")
            self._is_fitted = False

        return report

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Final olasılıkları tahmin et."""
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        if not self._is_fitted:
            # Fallback: ortalama al
            return np.mean(X, axis=1)

        try:
            # [:, 1] → pozitif sınıf (kazanma) olasılığı
            probs = self._meta_model.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs # Multiclass ise (Home, Draw, Away)
        except Exception as e:
            logger.warning(f"[Stacking] Tahmin hatası ({e}), ortalama kullanılıyor.")
            return np.mean(X, axis=1)

    def blend(self, predictions: dict[str, float],
              weights: dict[str, float] | None = None) -> float:
        """Basit ağırlıklı ortalama (Blending)."""
        if not predictions:
            return 0.5

        if weights is None:
            # Eşit ağırlık
            return float(np.mean(list(predictions.values())))

        total_weight = 0.0
        weighted_sum = 0.0

        for model, prob in predictions.items():
            w = weights.get(model, 1.0)
            weighted_sum += prob * w
            total_weight += w

        return weighted_sum / max(total_weight, 1e-9)
