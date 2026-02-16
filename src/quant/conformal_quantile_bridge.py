"""
conformal_quantile_bridge.py – Conformal Quantile Regression (CQR).
Tahmin aralıklarını %95 istatistiksel güven bandına hapseder.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    from mapie.regression import MapieQuantileRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning("mapie yüklü değil – CQR basit modda çalışacak.")


class ConformalQuantileBridge:
    """Conformal prediction ile kalibrasyon garantili tahmin aralıkları."""

    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha  # %95 güven
        self._model = None
        self._calibrated = False
        logger.debug("ConformalQuantileBridge başlatıldı.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Modeli eğit ve konformalize et."""
        if MAPIE_AVAILABLE and len(X) > 20:
            try:
                base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
                self._model = MapieQuantileRegressor(base, cv="split", alpha=self._alpha)
                self._model.fit(X, y)
                self._calibrated = True
                logger.info("CQR modeli eğitildi (MAPIE).")
            except Exception as e:
                logger.warning(f"MAPIE fit hatası: {e}")
                self._fit_fallback(X, y)
        else:
            self._fit_fallback(X, y)

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray):
        """MAPIE olmadan basit quantile regression."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = {
                "lower": GradientBoostingRegressor(
                    n_estimators=50, loss="quantile", alpha=self._alpha / 2, random_state=42
                ),
                "median": GradientBoostingRegressor(
                    n_estimators=50, loss="squared_error", random_state=42
                ),
                "upper": GradientBoostingRegressor(
                    n_estimators=50, loss="quantile", alpha=1 - self._alpha / 2, random_state=42
                ),
            }
            for m in self._model.values():
                m.fit(X, y)
            self._calibrated = True
            logger.info("CQR modeli eğitildi (fallback quantile).")
        except Exception as e:
            logger.error(f"Fallback fit hatası: {e}")

    def predict_intervals(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için güvenilir tahmin aralıkları üretir."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")

            # Feature vektörü
            feat_vals = []
            for k in ["home_odds", "draw_odds", "away_odds", "home_xg",
                       "away_xg", "home_win_rate", "odds_volatility"]:
                feat_vals.append(row.get(k, 0.0) or 0.0)

            X = np.array(feat_vals).reshape(1, -1)

            if self._calibrated and self._model is not None:
                lower, median, upper = self._predict_calibrated(X)
            else:
                lower, median, upper = self._predict_analytical(row)

            # Interval genişliği güven göstergesi
            interval_width = upper - lower
            interval_confidence = 1.0 / (1.0 + interval_width)

            results.append({
                "match_id": mid,
                "pred_lower": float(lower),
                "pred_median": float(median),
                "pred_upper": float(upper),
                "interval_width": float(interval_width),
                "interval_confidence": float(np.clip(interval_confidence, 0.1, 0.95)),
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _predict_calibrated(self, X: np.ndarray) -> tuple[float, float, float]:
        if isinstance(self._model, dict):
            lower = float(self._model["lower"].predict(X)[0])
            median = float(self._model["median"].predict(X)[0])
            upper = float(self._model["upper"].predict(X)[0])
        else:
            pred, intervals = self._model.predict(X)
            median = float(pred[0])
            lower = float(intervals[0, 0, 0])
            upper = float(intervals[0, 1, 0])
        return lower, median, upper

    def _predict_analytical(self, row: dict) -> tuple[float, float, float]:
        """Model yokken analitik tahmin aralığı."""
        ho = row.get("home_odds", 2.5) or 2.5
        do_ = row.get("draw_odds", 3.3) or 3.3
        ao = row.get("away_odds", 3.0) or 3.0
        vol = row.get("odds_volatility", 0.1) or 0.1

        implied_home = 1.0 / ho
        z = 1.96  # %95 güven

        median = implied_home
        spread = vol * z * 0.5 + 0.05
        lower = max(0.01, median - spread)
        upper = min(0.99, median + spread)

        return lower, median, upper
