"""
glm_model.py – Genelleştirilmiş Lineer Modeller (GLM).

Basit Poisson sadece "Gol Sayısı"na bakar.
GLM ise şu soruyu çözer:
  "Hava durumu yağmurluysa VE deplasman takımı uçakla 3 saat
   yol geldiyse VE 3 sakat varsa → gol beklentisi ne olur?"

Poisson regresyon + dış faktörler (covariates) = Düzeltilmiş xG.

Kütüphane: statsmodels.api (sm.GLM)
Hedef: Gol sayısı (count data → Poisson family)
Bağımsız Değişkenler:
  - Hava durumu (yağmur/güneş/soğuk)
  - Seyahat mesafesi (km)
  - Sakat oyuncu sayısı
  - Dinlenme günü (son maçtan bu yana)
  - Ev sahibi avantajı
  - Rakip defans gücü
  - xG ortalaması (ham)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Poisson, NegativeBinomial
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels yüklü değil – GLM heuristik modda.")

try:
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False


@dataclass
class GLMFeatures:
    """GLM için girdi feature'ları."""
    # Ham performans
    xg_avg: float = 1.2           # Son 5 maç xG ortalaması
    xga_avg: float = 1.0          # xG against ortalaması
    goals_avg: float = 1.3        # Gol ortalaması

    # Dış faktörler (covariates)
    is_home: int = 1              # 1=ev, 0=deplasman
    rest_days: int = 7            # Son maçtan bu yana gün
    injuries: int = 0             # Sakat oyuncu sayısı
    travel_km: float = 0.0        # Seyahat mesafesi
    weather_rain: int = 0         # 1=yağmur, 0=değil
    weather_cold: int = 0         # 1=soğuk (<5°C), 0=değil
    altitude_diff: float = 0.0    # Rakım farkı (metre)

    # Rakip gücü
    opponent_def_strength: float = 1.0  # 1.0=lig ortalaması, >1=güçlü
    opponent_xga: float = 1.2     # Rakibin xGA ortalaması
    league_avg_goals: float = 2.6 # Lig gol ortalaması

    # Form
    form_points: float = 1.5      # Son 5 maç puan ortalaması (0-3)
    streak_win: int = 0           # Galibiyet serisi
    streak_lose: int = 0          # Yenilgi serisi


class GLMGoalPredictor:
    """GLM tabanlı Düzeltilmiş xG (Adjusted xG) motoru.

    Kullanım:
        glm = GLMGoalPredictor()
        glm.fit(training_data)
        prediction = glm.predict(home_features, away_features)
        # prediction = {"adjusted_home_xg": 1.65, "adjusted_away_xg": 0.95, ...}
    """

    FEATURE_COLS = [
        "xg_avg", "xga_avg", "goals_avg", "is_home", "rest_days",
        "injuries", "travel_km", "weather_rain", "weather_cold",
        "altitude_diff", "opponent_def_strength", "opponent_xga",
        "league_avg_goals", "form_points", "streak_win", "streak_lose",
    ]

    def __init__(self, family: str = "poisson"):
        self._family = family
        self._model = None
        self._fitted = False
        self._coefficients: dict[str, float] = {}
        self._adjustment_factors: dict[str, float] = {}
        logger.debug(f"GLMGoalPredictor başlatıldı (family={family}).")

    # ═══════════════════════════════════════════
    #  EĞİTİM
    # ═══════════════════════════════════════════
    def fit(self, training_data: list[dict] | Any):
        """Geçmiş maç verileriyle modeli eğit.

        training_data: [
            {"goals_scored": 2, "xg_avg": 1.5, "is_home": 1, "rest_days": 7, ...},
            ...
        ]
        """
        if not STATSMODELS_OK:
            logger.info("[GLM] statsmodels yok – heuristik faktörler kullanılacak.")
            self._fit_heuristic(training_data)
            return

        if isinstance(training_data, list):
            if not training_data:
                return
            X_data = []
            y_data = []
            for row in training_data:
                features = [float(row.get(col, 0) or 0) for col in self.FEATURE_COLS]
                X_data.append(features)
                y_data.append(int(row.get("goals_scored", 0) or 0))

            X = np.array(X_data)
            y = np.array(y_data)
        else:
            return

        if len(X) < 50:
            logger.info(f"[GLM] Yetersiz veri ({len(X)}) – heuristik mod.")
            self._fit_heuristic(training_data)
            return

        try:
            X_const = sm.add_constant(X)

            if self._family == "negative_binomial":
                family = NegativeBinomial()
            else:
                family = Poisson()

            model = sm.GLM(y, X_const, family=family)
            result = model.fit(disp=0)

            self._model = result
            self._fitted = True

            # Katsayıları kaydet
            param_names = ["const"] + self.FEATURE_COLS
            for name, coef in zip(param_names, result.params):
                self._coefficients[name] = float(coef)

            logger.success(
                f"[GLM] Model eğitildi: {len(X)} kayıt, "
                f"AIC={result.aic:.1f}, deviance={result.deviance:.1f}"
            )

            # Katsayı yorumları
            self._log_coefficients()

        except Exception as e:
            logger.error(f"[GLM] Eğitim hatası: {e}")
            self._fit_heuristic(training_data)

    def _fit_heuristic(self, data: list[dict] | Any):
        """statsmodels yokken heuristik düzeltme faktörleri."""
        self._adjustment_factors = {
            "home_advantage": 0.25,    # Ev sahibi +0.25 xG
            "rest_bonus_per_day": 0.02,# Her ekstra dinlenme günü
            "injury_penalty": -0.08,   # Her sakat oyuncu
            "travel_penalty_per_100km": -0.01,  # Her 100km yol
            "rain_penalty": -0.10,     # Yağmur
            "cold_penalty": -0.05,     # Soğuk hava
            "form_bonus": 0.05,        # Form puanı başına
            "streak_win_bonus": 0.03,  # Galibiyet serisi başına
            "streak_lose_penalty": -0.05,  # Yenilgi serisi başına
        }
        self._fitted = True
        logger.info("[GLM] Heuristik düzeltme faktörleri aktif.")

    def _log_coefficients(self):
        """Katsayıları yorumla ve logla."""
        important = sorted(
            [(k, v) for k, v in self._coefficients.items() if k != "const"],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for name, coef in important[:5]:
            direction = "↑" if coef > 0 else "↓"
            logger.debug(f"[GLM] {name}: {direction} {abs(coef):.4f}")

    # ═══════════════════════════════════════════
    #  TAHMİN
    # ═══════════════════════════════════════════
    def predict(self, home_features: GLMFeatures | dict,
                away_features: GLMFeatures | dict) -> dict:
        """İki takımın düzeltilmiş xG'sini tahmin et."""
        if isinstance(home_features, dict):
            home_f = home_features
        else:
            home_f = home_features.__dict__

        if isinstance(away_features, dict):
            away_f = away_features
        else:
            away_f = away_features.__dict__

        if self._model and STATSMODELS_OK:
            home_xg = self._predict_glm(home_f)
            away_xg = self._predict_glm(away_f)
        else:
            home_xg = self._predict_heuristic(home_f)
            away_xg = self._predict_heuristic(away_f)

        # Maç olasılıkları (düzeltilmiş xG'den Poisson)
        from scipy.stats import poisson
        prob_home = prob_draw = prob_away = 0.0
        for h_goals in range(8):
            for a_goals in range(8):
                p = (poisson.pmf(h_goals, home_xg) *
                     poisson.pmf(a_goals, away_xg))
                if h_goals > a_goals:
                    prob_home += p
                elif h_goals == a_goals:
                    prob_draw += p
                else:
                    prob_away += p

        # Düzeltme detayları
        raw_home_xg = float(home_f.get("xg_avg", 1.2))
        raw_away_xg = float(away_f.get("xg_avg", 1.0))

        return {
            "adjusted_home_xg": round(home_xg, 3),
            "adjusted_away_xg": round(away_xg, 3),
            "raw_home_xg": raw_home_xg,
            "raw_away_xg": raw_away_xg,
            "home_xg_correction": round(home_xg - raw_home_xg, 3),
            "away_xg_correction": round(away_xg - raw_away_xg, 3),
            "prob_home": round(prob_home, 4),
            "prob_draw": round(prob_draw, 4),
            "prob_away": round(prob_away, 4),
            "expected_total_goals": round(home_xg + away_xg, 2),
            "method": "glm" if self._model else "heuristic",
        }

    def _predict_glm(self, features: dict) -> float:
        """GLM model ile düzeltilmiş xG tahmin et."""
        X = [float(features.get(col, 0) or 0) for col in self.FEATURE_COLS]
        X_const = np.array([[1.0] + X])  # constant ekle

        try:
            mu = self._model.predict(X_const)
            return float(max(mu[0], 0.1))
        except Exception as e:
            logger.debug(f"Exception caught: {e}")
            return self._predict_heuristic(features)

    def _predict_heuristic(self, features: dict) -> float:
        """Heuristik düzeltme ile xG hesapla."""
        base_xg = float(features.get("xg_avg", 1.2))
        adj = self._adjustment_factors

        correction = 0.0

        # Ev sahibi avantajı
        if features.get("is_home", 0):
            correction += adj.get("home_advantage", 0.25)

        # Dinlenme günü
        rest = int(features.get("rest_days", 7))
        if rest > 7:
            correction += (rest - 7) * adj.get("rest_bonus_per_day", 0.02)
        elif rest < 4:
            correction -= (4 - rest) * 0.05  # Yorgunluk cezası

        # Sakatlık
        injuries = int(features.get("injuries", 0))
        correction += injuries * adj.get("injury_penalty", -0.08)

        # Seyahat
        travel = float(features.get("travel_km", 0))
        correction += (travel / 100) * adj.get("travel_penalty_per_100km", -0.01)

        # Hava durumu
        if features.get("weather_rain"):
            correction += adj.get("rain_penalty", -0.10)
        if features.get("weather_cold"):
            correction += adj.get("cold_penalty", -0.05)

        # Form
        form = float(features.get("form_points", 1.5))
        correction += (form - 1.5) * adj.get("form_bonus", 0.05)

        # Seri
        correction += int(features.get("streak_win", 0)) * adj.get("streak_win_bonus", 0.03)
        correction += int(features.get("streak_lose", 0)) * adj.get("streak_lose_penalty", -0.05)

        # Rakip defans gücü
        opp_def = float(features.get("opponent_def_strength", 1.0))
        if opp_def != 0:
            correction *= (1.0 / opp_def)

        adjusted = max(base_xg + correction, 0.1)
        return adjusted

    # ═══════════════════════════════════════════
    #  TOPLU TAHMİN
    # ═══════════════════════════════════════════
    def predict_for_dataframe(self, df) -> list[dict]:
        """Polars DataFrame üzerinden toplu tahmin."""
        results = []
        if not hasattr(df, "iter_rows"):
            return results

        for row in df.iter_rows(named=True):
            home_f = {col: row.get(f"home_{col}", row.get(col, 0)) for col in self.FEATURE_COLS}
            away_f = {col: row.get(f"away_{col}", row.get(col, 0)) for col in self.FEATURE_COLS}
            home_f["is_home"] = 1
            away_f["is_home"] = 0

            pred = self.predict(home_f, away_f)
            pred["match_id"] = row.get("match_id", "")
            results.append(pred)

        return results

    @property
    def coefficients(self) -> dict:
        return dict(self._coefficients)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
