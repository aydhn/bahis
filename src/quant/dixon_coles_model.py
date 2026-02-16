"""
dixon_coles_model.py – Dixon-Coles düzeltilmiş Poisson modeli.

Standart Poisson, düşük skorlu beraberlikleri (0-0, 1-1) olduğundan
DÜŞÜK hesaplar. Dixon-Coles, bir τ(x,y,λ,μ,ρ) düzeltme faktörü ile
bu beraberlikleri gerçeğe yakınlaştırır.

Kaynak: Dixon & Coles (1997) "Modelling Association Football Scores
        and Inefficiencies in the Football Betting Market"
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from dataclasses import dataclass
from loguru import logger
import polars as pl


@dataclass
class DCParams:
    """Dixon-Coles model parametreleri."""
    home_attack: dict[str, float]   # Takım → atak gücü
    home_defence: dict[str, float]  # Takım → savunma gücü (düşük=iyi)
    away_attack: dict[str, float]
    away_defence: dict[str, float]
    home_advantage: float = 0.25    # Ev sahibi avantajı (log-scale)
    rho: float = -0.13             # Düşük skor düzeltme parametresi


class DixonColesModel:
    """Dixon-Coles düzeltilmiş Poisson modeli.

    Standart Poisson'dan farkı:
    - ρ (rho) parametresi ile 0-0, 1-0, 0-1, 1-1 skorlarının
      olasılıklarını düzeltir.
    - Exponential time decay ile eski maçları unutur.
    """

    def __init__(self, rho: float = -0.13, decay_rate: float = 0.005):
        self._rho = rho
        self._decay_rate = decay_rate      # Exponential time decay λ
        self._params: DCParams | None = None
        self._teams: set[str] = set()
        self._fitted = False
        logger.debug(f"DixonColesModel başlatıldı (ρ={rho}, decay={decay_rate}).")

    # ═══════════════════════════════════════════
    #  τ (tau) düzeltme fonksiyonu – modelin kalbi
    # ═══════════════════════════════════════════
    @staticmethod
    def tau(home_goals: int, away_goals: int,
            lambda_home: float, mu_away: float, rho: float) -> float:
        """Dixon-Coles τ düzeltme faktörü.
        Sadece düşük skorları (0-0, 1-0, 0-1, 1-1) düzeltir.
        Diğer skorlarda τ = 1 (nötr).
        """
        if home_goals == 0 and away_goals == 0:
            return 1.0 - lambda_home * mu_away * rho
        elif home_goals == 0 and away_goals == 1:
            return 1.0 + lambda_home * rho
        elif home_goals == 1 and away_goals == 0:
            return 1.0 + mu_away * rho
        elif home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        else:
            return 1.0

    def score_probability(self, home_goals: int, away_goals: int,
                          lambda_home: float, mu_away: float) -> float:
        """P(Home=x, Away=y) – Dixon-Coles düzeltilmiş."""
        p_home = poisson.pmf(home_goals, mu=max(lambda_home, 0.01))
        p_away = poisson.pmf(away_goals, mu=max(mu_away, 0.01))
        t = self.tau(home_goals, away_goals, lambda_home, mu_away, self._rho)
        return p_home * p_away * max(t, 0.001)

    def score_matrix(self, lambda_home: float, mu_away: float,
                     max_goals: int = 8) -> np.ndarray:
        """Tam skor olasılık matrisini döndürür."""
        mat = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                mat[i][j] = self.score_probability(i, j, lambda_home, mu_away)

        # Normalize (toplamı 1 yap)
        total = mat.sum()
        if total > 0:
            mat /= total
        return mat

    # ═══════════════════════════════════════════
    #  TIME DECAY – eski maçları unut
    # ═══════════════════════════════════════════
    def time_weight(self, days_ago: float) -> float:
        """W_t = e^(-λt) – exponential time decay."""
        return np.exp(-self._decay_rate * days_ago)

    # ═══════════════════════════════════════════
    #  FIT – modeli eğit
    # ═══════════════════════════════════════════
    def fit(self, matches: list[dict]):
        """Geçmiş maç verileriyle modeli eğitir.

        Her match dict'i: {home, away, home_goals, away_goals, days_ago}
        """
        if not matches:
            logger.warning("Dixon-Coles: eğitim verisi boş.")
            return

        # Takımları topla
        self._teams = set()
        for m in matches:
            self._teams.add(m["home"])
            self._teams.add(m["away"])

        teams_list = sorted(self._teams)
        n = len(teams_list)
        team_idx = {t: i for i, t in enumerate(teams_list)}

        # Parametreleri başlat: attack=1, defence=1, home_adv=0.25
        n_params = 2 * n + 1 + 1  # attack + defence + home_adv + rho
        x0 = np.ones(n_params)
        x0[-2] = 0.25   # home advantage
        x0[-1] = -0.13  # rho

        # Log-likelihood (negatifini minimize)
        def neg_log_likelihood(params):
            attacks = params[:n]
            defences = params[n:2*n]
            home_adv = params[-2]
            rho = np.clip(params[-1], -0.999, 0.999)

            ll = 0.0
            for m in matches:
                i_home = team_idx[m["home"]]
                i_away = team_idx[m["away"]]

                lambda_h = np.exp(attacks[i_home] + defences[i_away] + home_adv)
                mu_a = np.exp(attacks[i_away] + defences[i_home])

                lambda_h = np.clip(lambda_h, 0.01, 10)
                mu_a = np.clip(mu_a, 0.01, 10)

                hg = m["home_goals"]
                ag = m["away_goals"]

                p = self.score_probability(hg, ag, lambda_h, mu_a)
                p = max(p, 1e-12)

                weight = self.time_weight(m.get("days_ago", 0))
                ll += weight * np.log(p)

            # Constraint: sum(attacks) = n (ortalama 1)
            penalty = 100 * (np.sum(attacks) - n) ** 2
            return -ll + penalty

        try:
            result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                              options={"maxiter": 500, "ftol": 1e-6})
            if result.success or result.fun < neg_log_likelihood(x0):
                attacks = result.x[:n]
                defences = result.x[n:2*n]
                home_adv = result.x[-2]
                rho = np.clip(result.x[-1], -0.999, 0.999)

                self._params = DCParams(
                    home_attack={t: attacks[team_idx[t]] for t in teams_list},
                    home_defence={t: defences[team_idx[t]] for t in teams_list},
                    away_attack={t: attacks[team_idx[t]] for t in teams_list},
                    away_defence={t: defences[team_idx[t]] for t in teams_list},
                    home_advantage=home_adv,
                    rho=rho,
                )
                self._rho = rho
                self._fitted = True
                logger.info(f"Dixon-Coles eğitildi: {n} takım, ρ={rho:.3f}, home_adv={home_adv:.3f}")
            else:
                logger.warning("Dixon-Coles optimizasyon yakınsamadı.")
        except Exception as e:
            logger.error(f"Dixon-Coles fit hatası: {e}")

    # ═══════════════════════════════════════════
    #  PREDICT
    # ═══════════════════════════════════════════
    def predict(self, home: str, away: str,
                home_xg: float = 1.4, away_xg: float = 1.1) -> dict:
        """Maç olasılıklarını Dixon-Coles ile hesaplar."""
        if self._fitted and self._params:
            return self._predict_fitted(home, away)
        else:
            return self._predict_xg(home_xg, away_xg)

    def _predict_fitted(self, home: str, away: str) -> dict:
        p = self._params
        att_h = p.home_attack.get(home, 1.0)
        def_a = p.away_defence.get(away, 1.0)
        att_a = p.away_attack.get(away, 1.0)
        def_h = p.home_defence.get(home, 1.0)

        lambda_h = np.exp(att_h + def_a + p.home_advantage)
        mu_a = np.exp(att_a + def_h)

        lambda_h = np.clip(lambda_h, 0.01, 10)
        mu_a = np.clip(mu_a, 0.01, 10)

        return self._compute_probs(lambda_h, mu_a, home, away)

    def _predict_xg(self, home_xg: float, away_xg: float) -> dict:
        """Eğitim yokken xG tabanlı tahmin."""
        return self._compute_probs(home_xg, away_xg, "home", "away")

    def _compute_probs(self, lambda_h: float, mu_a: float,
                       home: str, away: str) -> dict:
        mat = self.score_matrix(lambda_h, mu_a)
        n = mat.shape[0]

        p_home = sum(mat[i][j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(mat[i][i] for i in range(n))
        p_away = sum(mat[i][j] for i in range(n) for j in range(n) if i < j)

        # Over/Under
        over25 = sum(mat[i][j] for i in range(n) for j in range(n) if i + j > 2.5)
        btts = sum(mat[i][j] for i in range(1, n) for j in range(1, n))

        # En olası skorlar
        scores = []
        for i in range(min(n, 6)):
            for j in range(min(n, 6)):
                scores.append({"score": f"{i}-{j}", "prob": float(mat[i][j])})
        scores.sort(key=lambda x: x["prob"], reverse=True)

        # Standart Poisson ile karşılaştır (rho=0)
        standard_draw = sum(
            poisson.pmf(i, lambda_h) * poisson.pmf(i, mu_a)
            for i in range(n)
        )
        dc_adjustment = p_draw - standard_draw

        return {
            "home_team": home,
            "away_team": away,
            "prob_home": float(p_home),
            "prob_draw": float(p_draw),
            "prob_away": float(p_away),
            "prob_over25": float(over25),
            "prob_btts": float(btts),
            "lambda_home": float(lambda_h),
            "mu_away": float(mu_a),
            "rho": float(self._rho),
            "dc_draw_adjustment": float(dc_adjustment),
            "top_scores": scores[:5],
        }

    def predict_for_dataframe(self, features: pl.DataFrame) -> pl.DataFrame:
        results = []
        for row in features.iter_rows(named=True):
            pred = self.predict(
                row.get("home_team", ""),
                row.get("away_team", ""),
                row.get("home_xg", 1.4) or 1.4,
                row.get("away_xg", 1.1) or 1.1,
            )
            pred["match_id"] = row.get("match_id", "")
            results.append(pred)
        return pl.DataFrame(results) if results else pl.DataFrame()

    def compare_with_standard_poisson(self, home_xg: float, away_xg: float) -> dict:
        """Dixon-Coles vs standart Poisson karşılaştırması."""
        dc = self._compute_probs(home_xg, away_xg, "home", "away")
        mat_std = np.outer(
            poisson.pmf(range(9), home_xg),
            poisson.pmf(range(9), away_xg),
        )
        std_draw = sum(mat_std[i][i] for i in range(9))
        std_home = sum(mat_std[i][j] for i in range(9) for j in range(9) if i > j)

        return {
            "dc_home": dc["prob_home"],
            "dc_draw": dc["prob_draw"],
            "dc_away": dc["prob_away"],
            "std_home": float(std_home),
            "std_draw": float(std_draw),
            "std_away": float(1 - std_home - std_draw),
            "draw_correction": dc["prob_draw"] - std_draw,
            "note": "Pozitif draw_correction = DC beraberliği daha yüksek tahmin ediyor (doğru).",
        }
