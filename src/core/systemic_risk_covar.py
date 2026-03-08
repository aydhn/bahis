"""
systemic_risk_covar.py – CoVaR ile sistemik risk ölçümü.
Portföydeki maçlar arasındaki bulaşıcı riskleri ölçer.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


class SystemicRiskCoVaR:
    """Conditional Value-at-Risk (CoVaR) tabanlı sistemik risk ölçümü."""

    def __init__(self, confidence_level: float = 0.95, n_simulations: int = 10000):
        self._alpha = confidence_level
        self._n_sim = n_simulations
        logger.debug("SystemicRiskCoVaR başlatıldı.")

    def measure(self, ensemble: list[dict]) -> dict:
        """Portföy genelinde CoVaR hesaplar."""
        if not ensemble:
            return {"covar": 0.0, "var": 0.0, "delta_covar": 0.0, "risk_level": "low"}

        n = len(ensemble)
        evs = np.array([m.get("best_ev", m.get("ev_home", 0)) for m in ensemble])
        confs = np.array([m.get("confidence", 0.5) for m in ensemble])
        probs = np.array([m.get("prob_home", 0.4) for m in ensemble])

        # Her bahsin PnL dağılımını simüle et
        pnl_matrix = self._simulate_pnl(ensemble)

        # VaR
        portfolio_pnl = pnl_matrix.sum(axis=1)
        var = float(np.percentile(portfolio_pnl, (1 - self._alpha) * 100))

        # CoVaR: i. bahis stresteyken portföy VaR'ı
        covar_values = []
        for i in range(n):
            cond_var = self._conditional_var(pnl_matrix, i)
            covar_values.append(cond_var)

        covar = float(np.mean(covar_values)) if covar_values else 0.0
        delta_covar = covar - var

        # Risk seviyesi
        risk_level = self._classify_risk(var, covar, delta_covar)

        return {
            "var": var,
            "covar": covar,
            "delta_covar": delta_covar,
            "risk_level": risk_level,
            "individual_covars": [float(c) for c in covar_values],
            "expected_pnl": float(portfolio_pnl.mean()),
            "pnl_std": float(portfolio_pnl.std()),
        }

    def _simulate_pnl(self, ensemble: list[dict]) -> np.ndarray:
        """Monte Carlo ile PnL simülasyonu."""
        n = len(ensemble)
        pnl = np.zeros((self._n_sim, n))

        for i, match in enumerate(ensemble):
            prob = match.get("prob_home", 0.4)
            odds = match.get("odds", 1.0 / max(prob, 0.05))
            stake = match.get("stake_pct", 0.01)

            if stake <= 0 or match.get("selection") == "skip":
                continue

            # Bernoulli: kazanır/kaybeder
            outcomes = np.random.binomial(1, prob, self._n_sim)
            pnl[:, i] = np.where(outcomes == 1, stake * (odds - 1), -stake)

        return pnl

    def _conditional_var(self, pnl_matrix: np.ndarray, stress_idx: int) -> float:
        """i. bahis stresteyken (kayıptayken) portföy VaR'ı."""
        stress_threshold = np.percentile(pnl_matrix[:, stress_idx], (1 - self._alpha) * 100)
        stress_mask = pnl_matrix[:, stress_idx] <= stress_threshold

        if stress_mask.sum() < 10:
            return float(np.percentile(pnl_matrix.sum(axis=1), (1 - self._alpha) * 100))

        conditioned_pnl = pnl_matrix[stress_mask].sum(axis=1)
        return float(np.percentile(conditioned_pnl, (1 - self._alpha) * 100))

    def _classify_risk(self, var: float, covar: float, delta_covar: float) -> str:
        if delta_covar > -0.01:
            return "low"
        elif delta_covar > -0.03:
            return "medium"
        elif delta_covar > -0.05:
            return "high"
        else:
            return "critical"

    def correlation_matrix(self, ensemble: list[dict]) -> np.ndarray:
        """Bahisler arası korelasyon matrisi."""
        pnl_matrix = self._simulate_pnl(ensemble)
        active_cols = [i for i in range(pnl_matrix.shape[1]) if pnl_matrix[:, i].std() > 1e-10]

        if len(active_cols) < 2:
            return np.eye(len(ensemble))

        active_pnl = pnl_matrix[:, active_cols]
        corr = np.corrcoef(active_pnl.T)

        full_corr = np.eye(len(ensemble))
        for i_new, i_old in enumerate(active_cols):
            for j_new, j_old in enumerate(active_cols):
                full_corr[i_old, j_old] = corr[i_new, j_new]

        return full_corr
