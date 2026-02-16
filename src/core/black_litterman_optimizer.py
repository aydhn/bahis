"""
black_litterman_optimizer.py – Black-Litterman portföy optimizasyonu.
Piyasa oranlarını (implied) botun özgün tahminleriyle dengeler.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


class BlackLittermanOptimizer:
    """Black-Litterman modeli: piyasa dengesi + subjektif görüşler."""

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self._delta = risk_aversion
        self._tau = tau
        logger.debug("BlackLittermanOptimizer başlatıldı.")

    def optimize(self, ensemble: list[dict], risk_metrics: dict) -> list[dict]:
        """Piyasa dengesini botun görüşleriyle birleştirerek optimal portföy oluşturur."""
        if not ensemble:
            return []

        n = len(ensemble)

        # Piyasa ağırlıkları (implied, oran bazlı)
        market_weights = self._market_equilibrium(ensemble)

        # Kovaryans matrisi (basitleştirilmiş)
        sigma = self._estimate_covariance(ensemble)

        # Denge getirileri: π = δΣw
        pi = self._delta * sigma @ market_weights

        # Bot görüşleri (P matris, Q vektör)
        P, Q, omega = self._build_views(ensemble)

        if P.shape[0] == 0:
            # Görüş yoksa market cap ağırlıkları kullan
            final_weights = market_weights
        else:
            # Black-Litterman formülü
            tau_sigma = self._tau * sigma
            try:
                inv_tau_sigma = np.linalg.inv(tau_sigma)
                inv_omega = np.linalg.inv(omega)

                M = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
                bl_returns = M @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
                bl_sigma = sigma + M

                # Optimal ağırlıklar
                final_weights = np.linalg.inv(self._delta * bl_sigma) @ bl_returns
            except np.linalg.LinAlgError:
                logger.warning("BL matris singüler – piyasa ağırlıklarına dönülüyor.")
                final_weights = market_weights

        # Normalize ve kısıtla
        final_weights = np.clip(final_weights, 0, 0.05)
        total = final_weights.sum()
        if total > 0.20:
            final_weights *= 0.20 / total

        # Sonuçlara yaz
        for i, match in enumerate(ensemble):
            match["stake_pct"] = float(round(final_weights[i], 5))
            match["bl_weight"] = float(round(final_weights[i], 5))
            match["market_weight"] = float(round(market_weights[i], 5))

        logger.info(f"BL optimizasyonu tamamlandı – toplam ağırlık: {final_weights.sum():.4f}")
        return ensemble

    def _market_equilibrium(self, ensemble: list[dict]) -> np.ndarray:
        """Piyasa dengesinden implied ağırlıklar çıkarır."""
        n = len(ensemble)
        weights = np.zeros(n)
        for i, m in enumerate(ensemble):
            ev = max(m.get("best_ev", m.get("ev_home", 0)), 0)
            conf = m.get("confidence", 0.5)
            weights[i] = ev * conf

        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(n) / n
        return weights

    def _estimate_covariance(self, ensemble: list[dict]) -> np.ndarray:
        """Basit kovaryans matrisi tahmini."""
        n = len(ensemble)
        sigma = np.eye(n) * 0.01

        for i in range(n):
            for j in range(i + 1, n):
                # Aynı lig / benzer zaman = korelasyon
                league_match = (
                    ensemble[i].get("league", "") == ensemble[j].get("league", "")
                    and ensemble[i].get("league", "") != ""
                )
                corr = 0.3 if league_match else 0.05
                sigma[i, j] = corr * 0.01
                sigma[j, i] = corr * 0.01

        return sigma

    def _build_views(self, ensemble: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Botun görüşlerini matris formuna dönüştürür."""
        n = len(ensemble)
        views_P = []
        views_Q = []
        confidences = []

        for i, m in enumerate(ensemble):
            ev = m.get("best_ev", m.get("ev_home", 0))
            conf = m.get("confidence", 0.5)

            if ev > 0.02 and conf > 0.4:
                row = np.zeros(n)
                row[i] = 1.0
                views_P.append(row)
                views_Q.append(ev * conf)
                confidences.append(1 - conf)

        if not views_P:
            return np.zeros((0, n)), np.zeros(0), np.eye(1)

        P = np.array(views_P)
        Q = np.array(views_Q)
        omega = np.diag(np.array(confidences) * 0.01 + 1e-6)

        return P, Q, omega
