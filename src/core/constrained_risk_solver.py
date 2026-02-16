"""
constrained_risk_solver.py – Lagrange çarpanları ile kısıtlı risk optimizasyonu.
Bahisleri katı finansal kısıtlar altında optimize eder.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from loguru import logger


class ConstrainedRiskSolver:
    """Kısıtlı optimizasyon ile portföy risk yönetimi."""

    def __init__(
        self,
        max_single_stake: float = 0.05,
        max_total_exposure: float = 0.20,
        max_correlated_exposure: float = 0.10,
        min_ev_threshold: float = 0.02,
    ):
        self._max_single = max_single_stake
        self._max_total = max_total_exposure
        self._max_corr = max_correlated_exposure
        self._min_ev = min_ev_threshold
        logger.debug("ConstrainedRiskSolver başlatıldı.")

    def solve(self, allocation: list[dict]) -> list[dict]:
        """Tahsis kararlarını kısıtlar altında optimize eder."""
        if not allocation:
            return []

        # Filtre: minimum EV eşiği
        candidates = [a for a in allocation if a.get("ev", 0) >= self._min_ev and a.get("selection") != "skip"]

        if not candidates:
            logger.info("EV eşiğini geçen bahis yok – portföy boş.")
            return allocation

        n = len(candidates)
        stakes = np.array([c.get("stake_pct", 0.01) for c in candidates])
        evs = np.array([c.get("ev", 0.0) for c in candidates])
        confidences = np.array([c.get("confidence", 0.5) for c in candidates])

        # Amaç: EV * confidence * stake toplamını maksimize et (negatifi minimize)
        def objective(x):
            return -np.sum(x * evs * confidences)

        # Kısıtlar
        constraints = []

        # 1) Toplam stake <= max_total_exposure
        constraints.append(LinearConstraint(np.ones(n), lb=0, ub=self._max_total))

        # 2) Her stake >= 0 ve <= max_single_stake
        bounds = [(0, self._max_single)] * n

        # 3) Gradient (Jacobian)
        def jac(x):
            return -(evs * confidences)

        try:
            result = minimize(
                objective,
                x0=stakes.clip(0, self._max_single),
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": lambda x: self._max_total - np.sum(x)}],
                jac=jac,
                options={"maxiter": 200, "ftol": 1e-10},
            )

            if result.success:
                optimized_stakes = result.x
                logger.info(f"Optimizasyon başarılı – toplam stake: {np.sum(optimized_stakes):.4f}")
            else:
                logger.warning(f"Optimizasyon yakınsamadı: {result.message}")
                optimized_stakes = stakes.clip(0, self._max_single)
        except Exception as e:
            logger.error(f"Optimizasyon hatası: {e}")
            optimized_stakes = stakes.clip(0, self._max_single)

        # Sonuçları geri yaz
        for i, c in enumerate(candidates):
            c["stake_pct"] = float(round(optimized_stakes[i], 5))
            c["optimized"] = True

        # Skip'leri de dahil et
        skipped = [a for a in allocation if a.get("ev", 0) < self._min_ev or a.get("selection") == "skip"]
        for s in skipped:
            s["stake_pct"] = 0.0
            s["optimized"] = False

        return candidates + skipped

    def validate(self, bets: list[dict]) -> dict:
        """Portföy kısıtlarını doğrular."""
        total_stake = sum(b.get("stake_pct", 0) for b in bets if b.get("selection") != "skip")
        max_stake = max((b.get("stake_pct", 0) for b in bets), default=0)
        active = sum(1 for b in bets if b.get("selection") != "skip" and b.get("stake_pct", 0) > 0)

        violations = []
        if total_stake > self._max_total * 1.01:
            violations.append(f"Toplam stake ({total_stake:.4f}) > limit ({self._max_total})")
        if max_stake > self._max_single * 1.01:
            violations.append(f"Max tekli stake ({max_stake:.4f}) > limit ({self._max_single})")

        return {
            "valid": len(violations) == 0,
            "total_stake": total_stake,
            "max_single_stake": max_stake,
            "active_bets": active,
            "violations": violations,
        }
