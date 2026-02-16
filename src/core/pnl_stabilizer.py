"""
pnl_stabilizer.py – PID kontrol döngüsü ile kasa eğrisini stabilize eder.
Drawdown'ları tespit eder, pozisyon boyutunu otomatik ayarlar.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

try:
    from simple_pid import PID
    PID_AVAILABLE = True
except ImportError:
    PID_AVAILABLE = False
    logger.warning("simple-pid yüklü değil – PnL stabilizer basit modda.")


class PnLStabilizer:
    """PID kontrol ile kasa eğrisini düzleştiren stabilizatör."""

    def __init__(
        self,
        target_daily_return: float = 0.002,
        max_drawdown_limit: float = 0.10,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.5,
    ):
        self._target = target_daily_return
        self._max_dd = max_drawdown_limit
        self._history: list[float] = []
        self._peak: float = 0.0

        if PID_AVAILABLE:
            self._pid = PID(kp, ki, kd, setpoint=target_daily_return)
            self._pid.output_limits = (0.1, 2.0)  # stake çarpanı [%10, %200]
        else:
            self._pid = None

        logger.debug("PnLStabilizer başlatıldı.")

    def stabilize(self, bets: list[dict]) -> list[dict]:
        """Bahis stake'lerini PID kontrol ile ayarlar."""
        if not bets:
            return bets

        # Mevcut drawdown hesapla
        dd = self._current_drawdown()
        multiplier = self._compute_multiplier(dd)

        for bet in bets:
            if bet.get("selection") == "skip":
                continue
            original_stake = bet.get("stake_pct", 0.0)
            adjusted = original_stake * multiplier

            # Drawdown limiti aşıldıysa agresif kes
            if dd < -self._max_dd:
                adjusted *= 0.3
                logger.warning(f"Drawdown limiti aşıldı ({dd:.2%}) – stake %70 azaltıldı.")

            bet["stake_pct"] = float(round(max(adjusted, 0), 5))
            bet["pid_multiplier"] = multiplier

        return bets

    def _compute_multiplier(self, current_drawdown: float) -> float:
        """PID kontrol ile stake çarpanını hesaplar."""
        if self._pid is not None:
            # Son PnL'i güncelle
            recent_return = self._recent_return()
            multiplier = self._pid(recent_return)
            return float(np.clip(multiplier, 0.1, 2.0))
        else:
            return self._fallback_multiplier(current_drawdown)

    def _fallback_multiplier(self, dd: float) -> float:
        """PID yokken basit kural tabanlı çarpan."""
        if dd < -0.08:
            return 0.2
        elif dd < -0.05:
            return 0.5
        elif dd < -0.02:
            return 0.8
        else:
            return 1.0

    def _current_drawdown(self) -> float:
        if not self._history:
            return 0.0
        current = self._history[-1]
        if current > self._peak:
            self._peak = current
        if self._peak == 0:
            return 0.0
        return (current - self._peak) / self._peak

    def _recent_return(self) -> float:
        if len(self._history) < 2:
            return 0.0
        return (self._history[-1] - self._history[-2]) / max(abs(self._history[-2]), 1e-8)

    def record_pnl(self, bankroll: float):
        self._history.append(bankroll)
        if bankroll > self._peak:
            self._peak = bankroll

    def metrics(self) -> dict:
        if not self._history:
            return {"drawdown": 0, "peak": 0, "current": 0, "history_len": 0}
        return {
            "drawdown": self._current_drawdown(),
            "peak": self._peak,
            "current": self._history[-1],
            "history_len": len(self._history),
        }
