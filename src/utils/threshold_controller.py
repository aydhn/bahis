"""
threshold_controller.py – Güven eşiği kontrolcüsü.
Telegram üzerinden botun EV/güven eşiklerini canlı yönetir.
"""
from __future__ import annotations

from loguru import logger


class ThresholdController:
    """Botun karar eşiklerini dinamik olarak yönetir."""

    def __init__(
        self,
        ev_threshold: float = 0.02,
        confidence_threshold: float = 0.40,
        max_stake: float = 0.05,
    ):
        self.ev_threshold = ev_threshold
        self.confidence_threshold = confidence_threshold
        self.max_stake = max_stake
        self._history: list[dict] = []
        logger.debug(
            f"ThresholdController başlatıldı – "
            f"EV>{ev_threshold}, Güven>{confidence_threshold}, MaxStake={max_stake}"
        )

    def adjust(self, delta: float, param: str = "ev"):
        """Eşiği artırır/azaltır."""
        old = getattr(self, f"{param}_threshold", None)
        if old is None:
            logger.warning(f"Bilinmeyen parametre: {param}")
            return

        new = max(0.0, min(1.0, old + delta))
        setattr(self, f"{param}_threshold", new)
        self._history.append({"param": param, "old": old, "new": new})
        logger.info(f"Eşik güncellendi: {param} {old:.3f} → {new:.3f}")

    def should_bet(self, ev: float, confidence: float) -> bool:
        """Verilen EV ve güven değeri ile bahis yapılmalı mı?"""
        return ev >= self.ev_threshold and confidence >= self.confidence_threshold

    def get_max_stake(self) -> float:
        return self.max_stake

    def set_max_stake(self, value: float):
        old = self.max_stake
        self.max_stake = max(0.001, min(0.20, value))
        logger.info(f"Max stake güncellendi: {old:.3f} → {self.max_stake:.3f}")

    def status(self) -> dict:
        return {
            "ev_threshold": self.ev_threshold,
            "confidence_threshold": self.confidence_threshold,
            "max_stake": self.max_stake,
            "adjustments": len(self._history),
        }
