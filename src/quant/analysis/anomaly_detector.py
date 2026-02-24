"""
anomaly_detector.py – Z-Score ile oran anomalisi tespiti.
Dropping Odds: bahis şirketlerinin oranlarındaki ani düşüşleri tespit eder.
"Smart Money" (büyük para) nereye akıyor takip eder.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from datetime import datetime, timedelta
from loguru import logger


class AnomalyDetector:
    """Z-Score tabanlı oran anomalisi ve dropping odds tespit modülü."""

    def __init__(self, z_threshold: float = 2.0, window_size: int = 20):
        self._z_threshold = z_threshold
        self._window = window_size
        self._alerts: list[dict] = []
        logger.debug(f"AnomalyDetector başlatıldı (Z>{z_threshold}).")

    def detect_dropping_odds(self, odds_history: pl.DataFrame) -> list[dict]:
        """Oran geçmişinde ani düşüşleri (dropping odds) tespit eder."""
        if odds_history.is_empty() or odds_history.height < 3:
            return []

        alerts = []
        if "odds" not in odds_history.columns:
            return []

        odds_arr = odds_history["odds"].to_numpy()
        match_ids = odds_history["match_id"].to_list() if "match_id" in odds_history.columns else [""] * len(odds_arr)
        selections = odds_history["selection"].to_list() if "selection" in odds_history.columns else [""] * len(odds_arr)

        # Rolling Z-Score hesapla
        for i in range(self._window, len(odds_arr)):
            window = odds_arr[max(0, i - self._window):i]
            current = odds_arr[i]

            mean = np.mean(window)
            std = np.std(window)

            if std < 1e-6:
                continue

            z_score = (current - mean) / std

            # Negatif Z-Score = oran düşüşü = smart money girişi
            if z_score < -self._z_threshold:
                pct_change = (current - mean) / mean * 100
                alert = {
                    "match_id": match_ids[i] if i < len(match_ids) else "",
                    "selection": selections[i] if i < len(selections) else "",
                    "z_score": float(z_score),
                    "current_odds": float(current),
                    "mean_odds": float(mean),
                    "pct_drop": float(pct_change),
                    "type": "DROPPING_ODDS",
                    "severity": self._classify_severity(z_score),
                    "smart_money_signal": True,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                alerts.append(alert)
                logger.warning(
                    f"[ANOMALI] Dropping Odds: {alert['match_id']} "
                    f"{alert['selection']} Z={z_score:.2f} ({pct_change:.1f}%)"
                )

            # Pozitif Z-Score = oran yükselişi = steam move (tersi)
            elif z_score > self._z_threshold:
                pct_change = (current - mean) / mean * 100
                alert = {
                    "match_id": match_ids[i] if i < len(match_ids) else "",
                    "selection": selections[i] if i < len(selections) else "",
                    "z_score": float(z_score),
                    "current_odds": float(current),
                    "mean_odds": float(mean),
                    "pct_rise": float(pct_change),
                    "type": "STEAM_MOVE",
                    "severity": self._classify_severity(z_score),
                    "smart_money_signal": False,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                alerts.append(alert)

        self._alerts.extend(alerts)
        return alerts

    def scan_all_matches(self, db) -> list[dict]:
        """Tüm maçların oran geçmişini tarar."""
        all_alerts = []

        try:
            matches = db.query("SELECT DISTINCT match_id FROM odds_history")
            for row in matches.iter_rows(named=True):
                mid = row.get("match_id", "")
                if not mid:
                    continue
                history = db.get_odds_history(mid)
                if not history.is_empty():
                    alerts = self.detect_dropping_odds(history)
                    all_alerts.extend(alerts)
        except Exception as e:
            logger.error(f"Anomali taraması hatası: {e}")

        if all_alerts:
            logger.info(f"Toplam {len(all_alerts)} oran anomalisi tespit edildi.")
        return all_alerts

    def analyze_line_movement(self, odds_history: pl.DataFrame) -> dict:
        """Oran hareketlerinin genel karakterini analiz eder."""
        if odds_history.is_empty() or odds_history.height < 5:
            return {"trend": "stable", "volatility": 0, "direction": 0}

        odds = odds_history["odds"].to_numpy()
        returns = np.diff(odds) / (np.abs(odds[:-1]) + 1e-8)

        trend = float(np.mean(returns))
        volatility = float(np.std(returns))

        # Ağırlıklı trend (son hareketler daha önemli)
        weights = np.linspace(0.5, 1.5, len(returns))
        weighted_trend = float(np.average(returns, weights=weights))

        if weighted_trend < -0.02:
            direction = "dropping"
        elif weighted_trend > 0.02:
            direction = "rising"
        else:
            direction = "stable"

        return {
            "trend": direction,
            "weighted_trend": weighted_trend,
            "volatility": volatility,
            "direction": float(np.sign(weighted_trend)),
            "n_observations": len(odds),
        }

    def smart_money_index(self, match_id: str, db) -> dict:
        """Smart Money Index: büyük paranın hangi yöne aktığını ölçer."""
        history = db.get_odds_history(match_id)
        if history.is_empty():
            return {"smi": 0.0, "direction": "neutral", "confidence": 0.0}

        selections = ["home", "draw", "away"]
        movements = {}

        for sel in selections:
            sel_data = history.filter(pl.col("selection") == sel)
            if sel_data.is_empty() or sel_data.height < 2:
                movements[sel] = 0.0
                continue

            odds = sel_data["odds"].to_numpy()
            pct_change = (odds[-1] - odds[0]) / (odds[0] + 1e-8)
            movements[sel] = float(pct_change)

        # En çok düşen oran = smart money yönü
        if not movements:
            return {"smi": 0.0, "direction": "neutral", "confidence": 0.0}

        most_dropped = min(movements, key=movements.get)
        smi_value = abs(movements[most_dropped])

        return {
            "smi": float(np.clip(smi_value, 0, 1)),
            "direction": most_dropped,
            "movements": movements,
            "confidence": float(np.clip(smi_value * 2, 0, 1)),
        }

    def _classify_severity(self, z_score: float) -> str:
        z = abs(z_score)
        if z > 4:
            return "critical"
        elif z > 3:
            return "high"
        elif z > 2.5:
            return "medium"
        else:
            return "low"

    @property
    def recent_alerts(self) -> list[dict]:
        return self._alerts[-50:]
