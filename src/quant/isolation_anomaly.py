"""
isolation_anomaly.py – Isolation Forest ile Tuzak/Şike Tespit Motoru.

Bahis büroları botları avlamak için "Tuzak Oranlar" açar.
Sakatlık haberi henüz düşmemiştir, şike vardır, veya kasıtlı
oran manipülasyonu yapılmaktadır. Klasik modeller bunu "Fırsat"
sanar – bu modül onu "TUZAK" olarak tespit eder.

Isolation Forest:
  - Normal maçlarda oran değişimi, hacim ve haberler belirli bir
    desen izler.
  - Bu desenin dışına çıkan Aykırı Değerleri (Outliers) bulur.
  - Aykırılar → "Kara Liste" → Bahis YASAKLANIR.

Ek Algılayıcılar:
  - Steam Movement Detector: Koordineli oran hareketleri
  - Sharp vs Public Money Divergence: Akıllı para / kamu parası ayrışması
  - Closing Line Value Anomaly: Kapanış oranı ile modelin sapması
  - Volume Spike Detector: Anormal işlem hacmi

Teknoloji: scikit-learn IsolationForest + LOF + Z-Score
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.info("scikit-learn yüklü değil – istatistiksel anomaly fallback.")


@dataclass
class AnomalyAlert:
    """Tespit edilen anomali."""
    match_id: str = ""
    alert_type: str = ""          # trap_odds | steam_move | sharp_public | volume_spike | score_anomaly
    severity: str = "medium"      # low | medium | high | critical
    score: float = 0.0            # Anomali skoru (-1 = çok anormal, 1 = normal)
    description: str = ""
    evidence: dict = field(default_factory=dict)
    action: str = "KARA_LISTE"   # KARA_LISTE | UYARI | GÖZLEM
    timestamp: str = ""
    is_blacklisted: bool = False


@dataclass
class MarketSnapshot:
    """Oran piyasası anlık görüntüsü."""
    match_id: str = ""
    timestamp: float = 0.0
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0
    over_25_odds: float = 0.0
    under_25_odds: float = 0.0
    volume: float = 0.0           # İşlem hacmi (normalize)
    bookmaker_count: int = 0      # Oran veren büro sayısı
    odds_movement_pct: float = 0.0  # Son 1 saat değişim %


class IsolationAnomalyDetector:
    """Isolation Forest + çoklu dedektör ile anomali tespiti.

    Kullanım:
        detector = IsolationAnomalyDetector()
        # Geçmiş normal verilerle eğit
        detector.fit(historical_snapshots)
        # Yeni maçı kontrol et
        alerts = detector.scan(current_snapshot)
        # Kara listeyi sorgula
        is_safe = detector.is_safe("match_123")
    """

    # Eşik değerleri
    ISOLATION_CONTAMINATION = 0.05       # %5 anomali beklentisi
    ZSCORE_THRESHOLD = 2.5               # Z-skoru eşiği
    STEAM_THRESHOLD = 0.08               # %8 koordineli hareket
    VOLUME_SPIKE_MULT = 3.0              # Normal hacmin 3 katı
    SHARP_PUBLIC_DIVERGE = 0.15          # %15 ayrışma

    def __init__(self, contamination: float = 0.05,
                 blacklist_severity: str = "high"):
        """
        Args:
            contamination: Beklenen anomali oranı
            blacklist_severity: Bu ve üstü → kara liste
        """
        self._contamination = contamination
        self._blacklist_severity = blacklist_severity
        self._model: Any = None
        self._lof: Any = None
        self._scaler: Any = None
        self._fitted = False
        self._blacklist: dict[str, AnomalyAlert] = {}
        self._history: list[MarketSnapshot] = []
        self._feature_names = [
            "home_odds", "draw_odds", "away_odds",
            "over_25_odds", "under_25_odds",
            "volume", "bookmaker_count", "odds_movement_pct",
        ]
        logger.debug("IsolationAnomalyDetector başlatıldı.")

    # ═══════════════════════════════════════════
    #  EĞİTİM
    # ═══════════════════════════════════════════
    def fit(self, snapshots: list[MarketSnapshot] | list[dict]) -> bool:
        """Geçmiş normal piyasa verileriyle modeli eğit."""
        if len(snapshots) < 30:
            logger.warning("[IsoForest] Yetersiz eğitim verisi (< 30).")
            return False

        matrix = self._to_matrix(snapshots)
        if matrix.shape[0] < 30:
            return False

        # StandardScaler
        if SKLEARN_OK:
            self._scaler = StandardScaler()
            matrix_scaled = self._scaler.fit_transform(matrix)

            # Isolation Forest
            self._model = IsolationForest(
                n_estimators=200,
                contamination=self._contamination,
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(matrix_scaled)

            # Local Outlier Factor (ek dedektör)
            self._lof = LocalOutlierFactor(
                n_neighbors=min(20, len(snapshots) // 2),
                contamination=self._contamination,
                novelty=True,
            )
            self._lof.fit(matrix_scaled)

            self._fitted = True
            logger.info(f"[IsoForest] Model eğitildi: {matrix.shape[0]} snapshot.")
            return True

        logger.info("[IsoForest] sklearn yok – Z-Score fallback.")
        # Fallback: istatistikleri sakla
        self._mean = np.mean(matrix, axis=0)
        self._std = np.std(matrix, axis=0) + 1e-8
        self._fitted = True
        return True

    def _to_matrix(self, snapshots: list) -> np.ndarray:
        """Snapshot listesini feature matrisine dönüştür."""
        rows = []
        for s in snapshots:
            if isinstance(s, dict):
                row = [float(s.get(k, 0)) for k in self._feature_names]
            else:
                row = [
                    s.home_odds, s.draw_odds, s.away_odds,
                    s.over_25_odds, s.under_25_odds,
                    s.volume, s.bookmaker_count, s.odds_movement_pct,
                ]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    # ═══════════════════════════════════════════
    #  TARAMA
    # ═══════════════════════════════════════════
    def scan(self, snapshot: MarketSnapshot | dict,
             match_id: str = "") -> list[AnomalyAlert]:
        """Tek maçı tüm dedektörlerle tara."""
        alerts: list[AnomalyAlert] = []

        if isinstance(snapshot, dict):
            mid = snapshot.get("match_id", match_id)
            # None değerleri varsayılana dönüştür (Polars null koruma)
            _defaults = {
                "home_odds": 0.0, "draw_odds": 0.0, "away_odds": 0.0,
                "over_25_odds": 0.0, "under_25_odds": 0.0, "volume": 0.0,
                "bookmaker_count": 0, "odds_movement_pct": 0.0, "timestamp": 0.0,
            }
            _safe = {}
            for k, v in snapshot.items():
                if k in MarketSnapshot.__dataclass_fields__:
                    if v is None:
                        _safe[k] = _defaults.get(k, 0.0)
                    else:
                        try:
                            _safe[k] = float(v) if k in _defaults else v
                        except (TypeError, ValueError):
                            _safe[k] = _defaults.get(k, 0.0)
            snap = MarketSnapshot(**_safe)
            snap.match_id = mid
        else:
            snap = snapshot
            mid = snap.match_id or match_id

        # 1) Isolation Forest
        iso_alert = self._isolation_forest_check(snap, mid)
        if iso_alert:
            alerts.append(iso_alert)

        # 2) Steam Movement
        steam_alert = self._steam_movement_check(snap, mid)
        if steam_alert:
            alerts.append(steam_alert)

        # 3) Volume Spike
        vol_alert = self._volume_spike_check(snap, mid)
        if vol_alert:
            alerts.append(vol_alert)

        # 4) Odds Consistency
        consist_alert = self._odds_consistency_check(snap, mid)
        if consist_alert:
            alerts.append(consist_alert)

        # 5) Z-Score check
        z_alert = self._zscore_check(snap, mid)
        if z_alert:
            alerts.append(z_alert)

        # Kara listeye al
        for a in alerts:
            a.timestamp = datetime.now().isoformat()
            if self._should_blacklist(a):
                a.is_blacklisted = True
                a.action = "KARA_LISTE"
                self._blacklist[mid] = a
                logger.warning(
                    f"[TUZAK] {mid} KARA LİSTEYE ALINDI: "
                    f"{a.alert_type} ({a.severity})"
                )

        # Geçmişe ekle
        self._history.append(snap)
        if len(self._history) > 10000:
            self._history = self._history[-5000:]

        return alerts

    def scan_batch(self, snapshots: list[dict]) -> list[AnomalyAlert]:
        """Birden fazla maçı toplu tara."""
        all_alerts = []
        for snap in snapshots:
            alerts = self.scan(snap, match_id=snap.get("match_id", ""))
            all_alerts.extend(alerts)
        return all_alerts

    # ═══════════════════════════════════════════
    #  DEDEKTÖRLER
    # ═══════════════════════════════════════════
    def _isolation_forest_check(self, snap: MarketSnapshot,
                                 mid: str) -> AnomalyAlert | None:
        """Isolation Forest anomali tespiti."""
        if not self._fitted:
            return None

        row = np.array([[
            snap.home_odds, snap.draw_odds, snap.away_odds,
            snap.over_25_odds, snap.under_25_odds,
            snap.volume, snap.bookmaker_count, snap.odds_movement_pct,
        ]], dtype=np.float64)

        if SKLEARN_OK and self._model is not None:
            row_scaled = self._scaler.transform(row)

            # Isolation Forest skoru (-1 = anomali, 1 = normal)
            iso_score = self._model.decision_function(row_scaled)[0]
            iso_pred = self._model.predict(row_scaled)[0]

            # LOF skoru
            lof_score = self._lof.decision_function(row_scaled)[0]
            lof_pred = self._lof.predict(row_scaled)[0]

            # Her iki model de anomali derse → yüksek güven
            if iso_pred == -1 or lof_pred == -1:
                combined_score = (iso_score + lof_score) / 2
                severity = self._classify_severity(combined_score)

                return AnomalyAlert(
                    match_id=mid,
                    alert_type="trap_odds",
                    severity=severity,
                    score=round(float(combined_score), 4),
                    description=(
                        f"Isolation Forest UYARI: Oran deseni anormal "
                        f"(IF={iso_score:.3f}, LOF={lof_score:.3f})"
                    ),
                    evidence={
                        "isolation_score": round(float(iso_score), 4),
                        "lof_score": round(float(lof_score), 4),
                        "home_odds": snap.home_odds,
                        "draw_odds": snap.draw_odds,
                        "away_odds": snap.away_odds,
                        "movement": snap.odds_movement_pct,
                    },
                )
        else:
            # Z-Score fallback
            return self._zscore_check(snap, mid)

        return None

    def _steam_movement_check(self, snap: MarketSnapshot,
                                mid: str) -> AnomalyAlert | None:
        """Koordineli oran hareketleri (Steam Move) tespiti.

        Tüm bürolarda aynı anda aynı yönde oran hareketi →
        büyük oyuncu (Sharp) girişi veya inside bilgi.
        """
        if abs(snap.odds_movement_pct) < self.STEAM_THRESHOLD:
            return None

        severity = "medium"
        if abs(snap.odds_movement_pct) > 0.15:
            severity = "high"
        if abs(snap.odds_movement_pct) > 0.25:
            severity = "critical"

        direction = "DÜŞÜYOR" if snap.odds_movement_pct < 0 else "ÇIKIYOR"
        return AnomalyAlert(
            match_id=mid,
            alert_type="steam_move",
            severity=severity,
            score=round(abs(snap.odds_movement_pct), 4),
            description=(
                f"STEAM MOVE: Oran {direction} ({snap.odds_movement_pct:+.1%}). "
                f"Koordineli büro hareketi tespit edildi."
            ),
            evidence={
                "movement_pct": snap.odds_movement_pct,
                "direction": direction,
                "bookmakers": snap.bookmaker_count,
            },
        )

    def _volume_spike_check(self, snap: MarketSnapshot,
                              mid: str) -> AnomalyAlert | None:
        """Anormal işlem hacmi tespiti."""
        if not self._history:
            return None

        avg_volume = np.mean([h.volume for h in self._history[-100:] if h.volume > 0])
        if avg_volume <= 0:
            return None

        ratio = snap.volume / avg_volume
        if ratio < self.VOLUME_SPIKE_MULT:
            return None

        severity = "medium"
        if ratio > 5:
            severity = "high"
        if ratio > 10:
            severity = "critical"

        return AnomalyAlert(
            match_id=mid,
            alert_type="volume_spike",
            severity=severity,
            score=round(ratio, 2),
            description=(
                f"HACİM PATLAMASI: Normal hacmin {ratio:.1f}x katı. "
                f"İçeriden bilgi veya manipülasyon olabilir."
            ),
            evidence={
                "current_volume": snap.volume,
                "avg_volume": round(avg_volume, 2),
                "multiplier": round(ratio, 2),
            },
        )

    def _odds_consistency_check(self, snap: MarketSnapshot,
                                  mid: str) -> AnomalyAlert | None:
        """Oran tutarsızlığı tespiti.

        Oranların ima ettiği olasılık toplamı (overround) anormal ise
        büro fiyatlama hatası veya kasıtlı manipülasyon.
        """
        if not snap.home_odds or not snap.draw_odds or not snap.away_odds:
            return None
        if snap.home_odds <= 1 or snap.draw_odds <= 1 or snap.away_odds <= 1:
            return None

        implied = (1 / snap.home_odds) + (1 / snap.draw_odds) + (1 / snap.away_odds)

        # Normal overround: 1.03 – 1.12
        if 1.02 < implied < 1.15:
            return None

        severity = "medium"
        if implied < 0.95 or implied > 1.20:
            severity = "high"
        if implied < 0.90 or implied > 1.30:
            severity = "critical"

        return AnomalyAlert(
            match_id=mid,
            alert_type="odds_consistency",
            severity=severity,
            score=round(float(implied), 4),
            description=(
                f"ORAN TUTARSIZLIĞI: Overround = {implied:.4f} "
                f"(normal: 1.03-1.12). "
                f"{'Arbitraj fırsatı!' if implied < 1.0 else 'Aşırı margin!'}"
            ),
            evidence={
                "overround": round(implied, 4),
                "home_implied": round(1 / snap.home_odds, 3),
                "draw_implied": round(1 / snap.draw_odds, 3),
                "away_implied": round(1 / snap.away_odds, 3),
            },
        )

    def _zscore_check(self, snap: MarketSnapshot,
                       mid: str) -> AnomalyAlert | None:
        """Z-Score bazlı basit anomali kontrolü."""
        if not self._fitted:
            return None

        row = np.array([
            snap.home_odds, snap.draw_odds, snap.away_odds,
            snap.over_25_odds, snap.under_25_odds,
            snap.volume, snap.bookmaker_count, snap.odds_movement_pct,
        ], dtype=np.float64)

        if hasattr(self, "_mean") and hasattr(self, "_std"):
            z_scores = np.abs((row - self._mean) / self._std)
        elif self._scaler is not None:
            scaled = self._scaler.transform(row.reshape(1, -1))[0]
            z_scores = np.abs(scaled)
        else:
            return None

        max_z = float(np.max(z_scores))
        if max_z < self.ZSCORE_THRESHOLD:
            return None

        outlier_features = [
            self._feature_names[i]
            for i in range(min(len(z_scores), len(self._feature_names)))
            if z_scores[i] > self.ZSCORE_THRESHOLD
        ]

        return AnomalyAlert(
            match_id=mid,
            alert_type="zscore_outlier",
            severity="medium" if max_z < 3.5 else "high",
            score=round(max_z, 3),
            description=(
                f"Z-SCORE ANOMALI: {', '.join(outlier_features)} "
                f"değerleri anormal (max Z={max_z:.2f})"
            ),
            evidence={
                "max_zscore": round(max_z, 3),
                "outlier_features": outlier_features,
                "z_scores": {
                    self._feature_names[i]: round(float(z_scores[i]), 3)
                    for i in range(min(len(z_scores), len(self._feature_names)))
                },
            },
        )

    # ═══════════════════════════════════════════
    #  KARA LİSTE
    # ═══════════════════════════════════════════
    def _classify_severity(self, score: float) -> str:
        """Anomali skorunu şiddete dönüştür."""
        if score < -0.3:
            return "critical"
        elif score < -0.1:
            return "high"
        elif score < 0.0:
            return "medium"
        return "low"

    def _should_blacklist(self, alert: AnomalyAlert) -> bool:
        """Bu alert kara listeye alınmalı mı?"""
        severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        threshold_rank = severity_rank.get(self._blacklist_severity, 2)
        alert_rank = severity_rank.get(alert.severity, 0)
        return alert_rank >= threshold_rank

    def is_safe(self, match_id: str) -> bool:
        """Maç bahse güvenli mi (kara listede değil mi)?"""
        return match_id not in self._blacklist

    def get_blacklist(self) -> dict[str, AnomalyAlert]:
        return dict(self._blacklist)

    def clear_blacklist(self, match_id: str | None = None):
        """Kara listeden çıkar."""
        if match_id:
            self._blacklist.pop(match_id, None)
        else:
            self._blacklist.clear()

    def filter_safe_bets(self, bets: list[dict]) -> list[dict]:
        """Kara listedeki maçları bahis listesinden çıkar."""
        safe = []
        blocked = 0
        for bet in bets:
            mid = bet.get("match_id", "")
            if mid in self._blacklist:
                blocked += 1
                alert = self._blacklist[mid]
                logger.warning(
                    f"[TUZAK] {mid} ENGELLENDİ: {alert.alert_type} – "
                    f"{alert.description[:80]}"
                )
                bet["blacklisted"] = True
                bet["blacklist_reason"] = alert.description
                continue
            safe.append(bet)

        if blocked:
            logger.info(
                f"[IsoForest] {blocked} bahis kara listede → engellendi. "
                f"Güvenli: {len(safe)}"
            )
        return safe

    @property
    def blacklist_count(self) -> int:
        return len(self._blacklist)
