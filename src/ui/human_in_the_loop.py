"""
human_in_the_loop.py – İnsan Filtresi Takip Sistemi.

Model "Bahis al" dedi → Telegram'dan [✅ Onayla] [❌ Reddet] [📈 Detay]
butonlarıyla karar verilir. Her karar veritabanına kaydedilir.

Böylece:
- "Model böyle dedi ama ben oynamadım" istatistiği tutulur
- Model vs İnsan performansı kıyaslanır
- Hangi senaryolarda insanın veto'su doğru? Hangi senaryolarda yanlış?

Bu veri, modelin güvenilirliğini ve insanın bias'larını ölçer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger


class Decision(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO = "auto"        # İnsan müdahale etmedi, otomatik işlendi


@dataclass
class SignalRecord:
    """Tek bir sinyalin tam hayat döngüsü."""
    signal_id: str
    match_id: str
    selection: str
    odds: float
    ev: float
    confidence: float
    stake_pct: float
    # Model tahmini
    model_prob: float = 0.0
    model_prediction: str = ""
    # İnsan kararı
    decision: Decision = Decision.PENDING
    decision_time: float = 0.0        # Kararın verildiği timestamp
    response_seconds: float = 0.0     # Ne kadar sürede karar verdi
    rejection_reason: str = ""        # Neden reddetti (opsiyonel)
    # Sonuç
    result: str = ""                  # won / lost / void / pending
    pnl: float = 0.0
    # Timestamps
    created_at: str = ""
    resolved_at: str = ""


class HumanInTheLoop:
    """Model vs İnsan performans karşılaştırma motoru.

    Her sinyal için:
    1. Model önerisini kaydet
    2. İnsan kararını bekle (approve/reject)
    3. Maç sonucunu kaydet
    4. İstatistik üret: "İnsanın veto'su doğru muydu?"
    """

    def __init__(self, auto_approve_ev: float = 0.0):
        self._records: dict[str, SignalRecord] = {}
        self._auto_approve_ev = auto_approve_ev  # Bu EV üstünde otomatik onayla
        self._counter = 0
        logger.debug("HumanInTheLoop başlatıldı.")

    def create_signal(self, match_id: str, selection: str,
                      odds: float, ev: float, confidence: float,
                      stake_pct: float, model_prob: float = 0.0,
                      model_prediction: str = "") -> str:
        """Yeni sinyal oluştur, onay bekleyen duruma al."""
        self._counter += 1
        signal_id = f"sig_{self._counter}_{int(time.time())}"

        record = SignalRecord(
            signal_id=signal_id,
            match_id=match_id,
            selection=selection,
            odds=odds,
            ev=ev,
            confidence=confidence,
            stake_pct=stake_pct,
            model_prob=model_prob,
            model_prediction=model_prediction,
            created_at=datetime.utcnow().isoformat(),
        )
        self._records[signal_id] = record
        return signal_id

    def approve(self, signal_id: str) -> bool:
        """Sinyali onayla."""
        rec = self._records.get(signal_id)
        if not rec or rec.decision != Decision.PENDING:
            return False

        rec.decision = Decision.APPROVED
        rec.decision_time = time.time()
        rec.response_seconds = time.time() - _parse_ts(rec.created_at)
        logger.info(f"[HITL] Sinyal ONAYLANDI: {signal_id} ({rec.match_id} {rec.selection})")
        return True

    def reject(self, signal_id: str, reason: str = "") -> bool:
        """Sinyali reddet."""
        rec = self._records.get(signal_id)
        if not rec or rec.decision != Decision.PENDING:
            return False

        rec.decision = Decision.REJECTED
        rec.decision_time = time.time()
        rec.response_seconds = time.time() - _parse_ts(rec.created_at)
        rec.rejection_reason = reason
        logger.info(f"[HITL] Sinyal REDDEDİLDİ: {signal_id} – {reason}")
        return True

    def auto_decide(self, signal_id: str) -> Decision:
        """Zaman aşımında veya auto-approve modunda otomatik karar."""
        rec = self._records.get(signal_id)
        if not rec or rec.decision != Decision.PENDING:
            return rec.decision if rec else Decision.PENDING

        if rec.ev >= self._auto_approve_ev and rec.confidence >= 0.6:
            rec.decision = Decision.AUTO
            logger.debug(f"[HITL] Otomatik onay: {signal_id}")
        else:
            rec.decision = Decision.AUTO
        return rec.decision

    def record_result(self, signal_id: str, result: str, pnl: float = 0.0):
        """Maç sonucunu kaydet."""
        rec = self._records.get(signal_id)
        if not rec:
            return
        rec.result = result
        rec.pnl = pnl
        rec.resolved_at = datetime.utcnow().isoformat()

    def record_result_by_match(self, match_id: str, selection: str,
                                result: str, pnl: float = 0.0):
        """match_id + selection ile sonuç kaydet."""
        for rec in self._records.values():
            if rec.match_id == match_id and rec.selection == selection:
                rec.result = result
                rec.pnl = pnl
                rec.resolved_at = datetime.utcnow().isoformat()

    # ═══════════════════════════════════════════
    #  İSTATİSTİKLER
    # ═══════════════════════════════════════════
    def performance_comparison(self) -> dict:
        """Model vs İnsan performans karşılaştırması."""
        resolved = [r for r in self._records.values() if r.result in ("won", "lost")]
        if not resolved:
            return {"status": "yeterli_veri_yok", "n": 0}

        # Model performansı (tüm sinyaller)
        model_wins = sum(1 for r in resolved if r.result == "won")
        model_total = len(resolved)
        model_pnl = sum(r.pnl for r in resolved
                        if r.decision in (Decision.APPROVED, Decision.AUTO))

        # İnsan onayladıklarının performansı
        approved = [r for r in resolved if r.decision == Decision.APPROVED]
        approved_wins = sum(1 for r in approved if r.result == "won")

        # İnsan reddettiği ama aslında kazanacakların
        rejected = [r for r in resolved if r.decision == Decision.REJECTED]
        rejected_would_won = sum(1 for r in rejected if r.result == "won")
        rejected_would_lost = sum(1 for r in rejected if r.result == "lost")

        # Veto doğruluğu: İnsan reddettiğinde haklı mıydı?
        veto_accuracy = (
            rejected_would_lost / max(len(rejected), 1)
        )

        # Kaçırılan fırsat: İnsan reddettiği ama kazanacak olanlar
        missed_value = sum(
            r.pnl for r in rejected if r.result == "won"
        )

        return {
            "total_signals": model_total,
            "model_win_rate": model_wins / max(model_total, 1),
            "human_approved": len(approved),
            "human_approved_win_rate": approved_wins / max(len(approved), 1),
            "human_rejected": len(rejected),
            "veto_accuracy": float(veto_accuracy),
            "veto_interpretation": (
                "İnsan filtresi değerli" if veto_accuracy > 0.6
                else "Model'e güvenmelisin" if veto_accuracy < 0.4
                else "Nötr – filtre etkisi belirsiz"
            ),
            "missed_value_pnl": float(missed_value),
            "model_total_pnl": float(model_pnl),
            "avg_response_time_sec": float(np.mean([
                r.response_seconds for r in resolved if r.response_seconds > 0
            ])) if any(r.response_seconds > 0 for r in resolved) else 0,
        }

    def pending_signals(self) -> list[dict]:
        """Onay bekleyen sinyaller."""
        return [
            {
                "signal_id": r.signal_id,
                "match_id": r.match_id,
                "selection": r.selection,
                "odds": r.odds,
                "ev": r.ev,
                "confidence": r.confidence,
                "created_at": r.created_at,
            }
            for r in self._records.values()
            if r.decision == Decision.PENDING
        ]

    def get_signal(self, signal_id: str) -> SignalRecord | None:
        return self._records.get(signal_id)


def _parse_ts(iso_str: str) -> float:
    try:
        return datetime.fromisoformat(iso_str).timestamp()
    except (ValueError, TypeError):
        return time.time()
