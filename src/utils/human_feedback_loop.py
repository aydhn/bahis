"""
human_feedback_loop.py – RLHF (Reinforcement Learning from Human Feedback).

ChatGPT'yi eğiten teknoloji. Bot bir tahminde bulunduğunda ona
sadece "Tuttu/Tutmadı" demeyin. Ona "Neden Katılmadığınızı" öğretin.

Akış:
  1. Bot: "GS Kazanır" → Telegram'a gönderilir
  2. Kullanıcı: [❌ Katılmıyorum] butonuna basar
  3. Bot: "Neden?" → [Sakatlık] [Hava] [Oran Düşük] [Sezgi]
  4. Cevap veritabanına kaydedilir
  5. RL ajanı bir sonraki eğitimde bu desene negatif ödül verir

Reward Shaping:
  - ONAY + TUTTU → +1.0 (tam ödül)
  - ONAY + YATTI → -0.5 (model ve insan yanıldı)
  - RET + TUTTU → -0.3 (insan haklıydı, model öğrenmeli)
  - RET + YATTI → +0.2 (model haklıydı ama insan dinlemedi)

Red Sebebi Penalty:
  - "sakatlik" → sakatlık bilgisini daha ağırlıkla kullan
  - "hava" → hava durumu feature'ını artır
  - "oran_dusuk" → value edge eşiğini yükselt
  - "sezgi" → insan sezgisi kaydedilir (meta-öğrenme)

Teknoloji: SQLite + Telegram Polls + Reward Function
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
FEEDBACK_DB = ROOT / "data" / "human_feedback.db"
FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class FeedbackEntry:
    """Tek bir geri bildirim kaydı."""
    match_id: str = ""
    selection: str = ""
    odds: float = 0.0
    model_prob: float = 0.0
    # İnsan kararı
    human_decision: str = ""       # "approve" | "reject"
    reject_reason: str = ""        # "sakatlik" | "hava" | "oran_dusuk" | "sezgi" | "form" | "hakem"
    reject_detail: str = ""        # Serbest metin açıklama
    confidence_override: float = 0.0  # İnsan güven skoru (0–1)
    # Sonuç
    outcome: str = ""              # "won" | "lost" | "void" | "pending"
    # Ödül
    reward: float = 0.0
    timestamp: float = 0.0


@dataclass
class RewardSignal:
    """RL ajanına gönderilecek ödül sinyali."""
    match_id: str = ""
    reward: float = 0.0
    penalty_features: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class FeedbackStats:
    """Geri bildirim istatistikleri."""
    total_feedback: int = 0
    approvals: int = 0
    rejections: int = 0
    # Ödül dağılımı
    avg_reward: float = 0.0
    total_reward: float = 0.0
    # Ret sebep dağılımı
    reason_distribution: dict = field(default_factory=dict)
    # Doğruluk
    human_accuracy: float = 0.0    # İnsan ne kadar haklıydı
    model_accuracy: float = 0.0    # Model ne kadar haklıydı
    agreement_rate: float = 0.0    # İnsan-model uyumu


# ═══════════════════════════════════════════════
#  ÖDÜL FONKSİYONU (Reward Shaping)
# ═══════════════════════════════════════════════
REWARD_TABLE = {
    ("approve", "won"):  +1.0,   # Model doğru, insan onayladı
    ("approve", "lost"): -0.5,   # İkisi de yanıldı
    ("reject", "won"):   -0.3,   # Model doğruydu, insan reddetti
    ("reject", "lost"):  +0.2,   # İnsan haklıydı
    ("approve", "void"):  0.0,
    ("reject", "void"):   0.0,
}

REASON_PENALTY: dict[str, dict[str, float]] = {
    "sakatlik": {"lineup_importance": 0.3, "injury_weight": 0.5},
    "hava": {"weather_weight": 0.3},
    "oran_dusuk": {"min_value_edge": 0.02},
    "form": {"form_weight": 0.2, "momentum_weight": 0.15},
    "hakem": {"referee_weight": 0.2},
    "sezgi": {"human_intuition": 0.1},
}


def compute_reward(decision: str, outcome: str,
                    reject_reason: str = "") -> float:
    """Ödül hesapla (reward shaping)."""
    base = REWARD_TABLE.get((decision, outcome), 0.0)

    # Ret sebebi bonusu
    if decision == "reject" and outcome == "lost" and reject_reason:
        base += 0.1  # İnsan doğru sebeple reddetti

    return base


# ═══════════════════════════════════════════════
#  FEEDBACK STORE (SQLite)
# ═══════════════════════════════════════════════
class FeedbackStore:
    """Geri bildirim veritabanı."""

    def __init__(self, db_path: Path | str = FEEDBACK_DB):
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    selection TEXT,
                    odds REAL,
                    model_prob REAL,
                    human_decision TEXT,
                    reject_reason TEXT,
                    reject_detail TEXT,
                    confidence_override REAL,
                    outcome TEXT DEFAULT 'pending',
                    reward REAL DEFAULT 0.0,
                    timestamp REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_match
                ON feedback(match_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_decision
                ON feedback(human_decision)
            """)

    def record(self, entry: FeedbackEntry) -> int:
        """Geri bildirim kaydet."""
        entry.timestamp = entry.timestamp or time.time()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute("""
                INSERT INTO feedback
                (match_id, selection, odds, model_prob, human_decision,
                 reject_reason, reject_detail, confidence_override,
                 outcome, reward, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.match_id, entry.selection, entry.odds,
                entry.model_prob, entry.human_decision,
                entry.reject_reason, entry.reject_detail,
                entry.confidence_override, entry.outcome,
                entry.reward, entry.timestamp,
            ))
            return cur.lastrowid or 0

    def update_outcome(self, match_id: str, outcome: str) -> int:
        """Maç sonucu geldiğinde ödülü güncelle."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, human_decision, reject_reason FROM feedback "
                "WHERE match_id = ? AND outcome = 'pending'",
                (match_id,),
            ).fetchall()

            updated = 0
            for row_id, decision, reason in rows:
                reward = compute_reward(decision, outcome, reason)
                conn.execute(
                    "UPDATE feedback SET outcome = ?, reward = ? WHERE id = ?",
                    (outcome, reward, row_id),
                )
                updated += 1

            return updated

    def get_rewards(self, limit: int = 500) -> list[RewardSignal]:
        """RL eğitimi için ödül sinyallerini al."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("""
                SELECT match_id, reward, reject_reason, human_decision,
                       odds, model_prob, outcome
                FROM feedback
                WHERE outcome != 'pending'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

        signals = []
        for mid, reward, reason, decision, odds, prob, outcome in rows:
            penalty_features = REASON_PENALTY.get(reason, {})
            signals.append(RewardSignal(
                match_id=mid,
                reward=reward,
                penalty_features=penalty_features,
                meta={
                    "decision": decision,
                    "reason": reason,
                    "odds": odds,
                    "model_prob": prob,
                    "outcome": outcome,
                },
            ))

        return signals

    def get_stats(self, days: int = 30) -> FeedbackStats:
        """İstatistik raporu."""
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT human_decision, reject_reason, outcome, reward "
                "FROM feedback WHERE timestamp > ?",
                (cutoff,),
            ).fetchall()

        stats = FeedbackStats(total_feedback=len(rows))
        reasons: dict[str, int] = {}
        human_correct = 0
        model_correct = 0
        agreements = 0

        for decision, reason, outcome, reward in rows:
            stats.total_reward += reward

            if decision == "approve":
                stats.approvals += 1
                if outcome == "won":
                    model_correct += 1
                    human_correct += 1
                    agreements += 1
                elif outcome == "lost":
                    agreements += 1  # İkisi de yanıldı
            elif decision == "reject":
                stats.rejections += 1
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                if outcome == "lost":
                    human_correct += 1
                elif outcome == "won":
                    model_correct += 1

        settled = sum(1 for _, _, o, _ in rows if o in ("won", "lost"))
        if settled > 0:
            stats.human_accuracy = round(human_correct / settled, 4)
            stats.model_accuracy = round(model_correct / settled, 4)
        if len(rows) > 0:
            stats.avg_reward = round(stats.total_reward / len(rows), 4)
            stats.agreement_rate = round(agreements / len(rows), 4)
        stats.reason_distribution = reasons

        return stats


# ═══════════════════════════════════════════════
#  HUMAN FEEDBACK LOOP (Ana Sınıf)
# ═══════════════════════════════════════════════
class HumanFeedbackLoop:
    """RLHF – İnsan geri bildirimi ile model eğitimi.

    Kullanım:
        hfl = HumanFeedbackLoop()

        # Geri bildirim kaydet (Telegram callback'ten)
        hfl.record_feedback(
            match_id="gs_fb_2026",
            selection="home",
            odds=1.85,
            model_prob=0.62,
            human_decision="reject",
            reject_reason="sakatlik",
        )

        # Maç sonucu geldiğinde
        hfl.update_outcome("gs_fb_2026", "lost")

        # RL eğitimi için ödüller
        rewards = hfl.get_reward_signals()
        for r in rewards:
            rl_agent.inject_reward(r.match_id, r.reward, r.penalty_features)

        # İstatistik
        stats = hfl.get_stats()
    """

    def __init__(self, db_path: Path | str | None = None):
        self._store = FeedbackStore(db_path or FEEDBACK_DB)
        logger.debug("[RLHF] HumanFeedbackLoop başlatıldı.")

    def record_feedback(self, match_id: str, selection: str = "",
                          odds: float = 0.0, model_prob: float = 0.0,
                          human_decision: str = "approve",
                          reject_reason: str = "",
                          reject_detail: str = "",
                          confidence_override: float = 0.0) -> int:
        """Geri bildirim kaydet."""
        entry = FeedbackEntry(
            match_id=match_id,
            selection=selection,
            odds=odds,
            model_prob=model_prob,
            human_decision=human_decision,
            reject_reason=reject_reason,
            reject_detail=reject_detail,
            confidence_override=confidence_override,
        )
        row_id = self._store.record(entry)

        logger.info(
            f"[RLHF] Feedback: {match_id} → {human_decision}"
            + (f" ({reject_reason})" if reject_reason else "")
        )
        return row_id

    def update_outcome(self, match_id: str, outcome: str) -> int:
        """Maç sonucunu kaydet ve ödülleri hesapla."""
        updated = self._store.update_outcome(match_id, outcome)
        if updated:
            logger.info(f"[RLHF] Sonuç: {match_id} → {outcome} ({updated} kayıt)")
        return updated

    def get_reward_signals(self, limit: int = 500) -> list[RewardSignal]:
        """RL ajanı için ödül sinyalleri."""
        return self._store.get_rewards(limit)

    def get_feature_adjustments(self, limit: int = 200) -> dict[str, float]:
        """Ret sebeplerine göre feature ağırlık ayarlamaları.

        RL ajanının feature importance'ını güncellemek için.
        """
        signals = self._store.get_rewards(limit)
        adjustments: dict[str, float] = {}

        for sig in signals:
            if sig.reward < 0 and sig.penalty_features:
                for feat, weight in sig.penalty_features.items():
                    adjustments[feat] = adjustments.get(feat, 0.0) + weight

        # Normalize
        if adjustments:
            max_val = max(adjustments.values()) or 1.0
            adjustments = {k: round(v / max_val, 4) for k, v in adjustments.items()}

        return adjustments

    def get_stats(self, days: int = 30) -> FeedbackStats:
        """İstatistik raporu."""
        return self._store.get_stats(days)

    def generate_telegram_report(self, days: int = 30) -> str:
        """Telegram formatında RLHF raporu."""
        stats = self.get_stats(days)

        lines = [
            "🧠 <b>RLHF Geri Bildirim Raporu</b>",
            f"📅 Son {days} gün",
            "",
            f"📊 Toplam: {stats.total_feedback} karar",
            f"  ✅ Onay: {stats.approvals}",
            f"  ❌ Ret: {stats.rejections}",
            "",
            f"🎯 İnsan Doğruluğu: {stats.human_accuracy:.1%}",
            f"🤖 Model Doğruluğu: {stats.model_accuracy:.1%}",
            f"🤝 Uyum Oranı: {stats.agreement_rate:.1%}",
            f"💰 Ort. Ödül: {stats.avg_reward:+.3f}",
        ]

        if stats.reason_distribution:
            lines.append("")
            lines.append("📋 <b>Ret Sebepleri:</b>")
            for reason, count in sorted(
                stats.reason_distribution.items(),
                key=lambda x: -x[1],
            ):
                lines.append(f"  • {reason}: {count}")

        return "\n".join(lines)
