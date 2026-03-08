"""
psycho_profiler.py – Yatırımcı Psikolojisi Profilleme (Behavioral Finance).

Bot mükemmel çalışsa bile, son kararı veren SİZSİNİZ.
Korkularınız ve açgözlülüğünüz sistemin en zayıf halkasıdır.

Bot sizi eğitmeli:
  "Mehmet Bey, son 1 ayda korktuğunuz için oynamadığınız
   maçlardan 5.000 TL kaçırdınız."

Takip Edilen Metrikler:
  - Omission Error: Reddedilen ama tutan bahisler
  - Commission Error: Onaylanan ama yatan bahisler
  - Risk Aversion: Risk iştahı skoru
  - Recency Bias: Son sonuçlara aşırı tepki
  - Loss Aversion: Kayıptan sonra aşırı temkinli olma
  - Overconfidence: Ardışık kazançtan sonra aşırı agresiflik
  - Streak Sensitivity: Seri halinde karar değişikliği
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
PROFILE_DIR = ROOT / "data" / "psychology"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Decision:
    """Tek bir karar kaydı."""
    decision_id: str = ""
    match_id: str = ""
    selection: str = ""
    odds: float = 0.0
    ev: float = 0.0
    confidence: float = 0.0
    model_recommendation: str = ""  # "BET" | "PASS"
    human_decision: str = ""        # "APPROVED" | "REJECTED"
    actual_result: str = ""         # "WIN" | "LOSS" | "PENDING"
    potential_pnl: float = 0.0      # Tutan/yatılan potential kâr/zarar
    stake: float = 0.0
    timestamp: float = 0.0
    emotion_tag: str = ""           # fear | greed | rational | impulsive


@dataclass
class PsychologyReport:
    """Psikoloji profil raporu."""
    period: str = ""               # "weekly" | "monthly"
    total_decisions: int = 0
    approved: int = 0
    rejected: int = 0
    # Hata metrikleri
    omission_errors: int = 0       # Reddedilen ama tutan
    commission_errors: int = 0     # Onaylanan ama yatan
    omission_cost: float = 0.0     # Kaçırılan kâr
    commission_cost: float = 0.0   # Gereksiz kayıp
    # Psikolojik metrikler
    risk_aversion_score: float = 0.5  # 0=çok agresif, 1=çok korku
    recency_bias: float = 0.0        # Son sonuçlara tepki
    loss_aversion: float = 0.0       # Kayıptan sonra temkinlilik
    overconfidence: float = 0.0      # Kazançtan sonra agresiflik
    streak_sensitivity: float = 0.0  # Seri sonrası karar değişikliği
    emotional_decision_rate: float = 0.0  # Duygusal karar oranı
    # Genel
    optimal_score: float = 0.0    # İnsan filtresi başarısı (0-1)
    bot_alone_roi: float = 0.0    # Bot tek başına ROI
    human_filter_roi: float = 0.0 # İnsan filtreli ROI
    recommendation: str = ""
    telegram_text: str = ""


class PsychoProfiler:
    """Yatırımcı psikolojisi profilleme motoru.

    Kullanım:
        profiler = PsychoProfiler()
        # Karar kaydet
        profiler.record_decision("GS_FB", "home", "APPROVED", odds=1.80, ev=0.05)
        # Sonuç kaydet
        profiler.settle_decision("GS_FB", "home", won=True)
        # Haftalık rapor
        report = profiler.weekly_report()
    """

    def __init__(self):
        self._decisions: list[Decision] = []
        self._load_history()
        logger.debug("[Psycho] Profiler başlatıldı.")

    # ═══════════════════════════════════════════
    #  KARAR KAYIT
    # ═══════════════════════════════════════════
    def record_decision(self, match_id: str, selection: str,
                         human_decision: str, odds: float = 0.0,
                         ev: float = 0.0, confidence: float = 0.0,
                         stake: float = 0.0,
                         model_rec: str = "BET",
                         emotion: str = "rational"):
        """İnsan kararını kaydet."""
        d = Decision(
            decision_id=f"{match_id}_{selection}_{int(time.time())}",
            match_id=match_id,
            selection=selection,
            odds=odds,
            ev=ev,
            confidence=confidence,
            model_recommendation=model_rec,
            human_decision=human_decision.upper(),
            stake=stake,
            timestamp=time.time(),
            emotion_tag=emotion,
        )
        self._decisions.append(d)
        self._save_history()

        logger.debug(
            f"[Psycho] Karar: {match_id} {selection} → "
            f"{human_decision} (model: {model_rec})"
        )

    def settle_decision(self, match_id: str, selection: str,
                          won: bool):
        """Kararın sonucunu kaydet."""
        for d in reversed(self._decisions):
            if (d.match_id == match_id and
                d.selection == selection and
                d.actual_result == "PENDING"):

                d.actual_result = "WIN" if won else "LOSS"

                if d.human_decision == "APPROVED":
                    d.potential_pnl = (
                        d.stake * (d.odds - 1) if won
                        else -d.stake
                    )
                elif d.human_decision == "REJECTED" and won:
                    d.potential_pnl = d.stake * (d.odds - 1)

                break

        self._save_history()

    # ═══════════════════════════════════════════
    #  RAPOR
    # ═══════════════════════════════════════════
    def weekly_report(self) -> PsychologyReport:
        """Son 7 günlük psikoloji raporu."""
        cutoff = time.time() - 7 * 86400
        return self._generate_report(
            [d for d in self._decisions if d.timestamp >= cutoff],
            period="weekly",
        )

    def monthly_report(self) -> PsychologyReport:
        """Son 30 günlük psikoloji raporu."""
        cutoff = time.time() - 30 * 86400
        return self._generate_report(
            [d for d in self._decisions if d.timestamp >= cutoff],
            period="monthly",
        )

    def _generate_report(self, decisions: list[Decision],
                          period: str = "weekly") -> PsychologyReport:
        """Rapor üret."""
        report = PsychologyReport(period=period)
        settled = [d for d in decisions if d.actual_result in ("WIN", "LOSS")]

        report.total_decisions = len(decisions)
        report.approved = sum(1 for d in decisions if d.human_decision == "APPROVED")
        report.rejected = sum(1 for d in decisions if d.human_decision == "REJECTED")

        if not settled:
            report.recommendation = "Henüz yeterli veri yok."
            return report

        # ── Omission Errors: Reddedilen ama tutan ──
        rejected_won = [
            d for d in settled
            if d.human_decision == "REJECTED" and d.actual_result == "WIN"
        ]
        report.omission_errors = len(rejected_won)
        report.omission_cost = round(sum(d.potential_pnl for d in rejected_won), 2)

        # ── Commission Errors: Onaylanan ama yatan ──
        approved_lost = [
            d for d in settled
            if d.human_decision == "APPROVED" and d.actual_result == "LOSS"
        ]
        report.commission_errors = len(approved_lost)
        report.commission_cost = round(sum(abs(d.potential_pnl) for d in approved_lost), 2)

        # ── Risk Aversion Score ──
        if report.total_decisions > 0:
            rejection_rate = report.rejected / report.total_decisions
            report.risk_aversion_score = round(rejection_rate, 3)

        # ── Recency Bias ──
        report.recency_bias = self._detect_recency_bias(settled)

        # ── Loss Aversion ──
        report.loss_aversion = self._detect_loss_aversion(settled)

        # ── Overconfidence ──
        report.overconfidence = self._detect_overconfidence(settled)

        # ── Streak Sensitivity ──
        report.streak_sensitivity = self._detect_streak_sensitivity(decisions)

        # ── Emotional Decision Rate ──
        emotional = sum(
            1 for d in decisions
            if d.emotion_tag in ("fear", "greed", "impulsive")
        )
        if decisions:
            report.emotional_decision_rate = round(emotional / len(decisions), 3)

        # ── Bot vs Human ROI ──
        bot_pnl = 0
        human_pnl = 0
        for d in settled:
            won = d.actual_result == "WIN"
            stake = d.stake or 100
            if d.model_recommendation == "BET":
                bot_pnl += stake * (d.odds - 1) if won else -stake
            if d.human_decision == "APPROVED":
                human_pnl += stake * (d.odds - 1) if won else -stake

        total_stake = len(settled) * 100
        report.bot_alone_roi = round(bot_pnl / max(total_stake, 1), 4)
        report.human_filter_roi = round(human_pnl / max(total_stake, 1), 4)

        # ── Optimal Score ──
        correct = sum(
            1 for d in settled
            if (d.human_decision == "APPROVED" and d.actual_result == "WIN") or
               (d.human_decision == "REJECTED" and d.actual_result == "LOSS")
        )
        report.optimal_score = round(correct / max(len(settled), 1), 3)

        # ── Tavsiye ──
        report.recommendation = self._generate_advice(report)
        report.telegram_text = self._format_telegram(report)

        return report

    # ═══════════════════════════════════════════
    #  PSİKOLOJİK TESPİTLER
    # ═══════════════════════════════════════════
    def _detect_recency_bias(self, decisions: list[Decision]) -> float:
        """Son sonuçlara aşırı tepki ölçümü."""
        if len(decisions) < 5:
            return 0.0

        changes = 0
        for i in range(1, len(decisions)):
            prev = decisions[i - 1]
            curr = decisions[i]
            if (prev.actual_result == "LOSS" and
                curr.human_decision == "REJECTED" and
                prev.human_decision == "APPROVED"):
                changes += 1
            elif (prev.actual_result == "WIN" and
                  curr.human_decision == "APPROVED" and
                  prev.human_decision == "REJECTED"):
                changes += 1

        return round(changes / max(len(decisions) - 1, 1), 3)

    def _detect_loss_aversion(self, decisions: list[Decision]) -> float:
        """Kayıptan sonra aşırı temkinlilik."""
        if len(decisions) < 5:
            return 0.0

        rejections_after_loss = 0
        total_after_loss = 0

        for i in range(1, len(decisions)):
            if decisions[i - 1].actual_result == "LOSS":
                total_after_loss += 1
                if decisions[i].human_decision == "REJECTED":
                    rejections_after_loss += 1

        if total_after_loss == 0:
            return 0.0
        return round(rejections_after_loss / total_after_loss, 3)

    def _detect_overconfidence(self, decisions: list[Decision]) -> float:
        """Kazançtan sonra aşırı agresiflik."""
        if len(decisions) < 5:
            return 0.0

        approvals_after_win = 0
        total_after_win = 0

        for i in range(1, len(decisions)):
            if decisions[i - 1].actual_result == "WIN":
                total_after_win += 1
                if decisions[i].human_decision == "APPROVED":
                    approvals_after_win += 1

        if total_after_win == 0:
            return 0.0
        return round(approvals_after_win / total_after_win, 3)

    def _detect_streak_sensitivity(self, decisions: list[Decision]) -> float:
        """Seri halinde karar değişikliği."""
        if len(decisions) < 5:
            return 0.0

        streaks = 0
        current_streak = 1
        for i in range(1, len(decisions)):
            if decisions[i].human_decision == decisions[i - 1].human_decision:
                current_streak += 1
            else:
                if current_streak >= 3:
                    streaks += 1
                current_streak = 1

        return round(streaks / max(len(decisions) // 3, 1), 3)

    # ═══════════════════════════════════════════
    #  TAVSİYE & FORMAT
    # ═══════════════════════════════════════════
    def _generate_advice(self, report: PsychologyReport) -> str:
        """Kişiselleştirilmiş psikoloji tavsiyesi."""
        advices = []

        if report.omission_cost > 500:
            advices.append(
                f"KORKU YÜZÜNDEN kaçırılan kâr: {report.omission_cost:,.0f} TL. "
                f"Modele daha fazla güvenin."
            )

        if report.loss_aversion > 0.7:
            advices.append(
                "Kayıptan sonra aşırı temkinlisiniz (Loss Aversion). "
                "Her bahis bağımsızdır – geçmiş kayıp gelecek fırsatları etkilemez."
            )

        if report.overconfidence > 0.8:
            advices.append(
                "Kazandıktan sonra çok agresifsiniz (Overconfidence). "
                "Ardışık kazanç yeteneğinizi kanıtlamaz – stake'i artırmayın."
            )

        if report.recency_bias > 0.3:
            advices.append(
                "Son sonuçlara aşırı tepki veriyorsunuz (Recency Bias). "
                "Model 3 yıllık veriye bakıyor, siz son 3 maça."
            )

        if report.emotional_decision_rate > 0.4:
            advices.append(
                f"Kararlarınızın %{report.emotional_decision_rate:.0%}'ı "
                f"duygusal. Mekanik kurallara bağlı kalın."
            )

        if report.bot_alone_roi > report.human_filter_roi:
            advices.append(
                f"Bot tek başına ({report.bot_alone_roi:.1%} ROI) "
                f"sizin filtrenizden ({report.human_filter_roi:.1%} ROI) "
                f"daha iyi performans gösteriyor. Müdahaleyi azaltın."
            )

        return " | ".join(advices) if advices else "Harika gidiyorsunuz. Disiplinli kalın."

    def _format_telegram(self, report: PsychologyReport) -> str:
        """Telegram HTML mesaj formatı."""
        period_name = "Haftalık" if report.period == "weekly" else "Aylık"

        text = (
            f"🧠 <b>{period_name} Psikoloji Raporu</b>\n"
            f"{'━' * 30}\n\n"
            f"📊 Toplam Karar: {report.total_decisions}\n"
            f"  ✅ Onaylanan: {report.approved}\n"
            f"  ❌ Reddedilen: {report.rejected}\n\n"
        )

        if report.omission_errors:
            text += (
                f"💸 <b>Kaçırılan Fırsat:</b> {report.omission_errors} maç\n"
                f"  Maliyet: {report.omission_cost:,.0f} TL\n\n"
            )

        if report.commission_errors:
            text += (
                f"📉 <b>Gereksiz Kayıp:</b> {report.commission_errors} maç\n"
                f"  Maliyet: {report.commission_cost:,.0f} TL\n\n"
            )

        text += (
            f"🎯 <b>Psikolojik Profil:</b>\n"
            f"  Risk İştahı: {'Düşük' if report.risk_aversion_score > 0.6 else 'Normal' if report.risk_aversion_score > 0.3 else 'Yüksek'}\n"
            f"  Kayıp Korkusu: {report.loss_aversion:.0%}\n"
            f"  Aşırı Güven: {report.overconfidence:.0%}\n"
            f"  Duygusal Karar: {report.emotional_decision_rate:.0%}\n\n"
            f"🤖 Bot ROI: {report.bot_alone_roi:.1%}\n"
            f"👤 İnsan+Bot ROI: {report.human_filter_roi:.1%}\n\n"
            f"💡 {report.recommendation[:200]}"
        )

        return text

    # ═══════════════════════════════════════════
    #  KALICILIK
    # ═══════════════════════════════════════════
    def _save_history(self):
        path = PROFILE_DIR / "decisions.json"
        try:
            data = [asdict(d) for d in self._decisions[-1000:]]
            path.write_text(json.dumps(data, ensure_ascii=False, default=str))
        except Exception:
            pass

    def _load_history(self):
        path = PROFILE_DIR / "decisions.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for d in data:
                self._decisions.append(Decision(**{
                    k: v for k, v in d.items()
                    if k in Decision.__dataclass_fields__
                }))
            logger.info(f"[Psycho] {len(self._decisions)} karar yüklendi.")
        except Exception:
            pass

    @property
    def total_decisions(self) -> int:
        return len(self._decisions)
