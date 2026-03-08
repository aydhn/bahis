"""
narrative_engine.py – Automated Investment Memo Generator.

Bu modül, sayısal analizleri (Quant, Risk, Philosophy) birleştirerek
insan tarafından okunabilir, profesyonel "Yatırım Notları" (Investment Memo) üretir.
Bill Benter'ın çalışma notları ile Wall Street raporlarının birleşimidir.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.quant.analysis.philosophical_engine import EpistemicReport
from src.quant.risk.volatility_analyzer import VolatilityReport


class NarrativeEngine:
    """
    Otonom anlatı motoru. Veriyi hikayeye çevirir.
    """

    def __init__(self):
        pass

    def generate_memo(self,
                      match_id: str,
                      selection: str,
                      odds: float,
                      stake: float,
                      confidence: float,
                      edge: float,
                      philo_report: Optional[EpistemicReport] = None,
                      vol_report: Optional[VolatilityReport] = None,
                      news_summary: str = "") -> str:
        """
        Detaylı yatırım notu oluşturur.
        """
        # 1. Başlık ve Karar
        emoji = "🟢" if confidence > 0.7 else "🟡"
        if confidence > 0.85: emoji = "🔥"

        lines = [
            f"# {emoji} YATIRIM NOTU: {match_id}",
            f"**Tarih:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Seçim:** {selection} @ {odds:.2f}",
            f"**Pozisyon:** {stake:.2f} TL (Edge: %{edge*100:.2f})",
            "",
            "## 1. Kantitatif Görünüm (The Numbers)",
            f"- **Model Güveni:** %{confidence*100:.1f}",
            f"- **Piyasa Beklentisi:** %{ (1/odds)*100:.1f} (Implied)",
            f"- **Matematiksel Avantaj:** %{edge*100:.2f} (Kelly Kriteri Onaylı)",
        ]

        # 2. Volatilite ve Rejim (The Context)
        if vol_report:
            regime_emoji = {
                "calm": "🌊 Sakin",
                "elevated": "⚠️ Yükselmiş",
                "storm": "⛈️ Fırtına",
                "crisis": "🚨 KRİZ",
                "insufficient_data": "❓ Bilinmiyor"
            }.get(vol_report.regime, vol_report.regime)

            lines.extend([
                "",
                "## 2. Piyasa Rejimi (Volatility Context)",
                f"- **Durum:** {regime_emoji}",
                f"- **GARCH Oynaklığı:** σ={vol_report.current_volatility:.4f} (Ort: {vol_report.avg_volatility:.4f})",
                f"- **Risk Çarpanı:** x{vol_report.kelly_multiplier:.2f}",
                f"- **Yorum:** {vol_report.recommendation}"
            ])

        # 3. Felsefi Derinlik (The Wisdom)
        if philo_report:
            philo_status = "✅ Onaylı" if philo_report.epistemic_approved else "❌ Red"
            lines.extend([
                "",
                "## 3. Epistemik Analiz (Philosophical Audit)",
                f"- **Karar:** {philo_status} (Skor: {philo_report.epistemic_score:.2f}/1.0)",
                f"- **Black Swan Riski:** {philo_report.black_swan_risk:.2f} (Düşük iyidir)",
                f"- **Antifragility:** {philo_report.antifragility:.2f} (Pozitif iyidir)",
                f"- **Piyasa Algısı:** {philo_report.crowd_vs_herd.replace('_', ' ').title()}",
                "",
                "### 💭 İçgörüler (Reflections):"
            ])
            for ref in philo_report.reflections:
                lines.append(f"> *\"{ref}\"*")

        # 4. Haber ve Sentiment (The Narrative)
        if news_summary:
            lines.extend([
                "",
                "## 4. Saha Dışı Faktörler (Narrative)",
                f"{news_summary}"
            ])

        # 5. Sonuç
        lines.extend([
            "",
            "---",
            "*Bu rapor Otonom Quant Sistemi tarafından üretilmiştir.*"
        ])

        return "\n".join(lines)

    def generate_short_rationale(self,
                                 edge: float,
                                 philo_report: Optional[EpistemicReport],
                                 vol_report: Optional[VolatilityReport]) -> str:
        """Kısa gerekçe cümlesi (Loglar için)."""
        reasons = [f"Edge: %{edge*100:.1f}"]

        if vol_report:
            reasons.append(f"Regime: {vol_report.regime}")

        if philo_report:
            reasons.append(f"Epistemic: {philo_report.epistemic_score:.2f}")
            if philo_report.crowd_vs_herd == "herd_behavior":
                reasons.append("Fade the Herd")

        return ", ".join(reasons)
