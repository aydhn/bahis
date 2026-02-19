"""
daily_briefing.py – Executive Daily Briefing (Yönetici Günlük Özeti).

Her sabah belirlenen saatte tüm analizleri, kasa durumunu
ve günün en önemli fırsatlarını tarayan bir CEO raporu sunar.

Format:
  📊 GÜNLÜK BRİFİNG – 16 Şubat 2026
  ─────────────────────────────
  💰 KASA: 12.450 TL (+3.2% dün)
  📈 ROI (30g): +8.7% | Sharpe: 1.42
  
  🎯 GÜNÜN EN İYİ 3 FIRSATI:
  1. GS vs FB – EV: +8%, Kelly: 3.2%
  2. BJK vs TS – EV: +5%, Kelly: 2.1%
  3. ADS vs GZT – EV: +4%, Kelly: 1.8%
  
  ⚠️ UYARILAR:
  - Model drift tespit (Wasserstein: 0.32)
  - GS defansı yorgun (stamina: 28%)
  
  🧠 SİSTEM DURUMU:
  - Uptime: 99.7% | Hata: 0.3%
  - Son başarı: %62 (7 gün)

Teknoloji: LLM özetleme + zamanlanmış iş (APScheduler)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

try:
    from src.utils.gemini_client import gemini_generate, GEMINI_OK as _GEMINI_OK
    GEMINI_OK = _GEMINI_OK
except ImportError:
    GEMINI_OK = False

OLLAMA_URL = "http://localhost:11434/api/generate"


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class BriefingData:
    """Brifing ham verileri."""
    # Kasa
    bankroll: float = 0.0
    bankroll_change_pct: float = 0.0
    roi_30d: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    # Performans
    win_rate_7d: float = 0.0
    total_bets_7d: int = 0
    profit_7d: float = 0.0
    # Fırsatlar
    top_opportunities: list[dict] = field(default_factory=list)
    # Uyarılar
    alerts: list[str] = field(default_factory=list)
    # Sistem
    uptime_pct: float = 99.0
    error_rate: float = 0.0
    model_accuracy: float = 0.0
    # Piyasa
    market_chaos_score: float = 0.0
    drift_detected: bool = False


@dataclass
class BriefingReport:
    """Günlük brifing raporu."""
    date: str = ""
    # Formatlanmış çıktı
    telegram_text: str = ""
    summary: str = ""
    # Meta
    generation_time_ms: float = 0.0
    method: str = ""  # "llm" | "template"


# ═══════════════════════════════════════════════
#  LLM ÖZETLEME
# ═══════════════════════════════════════════════
BRIEFING_SYSTEM_PROMPT = (
    "Sen bir finansal analistsin. Sana verilen ham verileri kurumsal "
    "Türkçe ile 1 paragrafta özetle. Gereksiz detay verme, sadece "
    "sonucu söyle. Kısa ve net ol. Emoji kullan ama abartma."
)


def _summarize_ollama(raw_text: str,
                        model: str = "llama3:8b") -> str:
    """Ollama ile özetleme."""
    if not HTTPX_OK:
        return ""
    try:
        resp = httpx.post(OLLAMA_URL, json={
            "model": model,
            "system": BRIEFING_SYSTEM_PROMPT,
            "prompt": f"Bu verileri özetle:\n{raw_text}",
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        }, timeout=30.0)
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception:
        pass
    return ""


def _summarize_gemini(raw_text: str) -> str:
    """Gemini ile özetleme."""
    if not GEMINI_OK:
        return ""
    try:
        result = gemini_generate(prompt=raw_text, system=BRIEFING_SYSTEM_PROMPT)
        return result
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════
#  DAILY BRIEFING (Ana Sınıf)
# ═══════════════════════════════════════════════
class DailyBriefing:
    """Yönetici günlük brifing üretici.

    Kullanım:
        db = DailyBriefing(llm_backend="auto")

        data = BriefingData(
            bankroll=12450,
            bankroll_change_pct=3.2,
            roi_30d=8.7,
            ...
        )

        report = db.generate(data)
        telegram_send(report.telegram_text)
    """

    def __init__(self, llm_backend: str = "auto"):
        self._backend = llm_backend
        logger.debug(f"[Briefing] Başlatıldı (backend={llm_backend})")

    def generate(self, data: BriefingData) -> BriefingReport:
        """Günlük brifing oluştur."""
        t0 = time.perf_counter()
        report = BriefingReport(
            date=datetime.now().strftime("%d %B %Y"),
        )

        # Template tabanlı metin
        template_text = self._build_template(data)

        # LLM özeti dene
        llm_summary = ""
        if self._backend in ("ollama", "auto"):
            llm_summary = _summarize_ollama(template_text)
        if not llm_summary and self._backend in ("gemini", "auto"):
            llm_summary = _summarize_gemini(template_text)

        if llm_summary:
            report.summary = llm_summary.strip()
            report.method = "llm"
        else:
            report.summary = self._build_short_summary(data)
            report.method = "template"

        # Telegram formatı
        report.telegram_text = self._format_telegram(data, report.summary)
        report.generation_time_ms = round(
            (time.perf_counter() - t0) * 1000, 1,
        )

        return report

    def collect_data(self, db: Any = None,
                       portfolio: Any = None,
                       health: Any = None,
                       chaos_filter: Any = None,
                       super_log: Any = None,
                       guardian: Any = None) -> BriefingData:
        """Sistemden veri topla – genişletilmiş metrikler."""
        data = BriefingData()

        # Portfolio metrikleri
        if portfolio:
            try:
                if hasattr(portfolio, "get_bankroll"):
                    data.bankroll = portfolio.get_bankroll()
                if hasattr(portfolio, "get_daily_pnl"):
                    pnl = portfolio.get_daily_pnl()
                    if data.bankroll > 0:
                        data.bankroll_change_pct = round(
                            pnl / data.bankroll * 100, 2,
                        )
                if hasattr(portfolio, "get_sharpe"):
                    data.sharpe_ratio = portfolio.get_sharpe()
                if hasattr(portfolio, "get_max_drawdown"):
                    data.max_drawdown = portfolio.get_max_drawdown()
            except Exception as e:
                logger.debug(f"[Briefing] Portfolio veri hatası: {e}")

        # Sağlık metrikleri
        if health:
            try:
                if hasattr(health, "get_win_rate"):
                    data.win_rate_7d = health.get_win_rate(days=7)
                if hasattr(health, "get_roi"):
                    data.roi_30d = health.get_roi(days=30)
                if hasattr(health, "get_model_accuracy"):
                    data.model_accuracy = health.get_model_accuracy()
                if hasattr(health, "get_uptime"):
                    data.uptime_pct = health.get_uptime()
            except Exception as e:
                logger.debug(f"[Briefing] Health veri hatası: {e}")

        # Fırsat taraması
        if db:
            try:
                if hasattr(db, "get_todays_opportunities"):
                    data.top_opportunities = db.get_todays_opportunities(
                        limit=3,
                    )
            except Exception as e:
                logger.debug(f"[Briefing] DB veri hatası: {e}")

        # SuperLogger metrikleri
        if super_log:
            try:
                if hasattr(super_log, "summarize_session"):
                    session = super_log.summarize_session()
                    data.error_rate = session.get("error_rate", 0.0) * 100
                    if session.get("total_errors", 0) > 10:
                        data.alerts.append(
                            f"Yüksek hata oranı: {session['total_errors']} hata"
                        )
            except Exception as e:
                logger.debug(f"[Briefing] SuperLog veri hatası: {e}")

        # Guardian metrikleri
        if guardian:
            try:
                if hasattr(guardian, "health_report"):
                    gh = guardian.health_report()
                    if gh.get("open_circuits"):
                        for circuit in gh["open_circuits"]:
                            data.alerts.append(f"Açık devre: {circuit}")
            except Exception as e:
                logger.debug(f"[Briefing] Guardian veri hatası: {e}")

        # Chaos filter
        if chaos_filter:
            try:
                if hasattr(chaos_filter, "get_chaos_score"):
                    data.market_chaos_score = chaos_filter.get_chaos_score()
                    if data.market_chaos_score > 0.7:
                        data.alerts.append(
                            f"Yüksek piyasa kaosu: {data.market_chaos_score:.2f}"
                        )
            except Exception as e:
                logger.debug(f"[Briefing] Chaos veri hatası: {e}")

        return data

    def _build_template(self, data: BriefingData) -> str:
        """Ham veri template metni."""
        opps = ""
        for i, opp in enumerate(data.top_opportunities[:3], 1):
            if isinstance(opp, dict):
                opps += (
                    f"{i}. {opp.get('home', '?')} vs {opp.get('away', '?')} "
                    f"– EV: {opp.get('ev', 0):+.1%}, "
                    f"Kelly: {opp.get('kelly', 0):.1%}\n"
                )

        alerts = "\n".join(f"- {a}" for a in data.alerts) if data.alerts else "Yok"

        return (
            f"Kasa: {data.bankroll:.0f} TL ({data.bankroll_change_pct:+.1f}% dün)\n"
            f"ROI (30g): {data.roi_30d:+.1f}%\n"
            f"Sharpe: {data.sharpe_ratio:.2f}\n"
            f"Drawdown: {data.max_drawdown:.1f}%\n"
            f"Başarı (7g): {data.win_rate_7d:.0%} ({data.total_bets_7d} bahis)\n"
            f"Kar (7g): {data.profit_7d:+.0f} TL\n\n"
            f"Fırsatlar:\n{opps}\n"
            f"Uyarılar:\n{alerts}\n"
            f"Sistem: Uptime {data.uptime_pct:.1f}%, "
            f"Hata {data.error_rate:.1f}%"
        )

    def _build_short_summary(self, data: BriefingData) -> str:
        """Kısa template özeti (LLM yoksa)."""
        if data.bankroll_change_pct > 0:
            trend = "pozitif"
        elif data.bankroll_change_pct < 0:
            trend = "negatif"
        else:
            trend = "yatay"

        n_opps = len(data.top_opportunities)
        return (
            f"Kasa {trend} seyretti ({data.bankroll_change_pct:+.1f}%). "
            f"30 günlük ROI {data.roi_30d:+.1f}%. "
            f"Bugün {n_opps} fırsat tespit edildi. "
            f"{'⚠️ ' + str(len(data.alerts)) + ' uyarı var.' if data.alerts else 'Sistem stabil.'}"
        )

    def _format_telegram(self, data: BriefingData,
                           summary: str) -> str:
        """Telegram HTML formatı."""
        date_str = datetime.now().strftime("%d %B %Y, %A")

        lines = [
            f"📊 <b>GÜNLÜK BRİFİNG – {date_str}</b>\n",
            "─" * 30,
            f"💰 <b>KASA:</b> {data.bankroll:,.0f} TL "
            f"({data.bankroll_change_pct:+.1f}% dün)",
            f"📈 ROI (30g): {data.roi_30d:+.1f}% | "
            f"Sharpe: {data.sharpe_ratio:.2f}",
            f"📉 Max Drawdown: {data.max_drawdown:.1f}%",
            f"🎯 Başarı (7g): {data.win_rate_7d:.0%} "
            f"({data.total_bets_7d} bahis, {data.profit_7d:+.0f} TL)\n",
        ]

        # Fırsatlar
        if data.top_opportunities:
            lines.append("🎯 <b>GÜNÜN EN İYİ FIRSATLARI:</b>")
            for i, opp in enumerate(data.top_opportunities[:3], 1):
                if isinstance(opp, dict):
                    lines.append(
                        f"  {i}. {opp.get('home', '?')} vs "
                        f"{opp.get('away', '?')} – "
                        f"EV: {opp.get('ev', 0):+.1%}, "
                        f"Kelly: {opp.get('kelly', 0):.1%}"
                    )
            lines.append("")

        # Uyarılar
        if data.alerts:
            lines.append("⚠️ <b>UYARILAR:</b>")
            for alert in data.alerts[:5]:
                lines.append(f"  • {alert}")
            lines.append("")

        # Sistem
        lines.extend([
            "🧠 <b>SİSTEM DURUMU:</b>",
            f"  Uptime: {data.uptime_pct:.1f}% | "
            f"Hata: {data.error_rate:.1f}%",
            f"  Model Doğruluğu: {data.model_accuracy:.0%}\n",
        ])

        # LLM özeti
        if summary:
            lines.extend([
                "─" * 30,
                f"💡 <b>ÖZET:</b> <i>{summary}</i>",
            ])

        return "\n".join(lines)
