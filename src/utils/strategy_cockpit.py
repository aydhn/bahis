"""
strategy_cockpit.py – Live Strategy Cockpit (Anlık Komuta Merkezi).

Telegram'dan tek bir komutla sistemin tüm durumunu, aktif bahisleri,
risk metriklerini ve model sağlığını tek bir "cockpit" görünümünde
sunar. Bir savaş uçağının HUD'u gibi.

Kavramlar:
  - HUD (Heads-Up Display): Tek bakışta tüm kritik bilgiler
  - System Heartbeat: CPU, RAM, uptime, aktif modüller
  - Risk Dashboard: Drawdown, exposure, Kelly rejimi
  - Model Health: Son 100 bahiste doğruluk, CLV, Sharpe
  - Active Positions: Açık bahisler ve P&L
  - Alerts: Aktif uyarılar ve anomaliler
  - Quick Actions: Komut butonları (durdur, devam, rapor)

Akış:
  1. /cockpit komutu gelir
  2. Tüm modüllerden anlık veri toplanır
  3. Tek bir zengin HTML mesajı oluşturulur
  4. Inline butonlar ile hızlı aksiyonlar sunulur
  5. Her 30 saniyede otomatik güncellenir (edit message)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class CockpitData:
    """Cockpit görünümü için toplanan veriler."""
    # Sistem
    uptime_hours: float = 0.0
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    ram_mb: float = 0.0
    active_modules: int = 0
    disabled_modules: int = 0
    # Kasa
    bankroll: float = 0.0
    peak: float = 0.0
    drawdown_pct: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    # Risk
    kelly_regime: str = ""       # "normal" | "reduced" | "frozen"
    vol_regime: str = ""         # "calm" | "elevated" | "storm" | "crisis"
    chaos_level: str = ""        # "stable" | "chaotic"
    daily_exposure_pct: float = 0.0
    # Performans
    total_bets: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    clv_avg: float = 0.0
    roi_pct: float = 0.0
    # Aktif bahisler
    active_bets: int = 0
    pending_signals: int = 0
    # Uyarılar
    alerts: list[str] = field(default_factory=list)
    # Zaman
    timestamp: str = ""
    cycle: int = 0


# ═══════════════════════════════════════════════
#  STRATEGY COCKPIT (Ana Sınıf)
# ═══════════════════════════════════════════════
class StrategyCockpit:
    """Live Strategy Cockpit — Telegram Komuta Merkezi.

    Kullanım:
        cockpit = StrategyCockpit()

        # Veri topla
        data = cockpit.collect(
            bankroll=10500, peak=11000,
            regime_kelly=regime_kelly,
            vol_analyzer=vol_analyzer,
            orchestrator=orchestrator,
            cycle=42,
        )

        # Telegram mesajı oluştur
        html = cockpit.render(data)
        await notifier.send(html, parse_mode="HTML")
    """

    def __init__(self, start_time: float | None = None):
        self._start = start_time or time.time()
        self._last_data: CockpitData | None = None
        self._message_id: int = 0

        logger.debug("[Cockpit] Komuta merkezi başlatıldı.")

    def collect(self, bankroll: float = 0, peak: float = 0,
                  daily_pnl: float = 0, weekly_pnl: float = 0,
                  total_bets: int = 0, win_rate: float = 0,
                  sharpe: float = 0, clv_avg: float = 0,
                  active_bets: int = 0, pending_signals: int = 0,
                  cycle: int = 0,
                  regime_kelly: object = None,
                  vol_analyzer: object = None,
                  chaos_filter: object = None,
                  orchestrator: object = None,
                  alerts: list[str] | None = None,
                  **kwargs) -> CockpitData:
        """Tüm modüllerden anlık veri topla."""
        data = CockpitData(timestamp=datetime.utcnow().isoformat())

        # Sistem
        data.uptime_hours = round((time.time() - self._start) / 3600, 2)
        if PSUTIL_OK:
            data.cpu_pct = round(psutil.cpu_percent(), 1)
            mem = psutil.virtual_memory()
            data.ram_pct = round(mem.percent, 1)
            data.ram_mb = round(
                psutil.Process().memory_info().rss / (1024 * 1024), 1,
            )

        # Kasa
        data.bankroll = round(bankroll, 2)
        data.peak = round(peak, 2)
        data.drawdown_pct = round(
            (bankroll - peak) / max(peak, 1) * 100, 2,
        ) if peak > 0 else 0
        data.daily_pnl = round(daily_pnl, 2)
        data.weekly_pnl = round(weekly_pnl, 2)

        # Performans
        data.total_bets = total_bets
        data.win_rate = round(win_rate, 4)
        data.sharpe = round(sharpe, 4)
        data.clv_avg = round(clv_avg, 4)
        data.roi_pct = round(
            (bankroll - 10000) / 10000 * 100, 2,
        ) if bankroll > 0 else 0
        data.active_bets = active_bets
        data.pending_signals = pending_signals
        data.cycle = cycle

        # Risk rejimleri
        if regime_kelly and hasattr(regime_kelly, 'get_stats'):
            rk_stats = regime_kelly.get_stats()
            dd = rk_stats.get("drawdown", 0)
            if dd < -0.25:
                data.kelly_regime = "🔴 FROZEN"
            elif dd < -0.10:
                data.kelly_regime = "🟡 REDUCED"
            else:
                data.kelly_regime = "🟢 NORMAL"
            data.daily_exposure_pct = round(
                rk_stats.get("daily_exposure", 0) / max(bankroll, 1) * 100, 1,
            )

        if vol_analyzer and hasattr(vol_analyzer, '_last_regime'):
            data.vol_regime = getattr(vol_analyzer, '_last_regime', 'unknown')

        # Orchestrator
        if orchestrator and hasattr(orchestrator, 'get_stats'):
            o_stats = orchestrator.get_stats()
            data.active_modules = o_stats.total_tasks_run
            data.disabled_modules = len(o_stats.disabled_tasks)

        # Alerts
        data.alerts = alerts or []
        if data.drawdown_pct < -15:
            data.alerts.append("⚠️ Ağır drawdown!")
        if data.cpu_pct > 90:
            data.alerts.append("⚠️ CPU %90+ kullanım!")

        self._last_data = data
        return data

    def render(self, data: CockpitData) -> str:
        """Cockpit görünümünü HTML olarak render et."""
        dd_emoji = (
            "🟢" if data.drawdown_pct > -5
            else "🟡" if data.drawdown_pct > -15
            else "🔴"
        )
        pnl_emoji = "📈" if data.daily_pnl >= 0 else "📉"

        lines = [
            "🛩️ <b>STRATEGY COCKPIT</b>",
            "<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>",
            "",
            "💰 <b>KASA</b>",
            f"  Bakiye: <b>{data.bankroll:,.2f}</b>",
            f"  Zirve:  {data.peak:,.2f}",
            f"  {dd_emoji} DD: <b>{data.drawdown_pct:+.1f}%</b>",
            f"  {pnl_emoji} Günlük: <b>{data.daily_pnl:+.2f}</b>",
            f"  📊 Haftalık: {data.weekly_pnl:+.2f}",
            f"  ROI: {data.roi_pct:+.1f}%",
            "",
            "<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>",
            "📊 <b>PERFORMANS</b>",
            f"  Toplam Bahis: {data.total_bets}",
            f"  Win Rate: {data.win_rate:.1%}",
            f"  Sharpe: {data.sharpe:.2f}",
            f"  CLV: {data.clv_avg:+.2%}",
            f"  Aktif: {data.active_bets} | Sinyal: {data.pending_signals}",
            "",
            "<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>",
            "⚡ <b>RİSK REJİMİ</b>",
            f"  Kelly: {data.kelly_regime or '?'}",
            f"  Volatilite: {data.vol_regime or '?'}",
            f"  Kaos: {data.chaos_level or '?'}",
            f"  Günlük Exp: {data.daily_exposure_pct:.1f}%",
            "",
            "<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>",
            "🖥️ <b>SİSTEM</b>",
            f"  Uptime: {data.uptime_hours:.1f}h",
            f"  CPU: {data.cpu_pct}% | RAM: {data.ram_mb:.0f}MB ({data.ram_pct}%)",
            f"  Döngü: #{data.cycle}",
        ]

        if data.alerts:
            lines.extend([
                "",
                "<code>━━━━━━━━━━━━━━━━━━━━━━━━</code>",
                "🚨 <b>UYARILAR</b>",
            ])
            for alert in data.alerts[:5]:
                lines.append(f"  {alert}")

        lines.extend([
            "",
            f"<code>🕐 {data.timestamp[:19]}</code>",
        ])

        return "\n".join(lines)

    def get_keyboard(self) -> InlineKeyboardMarkup | None:
        """Hızlı aksiyon butonları."""
        if not TELEGRAM_OK:
            return None

        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "🔄 Yenile", callback_data="cockpit_refresh",
                ),
                InlineKeyboardButton(
                    "⏸️ Durdur", callback_data="cockpit_pause",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📊 Detay", callback_data="cockpit_detail",
                ),
                InlineKeyboardButton(
                    "🔔 Sessiz", callback_data="cockpit_mute",
                ),
            ],
        ])

    async def send_cockpit(self, notifier, data: CockpitData | None = None,
                             chat_id: str | None = None) -> int:
        """Cockpit mesajını Telegram'a gönder."""
        if not notifier:
            return 0

        d = data or self._last_data
        if d is None:
            d = self.collect()

        html = self.render(d)
        keyboard = self.get_keyboard()

        try:
            msg_id = await notifier.send(
                html, chat_id=chat_id,
                parse_mode="HTML",
                reply_markup=keyboard,
                return_message_id=True,
            )
            if isinstance(msg_id, int):
                self._message_id = msg_id
            return msg_id if isinstance(msg_id, int) else 0
        except Exception as e:
            logger.debug(f"[Cockpit] Gönderim hatası: {e}")
            return 0

    async def update_cockpit(self, notifier,
                               data: CockpitData | None = None,
                               chat_id: str | None = None) -> None:
        """Mevcut cockpit mesajını güncelle (edit)."""
        if not notifier or not self._message_id:
            await self.send_cockpit(notifier, data, chat_id)
            return

        d = data or self._last_data
        if d is None:
            return

        html = self.render(d)
        keyboard = self.get_keyboard()

        try:
            bot = getattr(notifier, '_bot', None)
            target = chat_id or getattr(notifier, '_chat_id', '')
            if bot and target:
                await bot.edit_message_text(
                    text=html,
                    chat_id=target,
                    message_id=self._message_id,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
        except Exception as e:
            logger.debug(f"[Cockpit] Edit hatası: {e}")
