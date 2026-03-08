"""
telegram_mini_app.py – Profesyonel Telegram bot entegrasyonu.

- HTML parse modunda okunabilir mesajlar
- Value maç bulunduğunda otomatik push notification
- Sistem hatası olduğunda log'un son satırlarını gönderir
- İnteraktif butonlar ile eşik ayarı, sinyal onay/red
"""
from __future__ import annotations

import asyncio
import io
import os
import traceback
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.extensions.ceo_dashboard import CEODashboard

# Sabitler
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
ALERT_EMOJI = {
    "critical": "🔴",
    "warning": "🟡",
    "info": "🟢",
    "value": "💰",
    "error": "🚨",
    "system": "⚙️",
}


class TelegramNotifier:
    """Telegram bildirim motoru – tek sorumluluk: mesaj formatla ve gönder."""

    def __init__(self, token: str = "", chat_id: int | str = ""):
        self._token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = str(chat_id or os.getenv("TELEGRAM_CHAT_ID", ""))
        self._bot = None
        self._ready = False
        self._message_count = 0
        self._init_bot()

    def _init_bot(self):
        if not self._token:
            logger.info("Telegram token yok – bildirimler devre dışı.")
            return
        try:
            from telegram import Bot
            self._bot = Bot(token=self._token)
            self._ready = True
            logger.info("TelegramNotifier hazır.")
        except ImportError:
            logger.warning("python-telegram-bot yüklü değil.")
        except Exception as e:
            logger.error(f"Telegram bot init hatası: {e}")

    # ═══════════════════════════════════════════
    #  ANA MESAJ GÖNDERİM
    # ═══════════════════════════════════════════
    async def send(self, text: str, chat_id: str | None = None,
                   parse_mode: str = "HTML", reply_markup=None,
                   return_message_id: bool = False):
        """HTML formatlı mesaj gönderir. return_message_id=True ile msg ID döner."""
        if not self._ready or not self._bot:
            logger.debug(f"Telegram DEMO: {text[:80]}…")
            return False

        target = chat_id or self._chat_id
        if not target:
            logger.warning("Telegram chat_id belirtilmemiş.")
            return False

        try:
            msg = await self._bot.send_message(
                chat_id=target, text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
            self._message_count += 1
            return msg.message_id if return_message_id else True
        except Exception as e:
            logger.error(f"Telegram gönderim hatası: {e}")
            return 0 if return_message_id else False

    async def edit_message(self, message_id: int, text: str,
                           chat_id: str | None = None,
                           parse_mode: str = "HTML") -> bool:
        """Mevcut mesajı düzenle (Live Dashboard için)."""
        if not self._ready or not self._bot:
            return False
        target = chat_id or self._chat_id
        try:
            await self._bot.edit_message_text(
                chat_id=target, message_id=message_id,
                text=text, parse_mode=parse_mode,
            )
            return True
        except Exception as e:
            logger.debug(f"Mesaj düzenleme hatası: {e}")
            return False

    async def pin_message(self, message_id: int,
                          chat_id: str | None = None) -> bool:
        """Mesajı sabitle."""
        if not self._ready or not self._bot:
            return False
        target = chat_id or self._chat_id
        try:
            await self._bot.pin_chat_message(
                chat_id=target, message_id=message_id,
                disable_notification=True,
            )
            return True
        except Exception as e:
            logger.debug(f"Mesaj sabitleme hatası: {e}")
            return False

    # ═══════════════════════════════════════════
    #  VALUE MAÇ BİLDİRİMİ
    # ═══════════════════════════════════════════
    async def send_value_alert(self, signal: dict,
                              signal_id: str = "") -> bool:
        """Value maç bildirimi – interaktif butonlu (Human-in-the-Loop)."""
        ev = signal.get("ev", 0)
        conf = signal.get("confidence", 0)
        odds = signal.get("odds", 0)
        selection = signal.get("selection", "?")
        match_id = signal.get("match_id", "")
        home = signal.get("home_team", match_id.split("_")[0] if "_" in match_id else match_id)
        away = signal.get("away_team", match_id.split("_")[1] if "_" in match_id else "?")
        stake = signal.get("stake_pct", 0)
        mode = signal.get("trading_mode", "LIVE")

        if ev > 0.10:
            level = "critical"
            title = "KRİTİK VALUE FIRSATI"
        elif ev > 0.05:
            level = "value"
            title = "VALUE MAÇ TESPİT EDİLDİ"
        else:
            level = "info"
            title = "Olası Value Fırsatı"

        emoji = ALERT_EMOJI.get(level, "📊")
        conf_bar = self._progress_bar(conf, 10)
        now = datetime.now().strftime("%H:%M:%S")

        mode_text = ""
        if mode == "PAPER":
            mode_text = "\n🧪 <b>MOD: PAPER TRADING (Sanal)</b>\n"
        elif mode == "REDUCED":
            mode_text = "\n⚠️ <b>MOD: AZALTILMIŞ STAKE</b>\n"

        text = (
            f"{emoji} <b>{title}</b>\n"
            f"{'━' * 28}\n\n"
            f"⚽ <b>Maç:</b> {home} – {away}\n"
            f"🎯 <b>Seçim:</b> <code>{selection.upper()}</code>\n"
            f"📊 <b>Model Tahmini:</b> %{conf*100:.1f}\n"
            f"💰 <b>Piyasa Oranı:</b> {odds:.2f}\n"
            f"📈 <b>Value (EV):</b> <code>{ev*100:+.1f}%</code>\n"
            f"🏦 <b>Önerilen Stake:</b> %{stake*100:.1f}\n\n"
            f"<b>Güven:</b> {conf_bar} {conf*100:.0f}/100\n"
            f"{mode_text}\n"
            f"<i>🕐 {now}</i>"
        )

        # İnteraktif butonlar (Inline Keyboard)
        reply_markup = None
        if signal_id and self._ready:
            try:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                reply_markup = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("✅ Onayla", callback_data=f"hitl_approve_{signal_id}"),
                        InlineKeyboardButton("❌ Reddet", callback_data=f"hitl_reject_{signal_id}"),
                    ],
                    [
                        InlineKeyboardButton("📈 Detay Göster", callback_data=f"hitl_detail_{signal_id}"),
                        InlineKeyboardButton("📊 Model Karşılaştır", callback_data=f"hitl_compare_{signal_id}"),
                    ],
                ])
            except ImportError:
                pass

        return await self.send(text, reply_markup=reply_markup)

    # ═══════════════════════════════════════════
    #  SİSTEM HATASI BİLDİRİMİ
    # ═══════════════════════════════════════════
    async def send_error_alert(self, error: Exception,
                               module: str = "bahis.py",
                               include_log_tail: bool = True) -> bool:
        """Sistem hatası bildirimi + log'un son satırları."""
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        tb_short = "".join(tb[-3:])[:500]

        # Log dosyasının son satırlarını oku
        log_tail = ""
        if include_log_tail:
            log_tail = await self._read_log_tail(n_lines=10)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        text = (
            f"🚨 <b>SİSTEM HATASI</b>\n"
            f"{'━' * 28}\n\n"
            f"📦 <b>Modül:</b> <code>{module}</code>\n"
            f"❌ <b>Hata:</b> <code>{type(error).__name__}: {str(error)[:200]}</code>\n\n"
            f"📋 <b>Traceback:</b>\n"
            f"<pre>{tb_short}</pre>\n"
        )

        if log_tail:
            text += (
                f"\n📄 <b>Son Log Satırları:</b>\n"
                f"<pre>{log_tail}</pre>\n"
            )

        text += f"\n<i>🕐 {now}</i>"

        return await self.send(text)

    # ═══════════════════════════════════════════
    #  SCRAPER ÇÖKME BİLDİRİMİ
    # ═══════════════════════════════════════════
    async def send_scraper_down(self, scraper_name: str, error: str) -> bool:
        text = (
            f"⚙️ <b>SCRAPER ÇÖKTÜ</b>\n"
            f"{'━' * 28}\n\n"
            f"🔧 <b>Scraper:</b> <code>{scraper_name}</code>\n"
            f"❌ <b>Hata:</b> <code>{error[:300]}</code>\n"
            f"🔄 <b>Durum:</b> Circuit Breaker AÇIK – yeniden deneme bekleniyor.\n\n"
            f"<i>🕐 {datetime.now().strftime('%H:%M:%S')}</i>"
        )
        return await self.send(text)

    # ═══════════════════════════════════════════
    #  GÜNLÜK ÖZET RAPOR
    # ═══════════════════════════════════════════
    async def send_daily_summary(self, stats: dict) -> bool:
        """Günlük performans özeti."""
        bankroll = stats.get("bankroll", 10000)
        pnl = stats.get("pnl_today", 0)
        roi = stats.get("roi", 0)
        bets_placed = stats.get("bets_placed", 0)
        bets_won = stats.get("bets_won", 0)
        win_rate = bets_won / max(bets_placed, 1) * 100
        sharpe = stats.get("sharpe", 0)

        pnl_emoji = "📈" if pnl >= 0 else "📉"

        text = (
            f"📊 <b>GÜNLÜK RAPOR</b>\n"
            f"{'━' * 28}\n\n"
            f"🏦 <b>Kasa:</b> ₺{bankroll:,.0f}\n"
            f"{pnl_emoji} <b>Günlük PnL:</b> ₺{pnl:+,.0f}\n"
            f"📈 <b>ROI:</b> {roi:+.1f}%\n"
            f"📊 <b>Sharpe:</b> {sharpe:.2f}\n\n"
            f"🎯 <b>Bahisler:</b> {bets_placed}\n"
            f"✅ <b>Kazanan:</b> {bets_won} ({win_rate:.0f}%)\n"
            f"❌ <b>Kaybeden:</b> {bets_placed - bets_won}\n\n"
            f"<i>🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
        return await self.send(text)

    # ═══════════════════════════════════════════
    #  ANOMALI / DROPPING ODDS BİLDİRİMİ
    # ═══════════════════════════════════════════
    async def send_anomaly_alert(self, anomaly: dict) -> bool:
        """Oran düşüşü veya Smart Money tespitinde bildirim."""
        text = (
            f"🟡 <b>ANOMALİ TESPİTİ – Smart Money?</b>\n"
            f"{'━' * 28}\n\n"
            f"⚽ <b>Maç:</b> {anomaly.get('match', '?')}\n"
            f"📉 <b>Oran Düşüşü:</b> {anomaly.get('from_odds', 0):.2f} → {anomaly.get('to_odds', 0):.2f}\n"
            f"📊 <b>Z-Score:</b> {anomaly.get('z_score', 0):.2f}\n"
            f"⏱ <b>Süre:</b> Son {anomaly.get('hours', 0)} saatte\n"
            f"🎯 <b>Pazar:</b> {anomaly.get('market', '1X2')}\n\n"
            f"<i>🕐 {datetime.now().strftime('%H:%M:%S')}</i>"
        )
        return await self.send(text)

    # ═══════════════════════════════════════════
    #  YARDIMCI METODLAR
    # ═══════════════════════════════════════════
    @staticmethod
    def _progress_bar(value: float, width: int = 10) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    async def _read_log_tail(n_lines: int = 10) -> str:
        error_log = LOG_DIR / "error.log"
        main_log = LOG_DIR / "bahis.log"

        target = error_log if error_log.exists() else main_log
        if not target.exists():
            return "(Log dosyası bulunamadı)"
        try:
            content = await asyncio.to_thread(target.read_text, encoding="utf-8", errors="replace")
            lines = content.strip().splitlines()
            tail = lines[-n_lines:] if len(lines) > n_lines else lines
            return "\n".join(tail)[:1000]
        except Exception:
            return "(Log okunamadı)"


class TelegramApp:
    """Telegram bot entegrasyonu – interaktif komutlar + bildirimler + rich media.

    Komutlar:
      /start    – Bot'u başlat
      /durum    – Anlık kasa + sistem durumu
      /fikstur  – Günün maçları
      /signals  – Aktif value sinyalleri
      /report   – Günlük PnL/ROI raporu
      /clv      – CLV trend özeti
      /devreler – Circuit breaker durumları
      /log      – error.log dosyasını belge olarak gönder
      /volatility – Takım VIX
      /help     – Komut listesi
    """

    def __init__(self, threshold_ctrl=None, token: str = "",
                 chat_id: str = "", notifier: TelegramNotifier | None = None,
                 db=None, cb_registry=None, clv_tracker=None,
                 chart_sender=None, hitl=None, portfolio=None):
        self._token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._threshold = threshold_ctrl
        self._notifier = notifier or TelegramNotifier(self._token, self._chat_id)
        self._db = db
        self._cb_registry = cb_registry
        self._clv = clv_tracker
        self._chart = chart_sender
        self._hitl = hitl           # HumanInTheLoop
        self._portfolio = portfolio # PortfolioOptimizer
        self._bot = None
        self._app = None
        self.ceo_dashboard = CEODashboard()
        logger.debug("TelegramApp başlatıldı.")

    @property
    def notifier(self) -> TelegramNotifier:
        return self._notifier

    async def start(self, shutdown: asyncio.Event):
        if not self._token:
            logger.info("Telegram token yok – bot demo modunda.")
            await self._demo_mode(shutdown)
            return

        try:
            from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
            from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler

            self._app = ApplicationBuilder().token(self._token).build()

            commands = {
                "start": self._cmd_start,
                "help": self._cmd_help,
                "durum": self._cmd_durum,
                "fikstur": self._cmd_fikstur,
                "signals": self._cmd_signals,
                "status": self._cmd_durum,
                "report": self._cmd_report,
                "clv": self._cmd_clv,
                "devreler": self._cmd_circuit_breakers,
                "log": self._cmd_send_log,
                "volatility": self._cmd_volatility,
                "hitl": self._cmd_hitl_stats,
                "portfoy": self._cmd_portfolio,
                "vision": self._cmd_vision,
            }
            for cmd, handler in commands.items():
                self._app.add_handler(CommandHandler(cmd, handler))

            self._app.add_handler(CallbackQueryHandler(self._button_handler))

            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling()

            logger.info("Telegram bot başlatıldı.")
            while not shutdown.is_set():
                await asyncio.sleep(1)

            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        except ImportError:
            logger.warning("python-telegram-bot yüklü değil.")
            await self._demo_mode(shutdown)
        except Exception as e:
            logger.error(f"Telegram hatası: {e}")
            await self._notifier.send_error_alert(e, module="telegram_mini_app")

    async def _demo_mode(self, shutdown: asyncio.Event):
        while not shutdown.is_set():
            await asyncio.sleep(30)

    # ═══════════════════════════════════════════
    #  /start
    # ═══════════════════════════════════════════
    async def _cmd_start(self, update, context):
        text = (
            "🏟️ <b>Quant Betting Bot v3.0</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📊 /signals – Value sinyalleri\n"
            "⚡ /durum – Kasa + CPU/RAM durumu\n"
            "📅 /fikstur – Günün maçları\n"
            "📋 /report – PnL/ROI raporu\n"
            "📈 /clv – Closing Line Value trend\n"
            "🔧 /devreler – Circuit Breaker durumları\n"
            "📈 /volatility – Takım VIX\n"
            "📄 /log – error.log dosyasını gönder\n"
            "👁️ /vision – God Mode & CEO Dashboard\n"
            "❓ /help – Tüm komutlar\n\n"
            "<i>Bot otomatik olarak bildirir:\n"
            "💰 Value fırsatı, 📉 Anomali, 🚨 Hata</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_vision(self, update, context):
        report = self.ceo_dashboard.generate_report(None)
        await update.message.reply_text(report, parse_mode="Markdown")

    # ═══════════════════════════════════════════
    #  /durum – Anlık kasa + sistem
    # ═══════════════════════════════════════════
    async def _cmd_durum(self, update, context):
        now = datetime.now().strftime("%H:%M:%S")
        try:
            import psutil
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent if os.name != "nt" else psutil.disk_usage("C:\\").percent
        except ImportError:
            cpu, ram, disk = 0, 0, 0

        # Circuit breaker durumu
        open_cbs = []
        if self._cb_registry:
            open_cbs = self._cb_registry.open_breakers()

        cb_text = "🟢 Tümü aktif" if not open_cbs else "🔴 " + ", ".join(open_cbs)

        text = (
            f"⚡ <b>SİSTEM DURUMU</b> – {now}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🟢 <b>Durum:</b> Çalışıyor\n"
            f"🖥️ <b>CPU:</b> {cpu}%\n"
            f"🧠 <b>RAM:</b> {ram}%\n"
            f"💾 <b>Disk:</b> {disk}%\n"
            f"🔌 <b>Devreler:</b> {cb_text}\n"
            f"📨 <b>Mesaj:</b> {self._notifier._message_count}\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /fikstur – Günün maçları
    # ═══════════════════════════════════════════
    async def _cmd_fikstur(self, update, context):
        if not self._db:
            await update.message.reply_text(
                "📅 <b>FİKSTÜR</b>\n\n<i>Veritabanı bağlantısı yok.</i>",
                parse_mode="HTML",
            )
            return

        try:
            matches = self._db.get_upcoming_matches()
            if matches.is_empty():
                await update.message.reply_text(
                    "📅 <b>FİKSTÜR</b>\n\n<i>Yaklaşan maç bulunamadı.</i>",
                    parse_mode="HTML",
                )
                return

            text = "📅 <b>GÜNÜN MAÇLARI</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            for i, row in enumerate(matches.iter_rows(named=True)):
                if i >= 20:
                    text += f"\n<i>…ve {len(matches) - 20} maç daha</i>"
                    break
                home = row.get("home_team", "?")
                away = row.get("away_team", "?")
                league = row.get("league", "")[:15]
                ho = row.get("home_odds", 0) or 0
                do_ = row.get("draw_odds", 0) or 0
                ao = row.get("away_odds", 0) or 0

                text += f"⚽ <b>{home}</b> vs <b>{away}</b>\n"
                if league:
                    text += f"   🏆 {league}\n"
                if ho > 0:
                    text += f"   💰 {ho:.2f} | {do_:.2f} | {ao:.2f}\n"
                text += "\n"

            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Fikstur hatası: {e}", parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /signals
    # ═══════════════════════════════════════════
    async def _cmd_signals(self, update, context):
        text = (
            "📊 <b>AKTİF SİNYALLER</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<i>Analiz döngüsü çalışıyor…\n"
            "Value fırsatları otomatik gönderilir.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /report
    # ═══════════════════════════════════════════
    async def _cmd_report(self, update, context):
        await update.message.reply_text(
            "📋 <b>RAPOR</b>\n\n"
            "<i>Günlük PnL, ROI ve Sharpe bilgileri hazırlanıyor…</i>",
            parse_mode="HTML",
        )

    # ═══════════════════════════════════════════
    #  /clv – CLV trend özeti
    # ═══════════════════════════════════════════
    async def _cmd_clv(self, update, context):
        if not self._clv:
            await update.message.reply_text(
                "📈 <b>CLV</b>\n\n<i>CLV Tracker henüz başlatılmamış.</i>",
                parse_mode="HTML",
            )
            return

        stats = self._clv.aggregate_stats()
        if stats.get("status") == "yeterli veri yok":
            await update.message.reply_text(
                "📈 <b>CLV</b>\n\n<i>Henüz yeterli CLV verisi yok.</i>",
                parse_mode="HTML",
            )
            return

        text = (
            f"📈 <b>CLV (Closing Line Value)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 <b>Bahis Sayısı:</b> {stats['n_bets']}\n"
            f"📈 <b>Ort. CLV:</b> <code>{stats['avg_clv']:+.4f}</code>\n"
            f"📉 <b>CLV Std:</b> {stats['clv_std']:.4f}\n"
            f"✅ <b>Pozitif CLV Oranı:</b> {stats['positive_clv_rate']:.0%}\n"
            f"🎯 <b>Win Rate:</b> {stats['win_rate']:.0%}\n"
            f"📊 <b>Sharpe-like:</b> {stats['sharpe_like']:.2f}\n\n"
            f"💡 <i>{stats['interpretation']}</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /devreler – Circuit Breaker durumları
    # ═══════════════════════════════════════════
    async def _cmd_circuit_breakers(self, update, context):
        if not self._cb_registry:
            await update.message.reply_text(
                "🔧 <b>DEVRELER</b>\n\n<i>Registry yok.</i>",
                parse_mode="HTML",
            )
            return

        statuses = self._cb_registry.all_statuses()
        if not statuses:
            await update.message.reply_text(
                "🔧 <b>DEVRELER</b>\n\n<i>Kayıtlı devre yok.</i>",
                parse_mode="HTML",
            )
            return

        text = "🔧 <b>CIRCUIT BREAKER DURUMU</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        for s in statuses:
            state = s["state"]
            icon = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}.get(state, "⚪")
            remaining = s.get("time_until_recovery_min", 0)
            text += (
                f"{icon} <b>{s['name']}</b>\n"
                f"   Durum: <code>{state}</code>"
            )
            if state == "OPEN":
                text += f" (kalan: {remaining:.0f} dk)"
            text += (
                f"\n   Çağrı: {s['total_calls']} | "
                f"Hata: {s['failures']} | "
                f"Red: {s['total_rejected']}\n\n"
            )

        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /log – error.log dosyasını belge olarak gönder
    # ═══════════════════════════════════════════
    async def _cmd_send_log(self, update, context):
        """error.log dosyasını Telegram'dan sendDocument ile gönderir."""
        error_log = LOG_DIR / "error.log"

        if not error_log.exists():
            await update.message.reply_text(
                "📄 <b>LOG</b>\n\n<i>error.log dosyası bulunamadı (hata olmamış!).</i>",
                parse_mode="HTML",
            )
            return

        try:
            file_size = error_log.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                await update.message.reply_text(
                    "📄 Log dosyası çok büyük (>50MB). Son 1000 satır gönderiliyor…",
                    parse_mode="HTML",
                )
                content = await asyncio.to_thread(error_log.read_text, encoding="utf-8", errors="replace")
                lines = content.splitlines()
                tail = "\n".join(lines[-1000:])
                buf = io.BytesIO(tail.encode("utf-8"))
                buf.name = "error_tail.log"
                await update.message.reply_document(
                    document=buf,
                    caption="📄 error.log (son 1000 satır)",
                )
            else:
                data = await asyncio.to_thread(error_log.read_bytes)
                buf = io.BytesIO(data)
                buf.name = "error.log"
                await update.message.reply_document(
                    document=buf,
                    caption=f"📄 error.log ({file_size/1024:.0f} KB)",
                )
        except Exception as e:
            await update.message.reply_text(
                f"❌ Log gönderim hatası: <code>{e}</code>",
                parse_mode="HTML",
            )

    # ═══════════════════════════════════════════
    #  /volatility
    # ═══════════════════════════════════════════
    async def _cmd_volatility(self, update, context):
        text = (
            "📈 <b>TAKIM VOLATİLİTE ENDEKSİ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<i>Volatilite verileri yükleniyor…</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /help
    # ═══════════════════════════════════════════
    async def _cmd_help(self, update, context):
        text = (
            "❓ <b>KOMUT LİSTESİ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>📊 Analiz</b>\n"
            "/signals – Value sinyalleri\n"
            "/fikstur – Günün maçları\n"
            "/clv – Closing Line Value\n"
            "/volatility – Takım VIX\n\n"
            "<b>💼 Portföy</b>\n"
            "/portfoy – Kasa + drawdown durumu\n"
            "/hitl – Model vs İnsan istatistikleri\n\n"
            "<b>⚙️ Sistem</b>\n"
            "/durum – CPU/RAM/devre durumu\n"
            "/devreler – Circuit Breaker\n"
            "/report – PnL raporu\n"
            "/log – error.log gönder\n\n"
            "<b>🤖 Otomatik</b>\n"
            "💰 Value → [✅ Onayla] [❌ Reddet] butonlu\n"
            "📉 Dropping odds tespit edildiğinde\n"
            "📊 Maç analiz grafikleri\n"
            "🚨 Sistem hatası oluştuğunda\n"
            "🧪 Drawdown'da Paper Trading modu"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /hitl – Model vs İnsan istatistikleri
    # ═══════════════════════════════════════════
    async def _cmd_hitl_stats(self, update, context):
        if not self._hitl:
            await update.message.reply_text(
                "🤖 <b>HITL</b>\n\n<i>Human-in-the-Loop modülü başlatılmamış.</i>",
                parse_mode="HTML",
            )
            return

        stats = self._hitl.performance_comparison()
        if stats.get("status") == "yeterli_veri_yok":
            await update.message.reply_text(
                "🤖 <b>Model vs İnsan</b>\n\n<i>Henüz yeterli veri yok.</i>",
                parse_mode="HTML",
            )
            return

        text = (
            f"🤖 <b>MODEL vs İNSAN</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 <b>Toplam Sinyal:</b> {stats['total_signals']}\n"
            f"🎯 <b>Model Win Rate:</b> {stats['model_win_rate']:.0%}\n\n"
            f"✅ <b>Onaylanan:</b> {stats['human_approved']}\n"
            f"   Win Rate: {stats['human_approved_win_rate']:.0%}\n"
            f"❌ <b>Reddedilen:</b> {stats['human_rejected']}\n"
            f"   Veto Doğruluğu: {stats['veto_accuracy']:.0%}\n\n"
            f"💡 <i>{stats['veto_interpretation']}</i>\n\n"
            f"💰 <b>Model PnL:</b> ₺{stats['model_total_pnl']:+,.0f}\n"
            f"📉 <b>Kaçırılan:</b> ₺{stats['missed_value_pnl']:+,.0f}\n"
            f"⏱ <b>Ort. Yanıt:</b> {stats['avg_response_time_sec']:.0f}s"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /portfoy – Portföy durumu
    # ═══════════════════════════════════════════
    async def _cmd_portfolio(self, update, context):
        if not self._portfolio:
            await update.message.reply_text(
                "💼 <b>PORTFÖY</b>\n\n<i>Portföy modülü başlatılmamış.</i>",
                parse_mode="HTML",
            )
            return

        st = self._portfolio.status()
        mode_emoji = {"LIVE": "🟢", "REDUCED": "🟡", "PAPER": "🧪", "FROZEN": "🔴"
                      }.get(st["mode"], "⚪")

        text = (
            f"💼 <b>PORTFÖY DURUMU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🏦 <b>Kasa:</b> ₺{st['bankroll']:,.0f}\n"
            f"📈 <b>Zirve:</b> ₺{st['peak']:,.0f}\n"
            f"📉 <b>Drawdown:</b> {st['drawdown_pct']}\n"
            f"{mode_emoji} <b>Mod:</b> {st['mode']}\n\n"
            f"🎯 <b>Bahisler:</b> {st['total_bets']}\n"
            f"🧪 <b>Paper:</b> {st['paper_bets']}\n"
            f"💰 <b>Toplam PnL:</b> ₺{st['total_pnl']:+,.0f}\n"
            f"✅ <b>Win Rate:</b> {st['win_rate']:.0%}"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  BUTON HANDLER (HITL + diğer)
    # ═══════════════════════════════════════════
    async def _button_handler(self, update, context):
        query = update.callback_query
        await query.answer()
        data = query.data

        # ── HITL butonları ──
        if data.startswith("hitl_approve_"):
            signal_id = data.replace("hitl_approve_", "")
            if self._hitl:
                self._hitl.approve(signal_id)
            original = query.message.text or ""
            await query.edit_message_text(
                original + "\n\n✅ <b>ONAYLANDI</b> – Bahis kaydedildi.",
                parse_mode="HTML",
            )
            return

        if data.startswith("hitl_reject_"):
            signal_id = data.replace("hitl_reject_", "")
            if self._hitl:
                self._hitl.reject(signal_id, reason="Telegram'dan reddedildi")
            original = query.message.text or ""
            await query.edit_message_text(
                original + "\n\n❌ <b>REDDEDİLDİ</b> – Sonucu takip edeceğiz.",
                parse_mode="HTML",
            )
            return

        if data.startswith("hitl_detail_"):
            signal_id = data.replace("hitl_detail_", "")
            if self._hitl:
                sig = self._hitl.get_signal(signal_id)
                if sig:
                    detail = (
                        f"📈 <b>DETAY: {signal_id}</b>\n\n"
                        f"Maç: {sig.match_id}\n"
                        f"Seçim: {sig.selection}\n"
                        f"Oran: {sig.odds:.2f}\n"
                        f"EV: {sig.ev*100:+.1f}%\n"
                        f"Güven: {sig.confidence:.0%}\n"
                        f"Stake: {sig.stake_pct:.2%}\n"
                        f"Model Tahmin: {sig.model_prediction}\n"
                        f"Oluşturulma: {sig.created_at}"
                    )
                    await query.message.reply_text(detail, parse_mode="HTML")
            return

        if data.startswith("hitl_compare_"):
            await query.message.reply_text(
                "📊 <i>Model karşılaştırması hesaplanıyor…</i>",
                parse_mode="HTML",
            )
            return

        # ── Diğer butonlar ──
        if data == "threshold_up":
            if self._threshold:
                self._threshold.adjust(+0.01)
            await query.edit_message_text("✅ Eşik +0.01 artırıldı.", parse_mode="HTML")
        elif data == "threshold_down":
            if self._threshold:
                self._threshold.adjust(-0.01)
            await query.edit_message_text("✅ Eşik -0.01 azaltıldı.", parse_mode="HTML")
        elif data == "cb_reset_all":
            if self._cb_registry:
                self._cb_registry.reset_all()
            await query.edit_message_text("🔧 Tüm devreler sıfırlandı.", parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  ESKİ UYUMLULUK
    # ═══════════════════════════════════════════
    async def send_signal(self, chat_id: int | str, signal: dict):
        await self._notifier.send_value_alert(signal)
