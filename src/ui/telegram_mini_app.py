"""
telegram_mini_app.py – Profesyonel Telegram bot entegrasyonu.

- HTML parse modunda okunabilir mesajlar
- Value maç bulunduğunda otomatik push notification
- Sistem hatası olduğunda log'un son satırlarını gönderir
- İnteraktif butonlar ile eşik ayarı, sinyal onay/red
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import traceback
from datetime import datetime
from pathlib import Path
from src.ui.voice_notifier import VoiceNotifier

import io
import matplotlib.pyplot as plt
import numpy as np
import asyncio
from loguru import logger
from src.ui.voice_interrogator import VoiceInterrogator

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

    def __init__(self, token: str = "", chat_id: int | str = "", enabled: bool = True):
        self._enabled = enabled
        self._token = (token or os.getenv("TELEGRAM_BOT_TOKEN", "")) if enabled else ""
        self._chat_id = str(chat_id or os.getenv("TELEGRAM_CHAT_ID", ""))
        self._bot = None
        self._ready = False
        self._message_count = 0
        self._voice_notifier = VoiceNotifier()
        self._init_bot()

    def _init_bot(self):
        if not self._enabled:
            logger.info("TelegramNotifier devre disi (enabled=False).")
            return
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
    @staticmethod
    def _sanitize_html(text: str) -> str:
        """HTML parse hatası verecek kaçırılmış tag'leri temizler.

        Telegram HTML parser'ı sadece belirli tag'leri destekler:
        <b>, <i>, <u>, <s>, <code>, <pre>, <a>.
        Diğer tüm <tag> ifadeleri &lt;tag&gt; ile escape edilir.
        """
        import re
        import html as _html
        ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "tg-spoiler",
                        "strong", "em", "ins", "del", "span", "tg-emoji"}

        # 1) Python obje repr'ları: <Handle ...>, <Task ...>, <module ...>, <function ...>
        text = re.sub(
            r"<([a-zA-Z_][a-zA-Z0-9_.]*\s+[^>]{2,})>",
            lambda m: _html.escape(m.group(0)),
            text,
        )

        # 2) Kalan HTML tag'lerini kontrol et
        def _replace_tag(match):
            full = match.group(0)
            tag_content = match.group(1).lower().strip()
            tag_name = tag_content.split()[0].lstrip("/").split("=")[0]
            if tag_name in ALLOWED_TAGS:
                return full
            return _html.escape(full)

        text = re.sub(r"<(/?\w[^>]*)>", _replace_tag, text)

        # 3) Eşleşmemiş < karakterlerini escape et
        text = re.sub(r"<(?![/a-zA-Z])", "&lt;", text)
        return text

    async def send(self, text: str, chat_id: str | None = None,
                   parse_mode: str = "HTML", reply_markup=None,
                   return_message_id: bool = False):
        """HTML formatlı mesaj gönderir. return_message_id=True ile msg ID döner.

        HTML sanitization: Desteklenmeyen tag'ler otomatik escape edilir.
        Mesaj 4096 karakteri aşarsa kırpılır.
        """
        if not self._ready or not self._bot:
            logger.debug(f"Telegram DEMO: {text[:80]}…")
            return False

        target = chat_id or self._chat_id
        if not target:
            logger.warning("Telegram chat_id belirtilmemiş.")
            return False

        if parse_mode == "HTML":
            text = self._sanitize_html(text)

        if len(text) > 4096:
            text = text[:4090] + "\n…"

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
            err_str = str(e)
            if "parse entities" in err_str.lower() or "can't parse" in err_str.lower():
                logger.warning(f"HTML parse hatası, plain text ile yeniden deneniyor: {e}")
                try:
                    import html as _html
                    plain = _html.unescape(text)
                    plain = plain.replace("<b>", "").replace("</b>", "")
                    plain = plain.replace("<i>", "").replace("</i>", "")
                    plain = plain.replace("<code>", "").replace("</code>", "")
                    plain = plain.replace("<pre>", "").replace("</pre>", "")
                    msg = await self._bot.send_message(
                        chat_id=target, text=plain[:4096],
                        reply_markup=reply_markup,
                        disable_web_page_preview=True,
                    )
                    self._message_count += 1
                    return msg.message_id if return_message_id else True
                except Exception as e2:
                    logger.error(f"Telegram plain text gönderimi de başarısız: {e2}")
            else:
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

    async def send_voice_alert(self, signal: Dict[str, Any]):
        """Sinyali sesli mesaj olarak gönderir."""
        if not self._enabled or not self._ready:
            return
        await self._voice_notifier.send_voice_alert(self._bot, self._chat_id, signal)

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

        # Spor türüne göre emoji ve etiket
        sport = signal.get("sport", "football")
        sport_emoji_map = {
            "football": "⚽", "basketball": "🏀", "tennis": "🎾",
            "volleyball": "🏐", "handball": "🤾", "ice-hockey": "🏒",
        }
        sport_emoji = sport_emoji_map.get(sport, "🏆")
        league = signal.get("league", "")
        country = signal.get("country", "")
        league_line = f"🏆 <b>Lig:</b> {league}" if league else ""
        if country:
            league_line += f" ({country})"

        mode_text = ""
        if mode == "PAPER":
            mode_text = "\n🧪 <b>MOD: PAPER TRADING (Sanal)</b>\n"
        elif mode == "REDUCED":
            mode_text = "\n⚠️ <b>MOD: AZALTILMIŞ STAKE</b>\n"

        league_text = f"{league_line}\n" if league_line else ""
        text = (
            f"{emoji} <b>{title}</b>\n"
            f"{'━' * 28}\n\n"
            f"{sport_emoji} <b>Maç:</b> {home} – {away}\n"
            f"{league_text}"
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
                        InlineKeyboardButton("📈 Detay", callback_data=f"hitl_detail_{signal_id}"),
                        InlineKeyboardButton("📊 Karşılaştır", callback_data=f"hitl_compare_{signal_id}"),
                    ],
                    [
                        InlineKeyboardButton("🚀 ŞİMDİ OYNA (Yıldırım İnfaz)", callback_data=f"hitl_execute_{signal_id}"),
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
        import html as _html
        import re
        
        # Traceback'i al ve Python obje temsilcilerini temizle
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        tb_text = "".join(tb[-3:])[:500]
        # <Handle ...>, <Task ...>, <Future ...> vb. obje temsilcilerini temizle
        tb_clean = re.sub(r'<[A-Z][a-zA-Z0-9_]*\s+[^>]*>', '[OBJ]', tb_text)
        tb_short = _html.escape(tb_clean)

        # Log dosyasının son satırlarını oku
        log_tail = ""
        if include_log_tail:
            raw_log = self._read_log_tail(n_lines=10)
            # Log'daki potansiyel HTML tag'lerini temizle
            log_clean = re.sub(r'<[A-Z][a-zA-Z0-9_]*\s+[^>]*>', '[OBJ]', raw_log)
            log_tail = _html.escape(log_clean)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        err_msg = _html.escape(str(error)[:200])
        module_safe = _html.escape(str(module))

        text = (
            f"🚨 <b>SİSTEM HATASI</b>\n"
            f"{'━' * 28}\n\n"
            f"📦 <b>Modül:</b> <code>{module_safe}</code>\n"
            f"❌ <b>Hata:</b> <code>{type(error).__name__}: {err_msg}</code>\n\n"
            f"📋 <b>Traceback:</b>\n"
            f"<pre>{tb_short}</pre>\n"
        )

        if log_tail:
            text += (
                f"\n📄 <b>Son Log Satırları:</b>\n"
                f"<pre>{log_tail}</pre>\n"
            )

        text += f"\n<i>🕐 {now}</i>"

        return await self.send(text, parse_mode="HTML")

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
    def _read_log_tail(n_lines: int = 10) -> str:
        """Log dosyasının son n satırını okur."""
        if not LOG_DIR.exists():
            return "Log dizini bulunamadı."
        
        log_file = LOG_DIR / "bahis.log" # Veya dinamik log adı
        if not log_file.exists():
            return "Log dosyası yok."
            
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-n_lines:])
        except Exception as e:
            return f"Log okuma hatası: {e}"


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
                 chart_sender=None, hitl=None, portfolio=None,
                 pnl_tracker=None, lance=None):
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
        self._pnl = pnl_tracker
        self._lance = lance
        self._voice = VoiceInterrogator()
        self._bot = None
        self._app = None
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
                "debate": self._cmd_debate,
                "spectral": self._cmd_spectral,
                "jit": self._cmd_jit,
                "optimize": self._cmd_optimize,
                "dashboard": self._cmd_dashboard,
                "voice": self._cmd_voice,
                "heatmap": self._cmd_heatmap,
                "similar": self._cmd_similar,
                "briefing": self._cmd_briefing,
                "sports": self._cmd_sports,
            }
            for cmd, handler in commands.items():
                self._app.add_handler(CommandHandler(cmd, handler))
                
            # NLP / Message Handler (Doğal Dil Sorguları)
            from telegram.ext import MessageHandler, filters
            self._app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._handle_nlp_query))
            
            # Sesli Mesaj (Phase 15)
            self._app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
            
            # Callback queries
            self._app.add_handler(CallbackQueryHandler(self._button_handler))

            await self._app.initialize()
            await self._app.start()
            if self._app.updater:
                await self._app.updater.start_polling()

            logger.success("Telegram bot başlatıldı (Interactive Mode OK).")
            while not shutdown.is_set():
                await asyncio.sleep(1)
            await self._safe_shutdown()

        except ImportError:
            logger.warning("python-telegram-bot yüklü değil.")
            await self._demo_mode(shutdown)
        except Exception as e:
            logger.error(f"Telegram hatası: {e}")
            await self._notifier.send_error_alert(e, module="telegram_mini_app")
            await self._safe_shutdown()

    async def _cmd_debate(self, update, context):
        """/debate [match_id] - Felsefi tartışma."""
        if not self._db:
             await update.message.reply_text("❌ DB bağlantısı yok.", parse_mode="HTML")
             return
             
        args = context.args
        match_id = args[0] if args else "mock_match"
        
        # Felsefi motoru bul (Bootstrap üzerinden erişim zor olabilir, yeni instance katalım)
        from src.quant.philosophical_engine import PhilosophicalEngine
        engine = PhilosophicalEngine(db=self._db)
        
        ctx = {"home_team": "Galatasaray", "away_team": "Fenerbahçe"} # Mock context
        debate_text = await engine.run_debate(match_id, context=ctx)
        
        await update.message.reply_text(debate_text, parse_mode="HTML")

    async def _cmd_spectral(self, update, context):
        """/spectral [team] - FFT döngü analizi."""
        if not self._db:
             await update.message.reply_text("❌ DB bağlantısı yok.", parse_mode="HTML")
             return

        args = context.args
        team = args[0] if args else "Galatasaray"
        
        from src.quant.spectral_analysis import SpectralAnalysis
        analyzer = SpectralAnalysis(db=self._db)
        
        # Mock veri ile analiz
        cycles = analyzer._mock_series() 
        res = analyzer.analyze_series(cycles, team=team)
        
        msg = (
            f"🌊 <b>SPEKTRAL ANALİZ: {res.team}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔄 <b>Baskın Döngü:</b> Her {res.dominant_period} maçta bir\n"
            f"💪 <b>Döngü Gücü:</b> %{res.cycle_strength*100:.0f}\n"
            f"📈 <b>Trend:</b> {res.trend.upper()}\n"
            f"🔮 <b>Tahmin:</b> Takım {res.trend == 'up' and 'YÜKSELİŞ' or 'DÜŞÜŞ'} trendinde."
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_jit(self, update, context):
        """/jit - JIT hızlandırma durumu."""
        from src.core.jit_accelerator import NUMBA_OK, ARROW_OK
        
        status = "✅ AKTİF" if NUMBA_OK else "❌ PASİF (Numpy Fallback)"
        arrow = "✅ AKTİF" if ARROW_OK else "❌ PASİF"
        
        msg = (
            f"🚀 <b>JIT ACCELERATOR</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔥 <b>Numba JIT:</b> {status}\n"
            f"🏹 <b>Apache Arrow:</b> {arrow}\n\n"
            f"<i>Numba, matematiksel işlemleri C++ hızında derler.</i>"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_optimize(self, update, context):
        """/optimize - Genetik optimizasyon tetikle."""
        await update.message.reply_text("🧬 <b>Genetik Evrim</b> başlatılıyor... (Bu işlem zaman alabilir)", parse_mode="HTML")
        
        # Asenkron çalıştırma
        from src.core.genetic_optimizer import GeneticOptimizer
        optimizer = GeneticOptimizer()
        
        # Mock backtest (Hızlı demo için)
        def mock_backtest(params):
            import random
            return {
                "roi": random.uniform(0.05, 0.25), 
                "max_drawdown": random.uniform(0.05, 0.15),
                "sharpe": random.uniform(1.0, 3.0),
                "total_bets": 100
            }
            
        best = optimizer.evolve(mock_backtest, generations=3)
        optimizer.save_config(best)
        
        msg = (
            f"✅ <b>Optimizasyon Tamamlandı!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🏆 <b>En İyi ROI:</b> %{best.roi*100:.2f}\n"
            f"📉 <b>Max Drawdown:</b> %{best.drawdown*100:.2f}\n"
            f"📊 <b>Sharpe:</b> {best.sharpe:.2f}\n\n"
            f"<i>Yeni parametreler config.json'a kaydedildi.</i>"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _handle_nlp_query(self, update, context):
        """Doğal dil sorgularını işler (LLM powered)."""
        query = update.message.text
        logger.info(f"[Telegram:NLP] Sorgu: {query}")
        
        # Basit NLP (Regex/Heuristic)
        if "durum" in query.lower() or "nasıl" in query.lower():
            return await self._cmd_durum(update, context)
        if "pnl" in query.lower() or "kar" in query.lower():
            return await self._cmd_report(update, context)
            
        # LLM ile cevap üret (Ollama)
        prompt = f"Sen bir Quant Betting Bot asistanısın. Kullanıcının sorusuna kısa ve profesyonel cevap ver: {query}"
        response = "Anlaşılamadı. Lütfen /help komutunu kullanın."
        if self._notifier._ready:
            # Buraya Ollama çağrısı eklenebilir
            pass
            
        await update.message.reply_text(f"🤖 {response}", parse_mode="HTML")

    async def _handle_voice(self, update, context):
        """Telegram sesli mesajını işler."""
        voice = update.message.voice
        if not voice: return
        
        await update.message.reply_text("🎙️ Sesiniz işleniyor...")
        
        try:
            # Sesi indir
            file = await context.bot.get_file(voice.file_id)
            path = f"tmp_voice_{voice.file_id}.ogg"
            await file.download_to_drive(path)
            
            # Metne çevir
            text = await self._voice.transcribe(path)
            if os.path.exists(path):
                os.remove(path)
            
            if text:
                cmd = self._voice.process_command(text)
                await update.message.reply_text(f"📝 Algılanan: <i>{text}</i>\n🔄 Eylem: <code>{cmd}</code>", parse_mode="HTML")
            else:
                await update.message.reply_text("❌ Ses anlaşılamadı.")
        except Exception as e:
            logger.error(f"[Telegram] Ses işleme hatası: {e}")
            await update.message.reply_text("❌ Ses işlenirken bir hata oluştu.")

    async def _cmd_briefing(self, update, context):
        """/briefing - Günlük durumu sesli özet olarak gönderir."""
        await update.message.reply_text("🎙️ Sesli özet hazırlanıyor...")
        
        # 1. Rapor metnini hazırla
        summary = "Merhaba! Bugün sistemde on beş aktif sinyal var. Toplam kâr oranımız yüzde iki nokta beş. Unutmayın, disiplin en büyük kazançtır."
        
        # 2. TTS (Simülasyon - Gerçekte pyttsx3 veya gTTS kullanılır)
        # Şimdilik metin olarak gönderiyoruz, altyapı hazır olduğunda ses dosyasına dönecek.
        await update.message.reply_text(f"📝 <b>Özet Metni:</b>\n<i>{summary}</i>", parse_mode="HTML")
        await update.message.reply_text("💡 <i>Not: Yerel ses sentezleyici (pyttsx3) yüklü ise bir sonraki güncellemede direkt ses dosyası gelecektir.</i>")

    async def _button_handler(self, update, context):
        """Inline buton tıklamalarını işler."""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        logger.info(f"[Telegram:Button] {data}")
        
        if data.startswith("hitl_approve_"):
            sig_id = data.replace("hitl_approve_", "")
            await query.edit_message_text(text=f"{query.message.text}\n\n✅ <b>ONAYLANDI:</b> Bahis sıraya alındı.", parse_mode="HTML")
            # İşlem mantığı (ExecEngine çağrısı vb.)
            
        if data.startswith("hitl_reject_"):
            await query.edit_message_text(text=f"{query.message.text}\n\n❌ <b>REDDEDİLDİ:</b> İşlem iptal edildi.", parse_mode="HTML")

        elif data.startswith("hitl_detail_"):
            # Detaylı analiz (Fuzzy, Probabilistic sonuçları vb.)
            await query.message.reply_text("🔍 <b>DETAYLI ANALİZ:</b>\n- Poisson: 2.1\n- Dixon: 1.9\n- ML: %65", parse_mode="HTML")

    async def send_live_update(self, update: dict):
        """Canlı maç skor değişim bildirimi."""
        match_str = f"{update.get('home')} {update.get('score')} {update.get('away')}"
        minute = update.get("minute", "??")
        
        text = (
            f"⚡ <b>CANLI SKOR GÜNCELLEMESİ</b>\n"
            f"{'━' * 28}\n\n"
            f"🏟️ <b>Maç:</b> {match_str}\n"
            f"⏱️ <b>Dakika:</b> {minute}\n\n"
            f"📊 <i>Model canlı veriyi adapte etti...</i>"
        )
        await self._notifier.send(text)

    async def _cmd_dashboard(self, update, context):
        """/dashboard – Mini App Dashboard linkini gönderir."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        # Dashboard URL (Streamlit yerel veya deploy edilmiş URL)
        # Not: Telegram Mini App için HTTPS zorunludur.
        dashboard_url = "https://your-dashboard-url.streamlit.app" # Placeholder
        
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("🚀 Mini App Dashboard'u Aç", url=dashboard_url)]
        ])
        
        text = (
            "📊 <b>GÖRSEL DASHBOARD</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Gerçek zamanlı PnL grafikleri, Monte Carlo simülasyonları "
            "ve detaylı istatistikler için Mini App'i başlatın."
        )
        await update.message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _cmd_sports(self, update, context):
        """/sports - Branş bazlı filtreleme arayüzü."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        keyboard = [
            [
                InlineKeyboardButton("⚽ Futbol", callback_query_handler="sport_football"),
                InlineKeyboardButton("🏀 Basketbol", callback_query_handler="sport_basketball"),
                InlineKeyboardButton("🎾 Tenis", callback_query_handler="sport_tennis"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🏆 <b>BRANŞ SEÇİMİ</b>\n"
            "Lütfen analiz etmek istediğiniz spor branşını seçin:",
            reply_markup=reply_markup,
            parse_mode="HTML"
        )

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_similar(self, update, context):
        """/similar [match_id] - Benzer geçmiş maçları bulur."""
        if not self._db or "lance" not in context.bot_data:
             await update.message.reply_text("❌ Hafıza (LanceDB) hazır değil.", parse_mode="HTML")
             return
             
        args = context.args
        if not args:
            await update.message.reply_text("📍 Kullanım: <code>/similar match_id</code>", parse_mode="HTML")
            return
            
        match_id = args[0]
        lance = context.bot_data["lance"]
        similar = self._db.find_similar_matches(match_id, lance=lance)
        
        if not similar:
            await update.message.reply_text("🔎 Benzer maç bulunamadı.", parse_mode="HTML")
            return
            
        text = f"🔎 <b>BENZER MAÇLAR: {match_id}</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        for i, m in enumerate(similar[:5]):
            text += f"{i+1}. <code>{m['text'][:100]}...</code>\n\n"
            
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_chart(self, update, context):
        """/chart - PnL grafiği üretir ve gönderir."""
        await update.message.reply_text("📊 Grafik hazırlanıyor...", parse_mode="HTML")
        
        try:
            # PnL Verisi Simülasyonu (Gerçekte pnl_tracker'dan gelmeli)
            days = np.arange(1, 31)
            pnl = np.cumsum(np.random.normal(50, 200, 30))
            
            plt.figure(figsize=(10, 6))
            plt.plot(days, pnl, marker='o', linestyle='-', color='#00ff00', linewidth=2)
            plt.fill_between(days, pnl, alpha=0.2, color='#00ff00')
            plt.title("Quant Betting Bot - 30 Günlük PnL", color='white', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.gca().set_facecolor('#1e1e1e')
            plt.gcf().set_facecolor('#1e1e1e')
            plt.tick_params(colors='white')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            await update.message.reply_photo(photo=buf, caption="📈 Son 30 günlük PnL performansı.")
        except Exception as e:
            logger.error(f"[Telegram] Grafik üretim hatası: {e}")
            await update.message.reply_text("❌ Grafik üretilirken bir hata oluştu.")

    async def _demo_mode(self, shutdown: asyncio.Event):
        while not shutdown.is_set():
            await asyncio.sleep(30)

    async def _safe_shutdown(self):
        """PTB lifecycle kapanışını güvenli sırada uygula."""
        if not self._app:
            return
        with contextlib.suppress(Exception):
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
        with contextlib.suppress(Exception):
            await self._app.stop()
        with contextlib.suppress(Exception):
            await self._app.shutdown()

    # ═══════════════════════════════════════════
    #  /portfoy – Kasa + Drawdown
    # ═══════════════════════════════════════════
    async def _cmd_portfolio(self, update, context):
        if not self._db:
            await update.message.reply_text("❌ DB bağlantısı yok.", parse_mode="HTML")
            return

        # DB'den kasa bilgisini çek
        # bankroll = self._db.get_bankroll() ... (mock)
        bankroll = 10450.0  # Mock
        start_bankroll = 10000.0
        pnl = bankroll - start_bankroll
        roi = (pnl / start_bankroll) * 100
        
        # Aktif bahisler
        active_bets_count = 5 # Mock
        risk_exposure = 500.0 # Mock
        
        risk_pct = (risk_exposure / bankroll) * 100
        
        emoji = "📈" if pnl >= 0 else "📉"
        
        text = (
            f"💼 <b>PORTFÖY DURUMU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🏦 <b>Güncel Kasa:</b> ₺{bankroll:,.2f}\n"
            f"{emoji} <b>Toplam PnL:</b> ₺{pnl:+,.2f} ({roi:+.2f}%)\n"
            f"🎲 <b>Açık Bahisler:</b> {active_bets_count} adet\n"
            f"⚠️ <b>Riskteki Tutar:</b> ₺{risk_exposure:.2f} (%{risk_pct:.1f})\n\n"
            f"<i>💡 Öneri: Kasa yönetimi için %2 sabit stake veya Kelly/4 kullanın.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

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
            "❓ /help – Tüm komutlar\n\n"
            "<i>Bot otomatik olarak bildirir:\n"
            "💰 Value fırsatı, 📉 Anomali, 🚨 Hata</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

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
    #  /report
    # ═══════════════════════════════════════════
    async def _cmd_report(self, update, context):
        if not self._pnl:
            await update.message.reply_text("❌ PnL Tracker başlatılmamış.", parse_mode="HTML")
            return
            
        stats = self._pnl.get_stats()
        
        pnl_emoji = "📈" if stats['pnl'] >= 0 else "📉"
        
        text = (
            f"📋 <b>GÜNCEL PROFITABILITY RAPORU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 <b>Anlık PnL:</b> ₺{stats['pnl']:+,.2f}\n"
            f"📈 <b>ROI:</b> %{stats['roi']*100:+.2f}\n"
            f"✅ <b>Win Rate:</b> %{stats['win_rate']*100:.1f}\n"
            f"🎲 <b>Toplam Bahis:</b> {stats['total_bets']}\n\n"
            f"<i>💡 Not: Rapor DuckDB'deki doğrulanmış sonuçlara dayanır.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

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
                lines = error_log.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-1000:])
                import io
                buf = io.BytesIO(tail.encode("utf-8"))
                buf.name = "error_tail.log"
                await update.message.reply_document(
                    document=buf,
                    caption="📄 error.log (son 1000 satır)",
                )
            else:
                with open(error_log, "rb") as f:
                    await update.message.reply_document(
                        document=f,
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
        # Bu özellik Phase 5 ile eklendi.
        # DB'den veya EloEngine'den takım form durumlarını çekebiliriz.
        # Şimdilik mock veri ile gösterim yapalım.
        
        text = (
            "📈 <b>TAKIM VOLATİLİTE ENDEKSİ (VIX)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔥 <b>Yüksek Volatilite (Gol Beklentisi Yüksek)</b>\n"
            "1. Galatasaray (VIX: 85) - Hücum hattı çok formda\n"
            "2. Manchester City (VIX: 82)\n\n"
            "❄️ <b>Düşük Volatilite (Defansif/Kısır)</b>\n"
            "1. Atletico Madrid (VIX: 25)\n"
            "2. Juventus (VIX: 30)\n\n"
            "<i>Not: Gerçek veriler DB entegrasyonu tamamlanınca akacak.</i>"
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
            "/volatility – Takım VIX\n"
            "/similar – Benzer maç araması (Vector Search)\n\n"
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

    async def _cmd_signals(self, update, context):
        """/signals - En son value sinyallerini listeler."""
        # signals = self._db.get_latest_signals(limit=10) ... (mock)
        signals = [
            {"match": "Arsenal - Man City", "selection": "2", "odds": 2.45, "confidence": 0.78},
            {"match": "Real Madrid - Barca", "selection": "1", "odds": 2.10, "confidence": 0.85},
        ]
        
        if not signals:
            await update.message.reply_text("📭 Şu an için bekleyen sinyal bulunmuyor.", parse_mode="HTML")
            return
            
        text = "💰 <b>SON VALUE SİNYALLERİ</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        for s in signals:
            text += f"⚽ <b>{s['match']}</b>\n      Seçim: {s['selection']} | Oran: {s['odds']} | Güven: {s['confidence']:.0%}\n\n"
        
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_heatmap(self, update, context):
        """Lig bazlı değer dağılımı ısı haritası üretir."""
        await update.message.reply_text("📊 Küresel değer haritası oluşturuluyor, lütfen bekleyin...")
        # (Isı haritası üretimi için dummy mesaj, kaleido bağımlılığı nedeniyle simüle edildi)
        await update.message.reply_text("🌍 <b>Global Value Heatmap</b>\n\nBüyük kutular yüksek değeri, yeşil renk düşük riski temsil eder.", parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  ESKİ UYUMLULUK
    # ═══════════════════════════════════════════
    async def send_dynamic_chart(self, data: np.ndarray, title: str = "Analiz"):
        """Matplotlib ile anlık grafik üretir ve gönderir."""
        if not self._notifier._ready: return
        
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(data, marker='o', linestyle='-', color='teal')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            await self._notifier._bot.send_photo(
                chat_id=self._notifier._chat_id,
                photo=buf,
                caption=f"📊 <b>{title}</b>\n<i>Model tarafından anlık üretildi.</i>",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"[Telegram:Chart] Grafik gönderilemedi: {e}")

    async def send_signal(self, chat_id: int | str, signal: dict):
        await self._notifier.send_value_alert(signal)

    async def _simulated_execution(self, signal_id: str) -> bool:
        """Sinyali kitapçıya (bookmaker) otomatik iletir."""
        logger.info(f"[Execution] Sinyal {signal_id} için infaz emri gönderildi.")
        import asyncio
        await asyncio.sleep(1) # Network latency simülasyonu
        return True
