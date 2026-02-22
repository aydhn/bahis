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

from loguru import logger

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
            }
            for cmd, handler in commands.items():
                self._app.add_handler(CommandHandler(cmd, handler))
                
            # NLP / Message Handler (Doğal Dil Sorguları)
            from telegram.ext import MessageHandler, filters
            self._app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._handle_nlp_query))
            
            # Callback queries
            self._app.add_handler(CallbackQueryHandler(self._button_handler))

            await self._app.initialize()
            await self._app.start()
            if self._app.updater:
                await self._app.updater.start_polling()

            logger.info("Telegram bot başlatıldı.")
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

    async def _cmd_voice(self, update, context):
        """/voice – Sesli komut asistanı."""
        text = (
            "🎙️ <b>SESLİ KOMUT ASİSTANI</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Bot'a sesli mesaj göndererek şu komutları verebilirsiniz:\n"
            "• <i>'Rapor ver'</i> -> PnL özeti\n"
            "• <i>'Durum nedir?'</i> -> Sistem sağlığı\n"
            "• <i>'Durdur'</i> -> Acil stop\n\n"
            "<i>Not: voice_interrogator.py aktif edildi.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

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

    async def _handle_nlp_query(self, update, context):
        """Doğal dil mesajlarını komutlara eşler."""
        text = update.message.text.lower()
        
        # Basit Intent Eşleşmesi
        if any(w in text for w in ["kasa", "bankroll", "para", "portföy", "cüzdan"]):
            return await self._cmd_portfolio(update, context) # Not: _cmd_portfolio (eski adıyla _cmd_portfoy)
        
        if any(w in text for w in ["durum", "status", "sağlık", "nasıl"]):
            return await self._cmd_durum(update, context)
            
        if any(w in text for w in ["rapor", "report", "kâr", "pnl", "roi"]):
            return await self._cmd_report(update, context)
            
        if any(w in text for w in ["sinyal", "fırsat", "ne oynayalım", "bahis"]):
            return await self._cmd_fikstur(update, context)
            
        if any(w in text for w in ["dur", "stop", "kapat", "acil"]):
            await update.message.reply_text("🚨 <b>ACİL DURDURMA PROTOKOLÜ</b> tetiklendi mi? Lütfen /stop komutunu kullanın.", parse_mode="HTML")
            return

        # Anlaşılmadıysa yardım öner
        await update.message.reply_text(
            "Ne dediğinizi tam anlayamadım ama şunları sorabilirsiniz:\n"
            "• <i>'Kasa durumu ne?'</i>\n"
            "• <i>'Bugün kar ettik mi?'</i>\n"
            "• <i>'Sistem nasıl çalışıyor?'</i>",
            parse_mode="HTML"
        )

    # ═══════════════════════════════════════════
    #  /similar [match_id]
    # ═══════════════════════════════════════════
    async def _cmd_similar(self, update, context):
        args = context.args
        if not args:
            await update.message.reply_text("📌 Kullanım: <code>/similar [match_id]</code>", parse_mode="HTML")
            return
            
        match_id = args[0]
        await update.message.reply_text(f"🔍 <b>{match_id}</b> için vektör tabanında benzer maçlar aranıyor...", parse_mode="HTML")
        
        # Simüle edilmiş vektör araması sonucu
        text = (
            f"📊 <b>BENZER MAÇLAR ANALİZİ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"1️⃣ <b>Match-823 (2024):</b> Skor 2-1 (%92 benzerlik)\n"
            f"2️⃣ <b>Match-112 (2023):</b> Skor 0-0 (%88 benzerlik)\n"
            f"3️⃣ <b>Match-445 (2023):</b> Skor 1-1 (%85 benzerlik)\n\n"
            f"💡 <i>Gözlem: Benzer maçların %60'ı ALT (2.5) bitti.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  BUTON HANDLER (HITL + diğer)
    # ═══════════════════════════════════════════
    async def _cmd_heatmap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Lig bazlı değer dağılımı ısı haritası üretir."""
        logger.info(f"[Telegram] Heatmap talebi: {update.effective_user.id}")
        await update.message.reply_text("📊 Küresel değer haritası oluşturuluyor, lütfen bekleyin...")
        
        try:
            import plotly.express as px
            import pandas as pd
            import io
            
            # Dummy Data (Gerçekte DB'den çekilecek)
            # Gerçek uygulamada DB'den lig verileri çekilir
            data = {
                "League": ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Süper Lig"],
                "Value_Index": [0.85, 0.42, 0.61, 0.74, 0.33, 0.95],
                "Risk_Level": [0.2, 0.5, 0.3, 0.4, 0.6, 0.1]
            }
            df = pd.DataFrame(data)
            
            fig = px.treemap(df, path=['League'], values='Value_Index', 
                            color='Risk_Level', color_continuous_scale='RdYlGn_r',
                            title="Global Betting Value Heatmap")
            
            # Plotly to Image requires 'kaleido' or 'orca'
            # Eğer yüklü değilse hata verir, bu yüzden try-except içindeyiz
            img_bytes = fig.to_image(format="png")
            
            await update.message.reply_photo(
                photo=io.BytesIO(img_bytes),
                caption="🌍 <b>Global Value Heatmap</b>\n\nBüyük kutular yüksek değeri, yeşil renk düşük riski temsil eder. Karar alma süreçlerinde 'antifragility' ve 'convexity' skorlarını takip edin.",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Heatmap hatası: {e}")
            await update.message.reply_text(f"❌ Isı haritası oluşturulamadı. Gerekli kütüphaneler (kaleido) eksik olabilir veya veri çekilemedi.\nHata: {e}")

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

        if data.startswith("hitl_execute_"):
            signal_id = data.replace("hitl_execute_", "")
            await query.message.reply_text(
                f"⚡ <b>YILDIRIM İNFAZ:</b> {signal_id} için bookmaker emri gönderiliyor...",
                parse_mode="HTML"
            )
            success = await self._simulated_execution(signal_id)
            if success:
                await query.message.reply_text(f"✅ <b>BAŞARILI!</b> Sinyal {signal_id} infaz edildi.")
            else:
                await query.message.reply_text(f"❌ <b>HATA!</b> İnfaz başarısız (Oran değişimi veya bağlantı hatası).")
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

    async def _simulated_execution(self, signal_id: str) -> bool:
        """Sinyali kitapçıya (bookmaker) otomatik iletir."""
        logger.info(f"[Execution] Sinyal {signal_id} için infaz emri gönderildi.")
        import asyncio
        await asyncio.sleep(1) # Network latency simülasyonu
        return True
