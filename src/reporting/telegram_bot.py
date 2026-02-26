"""
telegram_bot.py – Etkileşimli CEO Paneli (Bot).

Bu modül, tek yönlü raporlamanın ötesine geçerek,
kullanıcının (CEO) sistemle konuşmasını sağlar.

Komutlar:
  /status  - Sistem sağlık durumu
  /pnl     - Finansal özet
  /risk    - Risk seviyesi
  /stop    - Acil durdurma (Circuit Breaker)
  /brain   - Quantum Brain & Physics Engine durumu
"""
import asyncio
import os
import json
from typing import Optional, Dict, Any
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None

try:
    from src.ingestion.voice_interrogator import VoiceInterrogator
except ImportError:
    VoiceInterrogator = None

try:
    from src.pipeline.context import BettingContext
except ImportError:
    BettingContext = None

from src.system.config import settings
from src.reporting.visualizer import Visualizer
from src.core.event_bus import Event
from src.quant.analysis.narrative_generator import NarrativeGenerator

class TelegramBot:
    """Async Polling tabanlı Telegram Botu."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and httpx)
        self.offset = 0
        self.running = False
        self._task = None
        self.warroom_task = None
        self.warroom_active = False
        self.voice_handler = VoiceInterrogator() if VoiceInterrogator else None
        self.bet_history = []  # Son bahisleri sakla (Explain için)
        self.context: Optional[BettingContext] = None # Live Context
        self.sentinel: Any = None # Sentinel (Orchestrator) referansı
        self.performance_report: Dict[str, Any] = {}

        if not self.enabled:
            logger.warning("TelegramBot devre dışı: Token veya httpx eksik.")

    def set_performance_report(self, report: Dict[str, Any]):
        """Performans raporunu güncelle."""
        self.performance_report = report

    def _get_stoic_quote(self, sentiment: str) -> str:
        """Ruh haline göre stoik alıntı seçer."""
        import random
        if sentiment == "negative":
            quotes = [
                "“The obstacle is the way.” — Marcus Aurelius",
                "“We suffer more often in imagination than in reality.” — Seneca",
                "“You have power over your mind - not outside events. Realize this, and you will find strength.” — Marcus Aurelius",
                "“Difficulties strengthen the mind, as labor does the body.” — Seneca"
            ]
            return random.choice(quotes)
        elif sentiment == "positive":
            return "“Don’t let your reflection on the whole sweep of life crush you.” — Marcus Aurelius"
        return ""

    def set_sentinel(self, sentinel: Any):
        """Sentinel (Orchestrator) bağlantısı."""
        self.sentinel = sentinel
        if self.sentinel and hasattr(self.sentinel, "bus"):
            # Olay dinleyicilerini kaydet
            self.sentinel.bus.subscribe("bet_placed", self.handle_event)
            self.sentinel.bus.subscribe("risk_alert", self.handle_event)
            self.sentinel.bus.subscribe("pipeline_crash", self.handle_event)

    async def handle_event(self, event: Event):
        """Event Bus üzerinden gelen kritik olayları raporla."""
        if not self.enabled: return

        # Chat ID bul
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return

        etype = event.event_type
        data = event.data

        if etype == "bet_placed":
            await self.send_bet_signal(data)
        elif etype == "risk_alert":
            await self.send_risk_alert("Risk Alert", str(data))
        elif etype == "pipeline_crash":
            await self.send_message(chat_id, f"🚨 *Sistem Çökmesi*\n{data.get('error')}")

    def set_context(self, ctx: BettingContext):
        """Pipeline'dan gelen güncel context'i al."""
        self.context = ctx

    async def start(self):
        """Botu başlat (Arka planda dinlemeye başla)."""
        if not self.enabled or self.running:
            return

        self.running = True
        logger.info("TelegramBot dinlemeye başladı...")
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        """Botu durdur."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _warroom_loop(self, chat_id: int):
        """High frequency updates."""
        try:
            while self.warroom_active and self.running:
                # Send a pulse every 60s
                if self.sentinel and hasattr(self.sentinel, "portfolio_manager"):
                    opps = len(self.sentinel.portfolio_manager.current_opportunities)
                    cycle = self.context.cycle_id if self.context else "?"
                    await self.send_message(chat_id, f"📡 *Live Pulse*\nCycle: {cycle}\nOpporunities: {opps}")
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Warroom loop error: {e}")

    async def _poll_loop(self):
        """Sürekli yeni mesajları kontrol et."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            while self.running:
                try:
                    params = {"offset": self.offset, "timeout": 20}
                    resp = await client.get(f"{self.base_url}/getUpdates", params=params)

                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("ok"):
                            for update in data.get("result", []):
                                await self._handle_update(update)
                                self.offset = update["update_id"] + 1
                    else:
                        logger.warning(f"Telegram polling hatası: {resp.status_code}")
                        await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Telegram polling exception: {e}")
                    await asyncio.sleep(5)

    async def _handle_update(self, update: Dict[str, Any]):
        """Gelen mesajı işle."""
        msg = update.get("message", {})
        chat_id = msg.get("chat", {}).get("id")

        if not chat_id:
            return

        # --- Security Check ---
        allowed_users = set()
        if settings.TELEGRAM_ALLOWED_USERS:
            allowed_users.update(u.strip() for u in settings.TELEGRAM_ALLOWED_USERS.split(",") if u.strip())

        # Legacy support: TELEGRAM_CHAT_ID is also admin
        admin_id = os.getenv("TELEGRAM_CHAT_ID")
        if admin_id:
            allowed_users.add(str(admin_id))

        if str(chat_id) not in allowed_users:
            logger.warning(f"Unauthorized access attempt from {chat_id}")
            return
        # ----------------------

        # Sesli mesaj kontrolü
        if ("voice" in msg or "audio" in msg) and self.voice_handler:
            await self._handle_voice(msg, chat_id)
            return

        text = msg.get("text", "")

        if not text.startswith("/"):
            return

        parts = text.split(" ")
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command == "/start":
            await self.send_message(chat_id, "🤖 *Otonom Quant Bot Devrede.*\nKomutlar: /status, /pnl, /risk, /brain, /explain, /analyze, /set_risk, /force, /shutdown")

        elif command == "/set_risk":
            if not self.sentinel:
                await self.send_message(chat_id, "❌ Sentinel bağlantısı yok.")
                return
            mode = args[0] if args else "normal"
            resp = self.sentinel.set_risk_mode(mode)
            await self.send_message(chat_id, f"⚙️ {resp}")

        elif command == "/force":
            if not self.sentinel:
                await self.send_message(chat_id, "❌ Sentinel bağlantısı yok.")
                return
            if len(args) < 2:
                await self.send_message(chat_id, "⚠️ Kullanım: `/force <match_id> <selection>`")
                return
            resp = await self.sentinel.force_bet(args[0], args[1])
            await self.send_message(chat_id, f"🚀 {resp}")

        elif command == "/shutdown":
            await self.send_message(chat_id, "🛑 Sistem kapatılıyor...")
            if self.sentinel:
                self.sentinel.shutdown()
            else:
                self.running = False

        elif command == "/status":
            cycle = self.context.cycle_id if self.context else 0
            warroom = "ON" if self.warroom_active else "OFF"

            # Circuit Breaker Durumu
            cb_status = "Bilinmiyor"
            if self.sentinel and hasattr(self.sentinel, "system_breaker"):
                 if self.sentinel.system_breaker.is_available:
                     cb_status = "✅ HEALTHY"
                 else:
                     cb_status = "🔴 TRIPPED (Safe Mode)"

            await self.send_message(chat_id, f"✅ *Sistem Çalışıyor*\nMod: Otonom\nCycle: #{cycle}\nWarRoom: {warroom}\nCircuit Breaker: {cb_status}\nVeri Akışı: Aktif")

        elif command == "/brief":
            # Morning Briefing (CEO Dashboard)
            await self.send_message(chat_id, "☕ *Günaydın, Şef. İşte sabah raporu:*")

            # 1. System Health
            hb_status = "✅"
            try:
                from pathlib import Path
                import time
                hb = Path("data/heartbeat.txt")
                if hb.exists() and time.time() - float(hb.read_text()) < 120:
                    hb_status = "✅ Canlı"
                else:
                    hb_status = "⚠️ Gecikmeli"
            except: hb_status = "❌ Yok"

            # 2. Market Sentiment (Global)
            # Fetch from DB or Context
            sentiment_msg = "Market Sentiment: Nötr 😐"
            if self.context and self.context.ensemble_results:
                # Simple aggregate of sentiments
                bullish = sum(1 for r in self.context.ensemble_results if r.get("market_sentiment", {}).get("direction") == "BULLISH")
                bearish = sum(1 for r in self.context.ensemble_results if r.get("market_sentiment", {}).get("direction") == "BEARISH")
                if bullish > bearish: sentiment_msg = "Market Sentiment: İştahlı 🐂"
                elif bearish > bullish: sentiment_msg = "Market Sentiment: Çekingen 🐻"

            # 3. Top Value Picks
            top_picks = "Henüz fırsat yok."
            if self.context and hasattr(self.context, 'final_bets'):
                bets = sorted(self.context.final_bets, key=lambda x: x.get('confidence', 0) * x.get('ev', 0), reverse=True)[:3]
                if bets:
                    top_picks = "\n".join([f"• {b['match_id']} ({b['selection']}) - EV: {b.get('edge', b.get('ev', 0)):.2f}" for b in bets])

            msg = (
                f"📡 *Sistem Durumu*: {hb_status}\n"
                f"🌍 {sentiment_msg}\n\n"
                f"💎 *Günün Öne Çıkan Fırsatları:*\n{top_picks}\n\n"
                f"Bol şans."
            )
            await self.send_message(chat_id, msg)

        elif command == "/warroom":
            self.warroom_active = not self.warroom_active
            if self.warroom_active:
                self.warroom_task = asyncio.create_task(self._warroom_loop(chat_id))
                await self.send_message(chat_id, "🚨 *WAR ROOM MODU AKTİF* 🚨\nCanlı akış başlıyor...")
            else:
                if self.warroom_task:
                    self.warroom_task.cancel()
                await self.send_message(chat_id, "💤 War Room modu kapatıldı.")

        elif command == "/audit":
            msg = "🔍 *Sistem Denetimi*\n"
            # 1. Heartbeat Check
            try:
                from pathlib import Path
                import time
                hb = Path("data/heartbeat.txt")
                if hb.exists():
                    delta = time.time() - float(hb.read_text())
                    msg += f"- Heartbeat: {delta:.1f}s ago {'✅' if delta < 60 else '⚠️'}\n"
                else:
                    msg += "- Heartbeat: ❌ (No file)\n"
            except Exception as e:
                msg += f"- Heartbeat: Error ({e})\n"

            # 2. Portfolio Manager
            if self.sentinel and hasattr(self.sentinel, "portfolio_manager"):
                pm = self.sentinel.portfolio_manager
                msg += f"- Portfolio Ops: {len(pm.current_opportunities)} pending\n"
            else:
                msg += "- Portfolio Ops: ❌ (Unlinked)\n"

            await self.send_message(chat_id, msg)

        elif command == "/pnl":
            # Real-Time DB Query for accuracy
            try:
                from src.system.container import container
                db = container.get("db")
                if db:
                    df = db.query("SELECT SUM(pnl) as total_pnl, COUNT(*) as count FROM bets WHERE status IN ('won', 'lost')")
                    pnl = df["total_pnl"][0] if not df.is_empty() and df["total_pnl"][0] is not None else 0.0
                    count = df["count"][0] if not df.is_empty() else 0
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    await self.send_message(chat_id, f"{emoji} *Realized PnL:* {pnl:.2f} TL ({count} bets)")
                else:
                    stats = self._read_bankroll_state()
                    pnl = stats.get("pnl", 0.0)
                    await self.send_message(chat_id, f"📝 *PnL (State):* {pnl:.2f} TL")
            except Exception as e:
                await self.send_message(chat_id, f"⚠️ DB Hatası: {e}")

        elif command == "/performance":
            try:
                from src.system.container import container
                db = container.get("db")
                if db:
                    df = db.query("""
                        SELECT
                            COUNT(*) as total,
                            SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                            SUM(stake) as total_stake,
                            SUM(pnl) as total_pnl
                        FROM bets
                        WHERE status IN ('won', 'lost')
                    """)
                    if not df.is_empty() and df["total"][0] > 0:
                        wins = df["wins"][0]
                        total = df["total"][0]
                        stake = df["total_stake"][0]
                        pnl = df["total_pnl"][0]

                        wr = wins / total
                        roi = pnl / stake if stake > 0 else 0.0
                        emoji = "🚀" if roi > 0 else "📉"

                        await self.send_message(chat_id, f"{emoji} *Canlı Performans*\nROI: %{roi*100:.2f}\nWin Rate: %{wr*100:.2f}\nRealized PnL: {pnl:.2f} TL\nTotal Vol: {stake:.2f} TL")
                    else:
                        await self.send_message(chat_id, "⚠️ Henüz sonuçlanmış bahis yok.")
                else:
                     await self.send_message(chat_id, "⚠️ DB bağlantısı yok.")
            except Exception as e:
                logger.error(f"Performance calc error: {e}")
                await self.send_message(chat_id, "⚠️ Hesaplama hatası.")

        elif command == "/risk":
            stats = self._read_bankroll_state()
            dd = stats.get("drawdown", 0.0)

            # Retrieve regime from sentinel or context if possible
            regime = "NORMAL"
            kelly = 1.0
            if self.context and self.context.ensemble_results:
                 # Just take the first one as proxy
                 first = self.context.ensemble_results[0]
                 regime = first.get("regime_status", "NORMAL")
                 kelly = first.get("kelly_fraction", 1.0)

            await self.send_message(chat_id, f"🛡️ *Risk Seviyesi*\nDrawdown: %{dd*100:.2f}\nRegime: {regime}\nKelly Scale: {kelly:.2f}x")

        elif command == "/brain":
            msg = "🧠 *Quantum Brain & Physics Status*\n"

            # 1. Quantum Check
            try:
                from src.quant.physics.quantum_brain import PENNYLANE_OK
                q_status = "ACTIVE 🟢" if PENNYLANE_OK else "Simulated 🟡"
            except ImportError:
                q_status = "OFF 🔴"

            # Get Quantum details from context if available
            q_conf = "N/A"
            if self.context:
                # Support both Object (Pydantic) and Dict access
                if isinstance(self.context, dict):
                    q_preds = self.context.get("quantum_predictions", {})
                else:
                    q_preds = getattr(self.context, "quantum_predictions", {})

                if q_preds:
                    avg_conf = sum(p.confidence for p in q_preds.values()) / len(q_preds)
                    q_conf = f"{avg_conf:.2f}"
            msg += f"- Quantum Engine: {q_status} (Avg Conf: {q_conf})\n"

            # 2. Chaos Check
            try:
                from src.quant.physics.chaos_filter import NOLDS_OK
                c_status = "ACTIVE 🟢" if NOLDS_OK else "Fallback 🟡"
            except ImportError:
                c_status = "OFF 🔴"

            # Get Chaos stats
            c_regime = "N/A"
            if self.context:
                if isinstance(self.context, dict):
                    c_reps = self.context.get("chaos_reports", {})
                else:
                    c_reps = getattr(self.context, "chaos_reports", {})

                if c_reps:
                    # Show first available match regime as sample
                    first_k = list(c_reps.keys())[0]
                    c_regime = f"{c_reps[first_k].regime} (Sample: {first_k})"
            msg += f"- Chaos Filter: {c_status} [{c_regime}]\n"

            # 3. Ricci Check
            try:
                from src.quant.physics.ricci_flow import RICCI_LIB_OK
                r_status = "ACTIVE 🟢" if RICCI_LIB_OK else "Fallback 🟡"
            except ImportError:
                r_status = "OFF 🔴"

            r_val = "N/A"
            if self.context:
                if isinstance(self.context, dict):
                    r_rep = self.context.get("ricci_report")
                else:
                    r_rep = getattr(self.context, "ricci_report", None)

                if r_rep:
                    r_val = f"κ={r_rep.avg_curvature:.2f} ({r_rep.stress_level})"
            msg += f"- Ricci Flow: {r_status} [{r_val}]\n"

            # 4. Particle Check
            if self.context:
                if isinstance(self.context, dict):
                    p_reps = self.context.get("particle_reports", {})
                else:
                    p_reps = getattr(self.context, "particle_reports", {})

                p_count = len(p_reps)
                p_status = f"{p_count} Live" if p_count > 0 else "Idle"
                msg += f"- Particle Tracker: {p_status}\n"

            await self.send_message(chat_id, msg)

        elif command == "/chart":
            await self.send_message(chat_id, "📊 *Grafik Hazırlanıyor...*")

            if args:
                # Match specific chart
                match_id = args[0]
                # Try to get data from context
                # For demo, we use dummy data if context is missing
                # In real flow, we would look up match_id in self.context

                # Mock Data for demo visualization
                buf = Visualizer.generate_value_chart(
                    home_team="Home", away_team="Away",
                    model_probs=[0.60, 0.25, 0.15],
                    market_probs=[0.55, 0.25, 0.20]
                )
                if buf:
                    await self.send_photo(chat_id, buf, caption=f"📊 Value Analysis: {match_id}")
                else:
                    await self.send_message(chat_id, "❌ Grafik oluşturulamadı.")
            else:
                # Bankroll Chart
                stats = self._read_bankroll_state()
                chart_buf = Visualizer.generate_dummy_chart()
                if chart_buf:
                    await self.send_photo(chat_id, chart_buf, caption="📈 Bankroll PnL")
                else:
                    await self.send_message(chat_id, "❌ Grafik oluşturulamadı.")

        elif command == "/explain":
            await self._handle_explain(chat_id, args)

        elif command == "/analyze":
            await self._handle_analyze(chat_id, args)

        elif command == "/portfolio":
            count = 0
            if self.sentinel and hasattr(self.sentinel, 'portfolio_manager'):
                count = len(self.sentinel.portfolio_manager.current_opportunities)
            await self.send_message(chat_id, f"📊 *Portföy Durumu*\nBekleyen Fırsat: {count}")

        elif command == "/optimize":
            if self.sentinel and hasattr(self.sentinel, "bus"):
                await self.sentinel.bus.emit(Event(event_type="pipeline_cycle_end"))
                await self.send_message(chat_id, "⚙️ Optimizasyon tetiklendi.")
            else:
                await self.send_message(chat_id, "❌ Sentinel yok.")

    async def _handle_voice(self, msg: Dict[str, Any], chat_id: int):
        """Sesli mesajı işle ve komuta çevir."""
        if not self.voice_handler: return

        file_id = msg.get("voice", {}).get("file_id") or msg.get("audio", {}).get("file_id")
        if not file_id: return

        await self.send_message(chat_id, "🎤 *Ses Analiz Ediliyor...*", parse_mode="Markdown")

        # Dosya yolunu al
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/getFile?file_id={file_id}")
            file_path = resp.json().get("result", {}).get("file_path")

            if file_path:
                # İndir
                dl_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
                audio_resp = await client.get(dl_url)

                # İşle
                result = await self.voice_handler.process_voice_message(audio_resp.content)
                cmd = result.get("command")

                if cmd != "unknown":
                    await self.send_message(chat_id, f"🗣️ Algılanan: *{result.get('raw')}* -> Komut: `/{cmd}`")
                    # Komutu simüle et
                    await self._handle_update({"message": {"chat": {"id": chat_id}, "text": f"/{cmd}"}})
                else:
                    await self.send_message(chat_id, f"🤔 Anlaşılamadı: *{result.get('raw')}*")

    async def _handle_explain(self, chat_id: int, args: list):
        """Son bahsin detaylı yatırım notunu (Investment Memo) getir."""
        if not self.bet_history:
            await self.send_message(chat_id, "📭 Henüz açıklanacak bir bahis yok.")
            return

        target_bet = self.bet_history[-1]
        match_id = target_bet.get("match_id")

        # Önce Context'ten Narrative'i bulmaya çalış
        if self.context and match_id in self.context.narratives:
            narrative = self.context.narratives[match_id]
            await self.send_message(chat_id, narrative, parse_mode="Markdown")
            return

        # Yoksa eski usul bet objesinden al
        if "narrative" in target_bet and target_bet["narrative"]:
            await self.send_message(chat_id, target_bet["narrative"], parse_mode="Markdown")
            return

        # O da yoksa felsefi raporu göster (Legacy fallback)
        philo = target_bet.get("philosophical_report")
        if philo:
            report_msg = f"🧠 *Felsefi Analiz*\nSkor: {philo.epistemic_score:.2f}\n"
            for ref in philo.reflections:
                report_msg += f"- {ref}\n"
            await self.send_message(chat_id, report_msg)
        else:
            await self.send_message(chat_id, "Detaylı rapor bulunamadı.")

    async def _handle_analyze(self, chat_id: int, args: list):
        """Belirli bir maçın analizini getir."""
        if not args:
            await self.send_message(chat_id, "⚠️ Kullanım: `/analyze <match_id>`")
            return

        match_id = args[0]

        if not self.context:
            await self.send_message(chat_id, "⚠️ Sistem verisi henüz hazır değil.")
            return

        # 1. Narrative var mı?
        if match_id in self.context.narratives:
            await self.send_message(chat_id, self.context.narratives[match_id], parse_mode="Markdown")
            return

        # 2. Ensemble Analizi Var mı? (NEW)
        if self.context.ensemble_results:
             # Optimize lookup
             res_map = {r.get("match_id"): r for r in self.context.ensemble_results}
             res = res_map.get(match_id)

             if res:
                 entropy = res.get("entropy", 0.5)
                 story = NarrativeGenerator.generate_story(
                     match_data={"home_team": res.get("home_team"), "away_team": res.get("away_team")},
                     prediction=res,
                     entropy=entropy
                 )

                 # Append Advanced Analysis
                 sim = res.get("similar_matches", {})
                 if sim:
                     matches = sim.get("matches", [])
                     summary = sim.get("summary", {})
                     story += f"\n\n👻 *Ghost Games ({len(matches)})*\n"
                     story += f"History: {summary.get('HOME',0)} Home, {summary.get('DRAW',0)} Draw, {summary.get('AWAY',0)} Away"

                 sent = res.get("market_sentiment", {})
                 if sent and sent.get("direction") != "NEUTRAL":
                     story += f"\n\n📉 *Market Sentiment*\n{sent.get('details')} ({sent.get('direction')})"

                 # Meta Labeler Quality
                 qual = res.get("meta_quality_score")
                 if qual is not None:
                     story += f"\n\n✅ *Meta-Quality Score:* {qual:.2f}"

                 # Bayesian Insight
                 details = res.get("details", {})
                 if "bayesian" in details:
                     bay = details["bayesian"]
                     conf = bay.get("confidence", 0.0)
                     prob = bay.get("prob_home", 0.0)
                     story += f"\n\n🔮 *Bayesian Insight*\nProb: %{prob*100:.1f}\nConf: %{conf*100:.1f} (Shrinkage: {1-conf:.2f})"

                 await self.send_message(chat_id, story)
                 return

        # 3. Volatilite Raporu var mı?
        if match_id in self.context.volatility_reports:
            vol = self.context.volatility_reports[match_id]
            msg = (
                f"📊 *Volatilite Analizi: {match_id}*\n"
                f"Rejim: {vol.regime}\n"
                f"GARCH Sigma: {vol.current_volatility:.4f}\n"
                f"Tavsiye: {vol.recommendation}"
            )
            await self.send_message(chat_id, msg)
            return

        await self.send_message(chat_id, f"❌ `{match_id}` için güncel analiz bulunamadı.")

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown"):
        """Mesaj gönder."""
        if not self.enabled: return

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{self.base_url}/sendMessage", json=payload)
        except Exception as e:
            logger.error(f"Mesaj gönderilemedi: {e}")

    async def send_photo(self, chat_id: int, photo: Any, caption: str = ""):
        """Fotoğraf gönder (BytesIO veya path)."""
        if not self.enabled: return

        try:
            async with httpx.AsyncClient() as client:
                # Telegram API expects 'photo' as multipart file
                files = {'photo': ('chart.png', photo, 'image/png')}
                data = {'chat_id': str(chat_id), 'caption': caption}
                await client.post(f"{self.base_url}/sendPhoto", data=data, files=files)
        except Exception as e:
            logger.error(f"Fotoğraf gönderilemedi: {e}")

    async def send_bet_signal(self, bet: Dict[str, Any]):
        """Bahis sinyali formatlayıp gönder (CEO Vision)."""
        # Tarihçeye ekle
        self.bet_history.append(bet)
        if len(self.bet_history) > 10:
            self.bet_history.pop(0)

        if not self.enabled: return

        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return

        match = bet.get("match_id", "Unknown")
        selection = bet.get("selection", "UNKNOWN")
        odds = bet.get("odds", 0.0)
        stake = bet.get("stake", 0.0)
        conf = bet.get("confidence", 0.0)
        reason = bet.get("reason", "")

        emoji = "🟢" if conf > 0.7 else "🟡"
        if conf > 0.85: emoji = "🔥"

        msg = (
            f"<b>{emoji} YENİ POZİSYON ALINDI</b>\n\n"
            f"⚽ <b>Maç:</b> {match}\n"
            f"🎯 <b>Seçim:</b> {selection}\n"
            f"📈 <b>Oran:</b> {odds:.2f}\n"
            f"💰 <b>Stake:</b> {stake:.2f} TL\n"
            f"🧠 <b>Güven:</b> %{conf*100:.1f}\n"
            f"📝 <b>Gerekçe:</b> {reason}\n"
            f"<i>🤖 Otonom Quant Sistemi</i>"
        )
        await self.send_message(chat_id, msg, parse_mode="HTML")

    async def send_risk_alert(self, alert_type: str, details: str):
        """Kritik risk uyarısı."""
        if not self.enabled: return
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return

        quote = self._get_stoic_quote("negative")
        msg = (
            f"<b>🚨 RİSK UYARISI: {alert_type.upper()}</b>\n\n"
            f"{details}\n\n"
            f"<i>{quote}</i>\n"
            f"<i>Sistem koruma protokolleri devrede.</i>"
        )
        await self.send_message(chat_id, msg, parse_mode="HTML")

    def _read_bankroll_state(self) -> Dict[str, Any]:
        """Disk'ten son durumu oku."""
        try:
            path = settings.DATA_DIR / "bankroll_state.json"
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"pnl": 0.0, "drawdown": 0.0}
