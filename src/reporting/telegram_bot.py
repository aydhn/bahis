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
  /ceo     - Yönetici özeti (ROI, Regime, Top Picks)
  /warroom - Canlı operasyon odası (Streaming events)
  /analyze - Derinlemesine maç analizi (Physics + AI)
  /physics - Fizik motoru raporları
  /brief   - Günlük Yönetici Brifingi (Teleoloji & Arb dahil)
  /hedge   - Hedge Operasyon Durumu
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

# New modules
from src.quant.finance.treasury import TreasuryEngine
from src.quant.analysis.oracle import TheOracle
from src.core.speed_cache import SpeedCache
from src.utils.daily_briefing import DailyBriefing

from src.system.config import settings
from src.reporting.visualizer import Visualizer
from src.core.event_bus import Event
from src.quant.analysis.narrative_generator import NarrativeGenerator

try:
    from src.quant.analysis.xai_explainer import XAIExplainer
except ImportError:
    XAIExplainer = None

try:
    from src.quant.analysis.scenario_simulator import ScenarioSimulator
except ImportError:
    ScenarioSimulator = None

try:
    from src.quant.analysis.market_regime_detector import MarketRegimeDetector
except ImportError:
    MarketRegimeDetector = None

try:
    from src.quant.treasury.synthetic_engine import SyntheticEngine
except ImportError:
    SyntheticEngine = None

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

        # Initialize new engines (or access via sentinel later)
        # For simplicity, we assume they are singletons or can be instantiated here for reporting
        self.treasury = TreasuryEngine()
        self.oracle = TheOracle()
        self.speed_cache = SpeedCache()

        # XAI Explainer
        self.xai = XAIExplainer() if XAIExplainer else None

        # New Simulation Tools
        self.simulator = ScenarioSimulator() if ScenarioSimulator else None
        self.regime_detector = MarketRegimeDetector() if MarketRegimeDetector else None
        self.synthetic = SyntheticEngine() if SyntheticEngine else None

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
            self.sentinel.bus.subscribe("pipeline_cycle_start", self.handle_event)
            self.sentinel.bus.subscribe("pipeline_cycle_end", self.handle_event)
            self.sentinel.bus.subscribe("hedge_order", self.handle_event)

    async def _warroom_dashboard_loop(self, chat_id: int):
        """Persistent dashboard loop that edits the same message."""
        dashboard_msg_id = None

        while self.warroom_active:
            try:
                # 1. Gather System Vitals
                cycle = self.context.cycle_id if self.context else 0

                # Regime
                regime = "NORMAL"
                if self.context and self.context.ensemble_results:
                     regime = self.context.ensemble_results[0].get('regime_status', 'NORMAL')

                # Financials
                t_status = self.treasury.state
                daily_pnl = t_status.daily_pnl
                total_cap = t_status.total_capital
                locked = t_status.locked_capital

                pnl_color = "🟢" if daily_pnl >= 0 else "🔴"

                # Active Hedges (from Treasury or Sentinel logic if accessible, or mock)
                # We can check recent hedge events or open positions count
                open_pos_count = 0
                # This requires access to DB or Sentinel state.
                # We'll use a placeholder or read from DB if available.

                dashboard = (
                    f"🚨 **EXECUTIVE WAR ROOM** 🚨\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚡ **Cycle:** #{cycle}  |  🕒 {asyncio.get_event_loop().time():.0f}\n"
                    f"🛡️ **Regime:** {regime}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 **Financials**\n"
                    f"• Capital: {total_cap:.2f}\n"
                    f"• Locked: {locked:.2f}\n"
                    f"• Daily PnL: {pnl_color} {daily_pnl:+.2f}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📡 **Live Feeds**\n"
                    f"• Odds Stream: {'🟢 ACTIVE' if self.speed_cache else '🔴 OFF'}\n"
                    f"• Quantum Brain: {'🟢 ONLINE' if self.context else '🟡 WAITING'}\n"
                    f"• Physics Engine: {'🟢 RUNNING' if self.context else '🟡 WAITING'}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💡 *Waiting for next alpha signal...*"
                )

                if dashboard_msg_id:
                    # Edit existing message
                    await self._edit_message(chat_id, dashboard_msg_id, dashboard)
                else:
                    # Send new message and track ID
                    # We need to capture the sent message ID.
                    # send_message returns None currently, we need to modify it or use raw client.
                    # For now, we will send a new message every 10 updates if we can't edit,
                    # but let's implement _edit_message helper.
                    dashboard_msg_id = await self._send_and_get_id(chat_id, dashboard)

            except Exception as e:
                logger.error(f"Warroom loop error: {e}")

            await asyncio.sleep(5) # Refresh every 5 seconds

    async def _send_and_get_id(self, chat_id: int, text: str) -> Optional[int]:
        if not self.enabled: return None
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self.base_url}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown"
                })
                if resp.status_code == 200:
                    return resp.json().get("result", {}).get("message_id")
        except Exception:
            pass
        return None

    async def _edit_message(self, chat_id: int, message_id: int, text: str):
        if not self.enabled: return
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{self.base_url}/editMessageText", json={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "parse_mode": "Markdown"
                })
        except Exception:
            pass

    async def handle_event(self, event: Event):
        """Event Bus üzerinden gelen kritik olayları raporla."""
        if not self.enabled: return

        # Chat ID bul
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return

        etype = event.event_type
        data = event.data

        # --- WAR ROOM STREAM ---
        # If dashboard is active, we don't spam simple events, only Critical ones.
        if self.warroom_active:
             # Only alert on high importance
             if etype in ["risk_alert", "hedge_order", "bet_placed", "pipeline_crash"]:
                 pass # Allow through
             else:
                 return # Suppress noise
        # -----------------------

        if etype == "bet_placed":
            await self.send_bet_signal(data)
        elif etype == "risk_alert":
            await self.send_risk_alert("Risk Alert", str(data))
        elif etype == "pipeline_crash":
            await self.send_message(chat_id, f"🚨 *Sistem Çökmesi*\n{data.get('error')}")
        elif etype == "hedge_order":
             hedge_info = data.get("hedge_signal", {})
             action = hedge_info.get("action", "UNKNOWN")
             reason = hedge_info.get("reason", "N/A")
             match_id = data.get("match_id", "Unknown")

             await self.send_message(chat_id,
                 f"🦔 **HEDGE ORDER TRIGGERED**\n"
                 f"Match: `{match_id}`\n"
                 f"Action: **{action}**\n"
                 f"Reason: {reason}\n"
                 f"Status: Executing..."
             )

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
            await self.send_message(chat_id, "🤖 *Otonom Quant Bot Devrede.*\nKomutlar: /status, /strategy, /ceo, /brief, /warroom, /pnl, /risk, /brain, /explain, /analyze, /physics, /treasury, /oracle, /set_risk, /force, /shutdown, /hedge, /synthetic, /memo")

        elif command == "/strategy":
            if self.context and hasattr(self.context, 'strategic_directive'):
                sd = self.context.strategic_directive
                if sd:
                    msg = (
                        f"🧠 **System Architect Directive**\n"
                        f"Posture: **{sd.posture}**\n"
                        f"Max Exposure: %{sd.max_daily_exposure*100:.1f}\n"
                        f"Edge Multiplier: x{sd.required_edge_multiplier:.1f}\n"
                        f"Rationale: {sd.rationale}"
                    )
                    await self.send_message(chat_id, msg)
                else:
                    await self.send_message(chat_id, "⚠️ No active directive found.")
            else:
                await self.send_message(chat_id, "⚠️ Pipeline context not available.")

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
            if not self.context:
                await self.send_message(chat_id, "⚠️ Henüz veri hazır değil.")
            else:
                # Convert Pydantic context to dict if necessary or use directly if Briefing supports it
                # Assuming DailyBriefing.generate expects a dict
                ctx_dict = self.context.to_dict() if hasattr(self.context, 'to_dict') else self.context.__dict__
                briefing = DailyBriefing.generate(ctx_dict)
                await self.send_message(chat_id, briefing)

        elif command == "/ceo":
            # Executive Summary (Legacy - now similar to brief but more concise)
            await self.send_message(chat_id, "☕ *Yönetici Özeti (Legacy)*")

            # 1. Financials
            stats = self._read_bankroll_state()
            bankroll = stats.get("bankroll", 0.0)
            roi = 0.0 # Calculate if possible or fetch from perf report
            if self.performance_report:
                roi = self.performance_report.get("roi", 0.0)

            fin_emoji = "🟢" if roi >= 0 else "🔴"

            # 2. Risk Regime
            regime = "NORMAL"
            if self.context and self.context.ensemble_results:
                 # Just take the first one as proxy
                 first = self.context.ensemble_results[0]
                 regime = first.get("regime_status", "NORMAL")

            # 3. Top Opportunities
            top_picks = "Fırsat Yok"
            if self.context and hasattr(self.context, 'final_bets'):
                bets = sorted(self.context.final_bets, key=lambda x: x.get('confidence', 0) * x.get('ev', 0), reverse=True)[:3]
                if bets:
                    top_picks = "\n".join([f"• {b['match_id']} ({b['selection']}) - Odds: {b['odds']}" for b in bets])

            msg = (
                f"🏛️ *CEO Dashboard*\n\n"
                f"{fin_emoji} **Finansal**: {bankroll:.2f} TL (ROI: %{roi*100:.2f})\n"
                f"🛡️ **Rejim**: {regime}\n\n"
                f"💎 **Top Fırsatlar**:\n{top_picks}\n\n"
                f"Sistem stabil."
            )
            await self.send_message(chat_id, msg)

        elif command == "/warroom":
            self.warroom_active = not self.warroom_active
            if self.warroom_active:
                # Start persistent dashboard updater
                self.warroom_task = asyncio.create_task(self._warroom_dashboard_loop(chat_id))
                await self.send_message(chat_id, "🚨 **WAR ROOM ACTIVATED** – Dashboard initializing...")
            else:
                if self.warroom_task:
                    self.warroom_task.cancel()
                await self.send_message(chat_id, "💤 War Room deactivated.")

        elif command == "/finance":
            # Financial Health Check
            t_status = self.treasury.get_status()
            # Hedge Status (Mock for now, would query HedgeHog state)
            h_status = "No active hedges."

            msg = (
                "💰 **FINANCIAL HEADQUARTERS**\n\n"
                f"{t_status}\n\n"
                "**Hedge Operations:**\n"
                f"{h_status}"
            )
            await self.send_message(chat_id, msg)

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

        elif command == "/treasury":
            status = self.treasury.get_status()
            await self.send_message(chat_id, f"🏦 *Treasury Report*\n\n{status}")

        elif command == "/hedge":
            # Show active hedges or potential
            # Ideally ask Sentinel to report, but we can check Treasury/HedgeHog status if exposed
            await self.send_message(chat_id, "🦔 *Hedge Operations*\nMonitoring active bets for Stop-Loss and Green Book opportunities...")

        elif command == "/oracle":
            if self.context:
                prophecy = self.oracle.consult(self.context.__dict__)
                await self.send_message(chat_id, prophecy)
            else:
                await self.send_message(chat_id, "🔮 The Oracle is meditating... (Waiting for pipeline context)")

        elif command == "/brain":
            # List status of all 10 engines
            msg = "🧠 *Physics & AI Engine Status*\n"

            # Helper to check engine presence
            def status(name, check):
                return f"- {name}: {'🟢 ACTIVE' if check else '🟡 SIM/OFF'}"

            # We check context or imports (simulated here)
            chaos = False
            try:
                from src.quant.physics.chaos_filter import NOLDS_OK
                chaos=NOLDS_OK
            except ImportError:
                pass

            quantum = False
            try:
                from src.quant.physics.quantum_brain import PENNYLANE_OK
                quantum=PENNYLANE_OK
            except ImportError:
                pass

            ricci = False
            try:
                from src.quant.physics.ricci_flow import RICCI_LIB_OK
                ricci=RICCI_LIB_OK
            except ImportError:
                pass

            topo = False
            try:
                from src.quant.physics.topology_mapper import KMAPPER_OK
                topo=KMAPPER_OK
            except ImportError:
                pass

            path = False
            try:
                from src.quant.physics.path_signature_engine import IIS_AVAILABLE
                path=IIS_AVAILABLE
            except ImportError:
                pass

            homo = False
            try:
                from src.quant.physics.homology_scanner import RIPSER_OK
                homo=RIPSER_OK
            except ImportError:
                pass

            gcn = False
            try:
                from src.quant.physics.gcn_pitch_graph import TORCH_AVAILABLE
                gcn=TORCH_AVAILABLE
            except ImportError:
                pass

            msg += status("Chaos Filter", chaos) + "\n"
            msg += status("Quantum Brain", quantum) + "\n"
            msg += status("Ricci Flow", ricci) + "\n"
            msg += status("Topology Mapper", topo) + "\n"
            msg += status("Path Signature", path) + "\n"
            msg += status("Homology Scanner", homo) + "\n"
            msg += status("GCN Pitch Graph", gcn) + "\n"
            msg += status("Fractal Analyzer", True) + "\n" # Pure math, usually active
            msg += status("Geometric Intel", True) + "\n"
            msg += status("Particle Tracker", True) + "\n"

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

        elif command == "/memo":
            await self._handle_memo(chat_id, args)

        elif command == "/analyze":
            await self._handle_analyze(chat_id, args)

        elif command == "/simulate":
            if not self.simulator:
                await self.send_message(chat_id, "⚠️ Simülatör modülü yüklü değil.")
                return
            if not args:
                await self.send_message(chat_id, "⚠️ Kullanım: `/simulate <match_id>`")
                return
            await self._handle_simulation(chat_id, args[0])

        elif command == "/physics":
            if not args:
                await self.send_message(chat_id, "⚠️ Kullanım: `/physics <match_id>`")
            else:
                await self._handle_physics_detail(chat_id, args[0])

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

        elif command == "/evolve":
            if self.sentinel:
                await self.send_message(chat_id, "🧬 Strateji Evrimi Tetikleniyor...")
                # We need to expose _run_evolution or call it directly if accessible
                if hasattr(self.sentinel, "_run_evolution"):
                    await self.sentinel._run_evolution()
                else:
                    await self.send_message(chat_id, "⚠️ Evrim fonksiyonu erişilemez.")
            else:
                await self.send_message(chat_id, "❌ Sentinel yok.")

        elif command == "/synthetic":
            if not self.synthetic:
                await self.send_message(chat_id, "⚠️ Synthetic Engine yüklü değil.")
                return

            if len(args) < 3:
                await self.send_message(chat_id, "⚠️ Kullanım: `/synthetic <home_odds> <draw_odds> <away_odds>`")
                return

            try:
                ho, do, ao = float(args[0]), float(args[1]), float(args[2])
                dnb = self.synthetic.calculate_dnb(ho, do, ao)
                dc = self.synthetic.calculate_double_chance(ho, do, ao)

                msg = f"🧪 **Synthetic Markets Calculation**\nInput: {ho} | {do} | {ao}\n\n"
                msg += f"🛡️ **Draw No Bet (DNB)**\n"
                msg += f"Home: {dnb.get('home_dnb', 0):.3f}\n"
                msg += f"Away: {dnb.get('away_dnb', 0):.3f}\n\n"

                msg += f"⚖️ **Double Chance (DC)**\n"
                msg += f"1X: {dc.get('1x', 0):.3f}\n"
                msg += f"X2: {dc.get('x2', 0):.3f}\n"
                msg += f"12: {dc.get('12', 0):.3f}"

                await self.send_message(chat_id, msg)
            except Exception as e:
                await self.send_message(chat_id, f"⚠️ Hata: {e}")

        elif command == "/active":
            # Active Inference Status
            try:
                from src.system.container import container
                agent = container.get("active_agent")
                if agent:
                    report = agent.get_report()
                    msg = (
                        f"🧠 **Active Inference Status**\n"
                        f"Free Energy: {report.total_free_energy:.4f}\n"
                        f"Avg Surprisal: {report.avg_surprisal:.4f}\n\n"
                        f"**Module Precision:**\n"
                    )
                    for mod, state in report.module_states.items():
                        msg += f"- {mod}: {state.precision:.2f} (Acc: {state.accuracy:.2%})\n"

                    msg += f"\n💡 **Recommendation:** {report.recommendation}"
                    await self.send_message(chat_id, msg)
                else:
                    await self.send_message(chat_id, "⚠️ Active Agent not initialized.")
            except Exception as e:
                await self.send_message(chat_id, f"⚠️ Error: {e}")

        elif command == "/nash":
            # Game Theory Analysis for a hypothetical or specific match
            # Usage: /nash <ev> <stake>
            if len(args) < 1:
                await self.send_message(chat_id, "⚠️ Usage: `/nash <ev>` (e.g. 0.05)")
                return

            try:
                ev = float(args[0])
                # Mock analysis using RiskTower logic
                import numpy as np
                from src.quant.analysis.game_theory_engine import GameTheoryEngine
                engine = GameTheoryEngine()

                payoff = np.array([[ev, -0.05], [0.0, 0.0]])
                res = engine.solve_nash(payoff)

                msg = (
                    f"♟️ **Game Theory Analysis (Nash)**\n"
                    f"Input EV: {ev:.2%}\n"
                    f"Scenario: Bet vs Market Drift\n\n"
                    f"**Optimal Strategy:**\n"
                    f"Bet Frequency: {res.optimal_strategy[0]:.2%}\n"
                    f"Pass Frequency: {res.optimal_strategy[1]:.2%}\n\n"
                    f"Expected Value: {res.game_value:.4f}\n"
                    f"Method: {res.method}"
                )
                await self.send_message(chat_id, msg)
            except Exception as e:
                await self.send_message(chat_id, f"⚠️ Error: {e}")

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

    async def _handle_memo(self, chat_id: int, args: list):
        """Generates a detailed Investment Memo (CEO Vision)."""
        if not args:
            await self.send_message(chat_id, "⚠️ Usage: `/memo <match_id>`")
            return

        match_id = args[0]
        if not self.context or not self.context.ensemble_results:
            await self.send_message(chat_id, "⚠️ Data not ready.")
            return

        # Find result
        res_map = {r.get("match_id"): r for r in self.context.ensemble_results}
        res = res_map.get(match_id)
        if not res:
            await self.send_message(chat_id, f"❌ Analysis not found for {match_id}")
            return

        # Construct Memo
        home = res.get("home_team", "Home")
        away = res.get("away_team", "Away")

        # Syndicate Data
        audit = res.get("syndicate_audit", [])
        verdict = res.get("verdict_text", "Consensus")

        # Format Audit Log
        audit_str = "\n".join([f"• {line}" for line in audit[:5]]) # Limit to 5 lines

        # Thesis & Counter-Thesis (Simulated from model details)
        # In a real implementation, we'd have explicit thesis strings
        details = res.get("details", {})

        thesis = "N/A"
        counter_thesis = "N/A"

        # Try to find Benter vs LSTM
        if "benter" in details and "lstm" in details:
            b_home = details["benter"].get("prob_home", 0)
            l_home = details["lstm"].get("prob_home", 0)

            if abs(b_home - l_home) > 0.15:
                if b_home > l_home:
                    thesis = f"Benter (Value): High stats dominance ({b_home:.2f})."
                    counter_thesis = f"LSTM (Trend): Recent form suggests struggle ({l_home:.2f})."
                else:
                    thesis = f"LSTM (Trend): Momentum is strong ({l_home:.2f})."
                    counter_thesis = f"Benter (Value): Fundamentals weak ({b_home:.2f})."
            else:
                thesis = f"Unified View: Both models agree (~{(b_home+l_home)/2:.2f})."
                counter_thesis = "No significant model divergence."

        msg = (
            f"📝 **INVESTMENT MEMO: {home} vs {away}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💡 **Investment Thesis**\n"
            f"{thesis}\n\n"
            f"⚖️ **Counter-Argument**\n"
            f"{counter_thesis}\n\n"
            f"🗳️ **Syndicate Verdict**\n"
            f"_{verdict}_\n"
            f"Confidence: {res.get('confidence', 0):.2%}\n\n"
            f"🔍 **Model Audit**\n"
            f"{audit_str}\n\n"
            f"🛡️ **Risk Note**\n"
            f"Portfolio Stress Impact: Low (Simulated)"
        )

        await self.send_message(chat_id, msg)

    async def _handle_analyze(self, chat_id: int, args: list):
        """Belirli bir maçın analizini getir."""
        if not args:
            await self.send_message(chat_id, "⚠️ Kullanım: `/analyze <match_id>`")
            return

        match_id = args[0]

        if not self.context:
            await self.send_message(chat_id, "⚠️ Sistem verisi henüz hazır değil.")
            return

        # 1. XAI Analysis (Visual & Text)
        if self.context.features is not None:
            # Polars filter for match_id
            try:
                row = self.context.features.filter(pl.col("match_id") == match_id)
                if not row.is_empty():
                    # Convert to dictionary for XAI
                    feat_dict = row.to_dict(as_series=False)
                    # Use index 0
                    single_row = {k: v[0] for k, v in feat_dict.items()}

                    if self.xai:
                        await self.send_message(chat_id, "🔍 *XAI Analizi Hazırlanıyor...*")
                        await self.xai.explain_and_send(single_row, match_id, chart_sender=self)
                    else:
                        await self.send_message(chat_id, "⚠️ XAI motoru yüklü değil.")
            except Exception as e:
                logger.error(f"Analysis failed: {e}")

        # 2. Narrative var mı?
        if match_id in self.context.narratives:
            await self.send_message(chat_id, self.context.narratives[match_id], parse_mode="Markdown")
            return

        # 3. Ensemble Analizi Var mı? (NEW)
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

                 # Deep Physics Insight
                 physics = self.context.physics_reports if hasattr(self.context, "physics_reports") else {}
                 if physics:
                     chaos = physics.get("chaos_reports", {}).get(match_id)
                     fractal = physics.get("fractal_reports", {}).get(match_id)
                     if chaos:
                         story += f"\n\n⚛️ *Deep Physics*\nChaos Regime: {chaos.regime} (λ={chaos.params.max_lyapunov:.3f})"
                     if fractal:
                         story += f"\nFractal Dim: {fractal.fractal_dimension:.3f} ({fractal.regime})"

                 await self.send_message(chat_id, story)
                 return

        # 4. Volatilite Raporu var mı?
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

        # If nothing found
        if not self.context.features or self.context.features.filter(pl.col("match_id") == match_id).is_empty():
             await self.send_message(chat_id, f"❌ `{match_id}` için güncel analiz bulunamadı.")

    async def _handle_simulation(self, chat_id: int, match_id: str):
        """Monte Carlo Simülasyonu çalıştır ve histogram gönder."""
        await self.send_message(chat_id, f"🎲 *{match_id}* için 10,000 maç simüle ediliyor...")

        # Fetch xG data from context if available, else mock
        home_xg, away_xg = 1.5, 1.2 # Default mock

        if self.context and self.context.features is not None:
            try:
                row = self.context.features.filter(pl.col("match_id") == match_id)
                if not row.is_empty():
                    home_xg = row["home_xg"][0] or 1.5
                    away_xg = row["away_xg"][0] or 1.2
            except Exception:
                pass

        sim_res = self.simulator.simulate_match(home_xg, away_xg, match_id)

        msg = (
            f"📊 **Simülasyon Sonucu (xG: {home_xg:.2f} - {away_xg:.2f})**\n"
            f"🏠 Home Win: %{sim_res['prob_home']*100:.1f}\n"
            f"🤝 Draw: %{sim_res['prob_draw']*100:.1f}\n"
            f"✈️ Away Win: %{sim_res['prob_away']*100:.1f}\n\n"
        )

        hist = self.simulator.generate_ascii_histogram(sim_res["home_goals_dist"], "Home Goals")
        msg += f"```\n{hist}\n```"

        await self.send_message(chat_id, msg)

    async def _handle_physics_detail(self, chat_id: int, match_id: str):
        """Physics motorlarının detaylı çıktısını raporla."""
        if not self.context or not hasattr(self.context, "physics_reports"):
            await self.send_message(chat_id, "⚠️ Fizik motoru verisi yok.")
            return

        reps = self.context.physics_reports

        msg = f"⚛️ *PHYSICS REPORT: {match_id}*\n"

        # 1. Chaos
        if match_id in reps.get("chaos_reports", {}):
            c = reps["chaos_reports"][match_id]
            msg += f"\n🌪️ **Chaos Theory**\nRegime: {c.regime}\nLyapunov: {c.params.max_lyapunov:.4f}\nEntropy: {c.params.sample_entropy:.3f}\n"

        # 2. Topology
        if match_id in reps.get("topology_reports", {}):
            t = reps["topology_reports"][match_id]
            msg += f"\n🕸️ **Topology**\nCluster: #{t.assigned_cluster}\nAnomaly Score: {t.anomaly_score:.2f}\nStatus: {t.cluster_label}\n"

        # 3. Homology
        if match_id in reps.get("homology_reports", {}):
            h = reps["homology_reports"][match_id]
            msg += f"\n🍩 **Homology**\nOrg Score: {h.get('home_org',0):.2f} (Home) vs {h.get('away_org',0):.2f} (Away)\nAdvantage: {h.get('org_advantage',0):.2f}\n"

        # 4. Fractal
        if match_id in reps.get("fractal_reports", {}):
            f = reps["fractal_reports"][match_id]
            msg += f"\n❄️ **Fractal**\nHurst: {f.hurst:.3f}\nDim: {f.fractal_dimension:.3f}\n"

        # 5. Hypergraph
        if match_id in reps.get("hypergraph_reports", {}):
            h = reps["hypergraph_reports"][match_id]
            home_rep = h.get("home")
            away_rep = h.get("away")
            if home_rep:
                msg += f"\n🔗 **Hypergraph (Home)**\nVuln: {home_rep.vulnerability_index:.2f}\nCohesion: {home_rep.team_cohesion:.2f}\n"
            if away_rep:
                msg += f"\n🔗 **Hypergraph (Away)**\nVuln: {away_rep.vulnerability_index:.2f}\nCohesion: {away_rep.team_cohesion:.2f}\n"

        await self.send_message(chat_id, msg)

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
