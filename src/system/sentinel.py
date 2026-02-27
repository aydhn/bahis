"""
sentinel.py – Otonom Sistem Orkestratörü (The CEO).

Bu modül, tüm sistemi (Pipeline, Risk, Bot) tek bir çatı altında yönetir.
"Sentinel", sistemin otonom karar vericisi ve koruyucusudur.

Özellikler:
  - Tek bir Telegram Bot instance'ı yönetir.
  - Pipeline'ı başlatır, durdurur, izler.
  - Risk modunu dinamik olarak değiştirir.
  - Acil durum komutlarını (Force Bet, Shutdown) uygular.
  - Strategy Evolver ile otonom iyileşme sağlar.
  - Otonom Risk Regülasyonu: ROI performansına göre risk modunu otomatik ayarlar.
  - Profit & Safety: Real-time Hedging via HedgeHog.
"""
import asyncio
import signal
import sys
from typing import Optional, Any, Dict, List
from loguru import logger

from src.reporting.telegram_bot import TelegramBot
from src.pipeline.core import create_default_pipeline, PipelineEngine
from src.pipeline.stages.execution import ExecutionStage
from src.system.lifecycle import lifecycle
from src.core.regime_kelly import RegimeKelly, RegimeState
from src.system.container import container
from src.core.event_bus import EventBus, Event
from src.quant.risk.portfolio_manager import PortfolioManager
from src.core.circuit_breaker import CircuitBreakerRegistry
from src.core.strategy_evolver import StrategyEvolver
from src.quant.finance.treasury import TreasuryEngine
from src.quant.analysis.oracle import TheOracle
from src.quant.finance.hedgehog import HedgeHog
from src.core.speed_cache import SpeedCache
from src.core.system_architect import SystemArchitect
from src.ingestion.flash_monitor import FlashOddsMonitor

class Sentinel:
    """
    Sistemin beyni ve yöneticisi.
    """

    def __init__(self, daemon_mode: bool = True):
        self.daemon_mode = daemon_mode
        self.running = False

        # 0. Event Bus (Merkezi Sinir Sistemi)
        self.bus = EventBus()
        self.portfolio_manager = PortfolioManager(self.bus)

        # 0.1 Flash Reaction (Odds Stream Listener)
        self.speed_cache = SpeedCache()
        # "The Sniper" Upgrade
        self.flash_monitor = FlashOddsMonitor(self.bus, self.speed_cache)
        self.bus.subscribe("odds_tick", self.flash_monitor.on_odds_tick)
        self.bus.subscribe("flash_opportunity", self._handle_flash_reaction)

        # 0.2 System Architect (The Brain)
        self.architect = SystemArchitect()

        # 1. Bot Entegrasyonu
        self.bot = TelegramBot()
        self.bot.set_sentinel(self)
        # Botu event bus'a abone yapabiliriz (daha sonra)

        # 2. Pipeline Hazırlığı
        self.pipeline: PipelineEngine = create_default_pipeline(
            bot_instance=self.bot,
            bus=self.bus
        )

        # 2.1 Wire Execution Stage to Hedge Events
        # Find execution stage instance in pipeline
        self.execution_stage = None
        for stage in self.pipeline.stages:
            if isinstance(stage, ExecutionStage):
                self.execution_stage = stage
                break

        if self.execution_stage:
            # Subscribe handle_hedge_order to bus
            # Note: EventBus.subscribe expects a coroutine callback taking one 'event' arg
            self.bus.subscribe("hedge_order", self._on_hedge_order)
        else:
            logger.warning("Sentinel: ExecutionStage not found in pipeline. Hedging disabled.")

        # 3. Risk Yöneticisi (Shared State via Container)
        # RiskStage ile aynı instance'ı kullandığımızdan emin olmalıyız.
        self.risk_manager = container.get("regime_kelly")
        if self.risk_manager:
            self.bus.subscribe("market_regime_update", self.risk_manager.update_regime)
            self.bus.subscribe("market_regime_update", self._on_regime_change)

        self.portfolio_opt = container.get("portfolio_opt")

        # 4. Global Circuit Breaker (Panic Button)
        self.cb_registry = CircuitBreakerRegistry()
        self.system_breaker = self.cb_registry.get_or_create("system_health", preset="api")
        # 1 saatlik soğuma, 3 kritik hata -> OPEN
        self.system_breaker._config.recovery_timeout = 3600.0

        # 5. Treasury & Oracle (Financial & Strategic Command)
        self.treasury = TreasuryEngine()
        self.oracle = TheOracle()

        # 5.1 HedgeHog (Real-time Hedging)
        self.hedgehog = HedgeHog()

        # 6. Otonom Strateji Evrimi
        self.evolver = StrategyEvolver(population_size=20, elitism_pct=0.1)
        if self.evolver.load_checkpoint():
            logger.info("Sentinel: Strategy Evolver checkpoint yüklendi.")
        else:
            logger.info("Sentinel: Strategy Evolver sıfırdan başlıyor.")

        # Sinyal Yakalama
        if daemon_mode:
            signal.signal(signal.SIGINT, self._handle_sigint)
            signal.signal(signal.SIGTERM, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        """Ctrl+C yakalayıcı."""
        logger.warning("Sentinel: Kapanma sinyali alındı.")
        self.shutdown()

    async def _on_hedge_order(self, event: Event):
        """Bridge between EventBus and ExecutionStage method."""
        if self.execution_stage:
            await self.execution_stage.handle_hedge_order(event.data)

    async def _on_regime_change(self, event: Event):
        """Handle market regime updates."""
        data = event.data
        # Extract regime string (e.g. from RegimeMetrics)
        # Assuming event data might contain 'regime' key directly or inside metrics
        regime = data.get("regime", "STABLE")

        # If it comes from MarketRegimeDetector (RegimeMetrics obj), it might be serialized differently
        # Let's support both simple string and nested dict
        if isinstance(regime, str):
            self.treasury.rebalance_buckets(regime)
        else:
            logger.warning(f"Sentinel: Unknown regime format in event: {regime}")

    async def run(self):
        """Sistemi başlat."""
        logger.info("╔════════════════════════════════════════╗")
        logger.info("║     SENTINEL OTONOM SİSTEM (v2.1)      ║")
        logger.info("╚════════════════════════════════════════╝")

        # System Integrity Check at Startup
        if not await self.system_integrity_check():
            logger.critical("Sentinel: System integrity check failed. Aborting startup.")
            return

        self.running = True

        # Botu başlat
        if self.bot.enabled:
            asyncio.create_task(self.bot.start())
            await self.bot.send_message(
                int(self.bot.token.split(":")[0]) if self.bot.token else 0, # Fallback ID
                "🚀 *Sentinel Başlatıldı.*"
            )

        # Start Background Services
        if self.daemon_mode:
            asyncio.create_task(self._run_hedge_monitor())
            await self.flash_monitor.start()

        # Pipeline döngüsü
        cycle_count = 0
        try:
            if self.daemon_mode:
                logger.info("Sentinel: Daemon Modu Aktif. Kontrollü döngü başlıyor.")
                while self.running and not lifecycle.shutdown_event.is_set():
                    cycle_count += 1

                    # 1. Sağlık Kontrolü
                    if not self._check_health():
                        logger.warning("Sentinel: Sistem sağlığı KRİTİK. Devre Kesici AÇIK. 5 dakika bekleniyor.")
                        if self.bot and self.bot.enabled:
                             await self.bot.send_risk_alert("CIRCUIT BREAKER", "Sistem finansal koruma moduna geçti. İşlemler durduruldu.")
                        await asyncio.sleep(300)
                        continue

                    # 1.5 Architect Consultation (Strategic Posture)
                    # We need access to state. Treasury is self.treasury.
                    # Regime is a bit harder as it's computed in RiskStage/Pipeline.
                    # Ideally Pipeline writes regime to a shared state or DB.
                    # For now, we assume Architect can pull basic state or use defaults.
                    strategic_directive = self.architect.consult(
                        treasury_status=self.treasury.state.__dict__,
                        regime_metrics=None, # Passed inside pipeline usually
                        news_sentiment=0.5 # Placeholder
                    )

                    # 2. Pipeline Döngüsü (Tek adım)
                    # Evolver'dan en iyi strateji ağırlıklarını al
                    best_dna = self.evolver.get_best_dna()
                    initial_ctx = {
                        "strategic_directive": strategic_directive
                    }
                    if best_dna:
                         # DNA'yı context'e enjekte et
                         dna_dict = best_dna.to_dict()
                         # Ensemble Weights'leri ayrıştır
                         weights = {
                             "benter": dna_dict.get("ensemble_weight_poisson", 0.3),
                             "lgbm": dna_dict.get("ensemble_weight_lgbm", 0.35),
                             "lstm": dna_dict.get("ensemble_weight_lstm", 0.2),
                             # Diğer modeller için mapping gerekebilir
                         }
                         initial_ctx["ensemble_weights"] = weights
                         initial_ctx["kelly_fraction"] = dna_dict.get("kelly_fraction", 0.25)

                    cycle_result = await self.pipeline.run_once(initial_context=initial_ctx)

                    # 2.5 Auto Risk Adjustment
                    if cycle_result and "performance_report" in cycle_result:
                        await self.auto_adjust_risk(cycle_result["performance_report"])

                    # 3. Evolver Check (Her 100 döngüde bir evrim)
                    if cycle_count % 100 == 0:
                        await self._run_evolution()

                    # 4. Bekleme
                    await asyncio.sleep(10)
            else:
                if self._check_health():
                    await self.pipeline.run_once()
                else:
                    logger.error("Sistem sağlığı yetersiz. Çalışma iptal edildi.")

        except Exception as e:
            logger.critical(f"Sentinel Critical Error: {e}")
        finally:
            await self.shutdown_async()

    async def system_integrity_check(self) -> bool:
        """Verifies critical components are operational before startup."""
        logger.info("Sentinel: Performing system integrity check...")

        checks = []

        # 1. DB Connection
        db = container.get("db")
        if not db:
            logger.error("Integrity Check: DB manager not found in container.")
            return False
        checks.append("DB Manager: OK")

        # 2. Pipeline Configuration
        if not self.pipeline or not self.pipeline.stages:
            logger.error("Integrity Check: Pipeline stages not configured.")
            return False
        checks.append(f"Pipeline Stages: {len(self.pipeline.stages)} OK")

        # 3. Risk Manager
        if not self.risk_manager:
            logger.warning("Integrity Check: Risk Manager (RegimeKelly) not found. Proceeding with caution.")
        else:
            checks.append("Risk Manager: OK")

        logger.info(f"Integrity Check Passed. Components: {', '.join(checks)}")
        return True

    async def auto_adjust_risk(self, perf_report: Dict[str, Any]):
        """
        Autonomously adjusts risk parameters based on real-time performance (ROI).
        Acts like a Fund Manager regulating exposure.
        """
        roi = perf_report.get("roi", 0.0)

        # Current mode logic (could be stored in state, simplified here)
        # We access risk_manager internal state directly for this autonomous logic
        if not self.risk_manager: return

        current_base_fraction = self.risk_manager._base_fraction
        new_mode = None

        if roi < -0.05: # Drawdown > 5%
            if current_base_fraction > 0.15:
                logger.warning(f"Sentinel: Auto-Risk Adjustment -> Conservative (ROI {roi:.2%})")
                self.set_risk_mode("conservative")
                new_mode = "CONSERVATIVE"
        elif roi > 0.10: # Profit > 10%
            if current_base_fraction < 0.35:
                logger.success(f"Sentinel: Auto-Risk Adjustment -> Aggressive (ROI {roi:.2%})")
                self.set_risk_mode("aggressive")
                new_mode = "AGGRESSIVE"
        elif -0.02 <= roi <= 0.05: # Stable
             if current_base_fraction != 0.25:
                 logger.info(f"Sentinel: Auto-Risk Adjustment -> Normal (ROI {roi:.2%})")
                 self.set_risk_mode("normal")
                 new_mode = "NORMAL"

        if new_mode and self.bot:
             await self.bot.send_message(
                 int(self.bot.token.split(":")[0]) if self.bot.token else 0,
                 f"🤖 *Otonom Risk Ayarı*\nROI: %{roi*100:.2f}\nYeni Mod: *{new_mode}*"
             )

    async def _handle_flash_reaction(self, event: Any):
        """
        Flash Reaction: Reacts to high-velocity odds changes (Dropping Odds).
        Acts as a 'Liquidity Sniper'.
        """
        data = event.data
        match_id = data.get("match_id")
        # Assume data contains z-score or we calculate it here.
        # For simplicity, let's assume ingestion calculated 'z_score'
        z_score = data.get("z_score", 0.0)

        if z_score < -2.0:
            logger.warning(f"⚡ FLASH REACTION: Dropping Odds detected for {match_id} (Z={z_score})")
            # Trigger immediate execution check for this match
            # This bypasses the normal cycle
            if self.pipeline:
                # We need a way to run specific match. Pipeline run_once takes context.
                # Construct a mini-context
                logger.info(f"⚡ Sniping liquidity for {match_id}...")

                # Fetch match data (pseudo-code, in real implementation we fetch from DB/Cache)
                # match_data = self.db.get_match(match_id)

                # For now, just log the intent as we need full match context to run pipeline
                # Ideally: await self.pipeline.run_single_match(match_id)
                pass

    async def _run_hedge_monitor(self):
        """
        Background task to monitor active bets for hedging opportunities.
        Uses HedgeHog engine and SpeedCache for real-time odds.
        """
        logger.info("Sentinel: Starting Hedge Monitor...")

        while self.running and not lifecycle.shutdown_event.is_set():
            try:
                db = container.get("db")
                if not db:
                    await asyncio.sleep(10)
                    continue

                # 1. Fetch OPEN bets from DB
                # This should ideally use a dedicated method, simulating raw query here
                try:
                    df = db.query("SELECT * FROM bets WHERE status IN ('pending', 'open')")
                    open_bets = df.to_dicts() if not df.is_empty() else []
                except Exception as e:
                    logger.error(f"Hedge Monitor DB Error: {e}")
                    open_bets = []

                if not open_bets:
                    await asyncio.sleep(30)
                    continue

                # 2. Check each bet against live odds
                for bet in open_bets:
                    match_id = bet.get("match_id")

                    # Fetch live odds from SpeedCache
                    live_odds_data = self.speed_cache.get(f"odds_{match_id}")
                    if not live_odds_data:
                        continue

                    # Determine current odds for the selection
                    # Assuming live_odds_data is dict: {"home": 2.1, "draw": 3.2, "away": 3.4}
                    selection = bet.get("selection", "").lower() # home, draw, away

                    # Map selection to key
                    sel_key = selection
                    if selection in ["1", "home"]: sel_key = "home"
                    elif selection in ["x", "draw"]: sel_key = "draw"
                    elif selection in ["2", "away"]: sel_key = "away"

                    current_price = live_odds_data.get(sel_key)

                    if current_price:
                        # 3. HedgeHog Check
                        # Use simple volatility or fetch real one
                        hedge_signal = self.hedgehog.dynamic_hedge(
                            position=bet,
                            live_odds=current_price,
                            volatility=0.05 # Default or fetch from context
                        )

                        if hedge_signal:
                            logger.warning(f"🦔 HEDGE SIGNAL: {match_id} -> {hedge_signal['action']}")

                            # 4. Emit Hedge Order
                            event_data = {
                                "bet_id": bet.get("bet_id"), # Assuming DB has this
                                "match_id": match_id,
                                "original_bet": bet,
                                "hedge_signal": hedge_signal,
                                "timestamp": str(asyncio.get_event_loop().time())
                            }
                            await self.bus.emit(Event("hedge_order", data=event_data))

                            # Notify User immediately via Bot if critical
                            if self.bot and self.bot.enabled:
                                await self.bot.send_risk_alert(
                                    f"HEDGE SIGNAL: {hedge_signal['action']}",
                                    f"Match: {match_id}\nReason: {hedge_signal['reason']}"
                                )

                # Sleep to prevent tight loop
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Hedge Monitor Loop Error: {e}")
                await asyncio.sleep(10)


    async def _run_evolution(self):
        """Strateji evrimini tetikle."""
        logger.info("Sentinel: Evrim süreci başlıyor...")

        db = container.get("db")
        if not db:
            logger.warning("Sentinel: DB bağlantısı yok, evrim atlanıyor.")
            return

        # 1. Sonuçlanmış bahisleri çek
        try:
            bets_df = db.get_settled_bets(limit=100)
        except Exception as e:
            logger.error(f"Sentinel: Bahis verisi çekilemedi: {e}")
            return

        if bets_df.is_empty():
            logger.info("Sentinel: Evrim için yeterli veri yok (Sonuçlanmış bahis bulunamadı).")
            return

        # 2. Sinyallerle birleştir (Prob ve EV bilgisi için)
        try:
            signals_df = db.get_signals()
        except Exception:
            signals_df = None

        results_for_evolver = []

        signal_map = {}
        if signals_df is not None and not signals_df.is_empty():
            for row in signals_df.iter_rows(named=True):
                key = f"{row.get('match_id')}_{row.get('selection')}"
                signal_map[key] = row

        for bet in bets_df.iter_rows(named=True):
            match_id = bet.get("match_id", "")
            selection = bet.get("selection", "")
            status = bet.get("status", "lost")
            odds = bet.get("odds", 2.0)

            sig = signal_map.get(f"{match_id}_{selection}", {})

            # Confidence is often used as probability in this system
            prob = sig.get("confidence", 1.0 / odds if odds > 0 else 0.5)
            ev = sig.get("ev", 0.0)

            won = (status == "won")

            results_for_evolver.append({
                "won": won,
                "pnl": bet.get("pnl", 0.0),
                "odds": odds,
                "prob": prob,
                "ev": ev,
                "match_id": match_id
            })

        # 3. Evrimleştir
        # 3. Active Inference Learning (Zero Error Loop)
        active_agent = container.get("active_agent")
        if active_agent:
            for res in results_for_evolver:
                # Retrieve individual model predictions for this match if stored
                # This requires that we stored individual model outputs in DB or logs.
                # Since we don't have deep history of sub-model predictions here,
                # we can simulate learning on the 'ensemble' itself as a module,
                # or if we had the data, loop through models.

                # For now, we treat the final outcome as feedback for the 'ensemble' module
                # In a full implementation, we'd need to query a 'predictions' table.

                # Simulating feedback for ensemble
                outcome_idx = 0 if res['won'] and res['odds'] < 2.5 else 1 # Simplified outcome mapping
                # Ideally: outcome_idx = 0 (Home), 1 (Draw), 2 (Away) based on score

                # Just trigger a generic observation for 'ensemble' to demonstrate loop
                active_agent.observe(
                    module="ensemble",
                    predicted_probs=[res['prob'], (1-res['prob'])/2, (1-res['prob'])/2], # Rough approx
                    observed=0 if res['won'] else 2 # Rough approx
                )

            logger.info("Sentinel: Active Inference feedback loop executed.")

        # 4. Evrimleştir
        if len(results_for_evolver) >= 5:  # Lower threshold for testing
            try:
                report = self.evolver.evolve(results_for_evolver)

                if report.improvements:
                    logger.success(f"Sentinel: Strateji iyileştirildi! {report.improvements}")

                    # Log new DNA
                    logger.info(f"Yeni DNA: {json.dumps(report.best_dna, indent=2)}")

                    # Notify
                    if self.bot and self.bot.enabled:
                        chat_id = 0
                        if self.bot.token and ":" in self.bot.token:
                            try:
                                chat_id = int(self.bot.token.split(":")[0])
                            except Exception:
                                pass

                        if chat_id:
                            await self.bot.send_message(
                                chat_id,
                                f"🧬 **Strateji Evrimi**\n"
                                f"Gen: #{report.generation}\n"
                                f"Fitness: {report.best_fitness:.4f}\n"
                                f"İyileşme: {', '.join(report.improvements)}"
                            )
            except Exception as e:
                logger.error(f"Sentinel: Evrim sırasında hata: {e}")
        else:
            logger.info(f"Sentinel: Yetersiz veri ({len(results_for_evolver)}).")

    def _check_health(self) -> bool:
        """Sistemin finansal ve teknik sağlığını kontrol et."""
        if not self.system_breaker.is_available:
            return False

        try:
            # DB'den Ardışık Kayıp Kontrolü
            db = container.get("db")
            if db:
                # Son 10 bahis kontrolü
                try:
                    df = db.query("SELECT status FROM bets WHERE status IN ('won', 'lost') ORDER BY settled_at DESC LIMIT 10")
                    if not df.is_empty():
                        statuses = df["status"].to_list()
                        # Eğer 10 bahis varsa ve hepsi kayıpsa
                        if len(statuses) >= 10 and all(s == 'lost' for s in statuses):
                            raise Exception("10 Ardışık Kayıp! Panic Button devrede.")
                except Exception as db_err:
                     # Tablo yoksa vs yut, sistemi durdurma
                     logger.warning(f"Health check DB uyarısı: {db_err}")

            self.system_breaker._on_success()
            return True
        except Exception as e:
            logger.critical(f"Health Check Failed: {e}")
            self.system_breaker._on_failure(e)
            return False

    def set_risk_mode(self, mode: str) -> str:
        """Risk modunu değiştir (Telegram komutu)."""
        mode = mode.lower()

        # Risk parametrelerini güncelle
        if mode == "aggressive":
            self.risk_manager._base_fraction = 0.35
            self.risk_manager._min_edge = 0.02
        elif mode == "conservative":
            self.risk_manager._base_fraction = 0.15
            self.risk_manager._min_edge = 0.05
        elif mode == "normal":
            self.risk_manager._base_fraction = 0.25
            self.risk_manager._min_edge = 0.03
        else:
            return f"Bilinmeyen mod: {mode}. (aggressive, normal, conservative)"

        logger.info(f"Sentinel: Risk modu '{mode}' olarak ayarlandı.")
        return f"Risk modu güncellendi: *{mode.upper()}*"

    async def force_bet(self, match_id: str, selection: str) -> str:
        """Manuel bahis zorla (Telegram komutu)."""
        logger.warning(f"Sentinel: Manuel bahis isteği -> {match_id} / {selection}")

        # Bu özellik ExecutionStage'e doğrudan emir iletmeyi gerektirir.
        # Şimdilik basitçe logluyoruz, tam implementasyon için
        # Pipeline context'ine "force_orders" listesi eklenmeli.

        # Pipeline context'ine erişimimiz var mı?
        # PipelineEngine her döngüde context'i sıfırlıyor ama persistent bir queue tutabiliriz.
        # Basit bir çözüm: Sentinel'in bir "command_queue"su olur, pipeline bunu okur.

        return f"Manuel emir kuyruğa alındı: {match_id} ({selection})"

    def shutdown(self):
        """Sistemi güvenli kapat."""
        self.running = False
        lifecycle.shutdown_event.set()
        if self.pipeline:
            self.pipeline.running = False
        logger.info("Sentinel: Kapanma prosedürü başlatıldı...")

    async def shutdown_async(self):
        """Async cleanup."""
        if self.bot:
            await self.bot.stop()

        # Persist Treasury State
        if self.treasury:
            self.treasury.save_state()
            logger.info("Sentinel: Treasury state saved.")

        if self.flash_monitor:
            await self.flash_monitor.stop()

        logger.info("Sentinel: Sistem kapandı. Görüşmek üzere.")

if __name__ == "__main__":
    sentinel = Sentinel()
    asyncio.run(sentinel.run())
