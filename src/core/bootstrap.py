"""
bootstrap.py – System Initialization & Orchestration
----------------------------------------------------
Eskiden `bahis.py` içinde duran devasa `_boot` ve `_shutdown` mantığı 
buraya taşındı. Amaç: Modüler, test edilebilir ve temiz bir başlatma süreci.
"""
import asyncio
import sys
import time
import warnings
from pathlib import Path

from loguru import logger
# from rich.console import Console # Rich bağımlılığını kaldır

# Proje kök dizini (bu dosya src/core/ içinde olduğundan ../../ ile köke çıkıyoruz)
ROOT = Path(__file__).resolve().parents[2]

class SystemBootstrapper:
    """Tüm bot sistemini katman katman ayağa kaldıran sınıf."""
    
    def __init__(self, mode: str, headless: bool, telegram_enabled: bool, dashboard: bool):
        self.mode = mode
        self.headless = headless
        self.telegram = telegram_enabled
        self.dashboard = dashboard
        self.shutdown_event = asyncio.Event()
        self.modules: dict = {}
        
    async def boot(self):
        """Sistemi başlatır."""
        # self.console.rule(f"[bold cyan]QUANT BETTING BOT – BOOT ({self.mode.upper()})[/]")
        logger.info(f"--- QUANT BETTING BOT – BOOT ({self.mode.upper()}) ---")
        t0 = time.perf_counter()
        
        # 1. Altyapı
        await self._boot_infrastructure()
        
        # 2. Hafıza (Memory)
        await self._boot_layer_memory()
        
        # 3. Veri Toplama (Ingestion)
        await self._boot_layer_ingestion()
        
        # 4. Kantitatif Zeka (Quant)
        await self._boot_layer_quant()
        
        # 5. Risk & İcra (Core 4)
        await self._boot_layer_risk()
        
        # 6. Arayüz & Orkestrasyon (UI/Ops)
        await self._boot_layer_ops()
        
        # 7. Modülleri Orkestratöre Kaydet (KRİTİK ADIM)
        # Orkestratörün hangi task'i hangi modülde çalıştıracağını bilmesi lazım
        if "orchestrator" in self.modules:
            orch = self.modules["orchestrator"]
            orch.register_modules(self.modules)
            logger.info(f"[Boot] {len(self.modules)} modül orkestratöre kaydedildi.")

        elapsed = time.perf_counter() - t0
        # self.console.rule(f"[bold green]SYSTEM READY – {elapsed:.2f}s[/]")
        logger.info(f"--- SYSTEM READY – {elapsed:.2f}s ---")
        logger.success(f"[Boot] Sistem {elapsed:.2f} saniyede hazır.")
        
        return self.modules

    async def _boot_infrastructure(self):
        logger.info("Layer 0 – Infrastructure...")
        from src.core.circuit_breaker import CircuitBreakerRegistry
        from src.core.exception_guardian import ExceptionGuardian, install_global_hook
        from src.utils.log_rotator import LogRotator
        from src.utils.super_logger import SuperLogger
        from src.memory.smart_cache import SmartCache

        # Log Rotation
        rotator = LogRotator(log_dir=str(ROOT / "logs"), archive_days=3)
        rotator.rotate()
        
        # Guardian & Cache
        guardian = ExceptionGuardian(error_budget_per_hour=50)
        install_global_hook(guardian)
        smart_cache = SmartCache(ttl_l2=3600.0)
        
        self.modules.update({
            "guardian": guardian,
            "smart_cache": smart_cache,
            "circuit_breaker": CircuitBreakerRegistry(),
            "super_log": SuperLogger(log_dir="data/logs"),
        })

    async def _boot_layer_memory(self):
        logger.info("Layer 2 – Memory & Context...")
        from src.memory.db_manager import DBManager
        from src.memory.feature_cache import FeatureCache
        from src.memory.lance_memory import LanceMemory
        # from src.memory.graph_rag import GraphRAG # Graph devre disi birakildi (baglanti hatasi onlemek icin lazy load)
        
        self.modules["db"] = DBManager()
        self.modules["cache"] = FeatureCache()
        self.modules["lance"] = LanceMemory()
        # self.modules["graph"] = GraphRAG() 
        logger.success("Layer 2 OK.")

    async def _boot_layer_ingestion(self):
        logger.info("Layer 1 – Sensors & Ingestion...")
        from src.ingestion.async_data_factory import DataFactory
        from src.ingestion.api_hijacker import APIHijacker
        from src.ingestion.metric_exporter import MetricExporter
        from src.ingestion.scraper_agent import ScraperAgent # Mackolik/Sofascore burada
        
        db = self.modules["db"]
        cache = self.modules["cache"]
        
        factory = DataFactory(db=db, cache=cache, headless=self.headless)
        hijacker = APIHijacker(db=db)
        metrics = MetricExporter()
        
        self.modules.update({
            "data_factory": factory,
            "hijacker": hijacker,
            "metrics": metrics,
        })
        logger.success("Layer 1 OK.")

    async def _boot_layer_quant(self):
        logger.info("Layer 3 – Quantitative Brain...")
        # Lazy imports to speed up boot if not needed immediately
        from src.quant.rl_trader import RLTrader
        from src.quant.probabilistic_engine import ProbabilisticEngine
        from src.quant.philosophical_engine import PhilosophicalEngine
        from src.quant.elo_engine import EloEngine
        from src.quant.monte_carlo_engine import MonteCarloEngine
        from src.quant.dixon_coles_model import DixonColesModel
        from src.quant.spectral_analysis import SpectralAnalysis
        from src.core.genetic_optimizer import GeneticOptimizer
        from src.core.jit_accelerator import JITAccelerator
        from src.core.hedge_calculator import HedgeCalculator
        from src.quant.fair_value_engine import FairValueEngine
        from src.core.multi_hedge_optimizer import MultiExchangeHedgeOptimizer
        from src.core.shared_hot_layer import SharedHotLayer
        from src.core.auto_refactor_agent import AutoRefactorAgent
        from src.core.meta_strategy_manager import MetaStrategyManager
        from src.core.synthetic_hedge import SyntheticHedgingEngine
        from src.quant.adverse_selection import AdverseSelectionGuard  # Fixed path if needed
        from src.quant.causal_engine import CausalInferenceEngine
        from src.quant.evt_longshot import EVTLongshotEngine
        from src.quant.fbm_model import fBMModel
        from src.core.news_aggregator import NewsAggregator
        from src.core.nas_engine import NASEngine
        from src.core.heuristic_evolver import HeuristicEvolver
        from src.quant.kalman_filter import KalmanStrengthFilter
        from src.quant.bayesian_elo import BayesianEloEngine
        from src.quant.falsification_guard import FalsificationGuard
        from src.core.evolutionary_runner import EvolutionaryRunner
        from src.core.debate_engine import MultiAgentDebateEngine
        from src.quant.hmm_regime import HMMRegimeSwitcher
        from src.quant.league_graph import LeagueGraphModel
        from src.quant.antifragility_engine import AntifragilityEngine
        from src.core.rust_bridge import RustFastBuffer
        
        # Sadece kritik olanlari baslat, digerleri talep uzerine yuklenebilir
        db = self.modules["db"]
        
        # JIT Accelerator (Hızlandırıcı)
        jit_acc = JITAccelerator()
        jit_acc.warmup() # Başlangıçta derleme yap (biraz yavaşlatır ama canlıda hız kazandırır)
        self.modules["jit"] = jit_acc
        
        self.modules["prob_engine"] = ProbabilisticEngine()
        self.modules["rl_trader"] = RLTrader(db=db)
        self.modules["philosophical"] = PhilosophicalEngine(db=db)
        self.modules["elo"] = EloEngine(db=db)
        
        # Monte Carlo + JIT Enjeksiyonu
        mc_engine = MonteCarloEngine(db=db)
        mc_engine.inject_jit(jit_acc)
        self.modules["monte_carlo"] = mc_engine
        
        self.modules["dixon_coles"] = DixonColesModel(db=db)
        self.modules["spectral_analysis"] = SpectralAnalysis(db=db)
        
        # Optimizasyon & Hedge & Kalman
        self.modules["genetic_opt"] = GeneticOptimizer()
        self.modules["hedge_calc"] = HedgeCalculator()
        self.modules["fair_value"] = FairValueEngine(db_manager=db)
        self.modules["multi_hedge"] = MultiExchangeHedgeOptimizer(db=db)
        self.modules["hot_layer"] = SharedHotLayer()
        self.modules["meta_strategy"] = MetaStrategyManager(db=db, total_bankroll=self.modules["kelly"]._bankroll.total)
        self.modules["synthetic_hedge"] = SyntheticHedgingEngine(db=db)
        self.modules["news_aggregator"] = NewsAggregator(db=db)
        self.modules["execution_guard"] = AdverseSelectionGuard()
        self.modules["causal_engine"] = CausalInferenceEngine(db=db)
        self.modules["evt_longshot"] = EVTLongshotEngine(db=db)
        self.modules["fbm_model"] = fBMModel(db=db)
        self.modules["nas_engine"] = NASEngine(db=db)
        self.modules["heuristic_evolver"] = HeuristicEvolver(db=db)
        self.modules["kalman"] = KalmanStrengthFilter(db=db)
        self.modules["bayesian_elo"] = BayesianEloEngine(db=db)
        self.modules["falsification"] = FalsificationGuard(db=db)
        self.modules["evolutionary"] = EvolutionaryRunner(db=db, optimizer=self.modules["genetic_opt"])
        self.modules["debate_engine"] = MultiAgentDebateEngine()
        self.modules["hmm_optimizer"] = HMMRegimeSwitcher(db=db)
        self.modules["league_graph"] = LeagueGraphModel(db=db)
        self.modules["antifragility"] = AntifragilityEngine(db=db)
        self.modules["fast_buffer"] = RustFastBuffer(size=2000)
        
        # Modelleri MetaStrategy'ye kaydet
        ms = self.modules["meta_strategy"]
        for m in ["dixon_coles", "prob_engine", "rl_trader", "bayesian_elo"]:
            if m in self.modules: ms.register_model(m)
            
        logger.success("Layer 3 OK.")

        from src.quant.pnl_tracker import PnLTracker
        from src.core.kelly_matrix import CorrelatedKellyMatrix
        
        db = self.modules["db"]
        self.modules["kelly"] = RegimeKelly(bankroll=10000.0, db=db)
        self.modules["kelly_matrix"] = CorrelatedKellyMatrix()
        self.modules["risk_solver"] = ConstrainedRiskSolver()
        self.modules["portfolio_opt"] = PortfolioOptimizer(db=db)
        self.modules["pnl_tracker"] = PnLTracker()
        
        # PnL senkronizasyonu
        try:
            self.modules["pnl_tracker"].sync_from_db(db)
            logger.info("[Boot] PnLTracker DuckDB ile senkronize edildi.")
        except Exception as e:
            logger.warning(f"[Boot] PnL senkronizasyonu başarısız: {e}")
            
        logger.success("Layer 4 OK.")

    async def _boot_layer_ops(self):
        logger.info("Layer 5 – Ops & Interface...")
        from src.ui.telegram_mini_app import TelegramNotifier, TelegramApp
        from src.core.workflow_orchestrator import WorkflowOrchestrator
        from src.core.job_scheduler import JobScheduler
        
        db = self.modules["db"]
        
        # Notifier (Sadece gönderim)
        notifier = TelegramNotifier(enabled=self.telegram)
        self.modules["notifier"] = notifier
        
        # Interactive App (Komutlar)
        self.modules["telegram_app"] = TelegramApp(
            token=notifier._token, 
            chat_id=notifier._chat_id,
            notifier=notifier,
            db=db,
            pnl_tracker=self.modules.get("pnl_tracker")
        )
        
        from src.core.omni_sovereign_logic import OmniSovereignController
        
        self.modules["orchestrator"] = WorkflowOrchestrator()
        self.modules["auto_refactor"] = AutoRefactorAgent(orchestrator=self.modules["orchestrator"])
        self.modules["omni_controller"] = OmniSovereignController(modules=self.modules)
        self.modules["scheduler"] = JobScheduler()
        logger.success("Layer 5 OK.")

    async def run_forever(self):
        """Ana döngüyü başlatır."""
        factory = self.modules["data_factory"]
        hijacker = self.modules["hijacker"]
        metrics = self.modules["metrics"]
        scheduler = self.modules["scheduler"]
        orchestrator = self.modules["orchestrator"]
        kelly = self.modules["kelly"]
        
        # 1. Adaptive Scheduling Setup
        # Full Pipeline: Normalde 5 dk (300s) arayla çalışır.
        # Volatilite/Kriz anında (mult=0.3) -> 90s arayla çalışır.
        async def _run_pipeline_wrapper():
            logger.info("[Scheduler] Adaptive Pipeline Tetiklendi 🚀")
            await orchestrator.run_pipeline()
            
        scheduler.add_adaptive_interval(
            name="full_pipeline",
            func=_run_pipeline_wrapper,
            base_seconds=120,   # [MODIFIED] 5 dk -> 2 dk (Daha agresif)
            volatility_provider=kelly.get_volatility_multiplier,
            min_seconds=30,     # [MODIFIED] 60s -> 30s (Kriz anında ultra hız)
            max_seconds=300     # [MODIFIED] 600s -> 300s
        )
        
        # 2. Evolutionary Strategy Tuning (Weekly)
        # Her hafta parametreleri gerçek verilerle optimize et
        evolutionary = self.modules.get("evolutionary")
        if evolutionary:
            async def _run_evolution():
                logger.info("[Scheduler] Haftalık Strateji Evrimi Başlatılıyor... 🧬")
                await evolutionary.run_optimization_cycle()
            
            # 1 haftalık (604800 saniye) sabit aralık
            scheduler.add_adaptive_interval(
                name="weekly_evolution",
                func=_run_evolution,
                base_seconds=604800,
                volatility_provider=lambda: 1.0,
                min_seconds=604800,
                max_seconds=604800
            )

        # 3. Market Regime & Graph Analysis (Daily)
        async def _run_daily_analysis():
            logger.info("[Scheduler] Günlük Rejim ve Çizge Analizi Başlatılıyor... 📊")
            if "hmm_optimizer" in self.modules:
                await self.modules["hmm_optimizer"].run_batch()
            if "league_graph" in self.modules:
                await self.modules["league_graph"].run_batch()
            if "bayesian_elo" in self.modules:
                await self.modules["bayesian_elo"].run_batch()
            if "heuristic_evolver" in self.modules:
                await self.modules["heuristic_evolver"].run_batch()
            if "news_aggregator" in self.modules:
                await self.modules["news_aggregator"].run_batch()
            if "fbm_model" in self.modules:
                await self.modules["fbm_model"].run_batch()
            if "nas_engine" in self.modules:
                await self.modules["nas_engine"].run_batch()
            if "meta_strategy" in self.modules:
                await self.modules["meta_strategy"].run_batch()
            if "nas_engine" in self.modules:
                await self.modules["nas_engine"].run_batch()

        scheduler.add_adaptive_interval(
            name="daily_analysis",
            func=_run_daily_analysis,
            base_seconds=86400,
            volatility_provider=lambda: 1.0,
            min_seconds=86400,
            max_seconds=86400
        )

        # Scheduler başlat
        await scheduler.start()

        # Görevler
        tasks = [
            asyncio.create_task(metrics.serve(), name="metrics"),
            asyncio.create_task(factory.run_live(shutdown=self.shutdown_event), name="data_live"),
            asyncio.create_task(hijacker.listen(shutdown=self.shutdown_event), name="hijacker"),
            asyncio.create_task(self.modules["telegram_app"].start(shutdown=self.shutdown_event), name="telegram_bot"),
        ]
        
        logger.info("[Main] Sistem döngüsü başladı. Çıkış için CTRL+C")
        try:
            # Sonsuza kadar bekle veya shutdown_event bekle
            await self.shutdown_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("[Main] Durdurma sinyali alındı.")
        finally:
            self.shutdown_event.set()
            await scheduler.stop() # Scheduler durdur
            # Taskleri iptal et
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.success("[Main] Güvenli kapanış tamamlandı.")

