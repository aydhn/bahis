#!/usr/bin/env python3
"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘  QUANT BETTING BOT â€“ ORCHESTRATOR (bahis.py)                â•‘
â•‘  Tek gÃ¶revi: uzantÄ± modÃ¼lleri baÅŸlatmak, durdurmak ve       â•‘
â•‘  yÃ¶netmek.  Ä°ÅŸ mantÄ±ÄŸÄ± SIFIR â€“ her ÅŸey src/ altÄ±nda.       â•‘
â•‘                                                              â•‘
â•‘  !! KULLANIM_KILAVUZU.txt GUNCELLEME HATIRLATMASI !!        â•‘
â•‘  Bu dosyada veya src/ altinda yapilan her degisiklikte       â•‘
â•‘  KULLANIM_KILAVUZU.txt belgesinin ilgili bolumu de           â•‘
â•‘  guncellenmelidir.                                           â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
import sys
import warnings
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

# â”€â”€ proje kÃ¶kÃ¼nÃ¼ PYTHONPATH'e ekle â”€â”€
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows terminal encoding kaynaklÄ± UnicodeEncodeError Ã§Ã¶kÃ¼ÅŸlerini engelle.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# â”€â”€ .env dosyasÄ±nÄ± yÃ¼kle â”€â”€
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# â”€â”€ GÃ¼rÃ¼ltÃ¼lÃ¼ stdlib loggerlarÄ± erken bastÄ±r (import Ã¶ncesi) â”€â”€
import logging as _logging
for _noisy in (
    # JAX / XLA
    "jax", "jax._src", "jax._src.dispatch", "jax._src.compiler",
    "jax._src.interpreters", "jax._src.interpreters.pxla",
    "jax._src.cache_key", "jax._src.compilation_cache",
    "jax._src.xla_bridge",
    # Neo4j
    "neo4j",
    # Numba â€“ byteflow/SSA DEBUG flood'unu kapat
    "numba", "numba.core", "numba.core.byteflow", "numba.core.ssa",
    "numba.core.interpreter", "numba.typed", "numba.np",
    # Prophet / Cmdstan / Stan
    "prophet", "cmdstanpy", "pystan", "stan",
    # Matplotlib
    "matplotlib", "matplotlib.font_manager",
    # TensorFlow (varsa)
    "tensorflow", "tf",
    # urllib3 / httpx
    "urllib3", "httpx", "httpcore",
):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

# Bilinen gÃ¼rÃ¼ltÃ¼ uyarÄ±larÄ±nÄ± bastÄ±r (hesap fallback ile devam ediyor).
warnings.filterwarnings("ignore", message="Mean of empty slice\\.", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)

# â”€â”€ Loguru yapÄ±landÄ±rmasÄ± â”€â”€
(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "logs" / "archive").mkdir(exist_ok=True)
(ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
logger.remove()

# Konsol Ã§Ä±ktÄ±sÄ± â€“ okunabilir ve renksiz gÃ¼venli
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> â€“ {message}",
    level="INFO",
    backtrace=True,
    diagnose=True,
)

# GÃ¼nlÃ¼k log dosyasÄ± â€“ DEBUG seviyesinde her ÅŸeyi yakala
logger.add(
    ROOT / "logs" / "bot_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} â€“ {message}",
    rotation="10 MB",
    retention="3 days",
    compression="gz",
    level="DEBUG",
    backtrace=True,
    diagnose=True,
    colorize=False,
)

# Hata logu â€“ sadece ERROR ve Ã¼stÃ¼
logger.add(
    ROOT / "logs" / "error.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} â€“ {message}",
    level="ERROR",
    rotation="5 MB",
    retention="7 days",
    backtrace=True,
    diagnose=True,
    colorize=False,
)

# Boot logu â€“ tam boot sÃ¼recini kaydet
logger.add(
    ROOT / "logs" / "boot.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} â€“ {message}",
    level="DEBUG",
    mode="w",
    colorize=False,
)

console = Console()
app = typer.Typer(
    name="bahis",
    help="Quant Betting Bot â€“ Komuta Merkezi",
    add_completion=False,
    pretty_exceptions_enable=True,
    invoke_without_command=True,
)


@app.callback()
def _default_callback(ctx: typer.Context):
    """Alt komut verilmezse otomatik olarak 'run' Ã§alÄ±ÅŸtÄ±rÄ±r."""
    if ctx.invoked_subcommand is None:
        # VarsayÄ±lan deÄŸerlerle run komutu Ã§alÄ±ÅŸtÄ±r
        ctx.invoke(run, mode="full", headless=True, telegram=True, dashboard=False)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Graceful Shutdown
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_shutdown_event = asyncio.Event()
_boot_complete = False
_signal_count = 0


def _handle_signal(*_):
    global _signal_count
    _signal_count += 1
    if _shutdown_event.is_set():
        logger.warning(
            f"Kapatma zaten tetiklenmiÅŸti (sinyal#{_signal_count}) "
            "â€“ gÃ¶revler kapanÄ±yorâ€¦"
        )
        return
    if not _boot_complete:
        logger.warning("Boot sÄ±rasÄ±nda kapatma sinyali alÄ±ndÄ± â€“ acil kapanÄ±ÅŸ baÅŸlatÄ±lÄ±yor.")
    else:
        logger.warning("Kapatma sinyali alÄ±ndÄ± â€“ modÃ¼ller durduruluyorâ€¦")
    _shutdown_event.set()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Helper: GÃ¼nlÃ¼k Brifing gÃ¶nder
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _send_daily_briefing(briefer, db, portfolio, health, notifier,
                         super_log=None, guardian=None):
    """Sabah brifingini oluÅŸtur ve Telegram'a gÃ¶nder."""
    try:
        import asyncio
        data = briefer.collect_data(
            db=db, portfolio=portfolio, health=health,
            super_log=super_log, guardian=guardian,
        )
        report = briefer.generate(data)
        if notifier and report.telegram_text:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(notifier.send(report.telegram_text))
            except RuntimeError:
                asyncio.run(notifier.send(report.telegram_text))
        logger.info(f"[Briefing] GÃ¼nlÃ¼k brifing gÃ¶nderildi ({report.method})")
    except Exception as e:
        logger.error(f"[Briefing] Hata: {e}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  BOOT: TÃ¼m katmanlarÄ± baÅŸlat
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
async def _boot(mode: str, headless: bool, telegram: bool, dashboard: bool):
    """Sistemi katman katman ayaÄŸa kaldÄ±rÄ±r."""
    global _boot_complete, _signal_count
    _boot_complete = False
    _signal_count = 0
    _shutdown_event.clear()
    # â”€â”€ LAYER 2: Memory â”€â”€
    from src.memory.db_manager import DBManager
    from src.memory.feature_cache import FeatureCache
    from src.memory.lance_memory import LanceMemory
    from src.memory.zero_copy_bridge import ZeroCopyBridge
    from src.memory.graph_rag import GraphRAG
    from src.memory.dvc_manager import DVCManager
    from src.memory.smart_cache import SmartCache

    # â”€â”€ LAYER 1: Ingestion â”€â”€
    from src.ingestion.async_data_factory import DataFactory
    from src.ingestion.api_hijacker import APIHijacker
    from src.ingestion.metric_exporter import MetricExporter
    from src.ingestion.auto_healer import AutoHealer
    from src.ingestion.vision_tracker import VisionTracker
    from src.ingestion.scraper_agent import ScraperAgent
    from src.ingestion.stealth_browser import StealthBrowser
    from src.ingestion.lineup_monitor import LineupMonitor
    from src.ingestion.news_rag import NewsRAGAnalyzer
    from src.ingestion.data_sources import DataSourceAggregator
    from src.ingestion.vision_live import LiveMatchVision

    # â”€â”€ LAYER 3: Quant â”€â”€
    from src.quant.probabilistic_engine import ProbabilisticEngine
    from src.quant.evt_tail_scanner import EVTTailScanner
    from src.quant.causal_discovery import CausalDiscovery
    from src.quant.conformal_quantile_bridge import ConformalQuantileBridge
    from src.quant.path_signature_engine import PathSignatureEngine
    from src.quant.jump_diffusion_model import JumpDiffusionModel
    from src.quant.geometric_intelligence import GeometricIntelligence
    from src.quant.multi_task_backbone import MultiTaskBackbone
    from src.quant.kan_interpreter import KANInterpreter
    from src.quant.rl_trader import RLTrader
    from src.quant.gcn_pitch_graph import GCNPitchGraph
    from src.quant.poisson_model import PoissonModel
    from src.quant.monte_carlo_engine import MonteCarloEngine
    from src.quant.elo_glicko_rating import EloGlickoSystem
    from src.quant.sentiment_analyzer import SentimentAnalyzer
    from src.quant.anomaly_detector import AnomalyDetector
    from src.quant.dixon_coles_model import DixonColesModel
    from src.quant.time_decay import ExponentialTimeDecay, TeamVolatilityIndex
    from src.quant.gradient_boosting import GradientBoostingModel, FeatureEngineer
    from src.quant.bayesian_hierarchical import BayesianHierarchicalModel, NPxGFilter
    from src.quant.clv_tracker import CLVTracker, CorrelationMatrix
    from src.quant.ensemble_stacking import EnsembleStacking
    from src.quant.lstm_trend import LSTMTrendAnalyzer
    from src.quant.xai_explainer import XAIExplainer
    from src.quant.glm_model import GLMGoalPredictor
    from src.quant.network_centrality import PassNetworkAnalyzer
    from src.quant.prophet_seasonality import ProphetSeasonalityAnalyzer
    from src.quant.uncertainty_quantifier import UncertaintyQuantifier
    from src.quant.kalman_tracker import KalmanTeamTracker
    from src.quant.rl_betting_env import RLBettingAgent
    from src.quant.vector_engine import VectorMatchEngine
    from src.quant.copula_risk import CopulaRiskAnalyzer
    from src.quant.isolation_anomaly import IsolationAnomalyDetector
    from src.quant.causal_reasoner import CausalReasoner
    from src.quant.topology_scanner import TopologyScanner
    from src.quant.nash_solver import NashGameSolver
    from src.quant.entropy_meter import EntropyMeter
    from src.quant.evt_risk_manager import EVTRiskManager
    from src.quant.digital_twin_sim import DigitalTwinSimulator
    from src.quant.bsts_impact import BSTSImpactAnalyzer
    from src.quant.fractal_analyzer import FractalAnalyzer
    from src.quant.transfer_learner import TransferLearner
    from src.quant.transport_metric import TransportMetric
    from src.quant.federated_trainer import FederatedTrainer
    from src.quant.hypergraph_unit import HypergraphUnitAnalyzer
    from src.quant.fluid_pitch import FluidPitchAnalyzer
    from src.quant.ricci_flow import RicciFlowAnalyzer
    from src.quant.regime_switcher import RegimeSwitcher
    from src.quant.sde_pricer import SDEPricer
    from src.quant.quantum_brain import QuantumBrain
    from src.quant.hawkes_momentum import HawkesMomentumAnalyzer
    from src.quant.survival_estimator import SurvivalEstimator
    from src.quant.fatigue_engine import FatigueEngine
    from src.quant.chaos_filter import ChaosFilter
    from src.quant.homology_scanner import HomologyScanner
    from src.quant.automl_engine import AutoMLEngine
    from src.quant.synthetic_trainer import SyntheticTrainer
    from src.quant.fuzzy_reasoning import FuzzyReasoningEngine, FuzzyInput
    from src.quant.uncertainty_separator import UncertaintySeparator
    from src.quant.topology_mapper import TopologyMapper
    from src.quant.probabilistic_engine import ProbabilisticEngine
    from src.quant.multifractal_logic import MultifractalAnalyzer
    from src.quant.symbolic_discovery import SymbolicDiscovery
    from src.quant.wavelet_denoiser import WaveletDenoiser
    from src.quant.volatility_analyzer import VolatilityAnalyzer
    from src.quant.particle_strength_tracker import ParticleStrengthTracker

    # â”€â”€ LAYER 4: Core â”€â”€
    from src.core.constrained_risk_solver import ConstrainedRiskSolver
    from src.core.systemic_risk_covar import SystemicRiskCoVaR
    from src.core.vector_backtester import VectorBacktester
    from src.core.black_litterman_optimizer import BlackLittermanOptimizer
    from src.core.pnl_stabilizer import PnLStabilizer
    from src.core.model_quantizer import ModelQuantizer
    from src.core.jax_accelerator import JAXAccelerator
    from src.core.circuit_breaker import CircuitBreakerRegistry
    from src.core.data_validator import DataValidator
    from src.core.dependency_container import DependencyContainer
    from src.core.job_scheduler import JobScheduler
    from src.core.portfolio_optimizer import PortfolioOptimizer
    from src.core.genetic_optimizer import GeneticOptimizer
    from src.core.hedge_calculator import HedgeCalculator
    from src.core.fair_value_engine import FairValueEngine
    from src.core.event_bus import EventBus, EventStore, ReplayEngine, Event
    from src.core.stream_processor import StreamProcessor
    from src.core.shadow_manager import ShadowManager
    from src.core.jit_accelerator import JITAccelerator
    from src.core.grpc_communicator import GRPCCommunicator
    from src.core.distributed_core import DistributedCore
    from src.core.mimic_engine import MimicEngine
    from src.core.quantum_annealer import QuantumAnnealer
    from src.core.auto_healer import SelfHealingEngine
    from src.core.telemetry_tracer import TelemetryTracer
    from src.core.blind_strategy import BlindStrategyEngine
    from src.core.rust_engine import RustEngine
    from src.core.active_inference_agent import ActiveInferenceAgent
    from src.core.stream_processor import StreamProcessor

    # â”€â”€ LAYER 6: Graph Intelligence â”€â”€
    from src.memory.neo4j_graph import Neo4jFootballGraph
    from src.memory.graph_rag import GraphRAG

    # â”€â”€ LAYER 5: Utils & UI â”€â”€
    from src.utils.strategy_health_report import StrategyHealthReport
    from src.utils.devils_advocate import DevilsAdvocate
    from src.utils.threshold_controller import ThresholdController
    from src.utils.auto_doc_generator import AutoDocGenerator
    from src.core.workflow_orchestrator import WorkflowOrchestrator
    from src.utils.podcast_producer import PodcastProducer
    from src.utils.telegram_live import TelegramLiveDashboard
    from src.utils.telegram_admin import TelegramAdmin
    from src.utils.telegram_scenario import ScenarioSimulator
    from src.utils.plot_animator import PlotAnimator
    from src.utils.psycho_profiler import PsychoProfiler
    from src.utils.query_assistant import QueryAssistant
    from src.utils.human_feedback_loop import HumanFeedbackLoop
    from src.utils.war_room import WarRoom
    from src.utils.daily_briefing import DailyBriefing
    from src.utils.decision_flow_gen import DecisionFlowGenerator
    from src.utils.super_logger import SuperLogger
    from src.utils.agent_poll_system import AgentPollSystem
    from src.utils.log_rotator import LogRotator

    # â”€â”€ YENÄ° MODÃœLLER (v2) â”€â”€
    from src.core.exception_guardian import ExceptionGuardian, install_global_hook
    from src.core.regime_kelly import RegimeKelly, RegimeState
    from src.core.strategy_evolver import StrategyEvolver
    from src.quant.fisher_geometry import FisherGeometry
    from src.quant.philosophical_engine import PhilosophicalEngine
    from src.utils.strategy_cockpit import StrategyCockpit

    import time as _time
    _boot_t0 = _time.perf_counter()

    console.rule("[bold cyan]QUANT BETTING BOT â€“ BAÅLATILIYOR[/]")
    logger.info(f"[Boot] Mod={mode}, headless={headless}, telegram={telegram}, dashboard={dashboard}")

    # â”€â”€ LAYER 0: AltyapÄ± (Circuit Breaker Registry, Smart Cache, Guardian) â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 0 â€“ Infrastructure baÅŸlatÄ±lÄ±yorâ€¦")
    cb_registry = CircuitBreakerRegistry()
    smart_cache = SmartCache(ttl_l2=3600.0, ttl_l3=86400.0)
    super_log = SuperLogger(
        log_dir="data/logs", rotation="100 MB",
        retention="30 days", compression="gz",
    )

    # Log Rotator â€” son 3 gÃ¼n aktif, eski loglar â†’ archive/
    log_rotator = LogRotator(
        log_dir=str(ROOT / "logs"),
        archive_days=3,
        compress=True,
    )
    log_rotator.rotate()  # Boot'ta hemen eski loglarÄ± arÅŸivle

    # Exception Guardian â€” sessiz hata yutmayÄ± ortadan kaldÄ±rÄ±r
    guardian = ExceptionGuardian(
        error_budget_per_hour=50, circuit_threshold=15,
        circuit_reset_seconds=300,
    )
    install_global_hook(guardian)

    # Regime-Aware Kelly Criterion v2
    regime_kelly = RegimeKelly(
        bankroll=10000.0, base_fraction=0.25,
        min_edge=0.04, max_stake_pct=0.05,
    )

    # Self-Evolving Strategy DNA
    strategy_evolver = StrategyEvolver(
        population_size=50, elitism_pct=0.10,
    )
    strategy_evolver.load_checkpoint()

    # Fisher Information Geometry
    fisher_geo = FisherGeometry(
        anomaly_threshold=2.0, regime_threshold=1.5,
    )

    # Philosophical Engine (Epistemic Reasoning)
    philo_engine = PhilosophicalEngine(
        calibration_window=200, min_epistemic_score=0.45,
    )

    # Strategy Cockpit (Telegram HUD)
    cockpit = StrategyCockpit()

    logger.success(f"Layer 0 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ LAYER 2: HafÄ±za â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 2 â€“ Memory & Context baÅŸlatÄ±lÄ±yorâ€¦")
    db = DBManager()
    cache = FeatureCache()
    lance = LanceMemory()
    bridge = ZeroCopyBridge()
    graph = GraphRAG(llm_backend="auto")
    dvc = DVCManager()
    logger.success(f"Layer 2 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ LAYER 1: Veri Toplama â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 1 â€“ Sensors & Ingestion baÅŸlatÄ±lÄ±yorâ€¦")
    factory = DataFactory(db=db, cache=cache, headless=headless)
    hijacker = APIHijacker(db=db)
    metrics = MetricExporter()
    healer = AutoHealer()
    vision = VisionTracker()
    stealth = StealthBrowser(headless=headless)
    logger.success(f"Layer 1 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ LAYER 3: Kantitatif Zeka â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 3 â€“ Quantitative Brain baÅŸlatÄ±lÄ±yorâ€¦")
    prob_engine = ProbabilisticEngine()
    evt_scanner = EVTTailScanner()
    causal = CausalDiscovery()
    conformal = ConformalQuantileBridge()
    path_sig = PathSignatureEngine()
    jump_diff = JumpDiffusionModel()
    geometric = GeometricIntelligence()
    mtl = MultiTaskBackbone()
    kan = KANInterpreter()
    rl = RLTrader()
    gcn = GCNPitchGraph()
    poisson = PoissonModel()
    monte_carlo = MonteCarloEngine()
    elo_system = EloGlickoSystem()
    sentiment = SentimentAnalyzer()
    anomaly = AnomalyDetector()

    # â”€â”€ Level 2 Advanced Stats â”€â”€
    time_decay = ExponentialTimeDecay(preset="moderate")
    dixon_coles = DixonColesModel(decay_rate=time_decay._lambda)
    team_vix = TeamVolatilityIndex(decay=time_decay)
    gb_model = GradientBoostingModel(engine="lightgbm")
    feat_eng = FeatureEngineer()

    # â”€â”€ Bayesian + CLV + Korelasyon + Stacking â”€â”€
    bayesian_model = BayesianHierarchicalModel(league="super_lig")
    npxg_filter = NPxGFilter()
    clv_tracker = CLVTracker()
    corr_matrix = CorrelationMatrix(max_portfolio_correlation=0.80)
    stacker = EnsembleStacking()

    # â”€â”€ Level 5: Deep Learning + XAI â”€â”€
    lstm_trend = LSTMTrendAnalyzer(hidden_size=32, num_layers=2)
    lstm_trend.load_model()
    xai = XAIExplainer(max_display=10)
    news_rag = NewsRAGAnalyzer()

    # â”€â”€ Level 7: GLM + Market Maker â”€â”€
    glm = GLMGoalPredictor(family="poisson")

    # â”€â”€ Level 8: Graph Intelligence & Mevsimsellik â”€â”€
    pass_network = PassNetworkAnalyzer()
    prophet_season = ProphetSeasonalityAnalyzer()

    # â”€â”€ Level 9: Reliability & Advanced Uncertainty â”€â”€
    uncertainty_q = UncertaintyQuantifier(alpha=0.05, abstain_threshold=0.50)
    kalman = KalmanTeamTracker(process_noise=2.0, measurement_noise=8.0)

    # â”€â”€ Level 10: RL & Vector Embeddings â”€â”€
    rl_agent = RLBettingAgent(initial_bankroll=10000.0)
    rl_agent.load()  # Daha Ã¶nce eÄŸitilmiÅŸ model varsa yÃ¼kle
    vector_engine = VectorMatchEngine(dim=32)
    copula_risk = CopulaRiskAnalyzer(copula_family="auto")
    scenario_sim = ScenarioSimulator()

    # â”€â”€ Level 11: Adversarial Resilience â”€â”€
    iso_anomaly = IsolationAnomalyDetector(contamination=0.05, blacklist_severity="high")

    # â”€â”€ Level 12: Causal Inference & Computer Vision â”€â”€
    vision_live = LiveMatchVision(model_size="n", fps=1.0)
    causal_reason = CausalReasoner(significance_level=0.05)
    topo_scanner = TopologyScanner(homology_dims=[0, 1])

    # â”€â”€ Level 13: Physics of Information & Zero-Latency â”€â”€
    nash_solver = NashGameSolver()
    entropy_meter = EntropyMeter(kill_threshold=2.50)

    # â”€â”€ Level 14: Digital Twin & EVT â”€â”€
    evt_risk = EVTRiskManager(threshold_quantile=0.90, kelly_reduction=0.50)
    digital_twin = DigitalTwinSimulator()

    # â”€â”€ Level 15: Distributed Computing & Fractal â”€â”€
    bsts_impact = BSTSImpactAnalyzer(significance_level=0.05)
    fractal = FractalAnalyzer()

    # â”€â”€ Level 16: Cognitive Mimicry & Optimal Transport â”€â”€
    transfer_learner = TransferLearner(input_dim=20, n_classes=3)
    transfer_learner.load("europe_base")
    transport_metric = TransportMetric(drift_threshold=0.25, kill_threshold=0.50)

    # â”€â”€ Level 17: Swarm Intelligence & Quantum Optimization â”€â”€
    fed_trainer = FederatedTrainer(
        leagues=["super_lig", "premier_league", "bundesliga", "la_liga", "serie_a"],
        input_dim=20, n_classes=3,
    )
    fed_trainer.load_global("global")
    hypergraph = HypergraphUnitAnalyzer(failure_threshold=0.35)

    # â”€â”€ Level 18: Autopoietic Self-Healing & Fluid Dynamics â”€â”€
    fluid_pitch = FluidPitchAnalyzer(diffusion_sigma=3.0, control_threshold=0.6)
    ricci_flow = RicciFlowAnalyzer(alpha=0.5, history_size=50)

    # â”€â”€ Level 19: Stochastic Calculus & Hidden Regimes â”€â”€
    regime_sw = RegimeSwitcher(n_regimes=3, n_iter=50)
    sde_pricer = SDEPricer(dt=1.0, n_sim_paths=1000, min_edge=0.04)

    # â”€â”€ Level 20: The Singularity & Market Microstructure â”€â”€
    quantum_brain = QuantumBrain(n_qubits=4, n_layers=2, lr=0.01)
    hawkes = HawkesMomentumAnalyzer(match_duration=90.0)

    # â”€â”€ Level 20+: Survival Analysis & Fatigue Modeling â”€â”€
    survival = SurvivalEstimator()
    fatigue = FatigueEngine(base_intensity=0.5)

    # â”€â”€ Level 23: Chaos Theory & Rust-Powered Core â”€â”€
    chaos_filter = ChaosFilter(emb_dim=3, lag=1)
    homology = HomologyScanner(max_dim=1, max_edge=50.0)
    rust_engine = RustEngine()

    # â”€â”€ Level 24: AutoML & Synthetic Data â”€â”€
    automl = AutoMLEngine(generations=5, population_size=50, cv_folds=5)
    synth_trainer = SyntheticTrainer(noise_scale=0.03)
    fuzzy = FuzzyReasoningEngine()

    # â”€â”€ Level 25: Epistemic Uncertainty & Topology â”€â”€
    uncertainty_sep = UncertaintySeparator(n_models=10, n_classes=3)
    topo_mapper = TopologyMapper(n_cubes=10, overlap=0.3)

    # â”€â”€ Level 26: Active Inference & Probabilistic Programming â”€â”€
    prob_engine = ProbabilisticEngine(n_samples=2000, n_tune=1000)
    active_inf = ActiveInferenceAgent(modules=[
        "poisson", "lightgbm", "lstm", "rl_trader", "ensemble", "sentiment",
    ])
    mf_analyzer = MultifractalAnalyzer(q_min=-5, q_max=5, q_step=0.5)

    # â”€â”€ Level 27: Symbolic Discovery & Wavelet Denoising â”€â”€
    symbolic = SymbolicDiscovery(max_complexity=20, n_iterations=40)
    wavelet = WaveletDenoiser(wavelet="db4", level=4)

    # â”€â”€ Level 28: Volatility Clustering â”€â”€
    vol_analyzer = VolatilityAnalyzer(model_type="GARCH", p=1, q=1, ewma_span=30)

    # â”€â”€ Level 29: Particle Filter & Deep Logging â”€â”€
    particle_tracker = ParticleStrengthTracker(
        n_particles=1000, process_noise=0.02, observation_noise=0.15,
    )
    logger.success(f"Layer 3 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ LAYER 6: Graph Database (Neo4j) â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 6 â€“ Graph Intelligence baÅŸlatÄ±lÄ±yorâ€¦")
    neo4j_graph = Neo4jFootballGraph()
    neo4j_graph.connect()
    graph_rag = graph  # Tek GraphRAG instance â€“ Layer 2'de oluÅŸturuldu
    logger.success(f"Layer 6 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ LAYER 4: Risk & Ä°cra â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 4 â€“ Risk & Execution baÅŸlatÄ±lÄ±yorâ€¦")
    risk_solver = ConstrainedRiskSolver()
    covar = SystemicRiskCoVaR()
    backtester = VectorBacktester()
    bl_opt = BlackLittermanOptimizer()
    pnl = PnLStabilizer()
    quantizer = ModelQuantizer()
    jax_acc = JAXAccelerator()
    jit_acc = JITAccelerator()
    jit_acc.warmup()
    grpc_comm = GRPCCommunicator(use_grpc=False)  # In-process bus (gRPC opsiyonel)
    dist_core = DistributedCore(max_workers=8)
    dist_core.start()
    mimic = MimicEngine(persona="random")  # Anti-Ban â€“ insansÄ± davranÄ±ÅŸ
    q_annealer = QuantumAnnealer(
        bankroll=10000.0, max_bets=10, max_risk=0.15, max_iter=10000,
    )
    feedback_loop = HumanFeedbackLoop()
    self_healer = SelfHealingEngine(llm_backend="auto")
    telemetry = TelemetryTracer(service_name="quant-betting-bot")
    blind_strategy = BlindStrategyEngine()
    war_room = WarRoom(llm_backend="auto")
    validator = DataValidator()
    portfolio_opt = PortfolioOptimizer(
        initial_bankroll=10000.0, max_portfolio_risk=0.15,
    )
    scheduler = JobScheduler(timezone="Europe/Istanbul")
    genetic = GeneticOptimizer(population_size=100)
    hedge_calc = HedgeCalculator(min_profit_pct=0.01)

    fair_value = FairValueEngine(min_edge=0.04, kelly_fraction=0.25)

    # â”€â”€ Level 28: Stream Processing â”€â”€
    stream_proc = StreamProcessor(max_queue=10000, window_sec=60.0, n_workers=4)

    # â”€â”€ Level 11: Event Bus, Shadow Manager, Animator â”€â”€
    event_store = EventStore()
    event_bus = EventBus(store=event_store)
    replayer = ReplayEngine(event_bus)
    shadow = ShadowManager()
    animator = PlotAnimator()

    # Genetik Algoritma: daha Ã¶nce optimize edilmiÅŸ parametreleri yÃ¼kle
    optimized_params = genetic.load_config()
    if optimized_params:
        logger.info(f"[GA] Optimize parametre yÃ¼klendi: {len(optimized_params)} gen.")
    logger.success(f"Layer 4 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ LAYER 5: ArayÃ¼z & Raporlama â”€â”€
    _layer_t0 = _time.perf_counter()
    logger.info("Layer 5 â€“ Ops & Interface baÅŸlatÄ±lÄ±yorâ€¦")
    health = StrategyHealthReport()
    devil = DevilsAdvocate()
    threshold = ThresholdController()
    autodoc = AutoDocGenerator()
    orchestrator = WorkflowOrchestrator(
        max_retries=3, backoff_base=2.0, task_timeout=120.0,
    )
    # TÃ¼m modÃ¼lleri Prefect Task olarak kaydet
    orchestrator.register_modules({
        # Stage 1: Ingestion
        "scraper_agent": factory,
        "api_hijacker": hijacker,
        "stealth_browser": stealth,
        "vision_tracker": vision,
        "data_sources": factory,
        "metric_exporter": metrics,
        # Stage 2: Memory
        "db_manager": db,
        "feature_cache": cache,
        "lance_memory": lance,
        "neo4j_graph": neo4j_graph,
        "graph_rag": graph_rag,
        "smart_cache": smart_cache,
        "dvc_manager": dvc,
        # Stage 3: Quant
        "poisson_model": poisson,
        "dixon_coles": dixon_coles,
        "gradient_boosting": gb_model,
        "glm_model": glm,
        "bayesian_model": bayesian_model,
        "lstm_trend": lstm_trend,
        "rl_trader": rl,
        "rl_betting_agent": rl_agent,
        "ensemble_stacking": stacker,
        "time_decay": time_decay,
        "kalman_tracker": kalman,
        "prophet_seasonality": prophet_season,
        "regime_switcher": regime_sw,
        "sde_pricer": sde_pricer,
        "wavelet_denoiser": wavelet,
        "volatility_analyzer": vol_analyzer,
        "anomaly_detector": anomaly,
        "isolation_anomaly": iso_anomaly,
        "uncertainty_quantifier": uncertainty_q,
        "uncertainty_separator": uncertainty_sep,
        "chaos_filter": chaos_filter,
        "entropy_meter": entropy_meter,
        "topology_scanner": topo_scanner,
        "topology_mapper": topo_mapper,
        "homology_scanner": homology,
        "network_centrality": pass_network,
        "hypergraph_unit": hypergraph,
        "digital_twin_sim": digital_twin,
        "fluid_pitch": fluid_pitch,
        "fatigue_engine": fatigue,
        "causal_reasoner": causal_reason,
        "bsts_impact": bsts_impact,
        "fractal_analyzer": fractal,
        "multifractal_logic": mf_analyzer,
        "copula_risk": copula_risk,
        "survival_estimator": survival,
        "hawkes_momentum": hawkes,
        "ricci_flow": ricci_flow,
        "transport_metric": transport_metric,
        "automl_engine": automl,
        "synthetic_trainer": synth_trainer,
        "symbolic_discovery": symbolic,
        "fuzzy_reasoning": fuzzy,
        "probabilistic_engine": prob_engine,
        "active_inference": active_inf,
        "quantum_brain": quantum_brain,
        "nash_solver": nash_solver,
        "sentiment_analyzer": sentiment,
        "xai_explainer": xai,
        "clv_tracker": clv_tracker,
        "vector_engine": vector_engine,
        "transfer_learner": transfer_learner,
        "federated_trainer": fed_trainer,
        "elo_glicko": elo_system,
        # Stage 4: Risk
        "fair_value_engine": fair_value,
        "portfolio_optimizer": portfolio_opt,
        "pnl_stabilizer": pnl,
        "constrained_risk_solver": risk_solver,
        "systemic_risk_covar": covar,
        "black_litterman": bl_opt,
        "hedge_calculator": hedge_calc,
        "quantum_annealer": q_annealer,
        "evt_risk_manager": evt_risk,
        "genetic_optimizer": genetic,
        "shadow_manager": shadow,
        "blind_strategy": blind_strategy,
    })

    # â”€â”€ Layer 5: Utils & UI â”€â”€
    podcast = PodcastProducer()
    psycho = PsychoProfiler()
    query_assist = QueryAssistant()
    daily_brief = DailyBriefing(llm_backend="auto")
    flow_gen = DecisionFlowGenerator()
    agent_poll = AgentPollSystem(notifier=None)  # notifier boot sonrasÄ± atanÄ±r

    orchestrator.register_modules({
        "daily_briefing": daily_brief,
        "decision_flow_gen": flow_gen,
        "psycho_profiler": psycho,
        "strategy_health": health,
        "auto_doc_gen": autodoc,
        "telemetry_tracer": telemetry,
    })
    logger.info(
        f"[Orchestrator] {len(orchestrator._modules)} modÃ¼l kaydedildi."
    )
    logger.success(f"Layer 5 hazÄ±r âœ“ ({_time.perf_counter() - _layer_t0:.2f}s)")

    # â”€â”€ Veri versiyonlama snapshot â”€â”€
    dvc.snapshot("boot")

    _boot_elapsed = _time.perf_counter() - _boot_t0
    console.rule(f"[bold green]TÃœM KATMANLAR HAZIR â€“ {_boot_elapsed:.1f}s[/]")
    logger.info(f"[Boot] TÃ¼m katmanlar {_boot_elapsed:.1f} saniyede ayaÄŸa kalktÄ±")

    _boot_complete = True
    _signal_count = 0

    # â”€â”€ ModlarÄ± Ã§alÄ±ÅŸtÄ±r â”€â”€
    tasks: list[asyncio.Task] = []

    # Metrik sunucusu her zaman Ã§alÄ±ÅŸÄ±r
    tasks.append(asyncio.create_task(metrics.serve(), name="metrics"))

    if mode in ("live", "full"):
        tasks.append(asyncio.create_task(factory.run_live(shutdown=_shutdown_event), name="data_live"))
        tasks.append(asyncio.create_task(hijacker.listen(shutdown=_shutdown_event), name="hijacker"))

    if mode in ("pre", "full"):
        tasks.append(asyncio.create_task(factory.run_prematch(shutdown=_shutdown_event), name="data_pre"))

    # â”€â”€ Telegram Bildirim Motoru (her zaman aktif) â”€â”€
    from src.ui.telegram_mini_app import TelegramApp, TelegramNotifier
    from src.ui.telegram_chart_sender import TelegramChartSender
    from src.ui.human_in_the_loop import HumanInTheLoop
    notifier = TelegramNotifier(enabled=telegram)
    chart_sender = TelegramChartSender(notifier=notifier)
    hitl = HumanInTheLoop()
    live_dash = TelegramLiveDashboard(notifier=notifier)
    agent_poll._notifier = notifier  # Agent Poll'a Telegram notifier ata
    # Veri Kaynaklari Merkezi (layered free providers dahil)
    data_agg = DataSourceAggregator(db=db)

    # -- PRE-FETCH: Boot sirasinda cok kaynakli veri doldur --
    try:
        logger.info("[PreFetch] Boot sirasinda TUM SPORLAR verisi cekiliyor...")
        from datetime import datetime as _pf_dt
        _pf_unique = await data_agg.fetch_today()

        _pf_added = 0
        _pf_sport_counts: dict[str, int] = {}
        if _pf_unique:
            for item in _pf_unique:
                if isinstance(item, dict) and item.get("home_team") and item.get("away_team"):
                    ts = item.get("start_timestamp", 0)
                    try:
                        kickoff = _pf_dt.fromtimestamp(ts).isoformat() if ts else _pf_dt.utcnow().isoformat()
                    except Exception:
                        kickoff = _pf_dt.utcnow().isoformat()
                    sport = item.get("sport", "football")
                    db.upsert_match({
                        "match_id": item.get("match_id", f"{item['home_team']}_{item['away_team']}"),
                        "sport": sport,
                        "league": item.get("league", "Unknown"),
                        "country": item.get("country", ""),
                        "competition_type": item.get("competition_type", "league"),
                        "home_team": item["home_team"],
                        "away_team": item["away_team"],
                        "kickoff": kickoff,
                        "status": "upcoming",
                    })
                    _pf_added += 1
                    _pf_sport_counts[sport] = _pf_sport_counts.get(sport, 0) + 1
            sport_str = ", ".join(f"{k}={v}" for k, v in _pf_sport_counts.items())
            logger.success(f"[PreFetch] {_pf_added} etkinlik DB'ye eklendi ({sport_str})")
        if _pf_added == 0:
            logger.warning("[PreFetch] Coklu kaynaklar bos - CSV fallback deneniyor...")
            _csv_src = data_agg.csv
            _csv_data = await _csv_src.get_historical("super_lig", "2526")
            if not _csv_data:
                _csv_data = await _csv_src.get_historical("epl", "2526")
            if _csv_data:
                for item in _csv_data[-50:]:
                    if isinstance(item, dict) and item.get("home_team"):
                        db.upsert_match({
                            "match_id": item.get("match_id", f"csv_{item['home_team']}_{item['away_team']}"),
                            "sport": "football",
                            "league": item.get("league", "super_lig"),
                            "home_team": item["home_team"],
                            "away_team": item["away_team"],
                            "kickoff": _pf_dt.utcnow().isoformat(),
                            "status": "upcoming",
                            "home_odds": item.get("home_odds"),
                            "draw_odds": item.get("draw_odds"),
                            "away_odds": item.get("away_odds"),
                        })
                logger.success(f"[PreFetch] CSV'den {min(len(_csv_data), 50)} mac yuklendi.")
    except Exception as _pf_err:
        logger.warning(f"[PreFetch] Hata: {_pf_err}")

    # Scraper Agent â€“ Circuit Breaker korumalÄ±
    scraper = ScraperAgent(db=db, notifier=notifier, cb_registry=cb_registry)
    tasks.append(asyncio.create_task(scraper.run_all(shutdown=_shutdown_event), name="scraper"))

    # Kadro Ä°zleme AjanÄ±
    lineup_mon = LineupMonitor(db=db, notifier=notifier, cb_registry=cb_registry)
    tasks.append(asyncio.create_task(lineup_mon.watch(shutdown=_shutdown_event), name="lineup_monitor"))
    async def _multisport_refresh_loop():
        """TÃ¼m branÅŸlar iÃ§in etkinlikleri periyodik yenile."""
        while not _shutdown_event.is_set():
            try:
                events = await data_agg.fetch_today()
                if events:
                    logger.info(f"[MultiSport] {len(events)} etkinlik gÃ¼ncellendi.")
            except Exception as e:
                logger.debug(f"[MultiSport] {type(e).__name__}: {e}")
            await asyncio.sleep(180)

    tasks.append(asyncio.create_task(_multisport_refresh_loop(), name="multisport_refresh"))

    if telegram:
        from src.ingestion.voice_interrogator import VoiceInterrogator
        # Admin paneli
        admin = TelegramAdmin(
            notifier=notifier, db=db, scraper=scraper,
            scheduler=scheduler, cache=smart_cache,
            genetic=genetic, stacker=stacker,
        )
        tg = TelegramApp(
            threshold_ctrl=threshold, notifier=notifier,
            db=db, cb_registry=cb_registry,
            clv_tracker=clv_tracker, chart_sender=chart_sender,
            hitl=hitl, portfolio=portfolio_opt,
        )
        # Admin komutlarÄ±nÄ± Telegram app'ine kaydet
        admin.register_handlers(tg._commands if hasattr(tg, "_commands") else {})
        voice = VoiceInterrogator()
        tasks.append(asyncio.create_task(tg.start(shutdown=_shutdown_event), name="telegram"))
        tasks.append(asyncio.create_task(voice.listen(shutdown=_shutdown_event), name="voice"))
    else:
        tg = None

    if dashboard:
        from src.ui.dashboard_tui import DashboardTUI
        dash = DashboardTUI()
        tasks.append(asyncio.create_task(dash.run(shutdown=_shutdown_event), name="dashboard"))

    # â”€â”€ Telegram Web App (Mini App) â”€â”€
    from src.ui.webapp_server import start_webapp_async
    tasks.append(asyncio.create_task(
        start_webapp_async(
            port=8080, db=db, portfolio=portfolio_opt,
            kalman=kalman, uncertainty=uncertainty_q,
            hedge_calc=hedge_calc,
            shutdown=_shutdown_event,
        ),
        name="webapp",
    ))

    # â”€â”€ APScheduler gÃ¶rev zamanlayÄ±cÄ± â”€â”€
    scheduler.add_cron("fikstur_cek", scraper._scrape_mackolik, hour=11, minute=0)
    scheduler.add_interval("oran_kontrol", scraper._scrape_sofascore, minutes=10)
    scheduler.add_cron("gunluk_rapor", health.generate_pdf, hour=23, minute=30)
    scheduler.add_cron("lstm_train", lambda: logger.info("[Scheduler] LSTM retrain tetiklendi"), hour=3, minute=0)
    scheduler.add_cron("veri_topla", lambda: logger.info("[Scheduler] Veri toplama tetiklendi"), hour=10, minute=0)  # GÃ¼n baÅŸÄ± toplu veri
    scheduler.add_cron("prophet_cache_temizle", prophet_season.clear_cache, hour=6, minute=0)  # GÃ¼nlÃ¼k cache sÄ±fÄ±rla
    scheduler.add_cron("rl_train", rl_agent.save, hour=4, minute=0)  # RL model checkpoint
    scheduler.add_cron("gunluk_brifing", lambda: _send_daily_briefing(daily_brief, db, portfolio_opt, health, notifier, super_log=super_log, guardian=guardian), hour=9, minute=0)  # CEO brifing
    scheduler.add_cron("log_rotate", log_rotator.rotate, hour=2, minute=0)  # Gece 02:00 log rotasyonu
    tasks.append(asyncio.create_task(scheduler.start(), name="scheduler"))

    # Ana analiz dÃ¶ngÃ¼sÃ¼
    tasks.append(asyncio.create_task(
        _analysis_loop(
            shutdown=_shutdown_event,
            db=db, cache=cache, smart_cache=smart_cache,
            lance=lance, bridge=bridge, graph=graph,
            dvc=dvc, vision=vision, gcn=gcn, podcast=podcast,
            prob_engine=prob_engine, evt_scanner=evt_scanner, causal=causal,
            conformal=conformal, path_sig=path_sig, jump_diff=jump_diff,
            geometric=geometric, mtl=mtl, kan=kan, rl=rl,
            poisson=poisson, monte_carlo=monte_carlo,
            elo_system=elo_system, sentiment=sentiment, anomaly=anomaly,
            dixon_coles=dixon_coles, time_decay=time_decay,
            team_vix=team_vix, gb_model=gb_model, feat_eng=feat_eng,
            bayesian_model=bayesian_model, npxg_filter=npxg_filter,
            clv_tracker=clv_tracker, corr_matrix=corr_matrix,
            chart_sender=chart_sender,
            stacker=stacker, hitl=hitl, portfolio_opt=portfolio_opt,
            lstm_trend=lstm_trend, xai=xai, news_rag=news_rag,
            hedge_calc=hedge_calc, live_dash=live_dash,
            glm=glm, fair_value=fair_value, data_agg=data_agg,
            neo4j_graph=neo4j_graph, pass_network=pass_network,
            prophet_season=prophet_season,
            uncertainty_q=uncertainty_q, kalman=kalman,
            rl_agent=rl_agent, vector_engine=vector_engine,
            copula_risk=copula_risk, scenario_sim=scenario_sim,
            iso_anomaly=iso_anomaly,
            vision_live=vision_live, causal_reason=causal_reason,
            topo_scanner=topo_scanner,
            nash_solver=nash_solver, entropy_meter=entropy_meter,
            jit_acc=jit_acc, psycho=psycho,
            evt_risk=evt_risk, digital_twin=digital_twin,
            grpc_comm=grpc_comm, query_assist=query_assist,
            bsts_impact=bsts_impact, fractal=fractal, dist_core=dist_core,
            transfer_learner=transfer_learner, transport_metric=transport_metric,
            mimic=mimic,
            fed_trainer=fed_trainer, hypergraph=hypergraph,
            q_annealer=q_annealer, feedback_loop=feedback_loop,
            fluid_pitch=fluid_pitch, ricci_flow=ricci_flow,
            self_healer=self_healer,
            regime_sw=regime_sw, sde_pricer=sde_pricer,
            telemetry=telemetry,
            quantum_brain=quantum_brain, hawkes=hawkes,
            survival=survival, fatigue=fatigue,
            chaos_filter=chaos_filter, homology=homology,
            rust_engine=rust_engine,
            automl=automl, synth_trainer=synth_trainer, fuzzy=fuzzy,
            uncertainty_sep=uncertainty_sep, topo_mapper=topo_mapper,
            graph_rag=graph_rag,
            active_inf=active_inf,
            mf_analyzer=mf_analyzer,
            symbolic=symbolic, wavelet=wavelet, flow_gen=flow_gen,
            vol_analyzer=vol_analyzer, stream_proc=stream_proc,
            particle_tracker=particle_tracker, super_log=super_log,
            agent_poll=agent_poll,
            blind_strategy=blind_strategy, war_room=war_room,
            event_bus=event_bus, shadow=shadow, animator=animator,
            validator=validator, notifier=notifier,
            risk_solver=risk_solver, covar=covar, bl_opt=bl_opt,
            pnl=pnl, jax_acc=jax_acc, quantizer=quantizer,
            health=health, devil=devil,
            # Yeni modÃ¼ller (v2)
            guardian=guardian, regime_kelly=regime_kelly,
            strategy_evolver=strategy_evolver,
            fisher_geo=fisher_geo, philo_engine=philo_engine,
            cockpit=cockpit,
        ),
        name="analysis",
    ))

    # Bekle â€“ sinyal gelene kadar
    await _shutdown_event.wait()
    logger.info("Shutdown event tetiklendi â€“ gÃ¼venli kapanÄ±ÅŸ baÅŸlatÄ±lÄ±yorâ€¦")

    async def _await_with_timeout(name: str, coro, timeout_s: float = 8.0):
        try:
            await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(f"[Shutdown] {name} timeout ({timeout_s:.0f}s)")
        except Exception as e:
            logger.debug(f"[Shutdown] {name} hatasÄ±: {type(e).__name__}: {e}")

    await _await_with_timeout("scheduler.stop", scheduler.stop())
    with contextlib.suppress(Exception):
        neo4j_graph.close()
    with contextlib.suppress(Exception):
        event_store.close()
    with contextlib.suppress(Exception):
        shadow._save_state()
    with contextlib.suppress(Exception):
        dist_core.shutdown()
    if hasattr(factory, "close"):
        await _await_with_timeout("data_factory.close", factory.close())

    for t in tasks:
        t.cancel()
    done, pending = await asyncio.wait(tasks, timeout=10)
    if pending:
        pending_names = [p.get_name() for p in pending]
        logger.warning(f"[Shutdown] ZamanÄ±nda kapanmayan task'lar: {pending_names}")
    for t in done:
        with contextlib.suppress(Exception):
            _ = t.result()
    console.rule("[bold red]SÄ°STEM DURDURULDU[/]")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  ANA ANALÄ°Z DÃ–NGÃœSÃœ
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
async def _analysis_loop(*, shutdown: asyncio.Event, **modules):
    """Her dÃ¶ngÃ¼de: veri Ã§ek â†’ doÄŸrula â†’ Ã¶zellik Ã¼ret â†’ model Ã§alÄ±ÅŸtÄ±r â†’ risk hesapla â†’ sinyal Ã¼ret â†’ bildir."""
    db             = modules["db"]
    cache          = modules["cache"]
    smart_cache    = modules["smart_cache"]
    lance          = modules["lance"]
    graph          = modules["graph"]
    dvc            = modules["dvc"]
    podcast        = modules["podcast"]
    prob_engine    = modules["prob_engine"]
    evt_scanner    = modules["evt_scanner"]
    causal         = modules["causal"]
    conformal      = modules["conformal"]
    path_sig       = modules["path_sig"]
    jump_diff      = modules["jump_diff"]
    geometric      = modules["geometric"]
    mtl            = modules["mtl"]
    kan            = modules["kan"]
    rl             = modules["rl"]
    poisson        = modules["poisson"]
    monte_carlo    = modules["monte_carlo"]
    elo_system     = modules["elo_system"]
    sentiment      = modules["sentiment"]
    anomaly        = modules["anomaly"]
    dixon_coles    = modules["dixon_coles"]
    time_decay     = modules["time_decay"]
    team_vix       = modules["team_vix"]
    gb_model       = modules["gb_model"]
    feat_eng       = modules["feat_eng"]
    bayesian_model = modules["bayesian_model"]
    npxg_filter    = modules["npxg_filter"]
    clv_tracker    = modules["clv_tracker"]
    corr_matrix    = modules["corr_matrix"]
    chart_sender   = modules["chart_sender"]
    stacker        = modules["stacker"]
    hitl           = modules["hitl"]
    portfolio_opt  = modules["portfolio_opt"]
    lstm_trend     = modules["lstm_trend"]
    xai            = modules["xai"]
    news_rag       = modules["news_rag"]
    hedge_calc     = modules["hedge_calc"]
    live_dash      = modules["live_dash"]
    glm            = modules["glm"]
    fair_value     = modules["fair_value"]
    data_agg       = modules["data_agg"]
    neo4j_graph    = modules["neo4j_graph"]
    pass_network   = modules["pass_network"]
    prophet_season = modules["prophet_season"]
    uncertainty_q  = modules["uncertainty_q"]
    kalman         = modules["kalman"]
    rl_agent       = modules["rl_agent"]
    vector_engine  = modules["vector_engine"]
    copula_risk    = modules["copula_risk"]
    scenario_sim   = modules["scenario_sim"]
    iso_anomaly    = modules["iso_anomaly"]
    vision_live    = modules["vision_live"]
    causal_reason  = modules["causal_reason"]
    topo_scanner   = modules["topo_scanner"]
    nash_solver    = modules["nash_solver"]
    entropy_meter  = modules["entropy_meter"]
    jit_acc        = modules["jit_acc"]
    psycho         = modules["psycho"]
    evt_risk       = modules["evt_risk"]
    digital_twin   = modules["digital_twin"]
    grpc_comm      = modules["grpc_comm"]
    query_assist   = modules["query_assist"]
    bsts_impact    = modules["bsts_impact"]
    fractal        = modules["fractal"]
    dist_core      = modules["dist_core"]
    event_bus      = modules["event_bus"]
    shadow         = modules["shadow"]
    animator       = modules["animator"]
    validator      = modules["validator"]
    notifier       = modules["notifier"]
    risk_solver    = modules["risk_solver"]
    covar          = modules["covar"]
    bl_opt         = modules["bl_opt"]
    pnl            = modules["pnl"]
    jax_acc        = modules["jax_acc"]
    health         = modules["health"]
    devil          = modules["devil"]

    graph_rag      = modules.get("graph_rag", graph)
    stream_proc    = modules.get("stream_proc")
    super_log      = modules.get("super_log")

    # Yeni modÃ¼ller (v2)
    guardian        = modules.get("guardian")
    regime_kelly    = modules.get("regime_kelly")
    strategy_evolver = modules.get("strategy_evolver")
    fisher_geo     = modules.get("fisher_geo")
    philo_engine   = modules.get("philo_engine")
    cockpit        = modules.get("cockpit")

    # â”€â”€ Eksik modÃ¼ller â€“ _analysis_loop scope'unda gerekli (NameError dÃ¼zeltmesi) â”€â”€
    fuzzy           = modules.get("fuzzy")
    vol_analyzer    = modules.get("vol_analyzer")
    ricci_flow      = modules.get("ricci_flow")
    mimic           = modules.get("mimic")
    blind_strategy  = modules.get("blind_strategy")
    chaos_filter    = modules.get("chaos_filter")
    hawkes          = modules.get("hawkes")
    survival        = modules.get("survival")
    fatigue         = modules.get("fatigue")
    quantum_brain   = modules.get("quantum_brain")
    homology        = modules.get("homology")
    rust_engine     = modules.get("rust_engine")
    automl          = modules.get("automl")
    synth_trainer   = modules.get("synth_trainer")
    uncertainty_sep = modules.get("uncertainty_sep")
    topo_mapper     = modules.get("topo_mapper")
    mf_analyzer     = modules.get("mf_analyzer")
    symbolic        = modules.get("symbolic")
    wavelet         = modules.get("wavelet")
    flow_gen        = modules.get("flow_gen")
    particle_tracker = modules.get("particle_tracker")
    agent_poll      = modules.get("agent_poll")
    war_room        = modules.get("war_room")
    active_inf      = modules.get("active_inf")
    fed_trainer     = modules.get("fed_trainer")
    hypergraph      = modules.get("hypergraph")
    q_annealer      = modules.get("q_annealer")
    feedback_loop   = modules.get("feedback_loop")
    fluid_pitch     = modules.get("fluid_pitch")
    self_healer     = modules.get("self_healer")
    regime_sw       = modules.get("regime_sw")
    sde_pricer      = modules.get("sde_pricer")
    telemetry       = modules.get("telemetry")
    transfer_learner = modules.get("transfer_learner")
    transport_metric = modules.get("transport_metric")
    quantizer       = modules.get("quantizer")

    # Eksik import'lar â€“ _analysis_loop scope'unda gerekli
    from src.core.event_bus import Event
    from src.quant.fuzzy_reasoning import FuzzyInput
    from src.core.regime_kelly import RegimeState

    import time as _time
    import math as _math

    def _sg(d: dict, key: str, default=0.0):
        """Safe get: None/NaN/inf -> default."""
        v = d.get(key)
        if v is None:
            return default
        try:
            f = float(v)
            return f if _math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    cycle = 0
    _cycle_times: list[float] = []
    while not shutdown.is_set():
        cycle += 1
        _cycle_start = _time.perf_counter()
        logger.info(f"â•â•â• Analiz DÃ¶ngÃ¼sÃ¼ #{cycle} â•â•â•")
        try:
            # â”€â”€ 1) GÃ¼ncel maÃ§ verisini al â”€â”€
            matches = db.get_upcoming_matches()
            if matches.is_empty():
                # DB boÅŸsa: tÃ¼m sporlardan Sofascore'dan Ã§ekmeyi dene
                logger.info("DB'de maÃ§ yok â€“ Sofascore'dan TÃœM SPORLAR Ã§ekiliyorâ€¦")
                try:
                    from src.ingestion.data_sources import SofascoreHiddenAPI as _SsAPI
                    _ss = _SsAPI()
                    _all_ev = await _ss.get_all_scheduled()
                    _live_ev = await _ss.get_all_live()
                    _all_ev = _all_ev + _live_ev
                    if _all_ev:
                        from datetime import datetime as _dt_cls
                        for item in _all_ev:
                            if isinstance(item, dict) and item.get("home_team"):
                                ts = item.get("start_timestamp", 0)
                                try:
                                    kickoff = _dt_cls.fromtimestamp(ts).isoformat() if ts else _dt_cls.utcnow().isoformat()
                                except Exception:
                                    kickoff = _dt_cls.utcnow().isoformat()
                                db.upsert_match({
                                    "match_id": item.get("match_id", f"{item['home_team']}_{item['away_team']}"),
                                    "sport": item.get("sport", "football"),
                                    "league": item.get("league", "Unknown"),
                                    "country": item.get("country", ""),
                                    "home_team": item["home_team"],
                                    "away_team": item["away_team"],
                                    "kickoff": kickoff,
                                    "status": "upcoming",
                                })
                        logger.info(f"[AutoFetch] {len(_all_ev)} etkinlik (Ã§ok sporlu) eklendi.")
                        matches = db.get_upcoming_matches()
                except Exception as _af_err:
                    logger.debug(f"[AutoFetch] Sofascore hatasÄ±: {_af_err}")

            if matches.is_empty():
                # HÃ¢lÃ¢ boÅŸsa: tÃ¼m maÃ§larÄ± al (status filter kaldÄ±r)
                try:
                    _all_matches = db.query("SELECT * FROM matches ORDER BY kickoff DESC LIMIT 100")
                    if not _all_matches.is_empty():
                        logger.info(f"[Fallback] DB'den {_all_matches.height} tarihsel etkinlik alÄ±ndÄ±.")
                        matches = _all_matches
                except Exception:
                    pass

            if matches.is_empty():
                logger.info("HiÃ§bir kaynaktan etkinlik bulunamadÄ± â€“ 120 s bekleniyor.")
                await asyncio.sleep(120)
                continue

            # â”€â”€ 1b) Pydantic veri doÄŸrulama (kirli veriyi filtrele) â”€â”€
            match_dicts = matches.to_dicts()
            # datetime â†’ ISO string dÃ¶nÃ¼ÅŸÃ¼mÃ¼
            for m in match_dicts:
                if 'kickoff' in m and hasattr(m['kickoff'], 'isoformat'):
                    m['kickoff'] = m['kickoff'].isoformat()
            
            validated_rows = validator.validate_batch(match_dicts, schema="match")
            if not validated_rows:
                logger.warning("TÃ¼m maÃ§ verisi doÄŸrulamadan reddedildi.")
                await asyncio.sleep(30)
                continue
            import polars as pl
            matches = pl.DataFrame(validated_rows)
            logger.debug(f"DoÄŸrulama: {len(validated_rows)} maÃ§ geÃ§erli.")

            # DÃ¶ngÃ¼ sÃ¼resini kontrol altÄ±nda tutmak iÃ§in iÅŸlenecek maÃ§ sayÄ±sÄ±nÄ± sÄ±nÄ±rla.
            max_matches = max(1, int(os.getenv("ANALYSIS_MAX_MATCHES", "12")))
            if getattr(matches, "height", 0) > max_matches:
                matches = matches.head(max_matches)
                logger.info(f"[Perf] MaÃ§ seti {max_matches} ile sÄ±nÄ±rlandÄ±rÄ±ldÄ±.")

            # â”€â”€ 2) Feature mÃ¼hendisliÄŸi (SmartCache L2 RAM â†’ L3 disk) â”€â”€
            features = smart_cache.get_or_compute(
                f"features_cycle_{cycle}",
                lambda: cache.get_or_compute("features", lambda: db.build_feature_matrix(matches)),
                persist=False,
            )

            # â”€â”€ 2b) Zaman aÄŸÄ±rlÄ±klandÄ±rma â”€â”€
            features = time_decay.apply_to_dataframe(features, date_col="kickoff")

            # â”€â”€ 2c) npxG filtresi (Non-Penalty xG dÃ¼zeltmesi) â”€â”€
            try:
                feature_dicts = features.to_dicts()
                feature_dicts = [npxg_filter.filter_features(f) for f in feature_dicts]
                import polars as pl
                features = pl.DataFrame(feature_dicts)
            except Exception as e:
                logger.debug(f"[Guardian] npxg_filter: {type(e).__name__}: {e}")

            # â”€â”€ 3) JAX ile hÄ±zlandÄ±rÄ±lmÄ±ÅŸ hesaplamalar â”€â”€
            features_acc = jax_acc.accelerate(features)

            # â”€â”€ 4) Model tahminleri â”€â”€
            prob_preds  = prob_engine.predict_for_dataframe(matches)
            mtl_preds   = mtl.predict(features_acc)
            kan_preds   = kan.predict(features_acc)
            poisson_preds = poisson.predict_for_dataframe(features_acc)
            mc_preds    = monte_carlo.predict_for_dataframe(features_acc)
            elo_preds   = elo_system.predict_for_dataframe(features_acc)

            # â”€â”€ 4b) Dixon-Coles dÃ¼zeltilmiÅŸ Poisson â”€â”€
            dc_preds    = dixon_coles.predict_for_dataframe(features_acc)

            # â”€â”€ 4c) Gradient Boosting (LightGBM) ML tahmini â”€â”€
            gb_preds    = gb_model.predict(features_acc)

            # â”€â”€ 4d-1) GLM DÃ¼zeltilmiÅŸ xG â”€â”€
            glm_preds = glm.predict_for_dataframe(features_acc)

            # â”€â”€ 4d) Bayesyen HiyerarÅŸik Model â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    bayes_pred = bayesian_model.predict(home, away)
                    if bayes_pred.get("shrinkage_home", 1) > 0.5:
                        logger.debug(
                            f"[Bayesian] {home}: prior aÄŸÄ±rlÄ±klÄ± "
                            f"(shrinkage={bayes_pred['shrinkage_home']:.2f}, "
                            f"n={bayes_pred['home_matches']})"
                        )

            # â”€â”€ 4e) TakÄ±m Volatilite analizi â”€â”€
            for row in matches.iter_rows(named=True):
                for team_key in ("home_team", "away_team"):
                    team = row.get(team_key, "")
                    if team:
                        history = db.get_team_history(team) if hasattr(db, "get_team_history") else []
                        team_vix.calculate(team, history)

            # â”€â”€ 4f) LSTM Momentum analizi â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    home_hist = db.get_team_history(home) if hasattr(db, "get_team_history") else []
                    away_hist = db.get_team_history(away) if hasattr(db, "get_team_history") else []
                    lstm_result = lstm_trend.predict_for_match(home, away, home_hist, away_hist)

                    # Stacking meta-modeline LSTM sinyali ekle
                    mid = row.get("match_id", f"{home}_{away}")
                    stacker.add_base_prediction(
                        "lstm", mid,
                        lstm_result["prob_home"], lstm_result["prob_draw"],
                        lstm_result["prob_away"],
                        confidence=lstm_result["confidence"],
                    )

            # â”€â”€ 4g) RAG Haber Sentiment analizi (LLM) â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    try:
                        rag_result = await news_rag.analyze_match(home, away)
                        mid = row.get("match_id", f"{home}_{away}")
                        # Sentiment farkÄ±nÄ± stacker'a ekle
                        s_home = rag_result.get("home_sentiment", 0.5)
                        s_away = rag_result.get("away_sentiment", 0.5)
                        s_diff = s_home - s_away
                        stacker.add_base_prediction(
                            "sentiment", mid,
                            0.33 + s_diff * 0.15,
                            0.34 - abs(s_diff) * 0.05,
                            0.33 - s_diff * 0.15,
                            confidence=rag_result.get("confidence", 0.3),
                        )
                    except Exception as e:
                        logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 4h) Network Centrality â€“ Eksik oyuncu etkisi â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    # Pas aÄŸÄ± verisi yÃ¼kle (varsa)
                    home_match = db.get_pass_network(home) if hasattr(db, "get_pass_network") else None
                    away_match = db.get_pass_network(away) if hasattr(db, "get_pass_network") else None
                    if home_match:
                        pass_network.load_passes(home, home_match)
                    elif hasattr(db, "get_lineup") and db.get_lineup(home):
                        pass_network.load_from_match_data(home, {"lineup": db.get_lineup(home)})
                    if away_match:
                        pass_network.load_passes(away, away_match)
                    elif hasattr(db, "get_lineup") and db.get_lineup(away):
                        pass_network.load_from_match_data(away, {"lineup": db.get_lineup(away)})

                    # Eksik oyuncu penalty hesapla
                    home_missing = row.get("home_missing", []) or []
                    away_missing = row.get("away_missing", []) or []
                    if home_missing:
                        home_penalty = pass_network.combined_penalty(home, home_missing)
                        logger.info(f"[Network] {home} eksik oyuncu penalty: {home_penalty:.2%}")
                    if away_missing:
                        away_penalty = pass_network.combined_penalty(away, away_missing)
                        logger.info(f"[Network] {away} eksik oyuncu penalty: {away_penalty:.2%}")

            # â”€â”€ 4i) Prophet Mevsimsellik â€“ Tarihsel dÃ¶nem analizi â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    season_analysis = prophet_season.analyze_match(
                        home, away, row.get("kickoff", None),
                    )
                    if season_analysis["avoid_match"]:
                        logger.warning(
                            f"[Prophet] âš ï¸ {home} vs {away}: "
                            f"Her iki takÄ±m da negatif sezonda â€“ AVOID sinyali!"
                        )
                    if season_analysis["insight"]:
                        logger.info(f"[Prophet] {home} vs {away}: {season_analysis['insight']}")

            # â”€â”€ 4j) Neo4j Graph â€“ MaÃ§ iliÅŸkilerini kaydet â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    neo4j_graph.create_match(row)
                    # Hakem yanlÄ±lÄ±k analizi
                    ref = row.get("referee", "")
                    if ref:
                        ref_bias = neo4j_graph.query_referee_bias(ref)
                        if ref_bias.get("is_strict_away"):
                            logger.info(
                                f"[Graph] Hakem {ref}: deplasmana sert "
                                f"(ort kart farkÄ±: {ref_bias['away_bias']:+.1f})"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 4k) Kalman Filtresi â€“ Dinamik gÃ¼Ã§ gÃ¼ncelleme â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    kalman_pred = kalman.predict_match(home, away)
                    if kalman_pred.get("reliability", 0) > 0.3:
                        mid = row.get("match_id", f"{home}_{away}")
                        stacker.add_base_prediction(
                            "kalman", mid,
                            kalman_pred["prob_home"],
                            kalman_pred["prob_draw"],
                            kalman_pred["prob_away"],
                            confidence=kalman_pred["reliability"],
                        )

            # â”€â”€ 4l) Vector Engine â€“ Tarihsel ikiz maÃ§larÄ± bul â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    similarity = vector_engine.find_similar(row, k=50)
                    if similarity.suggested_confidence > 0.60:
                        logger.info(
                            f"[Vector] {row.get('home_team','')} vs {row.get('away_team','')}: "
                            f"Tarihsel benzerlik â†’ {similarity.suggested_result} "
                            f"({similarity.suggested_confidence:.0%}), "
                            f"Ãœ2.5: {similarity.over_25_rate:.0%}"
                        )
                    # Ä°ndekse ekle
                    vector_engine.add_match(row)
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5) Ä°statistiksel analiz â”€â”€
            tail_risk    = evt_scanner.scan(features_acc)
            causal_edges = causal.discover(features_acc)
            intervals    = conformal.predict_intervals(features_acc)
            sig_features = path_sig.extract(features_acc)
            jumps        = jump_diff.detect(features_acc)
            geo_potential = geometric.compute_potential(features_acc)

            # â”€â”€ 5b) Anomali tespiti (Dropping Odds / Z-Score) â”€â”€
            odds_alerts = anomaly.scan_all_matches(db)
            if odds_alerts:
                logger.warning(f"[ANOMALI] {len(odds_alerts)} dropping odds tespit edildi!")
                for alert in odds_alerts[:3]:
                    await notifier.send_anomaly_alert(alert)

            # â”€â”€ 5b-2) Isolation Forest â€“ Tuzak/Åike taramasÄ± â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    iso_alerts = iso_anomaly.scan(row, match_id=row.get("match_id", ""))
                    for ia in iso_alerts:
                        await event_bus.emit(Event(
                            event_type="anomaly_detected",
                            source="isolation_forest",
                            match_id=ia.match_id,
                            data={"type": ia.alert_type, "severity": ia.severity,
                                  "score": ia.score, "description": ia.description},
                        ))
                        if ia.is_blacklisted:
                            await notifier.send(
                                f"ğŸš¨ <b>TUZAK TESPÄ°T:</b> {ia.match_id}\n"
                                f"{ia.description}\n"
                                f"<b>Aksiyon:</b> {ia.action}"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-3) Vision Live â€“ CanlÄ± maÃ§lar momentum kontrolÃ¼ â”€â”€
            for mid in vision_live.active_matches:
                momentum_rpt = vision_live.get_momentum_report(mid)
                if momentum_rpt.signal in ("GOAL_SMELL", "HIGH_PRESSURE"):
                    logger.warning(
                        f"[Vision] {mid}: {momentum_rpt.signal} "
                        f"(baskÄ±={momentum_rpt.avg_pressure:.0%}, "
                        f"ceza_sahasÄ±={momentum_rpt.ball_in_penalty_area_pct:.0%})"
                    )
                    await event_bus.emit(Event(
                        event_type="vision_signal",
                        source="vision_live",
                        match_id=mid,
                        data={
                            "signal": momentum_rpt.signal,
                            "pressure": momentum_rpt.avg_pressure,
                            "dominant": momentum_rpt.dominant_side,
                        },
                        cycle=cycle,
                    ))

            # â”€â”€ 5b-4) Causal Reasoning â€“ Nedensellik analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    causal_factors = causal_reason.get_match_causal_factors(row)
                    for cf in causal_factors:
                        if cf.is_significant:
                            logger.info(
                                f"[Causal] {row.get('home_team','')} vs "
                                f"{row.get('away_team','')}: "
                                f"{cf.treatment} â†’ {cf.outcome} "
                                f"(ATE={cf.ate:+.3f}, p={cf.p_value:.3f})"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-5) Nash Game Theory â€“ Oyun Teorisi analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    m_probs = {
                        "prob_home": row.get("prob_home", 0.33),
                        "prob_draw": row.get("prob_draw", 0.33),
                        "prob_away": row.get("prob_away", 0.34),
                    }
                    m_odds = {
                        "home": _sg(row, "home_odds", 2.0),
                        "draw": _sg(row, "draw_odds", 3.5),
                        "away": _sg(row, "away_odds", 4.0),
                    }
                    nash_analysis = nash_solver.analyze_match(
                        m_probs, m_odds, match_id=row.get("match_id", ""),
                    )
                    if nash_analysis.expected_value > 0:
                        logger.info(
                            f"[Nash] {row.get('home_team','')} vs "
                            f"{row.get('away_team','')}: "
                            f"Optimal={nash_analysis.optimal_action} "
                            f"(EV={nash_analysis.expected_value:+.1f}, "
                            f"exploit={nash_analysis.equilibrium.exploitability:.0%})"
                        )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-6) Entropy Meter â€“ Bilgi fiziÄŸi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    ent_report = entropy_meter.analyze_match(
                        match_id=row.get("match_id", ""),
                        model_probs={
                            "prob_home": row.get("prob_home", 0.33),
                            "prob_draw": row.get("prob_draw", 0.33),
                            "prob_away": row.get("prob_away", 0.34),
                        },
                        market_odds={
                            "home": _sg(row, "home_odds", 2.0),
                            "draw": _sg(row, "draw_odds", 3.5),
                            "away": _sg(row, "away_odds", 4.0),
                        },
                        home_team=row.get("home_team", ""),
                        away_team=row.get("away_team", ""),
                    )
                    if ent_report.kill_switch:
                        logger.warning(
                            f"[Entropy] KILL SWITCH: {row.get('match_id','')} â†’ "
                            f"{ent_report.match_entropy:.2f} bit (KAOTÄ°K)"
                        )
                    elif ent_report.kl_divergence > 0.10:
                        logger.info(
                            f"[Entropy] {row.get('home_team','')} vs "
                            f"{row.get('away_team','')}: "
                            f"H={ent_report.match_entropy:.2f} bit, "
                            f"KL={ent_report.kl_divergence:.3f} "
                            f"({ent_report.chaos_level})"
                        )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-7) Digital Twin â€“ SayÄ±sal Ä°kiz maÃ§ simÃ¼lasyonu â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    mid = row.get("match_id", "")
                    home_name = row.get("home_team", "Unknown")
                    away_name = row.get("away_team", "Unknown")
                    home_q = _sg(row, "home_rating", 70)
                    away_q = _sg(row, "away_rating", 70)
                    home_squad = digital_twin.generate_team(home_name, quality=home_q)
                    away_squad = digital_twin.generate_team(away_name, quality=away_q)
                    twin_report = digital_twin.simulate_match(
                        mid, home_squad, away_squad, n_sims=200,
                    )
                    if twin_report.prob_home > 0:
                        logger.info(
                            f"[Twin] {home_name} vs {away_name}: "
                            f"H={twin_report.prob_home:.0%} D={twin_report.prob_draw:.0%} "
                            f"A={twin_report.prob_away:.0%} "
                            f"(skor={twin_report.most_common_score}, "
                            f"Ãœ2.5={twin_report.prob_over25:.0%})"
                        )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-8) Fractal â€“ Hurst Exponent lig rejim analizi â”€â”€
            try:
                league_results = db.get_recent_results(limit=100) if hasattr(db, "get_recent_results") else []
                if league_results:
                    goals_series = [
                        r.get("home_goals", 0) + r.get("away_goals", 0)
                        for r in league_results
                    ]
                    hurst_result = fractal.compute_hurst(goals_series)
                    if hurst_result.regime != "random":
                        logger.info(
                            f"[Fractal] Lig rejimi: {hurst_result.regime} "
                            f"(H={hurst_result.hurst:.3f}, "
                            f"strateji={hurst_result.recommended_strategy[:60]})"
                        )
            except Exception as e:
                logger.debug(f"[Guardian] fractal: {type(e).__name__}: {e}")

            # â”€â”€ 5b-9) BSTS â€“ YapÄ±sal kÄ±rÄ±lma tespiti â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    if home:
                        perf_data = db.get_team_performance(home, limit=30) if hasattr(db, "get_team_performance") else []
                        if len(perf_data) >= 15:
                            breaks = bsts_impact.detect_breaks(perf_data)
                            for bp in breaks:
                                if bp.is_significant:
                                    logger.info(
                                        f"[BSTS] {home}: YapÄ±sal kÄ±rÄ±lma tespit! "
                                        f"idx={bp.index}, "
                                        f"Ã¶nceki={bp.pre_mean:.2f} â†’ sonraki={bp.post_mean:.2f} "
                                        f"({bp.change_pct:+.1f}%, p={bp.p_value:.3f})"
                                    )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-10) Hypergraph â€“ taktiksel birim analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    from src.quant.hypergraph_unit import TacticalUnit
                    import numpy as _np
                    home = row.get("home_team", "")
                    if home and hasattr(db, "get_team_lineup"):
                        lineup = db.get_team_lineup(home)
                        if lineup and len(lineup) >= 11:
                            ratings = _np.array(
                                [p.get("rating", 65) for p in lineup[:11]],
                                dtype=_np.float64,
                            )
                            units = [
                                TacticalUnit("Defans", "defense", [0, 1, 2, 3], weight=1.5),
                                TacticalUnit("Orta Saha", "midfield", [4, 5, 6], weight=1.2),
                                TacticalUnit("HÃ¼cum", "attack", [7, 8, 9, 10], weight=1.0),
                            ]
                            missing = [
                                i for i, p in enumerate(lineup[:11])
                                if p.get("status") == "injured"
                            ]
                            hg_report = hypergraph.analyze_team(
                                home, units, ratings, missing_players=missing,
                            )
                            if hg_report.defense_alert or hg_report.midfield_alert:
                                logger.warning(
                                    f"[Hypergraph] {home}: "
                                    f"{'SAVUNMA' if hg_report.defense_alert else ''}"
                                    f"{'|ORTA SAHA' if hg_report.midfield_alert else ''} "
                                    f"zayÄ±f! kÄ±rÄ±lganlÄ±k={hg_report.vulnerability_index:.2f}"
                                )
                            elif hg_report.vulnerability_index > 0.3:
                                logger.info(
                                    f"[Hypergraph] {home}: uyum={hg_report.team_cohesion:.2f}, "
                                    f"kÄ±rÄ±lganlÄ±k={hg_report.vulnerability_index:.2f}"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-11) Fluid Dynamics â€“ saha kontrol analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    if hasattr(db, "get_player_positions"):
                        from src.quant.fluid_pitch import PlayerState
                        positions = db.get_player_positions(
                            row.get("match_id", ""),
                        )
                        if positions and len(positions) >= 22:
                            players = [
                                PlayerState(
                                    player_id=p.get("id", ""),
                                    team=p.get("team", "home"),
                                    x=p.get("x", 52.5),
                                    y=p.get("y", 34.0),
                                    vx=p.get("vx", 0),
                                    vy=p.get("vy", 0),
                                    has_ball=p.get("has_ball", False),
                                    rating=p.get("rating", 70),
                                )
                                for p in positions[:22]
                            ]
                            fluid_report = fluid_pitch.analyze(players)
                            if fluid_report.momentum_score > 0.3 or fluid_report.momentum_score < -0.3:
                                logger.info(
                                    f"[Fluid] {row.get('home_team','')} vs "
                                    f"{row.get('away_team','')}: "
                                    f"kontrol={fluid_report.home_control_pct:.0f}% "
                                    f"xT_home={fluid_report.home_xt:.3f} "
                                    f"momentum={fluid_report.momentum_score:+.2f} "
                                    f"({fluid_report.flow_direction})"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-12) Regime Switcher â€“ gizli rejim tespiti â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    if home and hasattr(db, "get_match_stats_timeline"):
                        import numpy as _np
                        timeline = db.get_match_stats_timeline(
                            row.get("match_id", ""), team=home,
                        )
                        if timeline and len(timeline) >= 3:
                            obs = _np.array(timeline, dtype=_np.float64)
                            regime_report = regime_sw.analyze_match(
                                obs, team=home,
                                match_id=row.get("match_id", ""),
                            )
                            if regime_report.momentum_break:
                                logger.warning(
                                    f"[Regime] {home}: MOMENTUM KIRILMASI! "
                                    f"{regime_report.last_transition} "
                                    f"â†’ gol Ã§arpanÄ± x{regime_report.goal_adjustment:.2f}"
                                )
                            elif regime_report.current.regime_id != 1:
                                logger.info(
                                    f"[Regime] {home}: {regime_report.current.regime_name} "
                                    f"(gÃ¼ven={regime_report.current.confidence:.0%}, "
                                    f"stabilite={regime_report.stability:.2f})"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-13) SDE Pricer â€“ oran hareketi tahmini â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    if hasattr(db, "get_odds_history"):
                        odds_hist = db.get_odds_history(
                            row.get("match_id", ""), market="home",
                        )
                        if odds_hist and len(odds_hist) >= 5:
                            sde_forecast = sde_pricer.forecast(
                                odds_hist,
                                match_id=row.get("match_id", ""),
                                horizon_min=10,
                                fair_value_override=row.get("fair_odds"),
                            )
                            if sde_forecast.value_signal in ("BUY", "SELL"):
                                logger.info(
                                    f"[SDE] {row.get('home_team','')} vs "
                                    f"{row.get('away_team','')}: "
                                    f"{sde_forecast.value_signal} "
                                    f"oran={sde_forecast.current_odds} â†’ "
                                    f"beklenen={sde_forecast.predicted_odds} "
                                    f"({sde_forecast.expected_change_pct:+.1f}%, "
                                    f"Î¸={sde_forecast.params.theta:.3f}, "
                                    f"Ïƒ={sde_forecast.params.sigma:.3f})"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-14) Quantum Brain â€“ kuantum ML tahmini â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    away = row.get("away_team", "")
                    if hasattr(db, "get_match_features"):
                        feat = db.get_match_features(row.get("match_id", ""))
                        if feat and len(feat) >= 4:
                            import numpy as _np
                            q_pred = quantum_brain.predict_match(
                                _np.array(feat, dtype=_np.float64),
                                match_id=row.get("match_id", ""),
                            )
                            if q_pred.confidence > 0.5:
                                labels = {0: "Ev", 1: "Beraberlik", 2: "Deplasman"}
                                logger.info(
                                    f"[Quantum] {home} vs {away}: "
                                    f"{labels.get(q_pred.prediction, '?')} "
                                    f"(gÃ¼ven={q_pred.confidence:.0%}, "
                                    f"method={q_pred.method}, "
                                    f"{q_pred.n_qubits}q/{q_pred.circuit_depth}d, "
                                    f"{q_pred.compute_time_ms:.1f}ms)"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-15) Hawkes Momentum â€“ olay bulaÅŸÄ±cÄ±lÄ±ÄŸÄ± â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    if hasattr(db, "get_match_events"):
                        events_data = db.get_match_events(
                            row.get("match_id", ""), event_type="goal",
                        )
                        if events_data and len(events_data) >= 2:
                            hawkes_report = hawkes.analyze_match(
                                events_data,
                                match_id=row.get("match_id", ""),
                                current_min=row.get("minute", 90),
                            )
                            if hawkes_report.over_signal:
                                logger.warning(
                                    f"[Hawkes] {home}: ÃœST SÄ°NYALÄ°! "
                                    f"BR={hawkes_report.params.branching_ratio:.2f}, "
                                    f"momentum={hawkes_report.momentum_level}, "
                                    f"5dk={hawkes_report.next_event_prob_5min:.0%}"
                                )
                            elif hawkes_report.goal_burst_alert:
                                logger.warning(
                                    f"[Hawkes] {home}: GOL PATLAMASI! "
                                    f"ArdÄ±ÅŸÄ±k olaylar tespit."
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-16) Survival Estimator â€“ saÄŸkalÄ±m analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    if hasattr(db, "get_concede_times"):
                        durations = db.get_concede_times(
                            home, last_n=50,
                        )
                        if durations and len(durations) >= 5:
                            import numpy as _np
                            d_arr = _np.array(
                                [x["duration"] for x in durations],
                                dtype=_np.float64,
                            )
                            o_arr = _np.array(
                                [x.get("observed", 1) for x in durations],
                                dtype=_np.int32,
                            )
                            survival.fit(d_arr, o_arr)
                            sv_report = survival.analyze(
                                current_minute=row.get("minute", 0),
                                last_goal_minute=row.get("last_goal_min"),
                                team=home,
                                match_id=row.get("match_id", ""),
                            )
                            if sv_report.dam_breaking:
                                logger.warning(
                                    f"[Survival] {home}: BARAJ YIKILIYOR! "
                                    f"H(t)={sv_report.params.cumulative_hazard:.2f}, "
                                    f"S(t)={sv_report.params.survival_prob:.0%}, "
                                    f"5dk={sv_report.prob_concede_5min:.0%}"
                                )
                            elif sv_report.fortress_mode:
                                logger.info(
                                    f"[Survival] {home}: KALE SAÄLAM "
                                    f"S(t)={sv_report.params.survival_prob:.0%}, "
                                    f"alt bahis sinyali"
                                )
                            elif sv_report.risk_level in ("high", "critical"):
                                logger.info(
                                    f"[Survival] {home}: risk={sv_report.risk_level}, "
                                    f"beklenen gol={sv_report.expected_time_to_goal:.0f}dk"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-17) Fatigue Engine â€“ biyomekanik yorgunluk â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    if hasattr(db, "get_player_physical_data"):
                        player_data = db.get_player_physical_data(
                            row.get("match_id", ""), team=home,
                        )
                        if player_data and len(player_data) >= 5:
                            ft_report = fatigue.analyze_team(
                                player_data,
                                current_minute=row.get("minute", 0),
                                team=home,
                                match_id=row.get("match_id", ""),
                            )
                            if ft_report.defense_collapse_risk:
                                logger.warning(
                                    f"[Fatigue] {home}: SAVUNMA Ã‡Ã–KÃœYOR! "
                                    f"Def enerji={ft_report.defense_avg_stamina:.0f}%, "
                                    f"kÄ±rÄ±lganlÄ±k={ft_report.defense_vulnerability:.0%}, "
                                    f"en zayÄ±f={ft_report.weakest_player} "
                                    f"({ft_report.weakest_stamina:.0f}%)"
                                )
                            elif ft_report.late_goal_signal:
                                logger.info(
                                    f"[Fatigue] {home}: GEÃ‡ GOL RÄ°SKÄ°! "
                                    f"Ort enerji={ft_report.avg_stamina:.0f}%, "
                                    f"{ft_report.critical_count} kritik oyuncu"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-18) Chaos Filter â€“ Lyapunov kaos tespiti â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    mid = row.get("match_id", "")
                    if hasattr(db, "get_odds_history"):
                        odds_hist = db.get_odds_history(mid, market="home")
                        if odds_hist and len(odds_hist) >= 15:
                            chaos_report = chaos_filter.analyze(
                                odds_hist, match_id=mid, market="home_odds",
                            )
                            if chaos_report.kill_betting:
                                logger.warning(
                                    f"[Chaos] {home}: KAOS! "
                                    f"Î»={chaos_report.params.max_lyapunov:.4f}, "
                                    f"rejim={chaos_report.regime}. "
                                    f"TÃœM BAHÄ°SLER Ä°PTAL."
                                )
                            elif chaos_report.reduce_stake:
                                logger.info(
                                    f"[Chaos] {home}: SÄ±nÄ±rda, "
                                    f"Î»={chaos_report.params.max_lyapunov:.4f}, "
                                    f"stake %50 dÃ¼ÅŸÃ¼rÃ¼ldÃ¼."
                                )
                            elif chaos_report.boost_confidence:
                                logger.debug(
                                    f"[Chaos] {home}: STABÄ°L "
                                    f"Î»={chaos_report.params.max_lyapunov:.4f}"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-19) Homology Scanner â€“ topolojik organizasyon â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    if hasattr(db, "get_player_positions"):
                        import numpy as _np
                        positions = db.get_player_positions(
                            row.get("match_id", ""), team=home,
                        )
                        if positions and len(positions) >= 5:
                            pos_arr = _np.array(positions, dtype=_np.float64)
                            topo = homology.analyze_team(
                                pos_arr, team=home,
                                match_id=row.get("match_id", ""),
                            )
                            if topo.team_panicking:
                                logger.warning(
                                    f"[Homology] {home}: PANÄ°K! "
                                    f"gÃ¼rÃ¼ltÃ¼={topo.noise_ratio:.0%}, "
                                    f"Î²â‚€={topo.betti_0}, "
                                    f"org={topo.organization_score:.0%}"
                                )
                            elif topo.formation_broken:
                                logger.info(
                                    f"[Homology] {home}: Formasyon bozuk, "
                                    f"org={topo.organization_score:.0%}, "
                                    f"Î²â‚€={topo.betti_0}, Î²â‚={topo.betti_1}"
                                )
                            elif topo.organized_play:
                                logger.debug(
                                    f"[Homology] {home}: Organize oyun, "
                                    f"Î²â‚={topo.betti_1} dÃ¶ngÃ¼"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-20) Fuzzy Reasoning â€“ bulanÄ±k mantÄ±k risk deÄŸerlendirmesi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    fuzzy_input = FuzzyInput(
                        weather=row.get("weather_score", 0.5),
                        fatigue=row.get("fatigue_score", 0.3),
                        travel_distance=row.get("travel_score", 0.2),
                        injury_count=row.get("injury_score", 0.0),
                        motivation=row.get("motivation_score", 0.7),
                        form=row.get("form_score", 0.6),
                        crowd_factor=row.get("crowd_score", 0.5),
                    )
                    fuzzy_out = fuzzy.evaluate(fuzzy_input, team=home)
                    if fuzzy_out.risk_level in ("yÃ¼ksek", "Ã§ok_yÃ¼ksek"):
                        logger.info(
                            f"[Fuzzy] {home}: risk={fuzzy_out.risk_level} "
                            f"({fuzzy_out.risk_score:.0f}/100), "
                            f"gÃ¼ven x{fuzzy_out.confidence_modifier:.1f}, "
                            f"gol mod x{fuzzy_out.goal_expectation_mod:.2f}"
                        )
                        if fuzzy_out.active_rules:
                            for rule in fuzzy_out.active_rules[:2]:
                                logger.debug(f"  â””â”€ {rule}")
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-21) AutoML â€“ otonom model arama (periyodik) â”€â”€
            if cycle % 50 == 0 and hasattr(db, "get_training_data"):
                try:
                    import numpy as _np
                    train_data = db.get_training_data(limit=5000)
                    if train_data and len(train_data) > 100:
                        X_train = _np.array(
                            [d["features"] for d in train_data],
                            dtype=_np.float64,
                        )
                        y_train = _np.array(
                            [d["label"] for d in train_data],
                            dtype=_np.int64,
                        )
                        aml_result = automl.search(
                            X_train, y_train,
                            task="classify",
                            time_budget_min=3,
                        )
                        if aml_result.best_score > 0.5:
                            automl.deploy_best()
                            logger.info(
                                f"[AutoML] Yeni model: {aml_result.best_model_name} "
                                f"(skor={aml_result.best_score:.1%}, "
                                f"{aml_result.n_models_tried} denendi, "
                                f"{aml_result.search_time_sec:.0f}s)"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-23) Epistemic/Aleatoric Uncertainty â€“ belirsizlik ayrÄ±mÄ± â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    feat_keys = [
                        "xg", "shots", "possession", "form_score",
                        "fatigue_score", "motivation_score",
                    ]
                    features = [float(row.get(k, 0.0)) for k in feat_keys]
                    if any(f != 0 for f in features):
                        import numpy as _np
                        unc_report = uncertainty_sep.analyze(
                            _np.array(features),
                            match_id=row.get("match_id", ""),
                            team=home,
                        )
                        if unc_report.decision == "ABSTAIN":
                            logger.warning(
                                f"[Uncertainty] {home}: PAS GEÃ‡ â€“ "
                                f"epistemik={unc_report.epistemic:.3f} "
                                f"(bilgisizlik), "
                                f"aleatorik={unc_report.aleatoric:.3f} "
                                f"(ÅŸans). {unc_report.reason}"
                            )
                        elif unc_report.decision == "HALF_KELLY":
                            logger.info(
                                f"[Uncertainty] {home}: YARI KELLY â€“ "
                                f"aleatorik={unc_report.aleatoric:.3f}. "
                                f"Stake x{unc_report.confidence_modifier:.1f}"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-24) Topology Mapper â€“ topolojik kÃ¼me analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    feat_keys = [
                        "xg", "shots", "possession", "form_score",
                        "odds_home", "odds_draw", "odds_away",
                    ]
                    features = [float(row.get(k, 0.0)) for k in feat_keys]
                    if any(f != 0 for f in features):
                        import numpy as _np
                        topo_rep = topo_mapper.analyze_match(
                            _np.array(features),
                            match_id=row.get("match_id", ""),
                            team=home,
                        )
                        if topo_rep.is_anomalous:
                            logger.warning(
                                f"[Mapper] {home}: ANOMALÄ° KÃœME! "
                                f"Skor={topo_rep.anomaly_score:.0%}, "
                                f"kÃ¼me=#{topo_rep.assigned_cluster} "
                                f"({topo_rep.cluster_size} maÃ§). "
                                f"ÅÃ¼pheli hareket!"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-25) GraphRAG â€“ haber kaynaklÄ± kriz tespiti â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    away = row.get("away_team", "")
                    for team_name in (home, away):
                        if not team_name:
                            continue
                        crisis = graph_rag.analyze_crisis(
                            team_name,
                            match_id=row.get("match_id", ""),
                        )
                        if crisis.crisis_level in ("crisis", "meltdown"):
                            logger.warning(
                                f"[GraphRAG] {team_name}: {crisis.crisis_level.upper()}! "
                                f"Kriz={crisis.crisis_score:.0%}, "
                                f"negatif haber={crisis.negative_news_count}. "
                                f"{crisis.recommendation}"
                            )
                            if crisis.hidden_connections:
                                for hc in crisis.hidden_connections[:2]:
                                    logger.debug(f"  â””â”€ Gizli baÄŸ: {hc}")
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-26) Probabilistic Engine â€“ olasÄ±lÄ±ksal maÃ§ tahmini â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    home = row.get("home_team", "")
                    away = row.get("away_team", "")
                    if home and away and prob_engine._fitted:
                        prob_pred = prob_engine.predict(
                            home, away,
                            match_id=row.get("match_id", ""),
                        )
                        logger.info(
                            f"[ProbEngine] {home} vs {away}: "
                            f"Ev={prob_pred.p_home:.0%} / Ber={prob_pred.p_draw:.0%} / "
                            f"Dep={prob_pred.p_away:.0%} | "
                            f"Ev gol: {prob_pred.home_goals_mean:.1f} "
                            f"[{prob_pred.home_goals_hdi[0]:.1f}-{prob_pred.home_goals_hdi[1]:.1f}], "
                            f"Ãœ2.5: {prob_pred.p_over25:.0%}"
                        )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-27) MF-DFA â€“ Ã§oklu fraktal piyasa analizi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    odds_hist = row.get("odds_history", [])
                    if isinstance(odds_hist, (list, tuple)) and len(odds_hist) >= 50:
                        import numpy as _np
                        mf_report = mf_analyzer.analyze(
                            _np.array(odds_hist, dtype=_np.float64),
                            match_id=row.get("match_id", ""),
                            market="odds",
                        )
                        if mf_report.regime_change_signal:
                            logger.warning(
                                f"[MF-DFA] REJÄ°M DEÄÄ°ÅÄ°KLÄ°ÄÄ°: "
                                f"{row.get('home_team', '?')} vs {row.get('away_team', '?')} "
                                f"Î”h={mf_report.params.delta_h:.3f}, "
                                f"h(2)={mf_report.params.hurst_q2:.3f}. "
                                f"BÃ¼yÃ¼k sÃ¼rpriz riski!"
                            )
                        elif mf_report.regime == "strong_multifractal":
                            logger.info(
                                f"[MF-DFA] GÃ¼Ã§lÃ¼ Ã§ok-fraktal: "
                                f"Î”h={mf_report.params.delta_h:.3f}. "
                                f"Stake dÃ¼ÅŸÃ¼r."
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-28) Active Inference â€“ modÃ¼l gÃ¶zlem gÃ¼ncellemesi â”€â”€
            # (MaÃ§ sonuÃ§larÄ± geldiÄŸinde, her modÃ¼lÃ¼n surprisal'Ä±nÄ± gÃ¼ncelle)
            if hasattr(db, "get_recent_results"):
                try:
                    recent = db.get_recent_results(limit=10)
                    for result in (recent or []):
                        observed = result.get("outcome", -1)  # 0=home, 1=draw, 2=away
                        if observed < 0:
                            continue
                        for mod_name in ("lightgbm", "poisson", "ensemble"):
                            preds = result.get(f"{mod_name}_probs", None)
                            if preds and len(preds) == 3:
                                s = active_inf.observe(
                                    module=mod_name,
                                    predicted_probs=preds,
                                    observed=observed,
                                    match_id=result.get("match_id", ""),
                                )
                    # Yeniden eÄŸitim hedefleri
                    retrain = active_inf.get_retrain_targets()
                    if retrain:
                        logger.warning(
                            f"[ActiveInf] Yeniden eÄŸitim gerekli: "
                            f"{', '.join(retrain)}. "
                            f"Precision aÄŸÄ±rlÄ±klarÄ± gÃ¼ncellendi."
                        )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-29) Wavelet Denoising â€“ oran sinyali temizleme â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    odds_hist = row.get("odds_history", [])
                    if isinstance(odds_hist, (list, tuple)) and len(odds_hist) >= 20:
                        import numpy as _np
                        wav_report = wavelet.analyze(
                            _np.array(odds_hist, dtype=_np.float64),
                            match_id=row.get("match_id", ""),
                            market="odds",
                        )
                        if wav_report.fake_move_detected:
                            logger.warning(
                                f"[Wavelet] FAKE MOVE: "
                                f"{row.get('home_team', '?')} vs {row.get('away_team', '?')} "
                                f"t={wav_report.fake_move_times[:3]}, "
                                f"gÃ¼rÃ¼ltÃ¼={wav_report.result.noise_pct:.1f}%. "
                                f"GerÃ§ek trend: {wav_report.result.trend_direction} "
                                f"({wav_report.result.trend_slope:+.4f})"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-30) Symbolic Discovery â€“ periyodik formÃ¼l keÅŸfi â”€â”€
            if cycle % 100 == 0 and hasattr(db, "get_training_data"):
                try:
                    import numpy as _np
                    train_data = db.get_training_data(limit=3000)
                    if train_data and len(train_data) > 50:
                        X_sym = _np.array(
                            [d["features"] for d in train_data],
                            dtype=_np.float64,
                        )
                        y_sym = _np.array(
                            [d.get("goals", 0) for d in train_data],
                            dtype=_np.float64,
                        )
                        sym_report = symbolic.discover(
                            X_sym, y_sym,
                            feature_names=["xG", "shots", "poss", "form", "odds"],
                            target="goals",
                        )
                        if sym_report.best_formula.r2 > 0.3:
                            logger.info(
                                f"[Symbolic] FormÃ¼l keÅŸfedildi: "
                                f"{sym_report.best_formula.equation} "
                                f"(RÂ²={sym_report.best_formula.r2:.1%}, "
                                f"cplx={sym_report.best_formula.complexity})"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-32) GARCH Volatility Clustering â€“ risk rejimi tespiti â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    odds_hist = row.get("odds_history", [])
                    if isinstance(odds_hist, (list, tuple)) and len(odds_hist) >= 20:
                        import numpy as _np
                        odds_arr = _np.array(odds_hist, dtype=_np.float64)
                        # Log-return hesapla
                        if len(odds_arr) > 1:
                            log_ret = _np.diff(_np.log(odds_arr + 1e-8))
                            vol_report = vol_analyzer.analyze(
                                log_ret,
                                match_id=row.get("match_id", ""),
                                team=row.get("home_team", "?"),
                                market="odds",
                            )
                            if vol_report.regime in ("storm", "crisis"):
                                logger.warning(
                                    f"[GARCH] {vol_report.regime.upper()}: "
                                    f"{row.get('home_team', '?')} vs {row.get('away_team', '?')} "
                                    f"Ïƒ={vol_report.current_volatility:.4f}, "
                                    f"VaR95={vol_report.var_95:.4f}, "
                                    f"Kelly x{vol_report.kelly_multiplier:.1f}. "
                                    f"persistence={vol_report.params.persistence:.2f}"
                                )
                            elif vol_report.regime_change:
                                logger.info(
                                    f"[GARCH] Rejim deÄŸiÅŸimi: "
                                    f"{row.get('home_team', '?')} â†’ {vol_report.regime} "
                                    f"(Ïƒ={vol_report.current_volatility:.4f})"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-33) Stream Processing â€“ canlÄ± olay akÄ±ÅŸÄ± gÃ¼ncelleme â”€â”€
            try:
                for row in matches.iter_rows(named=True):
                    from src.core.stream_processor import StreamEvent
                    odds_val = _sg(row, "home_odds", 0.0)
                    if odds_val:
                        await stream_proc.emit(StreamEvent(
                            event_type="odds_update",
                            match_id=row.get("match_id", ""),
                            data={
                                "odds": float(odds_val),
                                "home_team": row.get("home_team", ""),
                                "away_team": row.get("away_team", ""),
                            },
                            source="analysis_loop",
                        ))
                # Windowed aggregate kontrol
                for mid in stream_proc.get_active_matches():
                    agg = stream_proc.get_window(mid)
                    if agg and agg.std_value > 0.05:
                        logger.info(
                            f"[Stream] {mid} son {agg.count} olay: "
                            f"avg={agg.avg_value:.3f}, "
                            f"std={agg.std_value:.4f}, "
                            f"range=[{agg.min_value:.3f}, {agg.max_value:.3f}]"
                        )
            except Exception as e:
                logger.debug(f"[Guardian] stream_agg: {type(e).__name__}: {e}")

            # â”€â”€ 5b-34) Particle Strength Tracker â€“ dinamik gÃ¼Ã§ takibi â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    live_data = row.get("live_stats", {})
                    if isinstance(live_data, dict) and live_data.get("minute", 0) > 0:
                        from src.quant.particle_strength_tracker import MatchObservation
                        obs = MatchObservation(
                            minute=int(live_data.get("minute", 0)),
                            home_shots=int(live_data.get("home_shots", 0)),
                            away_shots=int(live_data.get("away_shots", 0)),
                            home_possession=float(live_data.get("home_poss", 50)),
                            away_possession=float(live_data.get("away_poss", 50)),
                            home_dangerous_attacks=int(live_data.get("home_attacks", 0)),
                            away_dangerous_attacks=int(live_data.get("away_attacks", 0)),
                        )
                        # Prior'larÄ± modellerden al
                        home_prior = result.get("home_prob", 0.5) if isinstance(result, dict) else 0.5
                        if not particle_tracker._initialized:
                            particle_tracker.initialize(
                                home_prior=home_prior,
                                away_prior=1 - home_prior,
                            )
                        p_report = particle_tracker.update(
                            obs, match_id=row.get("match_id", ""),
                        )
                        if p_report.momentum_shift.detected:
                            logger.warning(
                                f"[Particle] MOMENTUM SHIFT dk.{p_report.minute}: "
                                f"{p_report.momentum_shift.direction} "
                                f"(Î”={p_report.momentum_shift.magnitude:.3f}), "
                                f"Home={p_report.state.home_power:.2f}, "
                                f"Away={p_report.state.away_power:.2f}, "
                                f"ESS={p_report.ess_ratio:.1%}"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-35) Causal Discovery â€“ nedensellik DAG gÃ¼ncelleme â”€â”€
            if cycle % 25 == 0 and hasattr(db, "get_training_data"):
                try:
                    import numpy as _np
                    train_data = db.get_training_data(limit=2000)
                    if train_data and len(train_data) > 50:
                        X_causal = _np.array(
                            [d["features"] for d in train_data],
                            dtype=_np.float64,
                        )
                        feat_names = ["xG", "shots", "poss", "corners",
                                      "fouls", "form", "odds", "goals"]
                        c_report = causal.analyze_match(
                            X_causal, feat_names[:X_causal.shape[1]],
                            target="goals",
                            match_id=f"cycle_{cycle}",
                        )
                        if c_report.goal_root_causes:
                            logger.info(
                                f"[Causal] Gol kÃ¶k nedenleri: "
                                f"{', '.join(c_report.goal_root_causes)} "
                                f"(gÃ¼ven={c_report.causal_confidence:.1%})"
                            )
                        if c_report.spurious_correlations:
                            logger.warning(
                                f"[Causal] Sahte korelasyon: "
                                f"{'; '.join(c_report.spurious_correlations[:2])}"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5b-31) Transfer Learning â€“ lig arasÄ± bilgi transferi â”€â”€
            try:
                if hasattr(db, "get_all_features"):
                    train_features = db.get_all_features(limit=5000)
                    if train_features and len(train_features) > 50:
                        import numpy as _np
                        X_ref = _np.array(train_features, dtype=_np.float64)
                        if X_ref.ndim == 2 and X_ref.shape[1] >= 5:
                            transport_metric.set_reference(X_ref)
                            logger.debug(
                                f"[Transport] Referans daÄŸÄ±lÄ±m gÃ¼ncellendi: "
                                f"{X_ref.shape}"
                            )
            except Exception as e:
                logger.debug(f"[Guardian] transport_ref: {type(e).__name__}: {e}")

            # â”€â”€ 5b-11) Optimal Transport â€“ model drift tespiti â”€â”€
            try:
                if hasattr(db, "get_live_features"):
                    live_features = db.get_live_features(limit=200)
                    if live_features and len(live_features) > 10:
                        import numpy as _np
                        X_live = _np.array(live_features, dtype=_np.float64)
                        drift_report = transport_metric.check_drift(X_live, name="cycle_check")
                        if drift_report.kill_betting:
                            logger.warning(
                                f"[Transport] SEVERE DRIFT: W={drift_report.wasserstein_2:.4f} "
                                f"â†’ BAHÄ°SLER DURDURULDU!"
                            )
                        elif drift_report.is_drifted:
                            logger.warning(
                                f"[Transport] Drift tespit: {drift_report.drift_severity} "
                                f"(W={drift_report.wasserstein_2:.4f}, "
                                f"boyutlar={drift_report.drift_dimensions[:3]})"
                            )
                        elif drift_report.drift_severity == "mild":
                            logger.info(
                                f"[Transport] Hafif kayma: W={drift_report.wasserstein_2:.4f}"
                            )
            except Exception as e:
                logger.debug(f"[Guardian] transport_drift: {type(e).__name__}: {e}")

            # â”€â”€ 5c) Duygu analizi (NLP) â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    sent = await sentiment.analyze_for_match_async(
                        row.get("home_team", ""), row.get("away_team", "")
                    )
                    lance.add_news(
                        f"Sentiment: {row.get('home_team','')} vs {row.get('away_team','')}",
                        f"edge={sent.get('sentiment_edge', 0):.2f}",
                        "sentiment_analyzer",
                    )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 5d) Volatilite bazlÄ± pazar Ã¶nerisi â”€â”€
            for row in matches.iter_rows(named=True):
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                if home and away:
                    market_rec = team_vix.recommend_market(home, away)
                    if market_rec.get("market") != "1X2":
                        logger.info(
                            f"[VIX] {home} vs {away} â†’ {market_rec['market']} "
                            f"({market_rec.get('reason', '')[:60]})"
                        )

            # â”€â”€ 5e) Bilgi grafiÄŸini gÃ¼ncelle (GraphRAG) â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    graph.add_match(
                        str(row.get("match_id") or ""),
                        str(row.get("home_team") or ""),
                        str(row.get("away_team") or ""),
                        str(row.get("kickoff") or ""),
                    )
                except Exception as e:
                    logger.debug(f"[Guardian] GraphRAG add_match: {type(e).__name__}: {e}")

            # â”€â”€ 5f) Semantik hafÄ±zaya kaydet (LanceDB) â”€â”€
            for row in matches.iter_rows(named=True):
                try:
                    odds_val = _sg(row, "home_odds", 0.0)
                    lance.add_odds_event(
                        str(row.get("match_id") or ""),
                        "1X2",
                        odds_val,
                        "pipeline",
                    )
                except Exception as e:
                    logger.debug(f"[Guardian] Lance add_odds: {type(e).__name__}: {e}")

            # â”€â”€ 6) Ensemble: tÃ¼m sinyalleri birleÅŸtir â”€â”€
            ensemble = prob_engine.ensemble(
                bayesian=prob_preds,
                mtl=mtl_preds,
                kan=kan_preds,
                tail=tail_risk,
                causal=causal_edges,
                intervals=intervals,
                signatures=sig_features,
                jumps=jumps,
                geometric=geo_potential,
            )

            # â”€â”€ 7) RL ajan kararÄ± â”€â”€
            action = rl.decide(ensemble)

            # â”€â”€ 8) Risk yÃ¶netimi â”€â”€
            covar_risk  = covar.measure(ensemble)
            allocation  = bl_opt.optimize(ensemble, covar_risk)
            constrained = risk_solver.solve(allocation)
            final_bets  = pnl.stabilize(constrained)
            if isinstance(final_bets, list):
                sanitized: list[dict] = []
                dropped = 0
                for bet in final_bets:
                    if not isinstance(bet, dict):
                        dropped += 1
                        continue
                    try:
                        odds = float(bet.get("odds", 0) or 0)
                    except Exception:
                        odds = 0.0
                    conf_raw = bet.get("confidence", bet.get("prob_home", 0.5))
                    try:
                        conf = float(conf_raw if conf_raw is not None else 0.5)
                    except Exception:
                        conf = 0.5
                    if not math.isfinite(conf):
                        conf = 0.5
                    if odds <= 1.01:
                        dropped += 1
                        continue
                    bet["confidence"] = min(max(conf, 1e-4), 1 - 1e-4)
                    sanitized.append(bet)
                if dropped > 0:
                    logger.warning(
                        f"[Sanity] GeÃ§ersiz olasÄ±lÄ±k/oran nedeniyle {dropped} sinyal elendi."
                    )
                final_bets = sanitized

            # â”€â”€ 8b) Fair Value Engine â€“ Sentetik Oran & Value Edge â”€â”€
            if isinstance(final_bets, list):
                for bet in final_bets:
                    if isinstance(bet, dict) and bet.get("prob_home"):
                        sel = bet.get("selection", "home")
                        prob_key = f"prob_{sel}" if sel in ("home", "draw", "away") else "prob_home"
                        model_prob = bet.get(prob_key, bet.get("confidence", 0.33))
                        m_odds = bet.get("odds", 0)
                        if model_prob > 0 and m_odds > 1.0:
                            fv = fair_value.analyze(model_prob, m_odds, sel, bet.get("match_id", ""))
                            bet["value_edge"] = fv.value_edge
                            bet["fair_odds"] = fv.fair_odds
                            bet["value_tier"] = fv.tier
                            bet["kelly_stake"] = fv.kelly_stake

            # â”€â”€ 8b-2) RL Ajan â€“ Stake kararÄ± (PPO) â”€â”€
            if isinstance(final_bets, list):
                for bet in final_bets:
                    if isinstance(bet, dict):
                        rl_obs = {
                            "model_prob_home": bet.get("prob_home", 0.33),
                            "model_prob_draw": bet.get("prob_draw", 0.33),
                            "model_prob_away": bet.get("prob_away", 0.34),
                            "odds_home": bet.get("odds", 2.0),
                            "value_edge": bet.get("value_edge", 0),
                            "confidence": bet.get("confidence", 0.5),
                            "bankroll_ratio": 1.0,
                            "volatility": 0.5,
                        }
                        rl_decision = rl_agent.predict(rl_obs)
                        bet["rl_action"] = rl_decision["action_name"]
                        bet["rl_stake_pct"] = rl_decision["stake_pct"]
                        if rl_decision["action"] == 0:
                            bet["rl_pass"] = True

            # â”€â”€ FILTER AUDIT: Filtre zinciri giriÅŸ sayÄ±sÄ±nÄ± kaydet â”€â”€
            _filter_entry_count = len(final_bets) if isinstance(final_bets, list) else 0
            _filter_log = []   # (filtre_adÄ±, giren, Ã§Ä±kan)

            # â”€â”€ 8c) Conformal Prediction â€“ soft-scoring (artÄ±k hard-kill DEÄÄ°L) â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                pre_count = len(final_bets)
                # Hard-kill yerine soft-scoring: belirsiz bahisleri sil deÄŸil, stake dÃ¼ÅŸÃ¼r
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            uq_score = uncertainty_q.uncertainty_score(
                                bet, prob_key="confidence",
                            ) if hasattr(uncertainty_q, "uncertainty_score") else None
                            if uq_score is not None:
                                bet["uq_uncertainty"] = uq_score
                                # Ã‡ok belirsiz â†’ stake %40'a dÃ¼ÅŸÃ¼r (silme)
                                if uq_score > 0.8:
                                    bet["uq_penalty"] = 0.4
                                elif uq_score > 0.6:
                                    bet["uq_penalty"] = 0.7
                                else:
                                    bet["uq_penalty"] = 1.0
                        except Exception:
                            bet["uq_penalty"] = 1.0
                _filter_log.append(("UQ_soft", pre_count, len(final_bets)))

            # â”€â”€ 8z-1) Fisher Geometry â€“ daÄŸÄ±lÄ±m anomali ve rejim tespiti â”€â”€
            if fisher_geo and isinstance(final_bets, list):
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            mid = bet.get("match_id", "")
                            if hasattr(db, "get_odds_history"):
                                hist = db.get_odds_history(mid, market="home")
                                current = [bet.get("odds", 2.0)]
                                if hist and len(hist) >= 10:
                                    import numpy as _np
                                    fr_report = fisher_geo.compare_distributions(
                                        _np.array(hist), _np.array(current),
                                        match_id=mid,
                                    )
                                    bet["fisher_rao_distance"] = fr_report.fisher_rao_distance
                                    bet["fisher_anomaly"] = fr_report.is_anomaly
                                    bet["fisher_regime_shift"] = fr_report.regime_shift
                                    if fr_report.regime_shift:
                                        bet["fisher_advice"] = fr_report.recommendation
                                        logger.warning(
                                            f"[Fisher] {mid}: REJÄ°M DEÄÄ°ÅÄ°MÄ° "
                                            f"FR={fr_report.fisher_rao_distance:.4f}"
                                        )
                        except Exception as e:
                            logger.debug(f"[Guardian] fisher_geo: {type(e).__name__}: {e}")

            # â”€â”€ 8z-2) Philosophical Engine â€“ soft-scoring (artÄ±k silmiyor) â”€â”€
            if philo_engine and isinstance(final_bets, list):
                _philo_pre = len(final_bets)
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            mid = bet.get("match_id", "")
                            prob = bet.get("confidence", bet.get("prob_home", 0.5))
                            conf = bet.get("model_confidence", 0.7)
                            sample_n = bet.get("sample_size", 100)
                            phi_report = philo_engine.evaluate(
                                probability=prob, confidence=conf,
                                sample_size=sample_n,
                                strategy_age_days=30,
                                model_count=5,
                                match_id=mid,
                            )
                            bet["epistemic_score"] = phi_report.epistemic_score
                            bet["epistemic_approved"] = phi_report.epistemic_approved
                            if phi_report.reflections:
                                bet["epistemic_reflection"] = phi_report.reflections[0]
                            if not phi_report.epistemic_approved:
                                # Silme, stake'i dÃ¼ÅŸÃ¼r
                                bet["philo_penalty"] = 0.5
                                logger.info(
                                    f"[Philo] {mid}: Epistemik uyarÄ± (skor={phi_report.epistemic_score:.2f}) "
                                    f"â€“ stake %50 dÃ¼ÅŸÃ¼rÃ¼ldÃ¼."
                                )
                            else:
                                bet["philo_penalty"] = 1.0
                        except Exception as e:
                            logger.debug(f"[Guardian] philo_engine: {type(e).__name__}: {e}")
                            bet["philo_penalty"] = 1.0
                _filter_log.append(("Philo_soft", _philo_pre, len(final_bets)))

            # â”€â”€ 8z-3) Regime Kelly v2 â€“ rejim-farkÄ±nda stake hesaplama â”€â”€
            if regime_kelly and isinstance(final_bets, list):
                regime_kelly.reset_daily()
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            prob = float(bet.get("confidence", bet.get("prob_home", 0.5)) or 0.5)
                            odds = float(bet.get("odds", 0) or 0)
                            mid = bet.get("match_id", "")
                            if odds <= 1.01:
                                bet["regime_kelly_stake"] = 0.0
                                bet["regime_kelly_approved"] = False
                                bet["kelly_reject_reason"] = "invalid_odds"
                                continue
                            prob = min(max(prob, 1e-4), 1 - 1e-4)

                            # Rejim durumunu modÃ¼llerden topla
                            regime_state = RegimeState()
                            if vol_analyzer and hasattr(vol_analyzer, '_last_regime'):
                                regime_state.volatility_regime = getattr(
                                    vol_analyzer, '_last_regime', 'calm'
                                )
                            if chaos_filter and hasattr(chaos_filter, '_last_regime'):
                                regime_state.chaos_regime = getattr(
                                    chaos_filter, '_last_regime', 'stable'
                                )

                            kelly_dec = regime_kelly.calculate(
                                probability=prob, odds=odds,
                                match_id=mid, regime=regime_state,
                            )
                            bet["regime_kelly_stake"] = kelly_dec.stake_amount
                            bet["regime_kelly_approved"] = kelly_dec.approved
                            bet["kelly_regime_multiplier"] = kelly_dec.regime_multiplier
                            if kelly_dec.adjustments:
                                bet["kelly_adjustments"] = kelly_dec.adjustments[:3]
                        except Exception as e:
                            logger.debug(f"[Guardian] regime_kelly: {type(e).__name__}: {e}")

            # â”€â”€ 9) SaÄŸlÄ±k raporu & ÅŸeytanÄ±n avukatÄ± â”€â”€
            health.update(final_bets, ensemble)
            devil.challenge(final_bets)

            # â”€â”€ 9b) Korelasyon filtresi â€“ portfÃ¶y varyansÄ±nÄ± dÃ¼ÅŸÃ¼r â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 1:
                _pre_corr = len(final_bets)
                final_bets = corr_matrix.filter_diversified(final_bets, max_bets=8)
                _filter_log.append(("Correlation", _pre_corr, len(final_bets)))

            # â”€â”€ 9c) Copula baÄŸÄ±mlÄ±lÄ±k filtresi â€“ kuyruk riski â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 1:
                _pre_cop = len(final_bets)
                coupon_report = copula_risk.analyze_coupon(final_bets)
                if coupon_report.dangerous_pairs:
                    logger.warning(
                        f"[Copula] {len(coupon_report.dangerous_pairs)} tehlikeli Ã§ift! "
                        f"DÃ¼zeltme: {coupon_report.risk_adjustment:.0%}"
                    )
                    final_bets = copula_risk.filter_safe_coupon(final_bets, max_size=8)
                _filter_log.append(("Copula", _pre_cop, len(final_bets)))

            # â”€â”€ 9d) Isolation Forest â€“ soft-scoring (tuzak uyarÄ±sÄ± ama silmez) â”€â”€
            if isinstance(final_bets, list):
                pre_iso = len(final_bets)
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            is_trap = iso_anomaly.is_anomaly(bet) if hasattr(iso_anomaly, "is_anomaly") else False
                            if is_trap:
                                bet["iso_trap"] = True
                                bet["iso_penalty"] = 0.3   # Stake %30'a indir
                                logger.warning(f"[IsoForest] {bet.get('match_id','?')}: TUZAK â€“ stake %30'a dÃ¼ÅŸÃ¼rÃ¼ldÃ¼")
                            else:
                                bet["iso_penalty"] = 1.0
                        except Exception:
                            bet["iso_penalty"] = 1.0
                _filter_log.append(("IsoForest_soft", pre_iso, len(final_bets)))

            # â”€â”€ 9e) Entropy â€“ soft-scoring (kaotik maÃ§larda stake dÃ¼ÅŸÃ¼r) â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                pre_ent = len(final_bets)
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            ent_val = entropy_meter.entropy_score(bet) if hasattr(entropy_meter, "entropy_score") else None
                            if ent_val is not None:
                                bet["entropy"] = ent_val
                                if ent_val > 0.9:
                                    bet["entropy_penalty"] = 0.2   # Ã‡ok kaotik â†’ stake %20
                                elif ent_val > 0.7:
                                    bet["entropy_penalty"] = 0.5
                                else:
                                    bet["entropy_penalty"] = 1.0
                            else:
                                bet["entropy_penalty"] = 1.0
                        except Exception:
                            bet["entropy_penalty"] = 1.0
                _filter_log.append(("Entropy_soft", pre_ent, len(final_bets)))

            # â”€â”€ 9f) Nash Game Theory â€“ optimal strateji filtresi â”€â”€
            if isinstance(final_bets, list):
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            nash_result = nash_solver.analyze_match(
                                model_probs={
                                    "prob_home": bet.get("prob_home", 0.33),
                                    "prob_draw": bet.get("prob_draw", 0.33),
                                    "prob_away": bet.get("prob_away", 0.34),
                                },
                                market_odds={
                                    "home": bet.get("odds", 2.0),
                                    "draw": bet.get("draw_odds", 3.5),
                                    "away": bet.get("away_odds", 4.0),
                                },
                                match_id=bet.get("match_id", ""),
                            )
                            bet["nash_action"] = nash_result.optimal_action
                            bet["nash_ev"] = nash_result.expected_value
                            bet["nash_exploitability"] = nash_result.equilibrium.exploitability
                        except Exception as e:
                            logger.debug(f"[Guardian] nash_enrich: {type(e).__name__}: {e}")

            # â”€â”€ 9g) EVT â€“ Kuyruk riski stake ayarlamasÄ± â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                final_bets = evt_risk.adjust_kelly_stakes(final_bets)
                # PortfÃ¶y VaR raporu
                p_var = evt_risk.portfolio_var(final_bets)
                if p_var.total_var_99 > 0:
                    logger.info(
                        f"[EVT] PortfÃ¶y VaR(%99): {p_var.total_var_99:.0f} TL, "
                        f"Max kayÄ±p: {p_var.max_loss_scenario:.0f} TL, "
                        f"Diversifikasyon: {p_var.diversification_benefit:.0%}"
                    )

            # â”€â”€ 9h) Fractal â€“ Hurst kelly Ã§arpanÄ± â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                for bet in final_bets:
                    if isinstance(bet, dict):
                        try:
                            league = bet.get("league", "")
                            if league and hasattr(db, "get_recent_results"):
                                lg_data = db.get_recent_results(league=league, limit=50)
                                if lg_data:
                                    goals = [r.get("home_goals", 0) + r.get("away_goals", 0) for r in lg_data]
                                    h_result = fractal.compute_hurst(goals, method="rs")
                                    bet["hurst"] = h_result.hurst
                                    bet["hurst_regime"] = h_result.regime
                                    if h_result.kelly_multiplier != 1.0:
                                        original = bet.get("kelly_stake", bet.get("stake", 100))
                                        bet["kelly_stake"] = round(
                                            original * h_result.kelly_multiplier, 2,
                                        )
                        except Exception as e:
                            logger.debug(f"[Guardian] hurst_kelly: {type(e).__name__}: {e}")

            # â”€â”€ 9h-2) Chaos Filter â€“ kaotik maÃ§larÄ± filtrele â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                try:
                    odds_histories = {}
                    for bet in final_bets:
                        if not isinstance(bet, dict):
                            continue
                        mid = bet.get("match_id", "")
                        if mid and hasattr(db, "get_odds_history"):
                            hist = db.get_odds_history(mid, market="home")
                            if hist and len(hist) >= 15:
                                odds_histories[mid] = hist
                    if odds_histories:
                        final_bets = chaos_filter.filter_bets(
                            final_bets, odds_histories,
                        )
                        chaos_killed = sum(
                            1 for b in final_bets
                            if isinstance(b, dict) and b.get("chaos_killed")
                        )
                        if chaos_killed > 0:
                            logger.warning(
                                f"[Chaos] {chaos_killed} bahis "
                                f"kaos nedeniyle iptal!"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9i) Transport Drift â€“ veri rejimi deÄŸiÅŸim filtresi â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                try:
                    if hasattr(db, "get_live_features"):
                        live_feat = db.get_live_features(limit=200)
                        if live_feat and len(live_feat) > 10:
                            import numpy as _np
                            X_lf = _np.array(live_feat, dtype=_np.float64)
                            final_bets = transport_metric.filter_drifted_bets(
                                final_bets, X_lf,
                            )
                            drift_killed = sum(
                                1 for b in final_bets
                                if isinstance(b, dict) and b.get("transport_killed")
                            )
                            if drift_killed > 0:
                                logger.warning(
                                    f"[Transport] {drift_killed} bahis drift nedeniyle iptal!"
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9j) Quantum Annealer â€“ optimal kupon sepeti â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 3:
                try:
                    qa_solution = q_annealer.optimize_from_bets(final_bets)
                    if qa_solution.n_bets > 0:
                        logger.info(
                            f"[Annealer] Optimal portfÃ¶y: {qa_solution.n_bets} bahis, "
                            f"Sharpe={qa_solution.sharpe_ratio:.2f}, "
                            f"Ã§eÅŸitlilik={qa_solution.diversification:.0%} "
                            f"({qa_solution.method}, {qa_solution.elapsed_ms:.0f}ms)"
                        )
                        # SeÃ§ilmeyen bahisleri iÅŸaretle
                        selected_ids = set(qa_solution.selected_matches)
                        for bet in final_bets:
                            if isinstance(bet, dict):
                                if bet.get("match_id", "") not in selected_ids:
                                    bet["annealer_excluded"] = True
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9k) Ricci Curvature â€“ sistemik risk kontrolÃ¼ â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                try:
                    match_list = [
                        b for b in final_bets if isinstance(b, dict)
                    ]
                    if match_list:
                        G = ricci_flow.build_market_graph(match_list)
                        if G and G.number_of_edges() > 0:
                            ricci_report = ricci_flow.analyze(G, name=f"cycle_{cycle}")
                            if ricci_report.kill_betting:
                                logger.warning(
                                    f"[Ricci] KRÄ°TÄ°K: Îº={ricci_report.avg_curvature:.4f} "
                                    f"â†’ SÄ°STEMÄ°K RÄ°SK! TÃ¼m bahisler durduruldu."
                                )
                                final_bets = ricci_flow.adjust_bets_by_curvature(
                                    final_bets, ricci_report,
                                )
                            elif ricci_report.stress_level in ("high", "moderate"):
                                logger.info(
                                    f"[Ricci] Stres={ricci_report.stress_level}: "
                                    f"Îº={ricci_report.avg_curvature:.4f}, "
                                    f"kriz={ricci_report.crisis_probability:.0%}, "
                                    f"stake x{ricci_report.stake_multiplier:.1f}"
                                )
                                final_bets = ricci_flow.adjust_bets_by_curvature(
                                    final_bets, ricci_report,
                                )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9l) Mimic â€“ insansÄ± gecikme enjeksiyonu â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                try:
                    if mimic.should_browse_first():
                        await mimic.human_delay(action="browse")
                    if mimic.should_idle():
                        await mimic.idle_pause()
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9m) Blind Strategy â€“ ÅŸifreli Kelly hesaplama â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                try:
                    probs = [
                        float(b.get("confidence", 0.5))
                        for b in final_bets if isinstance(b, dict)
                    ]
                    odds_list = [
                        float(b.get("odds", 2.0))
                        for b in final_bets if isinstance(b, dict)
                    ]
                    if probs and odds_list:
                        kelly_result = blind_strategy.blind_kelly(
                            blind_strategy.encrypt(probs, "kelly_probs"),
                            probs, odds_list,
                        )
                        if kelly_result.plaintext_result:
                            for idx, bet in enumerate(final_bets):
                                if isinstance(bet, dict) and idx < len(kelly_result.plaintext_result):
                                    bet["blind_kelly"] = kelly_result.plaintext_result[idx]
                            logger.debug(
                                f"[Blind] Åifreli Kelly tamamlandÄ±: "
                                f"{kelly_result.encryption_scheme or 'masked'}, "
                                f"{kelly_result.compute_time_ms:.1f}ms"
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9n) War Room â€“ multi-agent tartÄ±ÅŸma â”€â”€
            if isinstance(final_bets, list) and len(final_bets) > 0:
                try:
                    for bet in final_bets[:3]:
                        if not isinstance(bet, dict):
                            continue
                        ev = bet.get("ev", 0)
                        if ev > 0.05:
                            debate = war_room.debate(
                                match_info={
                                    "home": bet.get("home_team", ""),
                                    "away": bet.get("away_team", ""),
                                    "odds": bet.get("odds", 0),
                                    "prob": bet.get("confidence", 0),
                                    "ev": ev,
                                    "kelly": bet.get("kelly_fraction", 0),
                                    "confidence": bet.get("confidence", 0),
                                },
                                match_id=bet.get("match_id", ""),
                            )
                            bet["war_room_verdict"] = debate.majority_verdict
                            bet["war_room_consensus"] = debate.consensus
                            if debate.majority_verdict == "SKIP":
                                bet["war_room_skip"] = True
                                logger.warning(
                                    f"[WarRoom] {bet.get('home_team','')} vs "
                                    f"{bet.get('away_team','')}: "
                                    f"Ã‡oÄŸunluk SKIP! "
                                    f"({debate.bet_count}B/"
                                    f"{debate.skip_count}S/"
                                    f"{debate.hold_count}H)"
                                )
                            elif debate.consensus:
                                logger.info(
                                    f"[WarRoom] {bet.get('home_team','')} vs "
                                    f"{bet.get('away_team','')}: "
                                    f"OYBÄ°RLÄ°ÄÄ° {debate.majority_verdict}!"
                                )
                            # Agent Poll â†’ Telegram anketi gÃ¶nder
                            try:
                                match_info_poll = {
                                    "home": bet.get("home_team", ""),
                                    "away": bet.get("away_team", ""),
                                    "odds": bet.get("odds", 0),
                                    "prob": bet.get("confidence", 0),
                                    "ev": ev,
                                    "kelly": bet.get("kelly_fraction", 0),
                                }
                                council = agent_poll.create_council_decision(
                                    debate, match_info_poll,
                                )
                                await agent_poll.send_poll(council)
                                logger.info(
                                    f"[AgentPoll] Konsey anketi gÃ¶nderildi: "
                                    f"{council.home} vs {council.away} â†’ "
                                    f"{council.consensus_emoji} {council.council_verdict}"
                                )
                            except Exception as e:
                                logger.debug(f"[Guardian] agent_poll: {type(e).__name__}: {e}")
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 9z) Konsolide Ceza Ã‡arpanÄ± â€“ tÃ¼m soft-scoring penaltÄ±larÄ±nÄ± uygula â”€â”€
            if isinstance(final_bets, list):
                for bet in final_bets:
                    if isinstance(bet, dict):
                        # TÃ¼m penaltÄ± Ã§arpanlarÄ±nÄ± topla
                        combined_penalty = 1.0
                        for pkey in ("uq_penalty", "philo_penalty", "iso_penalty", "entropy_penalty"):
                            combined_penalty *= bet.get(pkey, 1.0)
                        # Minimum stake korumasÄ±: %15'in altÄ±na dÃ¼ÅŸÃ¼rme
                        combined_penalty = max(combined_penalty, 0.15)
                        bet["combined_penalty"] = round(combined_penalty, 3)
                        # Kelly stake'i Ã§arpanla ayarla
                        if "kelly_stake" in bet:
                            bet["kelly_stake_raw"] = bet["kelly_stake"]
                            bet["kelly_stake"] = round(bet["kelly_stake"] * combined_penalty, 4)
                        if "stake" in bet:
                            bet["stake_raw"] = bet["stake"]
                            bet["stake"] = round(bet["stake"] * combined_penalty, 2)

            # â”€â”€ FILTER AUDIT LOG â”€â”€
            _filter_exit_count = len(final_bets) if isinstance(final_bets, list) else 0
            if _filter_entry_count > 0:
                survival_rate = _filter_exit_count / max(_filter_entry_count, 1)
                logger.info(
                    f"[FilterAudit] GiriÅŸ={_filter_entry_count} â†’ Ã‡Ä±kÄ±ÅŸ={_filter_exit_count} "
                    f"(Hayatta kalma: {survival_rate:.0%}). "
                    f"Filtre detay: {_filter_log}"
                )
                if survival_rate < 0.2:
                    logger.warning(
                        f"[FilterAudit] DÄ°KKAT: Filtre zinciri Ã§ok agresif! "
                        f"Sinyallerin {1-survival_rate:.0%}'Ä± yok edildi."
                    )

            # â”€â”€ 10) SonuÃ§larÄ± kaydet â”€â”€
            db.save_signals(final_bets, cycle=cycle)
            logger.success(f"DÃ¶ngÃ¼ #{cycle} tamamlandÄ± â€“ {len(final_bets)} sinyal Ã¼retildi.")

            # â”€â”€ 10-ev) Event Bus + gRPC â€“ sinyalleri kaydet ve yayÄ±nla â”€â”€
            for bet in final_bets:
                if isinstance(bet, dict):
                    await event_bus.emit(Event(
                        event_type="signal_generated",
                        source="analysis_loop",
                        match_id=bet.get("match_id", ""),
                        data=bet,
                        cycle=cycle,
                    ))
                    # gRPC signal channel
                    await grpc_comm.send_signal(bet, source="analysis_loop")

            # â”€â”€ 10-sh) Shadow Testing â€“ gÃ¶lge stratejilere bahis at â”€â”€
            for bet in final_bets:
                if isinstance(bet, dict):
                    for strat_name in shadow.strategy_names:
                        if shadow.is_shadow(strat_name):
                            shadow.place_bet(
                                strat_name,
                                bet.get("match_id", ""),
                                bet.get("selection", ""),
                                bet.get("odds", 0),
                                bet.get("stake", 0),
                                prob=bet.get("confidence", 0),
                                value_edge=bet.get("value_edge", 0),
                            )

            # â”€â”€ 10b) CLV kaydÄ± + HITL + XAI + Telegram bildirimi â”€â”€
            for bet in final_bets:
                if not isinstance(bet, dict):
                    continue
                ev = bet.get("ev", 0)
                odds = bet.get("odds", 0)
                mid = bet.get("match_id", "")

                # CLV giriÅŸ kaydÄ±
                if odds > 1.0:
                    clv_tracker.record_entry(mid, bet.get("selection", ""), odds, ev)

                # Value bildirimi â€“ interaktif butonlu (Human-in-the-Loop)
                if ev > 0.02:
                    signal_id = hitl.create_signal(
                        match_id=mid,
                        selection=bet.get("selection", ""),
                        odds=odds, ev=ev,
                        confidence=bet.get("confidence", 0),
                        stake_pct=bet.get("stake_pct", 0),
                        model_prediction=bet.get("prediction", ""),
                    )
                    await notifier.send_value_alert(bet, signal_id=signal_id)

                    # Psycho Profiler â€“ karar kaydÄ± (model Ã¶nerisi)
                    psycho.record_decision(
                        mid, bet.get("selection", ""),
                        human_decision="PENDING",
                        odds=odds, ev=ev,
                        confidence=bet.get("confidence", 0),
                        stake=bet.get("stake", 100),
                        model_rec="BET",
                    )

                    # RLHF â€“ geri bildirim kaydÄ± (sonuÃ§ geldiÄŸinde Ã¶dÃ¼l hesaplanÄ±r)
                    feedback_loop.record_feedback(
                        match_id=mid,
                        selection=bet.get("selection", ""),
                        odds=odds,
                        model_prob=bet.get("confidence", 0),
                        human_decision="approve",
                    )

                    # Senaryo butonlarÄ±nÄ± gÃ¶nder
                    try:
                        scenario_markup = scenario_sim.build_inline_markup(mid)
                        if scenario_markup:
                            await notifier.send(
                                f"ğŸ“‹ <b>Senaryo Analizi:</b> {mid}",
                                reply_markup=scenario_markup,
                            )
                    except Exception as e:
                        logger.debug(f"[Guardian] {type(e).__name__}: {e}")

                    # XAI: model kararÄ±nÄ± aÃ§Ä±kla ve gÃ¶rsel gÃ¶nder
                    try:
                        await xai.explain_and_send(
                            bet, f"{mid} Analiz", chart_sender=chart_sender,
                        )
                    except Exception as e:
                        logger.debug(f"[Guardian] {type(e).__name__}: {e}")

            # â”€â”€ 10c) Hedge kontrolÃ¼ â€“ canlÄ± maÃ§larda fÄ±rsat tara â”€â”€
            try:
                active_bets = db.get_active_bets() if hasattr(db, "get_active_bets") else []
                live_odds = db.get_live_odds() if hasattr(db, "get_live_odds") else {}
                if active_bets and live_odds:
                    hedge_opps = hedge_calc.scan_active_bets(active_bets, live_odds)
                    if hedge_opps:
                        await hedge_calc.notify_opportunities(hedge_opps, notifier)
            except Exception as e:
                logger.debug(f"[Guardian] hedge_scan: {type(e).__name__}: {e}")

            # â”€â”€ 10z-1) Strategy Cockpit â€“ Telegram HUD gÃ¼ncelle â”€â”€
            if cockpit and notifier:
                try:
                    status = portfolio_opt.status() if portfolio_opt else {}
                    rk_stats = regime_kelly.get_stats() if regime_kelly else {}
                    cockpit_data = cockpit.collect(
                        bankroll=status.get("bankroll", 10000),
                        peak=rk_stats.get("peak", 10000),
                        daily_pnl=status.get("total_pnl", 0),
                        total_bets=rk_stats.get("total_bets", 0),
                        win_rate=rk_stats.get("win_rate", 0),
                        active_bets=len(final_bets),
                        cycle=cycle,
                        regime_kelly=regime_kelly,
                        vol_analyzer=vol_analyzer,
                    )
                    # Her 5 dÃ¶ngÃ¼de cockpit gÃ¼ncelle (spam Ã¶nleme)
                    if cycle % 5 == 0:
                        await cockpit.update_cockpit(notifier)
                except Exception as e:
                    logger.debug(f"[Guardian] cockpit: {type(e).__name__}: {e}")

            # â”€â”€ 10z-2) Strategy Evolver â€“ her 100 dÃ¶ngÃ¼de DNA evrimleÅŸtir â”€â”€
            if strategy_evolver and cycle % 100 == 0:
                try:
                    # Son 100 bahis sonucunu topla
                    recent = list(regime_kelly._results) if regime_kelly else []
                    if len(recent) >= 20:
                        evo_results = [
                            {
                                "won": r["won"],
                                "pnl": r["pnl"],
                                "ev": 0,
                                "odds": 2.0,
                                "prob": 0.5,
                            }
                            for r in recent[-100:]
                        ]
                        evo_report = strategy_evolver.evolve(evo_results)
                        logger.info(
                            f"[Evolver] Nesil #{evo_report.generation}: "
                            f"best={evo_report.best_fitness:.4f}, "
                            f"avg={evo_report.avg_fitness:.4f}"
                        )
                        # En iyi DNA'yÄ± sisteme uygula
                        best_dna = strategy_evolver.get_best_dna()
                        best_params = best_dna.to_dict()
                        if regime_kelly:
                            regime_kelly._base_fraction = best_params.get(
                                "kelly_fraction", 0.25
                            )
                            regime_kelly._min_edge = best_params.get(
                                "min_edge", 0.03
                            )
                except Exception as e:
                    logger.debug(f"[Guardian] evolver: {type(e).__name__}: {e}")

            # â”€â”€ 10z-3) Guardian heartbeat & saÄŸlÄ±k kontrolÃ¼ â”€â”€
            if guardian:
                guardian.heartbeat("analysis_loop")
                if cycle % 20 == 0:
                    silent = guardian.check_heartbeats(timeout=600)
                    if silent:
                        logger.warning(
                            f"[Guardian] Sessiz modÃ¼ller: {', '.join(silent)}"
                        )
                    gh_report = guardian.health_report()
                    if gh_report["open_circuits"]:
                        logger.error(
                            f"[Guardian] AÃ§Ä±k devreler: "
                            f"{', '.join(gh_report['open_circuits'])}"
                        )

            # â”€â”€ 11) Her 10 dÃ¶ngÃ¼de: snapshot + sesli bÃ¼lten + gÃ¼nlÃ¼k rapor â”€â”€
            if cycle % 10 == 0:
                dvc.snapshot(f"cycle_{cycle}")
                await podcast.produce(final_bets)
                status = portfolio_opt.status() if portfolio_opt else {}
                await notifier.send_daily_summary({
                    "bankroll": status.get("bankroll", 10000),
                    "pnl_today": status.get("total_pnl", 0),
                    "roi": 0,
                    "bets_placed": len(final_bets),
                    "bets_won": status.get("win_rate", 0),
                })

            # â”€â”€ 11b) HaftalÄ±k: Psikoloji raporu (her 50 dÃ¶ngÃ¼de) â”€â”€
            if cycle % 50 == 0:
                try:
                    psych_report = psycho.weekly_report()
                    if psych_report.total_decisions > 0:
                        await notifier.send(psych_report.telegram_text)
                except Exception as e:
                    logger.debug(f"[Guardian] {type(e).__name__}: {e}")

        except Exception as e:
            logger.exception(f"DÃ¶ngÃ¼ #{cycle} hatasÄ±: {e}")
            # Hata bildirimi â†’ Telegram
            await notifier.send_error_alert(e, module="analysis_loop")

        # DÃ¶ngÃ¼ performans metrikleri
        _cycle_elapsed = (_time.perf_counter() - _cycle_start) * 1000
        _cycle_times.append(_cycle_elapsed)
        if len(_cycle_times) > 100:
            _cycle_times = _cycle_times[-100:]
        _avg_cycle = sum(_cycle_times) / len(_cycle_times)
        logger.info(
            f"[Perf] DÃ¶ngÃ¼ #{cycle}: {_cycle_elapsed:.0f}ms "
            f"(ort={_avg_cycle:.0f}ms, max={max(_cycle_times):.0f}ms)"
        )
        if super_log and hasattr(super_log, "log_library_event"):
            super_log.log_library_event(
                "analysis_loop", f"cycle_{cycle}",
                cycle_ms=round(_cycle_elapsed, 1),
                avg_ms=round(_avg_cycle, 1),
            )

        # DÃ¶ngÃ¼ arasÄ± bekleme (canlÄ±da kÄ±sa, pre-match'te uzun)
        await asyncio.sleep(30)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CLI KOMUTLARI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def run(
    mode: str = typer.Option("full", help="Ã‡alÄ±ÅŸma modu: live | pre | full"),
    headless: bool = typer.Option(True, help="TarayÄ±cÄ± headless mi?"),
    telegram: bool = typer.Option(True, help="Telegram botunu baÅŸlat"),
    dashboard: bool = typer.Option(False, help="TUI dashboard aÃ§"),
):
    """Botu baÅŸlatÄ±r ve TÃœM katmanlarÄ± ayaÄŸa kaldÄ±rÄ±r.

    VarsayÄ±lan: mode=full, telegram=True â†’ TÃ¼m Ã¶zellikler aktif.
    'python bahis.py' veya 'python bahis.py run' ile Ã§alÄ±ÅŸtÄ±rÄ±labilir.
    """
    sigs = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        sigs.append(signal.SIGBREAK)
    for sig in sigs:
        try:
            signal.signal(sig, _handle_signal)
        except (OSError, ValueError):
            pass
    logger.info(
        f"[CLI] BaÅŸlatÄ±lÄ±yor: mode={mode}, headless={headless}, "
        f"telegram={telegram}, dashboard={dashboard}"
    )
    try:
        asyncio.run(_boot(mode=mode, headless=headless, telegram=telegram, dashboard=dashboard))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt alÄ±ndÄ± â€“ sÃ¼reÃ§ sonlandÄ±rÄ±lÄ±yor.")


@app.command()
def backtest(
    start: str = typer.Option("2024-01-01", help="BaÅŸlangÄ±Ã§ tarihi"),
    end: str = typer.Option("2026-01-01", help="BitiÅŸ tarihi"),
):
    """GeÃ§miÅŸ veri Ã¼zerinde strateji testi Ã§alÄ±ÅŸtÄ±rÄ±r."""
    from src.core.vector_backtester import VectorBacktester
    from src.memory.db_manager import DBManager

    console.rule("[bold cyan]BACKTEST BAÅLATILIYOR[/]")
    db = DBManager()
    bt = VectorBacktester()
    results = bt.run(db=db, start=start, end=end)
    console.print(results)
    console.rule("[bold green]BACKTEST TAMAMLANDI[/]")


@app.command()
def report():
    """Strateji saÄŸlÄ±k raporu Ã¼retir (PDF)."""
    from src.utils.strategy_health_report import StrategyHealthReport

    console.rule("[bold cyan]RAPOR ÃœRETÄ°LÄ°YOR[/]")
    rpt = StrategyHealthReport()
    path = rpt.generate_pdf()
    console.print(f"[green]Rapor kaydedildi:[/] {path}")


@app.command()
def doctor():
    """Sistem bileÅŸenlerini kontrol eder."""
    from src.ingestion.auto_healer import AutoHealer

    console.rule("[bold cyan]SÄ°STEM SAÄLIK KONTROLÃœ[/]")
    healer = AutoHealer()
    healer.diagnose()
    console.rule("[bold green]KONTROL TAMAMLANDI[/]")


@app.command()
def docs():
    """Otomatik dokÃ¼mantasyonu gÃ¼nceller."""
    from src.utils.auto_doc_generator import AutoDocGenerator

    gen = AutoDocGenerator()
    gen.generate()
    console.print("[green]DokÃ¼mantasyon gÃ¼ncellendi.[/]")


@app.command()
def web():
    """Streamlit web dashboard'u baÅŸlatÄ±r."""
    import subprocess
    dashboard_path = ROOT / "src" / "ui" / "streamlit_dashboard.py"
    console.print(f"[cyan]Streamlit baÅŸlatÄ±lÄ±yor â†’[/] {dashboard_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


@app.command()
def analyze(
    home: str = typer.Argument(..., help="Ev sahibi takÄ±m"),
    away: str = typer.Argument(..., help="Deplasman takÄ±mÄ±"),
    home_xg: float = typer.Option(1.4, help="Ev sahibi xG"),
    away_xg: float = typer.Option(1.1, help="Deplasman xG"),
):
    """Tek bir maÃ§Ä± derinlemesine analiz eder (Poisson + Dixon-Coles + MC + Elo + GB + VIX)."""
    from src.quant.poisson_model import PoissonModel
    from src.quant.monte_carlo_engine import MonteCarloEngine
    from src.quant.elo_glicko_rating import EloGlickoSystem
    from src.quant.dixon_coles_model import DixonColesModel
    from src.quant.time_decay import TeamVolatilityIndex, ExponentialTimeDecay
    from src.quant.gradient_boosting import GradientBoostingModel
    from rich.table import Table
    from rich.panel import Panel

    console.rule(f"[bold cyan]{home} vs {away}[/]")

    poisson_m = PoissonModel()
    mc = MonteCarloEngine()
    elo = EloGlickoSystem()
    dc = DixonColesModel()
    gb = GradientBoostingModel()
    td = ExponentialTimeDecay(preset="moderate")
    vix = TeamVolatilityIndex(decay=td)

    # Standart Poisson
    probs = poisson_m.match_outcome_probs(home_xg, away_xg)
    ou = poisson_m.over_under_probs(home_xg, away_xg)
    btts = poisson_m.btts_probs(home_xg, away_xg)
    scores = poisson_m.most_likely_scores(home_xg, away_xg, 5)

    # Dixon-Coles dÃ¼zeltilmiÅŸ Poisson
    dc_pred = dc.predict(home, away, home_xg, away_xg)
    dc_comparison = dc.compare_with_standard_poisson(home_xg, away_xg)

    # Monte Carlo
    sim = mc.simulate_match(home_xg, away_xg)

    # Elo
    elo_pred = elo.predict(home, away)

    # Gradient Boosting (heuristic mode)
    import polars as pl
    gb_input = pl.DataFrame([{
        "match_id": f"{home}_vs_{away}",
        "home_odds": round(1 / max(probs["prob_home"], 0.01), 2),
        "draw_odds": round(1 / max(probs["prob_draw"], 0.01), 2),
        "away_odds": round(1 / max(probs["prob_away"], 0.01), 2),
        "home_xg": home_xg, "away_xg": away_xg,
    }])
    gb_preds = gb.predict(gb_input)

    # â”€â”€ Model KarÅŸÄ±laÅŸtÄ±rma Tablosu â”€â”€
    table = Table(title="ğŸ“Š Model KarÅŸÄ±laÅŸtÄ±rmasÄ±", show_lines=True)
    table.add_column("Model", style="cyan", width=16)
    table.add_column("Ev (%)", justify="right")
    table.add_column("Beraberlik (%)", justify="right")
    table.add_column("Deplasman (%)", justify="right")

    table.add_row("Std Poisson", f"{probs['prob_home']:.1%}", f"{probs['prob_draw']:.1%}", f"{probs['prob_away']:.1%}")
    table.add_row("[bold]Dixon-Coles[/]", f"{dc_pred['prob_home']:.1%}", f"{dc_pred['prob_draw']:.1%}", f"{dc_pred['prob_away']:.1%}")
    table.add_row("Monte Carlo", f"{sim['prob_home']:.1%}", f"{sim['prob_draw']:.1%}", f"{sim['prob_away']:.1%}")
    table.add_row("Elo/Glicko", f"{elo_pred['prob_home']:.1%}", f"{elo_pred['prob_draw']:.1%}", f"{elo_pred['prob_away']:.1%}")

    if not gb_preds.is_empty():
        gbr = gb_preds.to_dicts()[0]
        table.add_row("LightGBM", f"{gbr.get('gb_prob_home',0):.1%}", f"{gbr.get('gb_prob_draw',0):.1%}", f"{gbr.get('gb_prob_away',0):.1%}")

    console.print(table)

    # â”€â”€ Dixon-Coles DÃ¼zeltme Paneli â”€â”€
    correction = dc_comparison.get("draw_correction", 0)
    correction_text = (
        f"[yellow]Dixon-Coles beraberlik dÃ¼zeltmesi:[/] {correction:+.3f}\n"
        f"Standart Poisson beraberlik: {dc_comparison.get('std_draw', 0):.1%}\n"
        f"Dixon-Coles beraberlik: {dc_pred['prob_draw']:.1%}\n"
        f"Ï (rho) parametresi: {dc_pred.get('rho', 0):.3f}\n\n"
        f"[dim]{dc_comparison.get('note', '')}[/]"
    )
    console.print(Panel(correction_text, title="ğŸ”§ Dixon-Coles DÃ¼zeltme", border_style="yellow"))

    # â”€â”€ Gol & Detay â”€â”€
    console.print(f"\n[yellow]Ãœ 2.5:[/] Poisson={ou.get('over_25', 0):.1%} | DC={dc_pred.get('prob_over25', 0):.1%} | MC={sim['prob_over_25']:.1%}")
    console.print(f"[yellow]KG Var:[/] Poisson={btts['btts_yes']:.1%} | DC={dc_pred.get('prob_btts', 0):.1%} | MC={sim['prob_btts']:.1%}")
    console.print(f"[yellow]Ort. Gol (MC):[/] {sim['avg_total_goals']:.2f}")

    # â”€â”€ En olasÄ± skorlar (Dixon-Coles) â”€â”€
    console.print(f"\n[cyan]En olasÄ± skorlar (Dixon-Coles):[/]")
    for s in dc_pred.get("top_scores", [])[:5]:
        console.print(f"  {s['score']} â†’ {s['prob']:.1%}")

    # â”€â”€ Zaman Decay bilgisi â”€â”€
    console.print(f"\n[magenta]Zaman Decay:[/] yarÄ±-Ã¶mÃ¼r = {td.half_life_days:.0f} gÃ¼n")
    console.print(f"  30 gÃ¼n Ã¶nceki verinin aÄŸÄ±rlÄ±ÄŸÄ±: {td.weight(30):.2%}")
    console.print(f"  180 gÃ¼n Ã¶nceki verinin aÄŸÄ±rlÄ±ÄŸÄ±: {td.weight(180):.2%}")
    console.print(f"  365 gÃ¼n Ã¶nceki verinin aÄŸÄ±rlÄ±ÄŸÄ±: {td.weight(365):.2%}")


@app.command()
def optimize(
    generations: int = typer.Option(50, help="Genetik algoritma jenerasyon sayÄ±sÄ±"),
    population: int = typer.Option(100, help="PopÃ¼lasyon bÃ¼yÃ¼klÃ¼ÄŸÃ¼"),
):
    """Genetik Algoritma ile strateji parametrelerini optimize eder."""
    from src.core.genetic_optimizer import GeneticOptimizer
    from src.core.vector_backtester import VectorBacktester
    from src.memory.db_manager import DBManager

    console.rule("[bold cyan]GENETÄ°K ALGORÄ°TMA â€“ PARAMETRE OPTÄ°MÄ°ZASYONU[/]")

    db = DBManager()
    bt = VectorBacktester()
    ga = GeneticOptimizer(population_size=population)

    def backtest_fn(params: dict) -> dict:
        """Her birey iÃ§in backtest Ã§alÄ±ÅŸtÄ±r."""
        return bt.run(db=db, params=params, start="2025-01-01", end="2026-01-01")

    best = ga.evolve(backtest_fn, generations=generations)
    ga.save_config(best)

    console.print(f"\n[green]En Ä°yi Parametreler:[/]")
    from rich.table import Table
    table = Table(show_lines=True)
    table.add_column("Parametre", style="cyan")
    table.add_column("DeÄŸer", justify="right")
    for k, v in best.genes.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    console.print(f"\n[bold green]ROI: {best.roi:.2%} | Drawdown: {best.drawdown:.1%} | Sharpe: {best.sharpe:.2f}[/]")
    console.rule("[bold green]OPTÄ°MÄ°ZASYON TAMAMLANDI â€“ config.json gÃ¼ncellendi[/]")


@app.command()
def hedge(
    stake: float = typer.Argument(..., help="Orijinal bahis miktarÄ± (â‚º)"),
    odds: float = typer.Argument(..., help="Orijinal bahis oranÄ±"),
    selection: str = typer.Argument(..., help="Orijinal seÃ§im: home/draw/away"),
    live_home: float = typer.Option(0, help="CanlÄ± ev sahibi oranÄ±"),
    live_draw: float = typer.Option(0, help="CanlÄ± beraberlik oranÄ±"),
    live_away: float = typer.Option(0, help="CanlÄ± deplasman oranÄ±"),
):
    """CanlÄ± maÃ§ta hedge/arbitraj fÄ±rsatÄ± hesaplar."""
    from src.core.hedge_calculator import HedgeCalculator

    console.rule("[bold cyan]HEDGE HESAPLAYICI[/]")

    calc = HedgeCalculator()

    # Surebet kontrolÃ¼
    if live_home > 0 and live_draw > 0 and live_away > 0:
        surebet = calc.check_surebet(live_home, live_draw, live_away)
        if surebet:
            console.print(f"[bold red]ğŸš¨ SUREBET TESPÄ°T EDÄ°LDÄ°![/]")
            console.print(surebet.action_text)

    # Hedge hesaplama
    live_odds = {}
    if live_home > 0:
        live_odds["home"] = live_home
    if live_draw > 0:
        live_odds["draw"] = live_draw
    if live_away > 0:
        live_odds["away"] = live_away

    if live_odds:
        opp = calc.calculate_hedge(stake, odds, selection, live_odds)
        if opp:
            console.print(f"\n[bold green]ğŸ’° HEDGE FIRSATI:[/]")
            console.print(opp.action_text)
            console.print(f"Garanti KÃ¢r: â‚º{opp.guaranteed_profit:.2f} ({opp.guaranteed_profit_pct:.1%})")
        else:
            console.print("[yellow]Åu an kÃ¢rlÄ± hedge fÄ±rsatÄ± yok.[/]")

    # Cash-out
    if live_odds.get(selection, 0) > 0:
        co = calc.calculate_cashout(stake, odds, live_odds[selection])
        console.print(f"\n[cyan]Cash-out deÄŸeri: â‚º{co['cashout_value']:.2f} ({co['recommendation']})[/]")


@app.command()
def xai(
    home: str = typer.Argument(..., help="Ev sahibi takÄ±m"),
    away: str = typer.Argument(..., help="Deplasman takÄ±mÄ±"),
):
    """MaÃ§ tahminini SHAP ile aÃ§Ä±klar (Explainable AI)."""
    from src.quant.xai_explainer import XAIExplainer

    console.rule(f"[bold cyan]XAI â€“ {home} vs {away}[/]")
    explainer = XAIExplainer()

    # Basit feature set (demo)
    features = {
        "home_impl_prob": 0.45, "draw_impl_prob": 0.28,
        "away_impl_prob": 0.27, "xg_diff": 0.3,
        "home_win_rate": 0.6, "away_win_rate": 0.35,
        "form_diff": 4, "possession_diff": 5,
    }

    result = explainer.explain(features)
    console.print(f"\n[bold]Model KararÄ±:[/]")
    console.print(result["explanation_text"])

    fig = explainer.plot_waterfall(features, title=f"{home} vs {away}")
    if fig:
        import matplotlib.pyplot as plt
        plt.show()


@app.command()
def fair_odds(
    home_odds: float = typer.Argument(..., help="Ev sahibi oranÄ±"),
    draw_odds: float = typer.Argument(..., help="Beraberlik oranÄ±"),
    away_odds: float = typer.Argument(..., help="Deplasman oranÄ±"),
    model_home: float = typer.Option(0, help="Model ev sahibi olasÄ±lÄ±ÄŸÄ±"),
):
    """Fair Value / No-Vig analizi â€“ bahisÃ§inin marjÄ±nÄ± Ã§Ä±kar."""
    from src.core.fair_value_engine import FairValueEngine
    from rich.table import Table
    from rich.panel import Panel

    console.rule("[bold cyan]FAIR VALUE ANALÄ°ZÄ°[/]")
    engine = FairValueEngine()

    # No-vig hesaplama
    novig = engine.remove_vig(home_odds, draw_odds, away_odds)
    console.print(f"\n[yellow]BahisÃ§i MarjÄ±:[/] %{novig.get('raw_margin', 0):.2f}")

    table = Table(title="MarjsÄ±z (No-Vig) Oranlar", show_lines=True)
    table.add_column("SeÃ§im", style="cyan")
    table.add_column("Piyasa", justify="right")
    table.add_column("No-Vig Oran", justify="right", style="green")
    table.add_column("GerÃ§ek OlasÄ±lÄ±k", justify="right")

    table.add_row("Ev", f"{home_odds:.2f}", f"{novig.get('novig_home_odds', 0):.3f}", f"{novig.get('novig_home', 0):.1%}")
    table.add_row("Ber", f"{draw_odds:.2f}", f"{novig.get('novig_draw_odds', 0):.3f}", f"{novig.get('novig_draw', 0):.1%}")
    table.add_row("Dep", f"{away_odds:.2f}", f"{novig.get('novig_away_odds', 0):.3f}", f"{novig.get('novig_away', 0):.1%}")
    console.print(table)

    # Model olasÄ±lÄ±ÄŸÄ± verilmiÅŸse Value Edge hesapla
    if model_home > 0:
        fv = engine.analyze(model_home, home_odds, "home")
        edge_color = "green" if fv.is_value else "red"
        console.print(Panel(
            f"Fair Odds: {fv.fair_odds:.3f}\n"
            f"Value Edge: [{edge_color}]{fv.value_edge_pct}[/]\n"
            f"Tier: {fv.tier.upper()}\n"
            f"Kelly Stake: {fv.kelly_stake:.2%}",
            title="ğŸ’° Value Analizi", border_style=edge_color,
        ))


@app.command()
def fetch_data(
    league: str = typer.Option("super_lig", help="Lig kodu"),
    season: str = typer.Option("2526", help="Sezon kodu"),
):
    """TÃ¼m Ã¼cretsiz kaynaklardan veri toplar."""
    from src.ingestion.data_sources import DataSourceAggregator
    from src.memory.db_manager import DBManager

    console.rule("[bold cyan]VERÄ° TOPLAMA â€“ TÃœM KAYNAKLAR[/]")
    db = DBManager()
    agg = DataSourceAggregator(db=db)

    results = asyncio.run(agg.fetch_all(league, season))
    for source, data in results.items():
        n = len(data) if isinstance(data, list) else 0
        console.print(f"  {'âœ“' if n > 0 else 'âœ—'} [cyan]{source}[/]: {n} kayÄ±t")

    console.rule("[bold green]VERÄ° TOPLAMA TAMAMLANDI[/]")


@app.command()
def graph_query(
    query_type: str = typer.Argument(..., help="Sorgu tipi: referee_bias | h2h | player_impact"),
    arg1: str = typer.Argument("", help="Ä°lk argÃ¼man (hakem adÄ± / takÄ±m A)"),
    arg2: str = typer.Argument("", help="Ä°kinci argÃ¼man (takÄ±m B)"),
):
    """Neo4j Graph DB Ã¼zerinde sorgu Ã§alÄ±ÅŸtÄ±rÄ±r."""
    from src.memory.neo4j_graph import Neo4jFootballGraph
    from rich.table import Table

    console.rule("[bold cyan]GRAPH INTELLIGENCE SORGUSU[/]")
    g = Neo4jFootballGraph()
    g.connect()

    if query_type == "referee_bias":
        result = g.query_referee_bias(arg1)
        console.print(f"[cyan]Hakem:[/] {arg1}")
        console.print(f"  Ort. Ev KartÄ±:  {result.get('avg_home_cards', 0):.1f}")
        console.print(f"  Ort. Dep KartÄ±: {result.get('avg_away_cards', 0):.1f}")
        console.print(f"  Deplasman YanlÄ±lÄ±ÄŸÄ±: {result.get('away_bias', 0):+.1f}")
        console.print(f"  Toplam MaÃ§: {result.get('total_matches', 0)}")
        if result.get("is_strict_away"):
            console.print("[red]âš ï¸ Bu hakem deplasmana sert![/]")

    elif query_type == "h2h":
        results = g.query_h2h_graph(arg1, arg2)
        table = Table(title=f"{arg1} vs {arg2} â€“ Son MaÃ§lar")
        table.add_column("Tarih", style="cyan")
        table.add_column("Skor", justify="center")
        for r in results[:10]:
            table.add_row(
                str(r.get("date", "")),
                f"{r.get('home_goals', 0)}-{r.get('away_goals', 0)}"
            )
        console.print(table)

    elif query_type == "player_impact":
        results = g.query_key_player_impact(arg1, top_n=5)
        table = Table(title=f"{arg1} â€“ Kilit Oyuncular (Graph)")
        table.add_column("Oyuncu", style="cyan")
        table.add_column("Poz.", justify="center")
        table.add_column("Ort. Rating", justify="right")
        table.add_column("Gol", justify="right")
        table.add_column("Asist", justify="right")
        for r in results:
            table.add_row(
                str(r.get("player", "")),
                str(r.get("position", "")),
                f"{r.get('avg_rating', 0):.1f}",
                str(r.get("goals", 0)),
                str(r.get("assists", 0)),
            )
        console.print(table)
    else:
        console.print(f"[red]Bilinmeyen sorgu tipi: {query_type}[/]")
        console.print("GeÃ§erli tipler: referee_bias, h2h, player_impact")

    console.print(f"\n[dim]Graph Stats: {g.stats()}[/]")
    g.close()


@app.command()
def network(
    team: str = typer.Argument(..., help="TakÄ±m adÄ±"),
    missing: str = typer.Option("", help="Eksik oyuncular (virgÃ¼lle ayÄ±r)"),
):
    """Pas aÄŸÄ± analizi â€“ oyuncu Ã¶nem sÄ±ralamasÄ± ve eksiklik etkisi."""
    from src.quant.network_centrality import PassNetworkAnalyzer
    from rich.table import Table

    console.rule(f"[bold cyan]AÄ ANALÄ°ZÄ° â€“ {team}[/]")
    analyzer = PassNetworkAnalyzer()

    # Demo: basit kadro ile aÄŸ oluÅŸtur
    demo_lineup = [
        {"name": "Kaleci", "position": "GK"},
        {"name": "Defans_1", "position": "DEF"},
        {"name": "Defans_2", "position": "DEF"},
        {"name": "Defans_3", "position": "DEF"},
        {"name": "Defans_4", "position": "DEF"},
        {"name": "Ortasaha_1", "position": "MID"},
        {"name": "Ortasaha_2", "position": "MID"},
        {"name": "Ortasaha_3", "position": "MID"},
        {"name": "Forvet_1", "position": "FWD"},
        {"name": "Forvet_2", "position": "FWD"},
        {"name": "Forvet_3", "position": "FWD"},
    ]
    analyzer.load_from_match_data(team, {"lineup": demo_lineup})

    rankings = analyzer.calculate_centrality(team)
    table = Table(title="Oyuncu Ã–nem SÄ±ralamasÄ± (Centrality)", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Oyuncu", style="cyan")
    table.add_column("Pozisyon", justify="center")
    table.add_column("PageRank", justify="right")
    table.add_column("Betweenness", justify="right")
    table.add_column("Composite", justify="right", style="green")
    table.add_column("Kilit?", justify="center")

    for r in rankings:
        table.add_row(
            str(r.rank_in_team),
            r.name, r.position,
            f"{r.pagerank:.4f}",
            f"{r.betweenness:.4f}",
            f"{r.composite_score:.4f}",
            "â­" if r.is_key_player else "",
        )
    console.print(table)

    if missing:
        missing_list = [m.strip() for m in missing.split(",") if m.strip()]
        penalties = analyzer.absence_impact(team, missing_list)
        console.print(f"\n[yellow]Eksik oyuncu analizi:[/]")
        for p in penalties:
            color = {"critical": "red", "high": "yellow", "medium": "cyan"}.get(p.severity, "dim")
            console.print(
                f"  [{color}]{p.player}[/]: xG Ã§arpanÄ±={p.penalty_factor:.2%}, "
                f"ÅŸiddet={p.severity} ({p.reason})"
            )
        combined = analyzer.combined_penalty(team, missing_list)
        console.print(f"\n[bold]Toplam xG dÃ¼zeltmesi: {combined:.2%}[/]")


@app.command()
def seasonality(
    team: str = typer.Argument(..., help="TakÄ±m adÄ±"),
    date: str = typer.Option("", help="Hedef tarih (YYYY-MM-DD). BoÅŸsa bugÃ¼n."),
):
    """Mevsimsellik analizi â€“ takÄ±mÄ±n dÃ¶nemsel performansÄ±."""
    from src.quant.prophet_seasonality import ProphetSeasonalityAnalyzer
    from rich.panel import Panel

    console.rule(f"[bold cyan]MEVSÄ°MSELLÄ°K â€“ {team}[/]")
    analyzer = ProphetSeasonalityAnalyzer()

    target = date if date else None
    result = analyzer.predict(team, target)

    color = "red" if result.avoid_signal else ("yellow" if result.is_negative_season else "green")
    console.print(Panel(
        f"Tarih: {result.current_date}\n"
        f"Mevsimsel Etki: {result.seasonal_effect:+.1%}\n"
        f"Trend: {result.trend:+.1%}\n"
        f"Negatif Sezon: {'EVET' if result.is_negative_season else 'HayÄ±r'}\n"
        f"AVOID Sinyali: {'âš ï¸ EVET' if result.avoid_signal else 'HayÄ±r'}\n"
        f"GÃ¼ven: {result.confidence:.0%}\n"
        f"Metod: {result.method}\n\n"
        f"{result.explanation}",
        title=f"ğŸ“… {team} â€“ Mevsimsellik Raporu",
        border_style=color,
    ))

    # Decomposition detaylarÄ±
    if result.decomposition:
        console.print(f"\n[dim]Decomposition: {result.decomposition}[/]")


@app.command()
def kalman_power(
    top_n: int = typer.Option(20, help="GÃ¶sterilecek takÄ±m sayÄ±sÄ±"),
):
    """Kalman Filtresi gÃ¼Ã§ sÄ±ralamasÄ±."""
    from src.quant.kalman_tracker import KalmanTeamTracker
    from rich.table import Table

    console.rule("[bold cyan]KALMAN FÄ°LTRESÄ° â€“ GÃœÃ‡ SIRALAMASI[/]")
    tracker = KalmanTeamTracker()

    # Demo veri yÃ¼kle
    demo_results = [
        {"home": "Galatasaray", "away": "Fenerbahce", "hg": 2, "ag": 1, "date": "2026-02-01"},
        {"home": "Besiktas", "away": "Trabzonspor", "hg": 1, "ag": 1, "date": "2026-02-01"},
        {"home": "Fenerbahce", "away": "Besiktas", "hg": 3, "ag": 0, "date": "2026-02-08"},
        {"home": "Trabzonspor", "away": "Galatasaray", "hg": 0, "ag": 2, "date": "2026-02-08"},
        {"home": "Galatasaray", "away": "Besiktas", "hg": 1, "ag": 0, "date": "2026-02-15"},
        {"home": "Fenerbahce", "away": "Trabzonspor", "hg": 2, "ag": 2, "date": "2026-02-15"},
    ]
    tracker.bulk_update(demo_results)

    rankings = tracker.power_rankings(top_n)
    table = Table(title="Kalman Power Rankings", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("TakÄ±m", style="cyan", min_width=15)
    table.add_column("GÃ¼Ã§", justify="right", style="green")
    table.add_column("Momentum", justify="right")
    table.add_column("Belirsizlik", justify="right")
    table.add_column("Trend", justify="center")

    trend_icons = {"rising": "[green]â–²[/]", "falling": "[red]â–¼[/]", "stable": "[dim]â—[/]"}
    for r in rankings:
        table.add_row(
            str(r["rank"]),
            r["team"],
            f"{r['strength']:.1f}",
            f"{r['momentum']:+.2f}",
            f"Â±{r['uncertainty']:.1f}",
            trend_icons.get(r["trend"], "â—"),
        )
    console.print(table)

    # MaÃ§ tahmini demo
    if len(rankings) >= 2:
        t1 = rankings[0]["team"]
        t2 = rankings[-1]["team"]
        pred = tracker.predict_match(t1, t2)
        console.print(f"\n[cyan]{t1} vs {t2}:[/]")
        console.print(
            f"  Ev: {pred['prob_home']:.1%} | Ber: {pred['prob_draw']:.1%} | "
            f"Dep: {pred['prob_away']:.1%} (gÃ¼venilirlik: {pred['reliability']:.0%})"
        )


@app.command()
def uncertainty(
    prob: float = typer.Argument(..., help="Model olasÄ±lÄ±ÄŸÄ± (0-1)"),
    match_id: str = typer.Option("test", help="MaÃ§ ID"),
    selection: str = typer.Option("home", help="SeÃ§im: home/draw/away"),
):
    """Conformal Prediction ile gÃ¼ven aralÄ±ÄŸÄ± hesapla."""
    from src.quant.uncertainty_quantifier import UncertaintyQuantifier
    from rich.panel import Panel

    console.rule("[bold cyan]CONFORMAL PREDICTION â€“ BELÄ°RSÄ°ZLÄ°K ANALÄ°ZÄ°[/]")
    uq = UncertaintyQuantifier()

    result = uq.quantify(match_id, selection, prob)

    color = "green" if result.is_certain else ("red" if result.abstain else "yellow")
    decision = (
        "[bold red]ABSTAIN â€“ Bahis yapma![/]" if result.abstain
        else ("[bold green]CERTAIN â€“ Devam et[/]" if result.is_certain
              else "[yellow]MODERATE â€“ Dikkatli ol[/]")
    )

    console.print(Panel(
        f"Nokta Tahmini: {result.point_estimate:.1%}\n"
        f"GÃ¼ven AralÄ±ÄŸÄ±: [{result.lower_bound:.1%} â€“ {result.upper_bound:.1%}]\n"
        f"AralÄ±k GeniÅŸliÄŸi: {result.interval_width:.1%}\n"
        f"GÃ¼ven Seviyesi: {result.confidence_level:.0%}\n"
        f"GÃ¼venilirlik: {result.reliability_score:.0%}\n\n"
        f"Karar: {decision}\n"
        f"{result.abstain_reason if result.abstain_reason else ''}\n\n"
        f"Metod: {result.method}",
        title=f"Belirsizlik â€“ {match_id} ({selection})",
        border_style=color,
    ))


@app.command()
def chaos_monkey(
    target: str = typer.Option("all", help="Hedef: scraper|database|validator|network|memory|asyncio|disk|all"),
    intensity: str = typer.Option("medium", help="Åiddet: low|medium|high"),
):
    """Chaos Engineering testlerini Ã§alÄ±ÅŸtÄ±rÄ±r."""
    console.rule("[bold red]CHAOS MONKEY â€“ KAOS TESTLERÄ°[/]")
    console.print("[yellow]tests/chaos_monkey.py Ã§alÄ±ÅŸtÄ±rÄ±lÄ±yor...[/]")

    import subprocess
    subprocess.run([
        sys.executable, str(ROOT / "tests" / "chaos_monkey.py"),
        "run", "--target", target, "--intensity", intensity,
    ])


@app.command()
def rl_train(
    timesteps: int = typer.Option(100000, help="EÄŸitim adÄ±m sayÄ±sÄ±"),
    bankroll: float = typer.Option(10000, help="BaÅŸlangÄ±Ã§ kasasÄ±"),
):
    """RL ajanÄ±nÄ± (PPO) geÃ§miÅŸ veri ile eÄŸitir."""
    from src.quant.rl_betting_env import RLBettingAgent, BettingMatch
    import random

    console.rule("[bold cyan]RL AJAN EÄÄ°TÄ°MÄ° â€“ PPO[/]")

    agent = RLBettingAgent(initial_bankroll=bankroll)

    # Demo eÄŸitim verisi (gerÃ§ek kullanÄ±mda DB'den gelir)
    demo_matches = []
    for i in range(500):
        ph = random.uniform(0.25, 0.65)
        pd = random.uniform(0.15, 0.35)
        pa = 1.0 - ph - pd
        result = random.choices([0, 1, 2], weights=[ph, pd, pa])[0]
        demo_matches.append(BettingMatch(
            match_id=f"demo_{i}",
            model_prob_home=ph, model_prob_draw=pd, model_prob_away=pa,
            odds_home=round(1.0 / max(ph, 0.1), 2),
            odds_draw=round(1.0 / max(pd, 0.1), 2),
            odds_away=round(1.0 / max(pa, 0.1), 2),
            value_edge=round(random.uniform(-0.05, 0.15), 3),
            confidence=round(random.uniform(0.3, 0.8), 2),
            home_form=round(random.uniform(0.3, 0.8), 2),
            result=result,
        ))

    result = agent.train(demo_matches, total_timesteps=timesteps)
    agent.save()

    console.print(f"\n[green]EÄŸitim SonuÃ§larÄ±:[/]")
    console.print(f"  ROI: {result.get('roi', 0):.2%}")
    console.print(f"  Win Rate: {result.get('win_rate', 0):.0%}")
    console.print(f"  Method: {result.get('status', '?')}")
    console.rule("[bold green]RL EÄÄ°TÄ°MÄ° TAMAMLANDI[/]")


@app.command()
def similar(
    home: str = typer.Argument(..., help="Ev sahibi takÄ±m"),
    away: str = typer.Argument(..., help="Deplasman takÄ±mÄ±"),
    k: int = typer.Option(20, help="Benzer maÃ§ sayÄ±sÄ±"),
):
    """FAISS ile tarihsel benzer maÃ§larÄ± bulur."""
    from src.quant.vector_engine import VectorMatchEngine
    from rich.table import Table
    from rich.panel import Panel

    console.rule(f"[bold cyan]TARÄ°HSEL Ä°KÄ°ZLER â€“ {home} vs {away}[/]")
    engine = VectorMatchEngine(dim=32)

    # Demo indeks (gerÃ§ek kullanÄ±mda DB'den yÃ¼klenir)
    import random
    demo_history = []
    for i in range(200):
        hg = random.randint(0, 4)
        ag = random.randint(0, 3)
        demo_history.append({
            "match_id": f"hist_{i}",
            "home_team": random.choice(["GS", "FB", "BJK", "TS"]),
            "away_team": random.choice(["GS", "FB", "BJK", "TS"]),
            "home_goals": hg, "away_goals": ag,
            "home_xg": round(random.uniform(0.5, 2.5), 2),
            "away_xg": round(random.uniform(0.5, 2.0), 2),
            "home_odds": round(random.uniform(1.3, 4.0), 2),
            "draw_odds": round(random.uniform(2.5, 4.5), 2),
            "away_odds": round(random.uniform(1.5, 6.0), 2),
            "date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        })
    engine.index_matches(demo_history)

    query = {"home_team": home, "away_team": away, "home_xg": 1.4, "away_xg": 1.1}
    report = engine.find_similar(query, k=k)

    console.print(Panel(
        f"Benzer MaÃ§: {report.k} | Metod: {report.method}\n"
        f"Ã–nerilen SonuÃ§: {report.suggested_result} "
        f"(gÃ¼ven: {report.suggested_confidence:.0%})\n"
        f"DaÄŸÄ±lÄ±m: Ev={report.distribution.get('H',0)} | "
        f"Ber={report.distribution.get('D',0)} | "
        f"Dep={report.distribution.get('A',0)}\n"
        f"Ãœ2.5: {report.over_25_rate:.0%} | "
        f"KG Var: {report.btts_rate:.0%} | "
        f"Ort. Gol: {report.avg_total_goals:.1f}",
        title="Tarihsel Benzerlik Raporu",
        border_style="cyan",
    ))

    if report.similar_matches:
        table = Table(title="En Benzer MaÃ§lar", show_lines=True)
        table.add_column("Tarih", style="dim")
        table.add_column("MaÃ§", style="cyan")
        table.add_column("Skor", justify="center")
        table.add_column("Benzerlik", justify="right", style="green")
        for m in report.similar_matches[:10]:
            table.add_row(
                m.date, f"{m.home_team} vs {m.away_team}",
                f"{m.home_goals}-{m.away_goals}",
                f"{m.similarity:.0%}",
            )
        console.print(table)


@app.command()
def copula(
    n_matches: int = typer.Option(3, help="Kupondaki maÃ§ sayÄ±sÄ±"),
):
    """Copula baÄŸÄ±mlÄ±lÄ±k analizi â€“ kupon riski."""
    from src.quant.copula_risk import CopulaRiskAnalyzer
    from rich.panel import Panel

    console.rule("[bold cyan]COPULA BAÄIMLILIK ANALÄ°ZÄ°[/]")

    analyzer = CopulaRiskAnalyzer()

    # Demo kupon
    import random
    matches = [
        {"match_id": f"m{i}", "prob": round(random.uniform(0.45, 0.70), 2),
         "odds": round(random.uniform(1.5, 2.5), 2),
         "league": random.choice(["super_lig", "premier_league"]),
         "country": "TR",
         "date": "2026-02-16"}
        for i in range(n_matches)
    ]

    report = analyzer.analyze_coupon(matches)

    color = "red" if report.dangerous_pairs else "green"
    console.print(Panel(
        f"MaÃ§ SayÄ±sÄ±: {report.n_matches}\n"
        f"Analiz Edilen Ã‡ift: {report.pairs_analyzed}\n"
        f"Tehlikeli Ã‡ift: {len(report.dangerous_pairs)}\n\n"
        f"Naif BirleÅŸik OlasÄ±lÄ±k: {report.naive_combined_prob:.4%}\n"
        f"Copula DÃ¼zeltilmiÅŸ: {report.copula_combined_prob:.4%}\n"
        f"Risk DÃ¼zeltmesi: {report.risk_adjustment:.0%}\n\n"
        f"Tavsiye: {report.recommendation}",
        title="Kupon BaÄŸÄ±mlÄ±lÄ±k Raporu",
        border_style=color,
    ))


@app.command()
def scenario(
    home: str = typer.Argument(..., help="Ev sahibi takÄ±m"),
    away: str = typer.Argument(..., help="Deplasman takÄ±mÄ±"),
    scenario_id: str = typer.Option("early_goal_home", help="Senaryo ID"),
):
    """Ä°nteraktif senaryo simÃ¼lasyonu."""
    from src.utils.telegram_scenario import ScenarioSimulator
    from rich.panel import Panel

    console.rule(f"[bold cyan]SENARYO â€“ {home} vs {away}[/]")

    sim = ScenarioSimulator()
    features = {"home_xg": 1.4, "away_xg": 1.1, "home_morale": 0.60, "away_morale": 0.50}
    result = sim.simulate(f"{home}_{away}", scenario_id, features)

    console.print(Panel(
        result.explanation or "SonuÃ§ yok.",
        title=result.scenario_label,
        border_style="cyan",
    ))

    console.print(f"\n[dim]Mevcut senaryolar: {', '.join(sim.available_scenarios)}[/]")


@app.command()
def shadow_report():
    """Shadow Testing â€“ tÃ¼m gÃ¶lge stratejilerin performans raporu."""
    from src.core.shadow_manager import ShadowManager
    from rich.table import Table
    from rich.panel import Panel

    console.rule("[bold cyan]SHADOW TESTING â€“ STRATEJÄ° KARÅILAÅTIRMA[/]")
    mgr = ShadowManager()

    if not mgr.strategy_names:
        console.print("[yellow]HenÃ¼z kayÄ±tlÄ± gÃ¶lge strateji yok.[/]")
        console.print("[dim]KullanÄ±m: shadow.register_strategy('strateji_adi')[/]")
        return

    comparison = mgr.compare_all()

    table = Table(title="Strateji PerformanslarÄ±", show_lines=True)
    table.add_column("Strateji", style="cyan")
    table.add_column("Mod", justify="center")
    table.add_column("Kasa", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("ROI", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("Bahis", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")

    for name, data in comparison.get("strategies", {}).items():
        pnl = data.get("total_pnl", 0)
        color = "green" if pnl >= 0 else "red"
        mode_color = "bold green" if data.get("mode") == "LIVE" else "yellow"
        table.add_row(
            name,
            f"[{mode_color}]{data.get('mode', '?')}[/]",
            f"{data.get('current_bankroll', 0):,.0f}",
            f"[{color}]{pnl:+,.0f}[/]",
            data.get("roi", "0%"),
            data.get("win_rate", "0%"),
            str(data.get("total_bets", 0)),
            str(data.get("sharpe", 0)),
            data.get("max_drawdown", "0%"),
        )

    console.print(table)
    console.print(Panel(
        comparison.get("recommendation", ""),
        title="Tavsiye",
        border_style="green",
    ))


@app.command()
def events(
    event_type: str = typer.Option("", help="Olay tipi filtresi (boÅŸ = hepsi)"),
    match_id: str = typer.Option("", help="MaÃ§ ID filtresi"),
    limit: int = typer.Option(20, help="GÃ¶sterilecek olay sayÄ±sÄ±"),
):
    """Event Store â€“ kaydedilmiÅŸ olaylarÄ± listeler."""
    from src.core.event_bus import EventStore
    from rich.table import Table
    from datetime import datetime

    console.rule("[bold cyan]EVENT STORE â€“ OLAY GEÃ‡MÄ°ÅÄ°[/]")
    store = EventStore()

    results = store.query(
        event_type=event_type or None,
        match_id=match_id or None,
        limit=limit,
    )

    if not results:
        console.print("[yellow]KayÄ±tlÄ± olay bulunamadÄ±.[/]")
        store.close()
        return

    table = Table(title=f"Son {len(results)} Olay", show_lines=True)
    table.add_column("Zaman", style="dim", width=19)
    table.add_column("Tip", style="cyan", width=18)
    table.add_column("Kaynak", width=14)
    table.add_column("MaÃ§", width=16)
    table.add_column("Detay", width=40)

    for ev in results[-limit:]:
        ts = datetime.fromtimestamp(ev.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        detail = str(ev.data)[:80] if ev.data else ""
        table.add_row(ts, ev.event_type, ev.source, ev.match_id, detail)

    console.print(table)
    console.print(f"\n[dim]Toplam kayÄ±tlÄ± olay: {store.count()}[/]")
    store.close()


@app.command()
def replay(
    match_id: str = typer.Argument(..., help="Yeniden oynatÄ±lacak maÃ§ ID"),
    speed: float = typer.Option(10.0, help="Oynatma hÄ±zÄ± (10x = 10 kat hÄ±zlÄ±)"),
):
    """Event Replay â€“ geÃ§miÅŸ maÃ§Ä± bugÃ¼nkÃ¼ kodla tekrar oynatÄ±r."""
    from src.core.event_bus import EventBus, EventStore, ReplayEngine

    console.rule(f"[bold cyan]REPLAY â€“ {match_id}[/]")

    store = EventStore()
    bus = EventBus(store=store, persist=False)
    replayer = ReplayEngine(bus)

    events = store.get_match_timeline(match_id)
    if not events:
        console.print(f"[yellow]{match_id} iÃ§in kayÄ±tlÄ± olay bulunamadÄ±.[/]")
        store.close()
        return

    console.print(f"[cyan]{len(events)} olay bulundu â€“ oynatÄ±lÄ±yor ({speed}x)...[/]")
    result = asyncio.run(replayer.replay(events, speed=speed, real_time=True))

    console.print(f"\n[green]Replay tamamlandÄ±:[/]")
    console.print(f"  OynatÄ±lan: {result.get('events_replayed', 0)}")
    console.print(f"  Hatalar: {result.get('errors', 0)}")
    console.print(f"  SÃ¼re: {result.get('elapsed_seconds', 0):.1f}s")
    store.close()


@app.command()
def trap_scan(
    home_odds: float = typer.Option(1.80, help="Ev sahibi oranÄ±"),
    draw_odds: float = typer.Option(3.50, help="Beraberlik oranÄ±"),
    away_odds: float = typer.Option(4.20, help="Deplasman oranÄ±"),
    movement: float = typer.Option(0.0, help="Son 1 saat oran deÄŸiÅŸimi (%)"),
):
    """Isolation Forest â€“ oran tuzak taramasÄ±."""
    from src.quant.isolation_anomaly import IsolationAnomalyDetector, MarketSnapshot
    from rich.panel import Panel

    console.rule("[bold red]TUZAK TARAMA â€“ ISOLATION FOREST[/]")

    detector = IsolationAnomalyDetector()
    snap = MarketSnapshot(
        match_id="manual_scan",
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        odds_movement_pct=movement,
        bookmaker_count=15,
        volume=1.0,
    )

    alerts = detector.scan(snap, match_id="manual_scan")

    if not alerts:
        console.print(Panel(
            f"Oran deseni NORMAL gÃ¶rÃ¼nÃ¼yor.\n\n"
            f"Ev: {home_odds} | Ber: {draw_odds} | Dep: {away_odds}\n"
            f"Hareket: {movement:+.1%}",
            title="Tuzak Tarama Sonucu",
            border_style="green",
        ))
    else:
        for a in alerts:
            color = "red" if a.severity in ("high", "critical") else "yellow"
            console.print(Panel(
                f"Tip: {a.alert_type}\n"
                f"Åiddet: {a.severity.upper()}\n"
                f"Skor: {a.score}\n"
                f"Aksiyon: {a.action}\n\n"
                f"{a.description}",
                title=f"UYARI: {a.alert_type}",
                border_style=color,
            ))


@app.command()
def causal(
    treatment: str = typer.Argument("red_card", help="MÃ¼dahale: red_card | injury_key_player | rain | home_advantage"),
    outcome: str = typer.Option("goals", help="SonuÃ§ deÄŸiÅŸkeni"),
):
    """Causal Inference â€“ nedensellik analizi."""
    from src.quant.causal_reasoner import CausalReasoner
    from rich.panel import Panel
    from rich.table import Table

    console.rule(f"[bold cyan]NEDENSELLÄ°K ANALÄ°ZÄ° â€“ {treatment} â†’ {outcome}[/]")

    reasoner = CausalReasoner()

    if treatment == "all":
        effects = reasoner.analyze_all_treatments(outcome)
        table = Table(title="TÃ¼m Nedensel Etkiler", show_lines=True)
        table.add_column("MÃ¼dahale", style="cyan")
        table.add_column("ATE", justify="right")
        table.add_column("AnlamlÄ±", justify="center")
        table.add_column("Metod")
        table.add_column("Yorum", width=40)

        for e in effects:
            sig_color = "green" if e.is_significant else "dim"
            table.add_row(
                e.treatment,
                f"{e.ate:+.3f}",
                f"[{sig_color}]{'EVET' if e.is_significant else 'HAYIR'}[/]",
                e.method,
                e.interpretation[:60] + "..." if len(e.interpretation) > 60 else e.interpretation,
            )
        console.print(table)
    else:
        effect = reasoner.estimate_effect(treatment, outcome)
        color = "green" if effect.is_significant else "yellow"
        console.print(Panel(
            f"MÃ¼dahale: {effect.treatment}\n"
            f"SonuÃ§: {effect.outcome}\n\n"
            f"ATE (Ortalama Tedavi Etkisi): {effect.ate:+.4f}\n"
            f"GÃ¼ven AralÄ±ÄŸÄ±: [{effect.ate_ci_lower:.3f}, {effect.ate_ci_upper:.3f}]\n"
            f"p-deÄŸeri: {effect.p_value:.4f}\n"
            f"Ä°statistiksel AnlamlÄ±lÄ±k: {'EVET' if effect.is_significant else 'HAYIR'}\n"
            f"Metod: {effect.method}\n\n"
            f"{effect.interpretation}",
            title="Nedensel Etki Raporu",
            border_style=color,
        ))


@app.command()
def counterfactual(
    match_id: str = typer.Argument(..., help="MaÃ§ ID"),
    treatment: str = typer.Option("red_card", help="MÃ¼dahale"),
    value: int = typer.Option(0, help="KarÅŸÄ±-olgusal deÄŸer (0=olmasaydÄ±)"),
):
    """Counterfactual analiz â€“ 'EÄŸer X olmasaydÄ± ne olurdu?'"""
    from src.quant.causal_reasoner import CausalReasoner
    from rich.panel import Panel

    console.rule(f"[bold cyan]KARÅI-OLGUSAL â€“ {match_id}[/]")

    reasoner = CausalReasoner()
    actual = {treatment: 1, "goals_home": 2, "goals_away": 1}
    cf = reasoner.counterfactual(match_id, {treatment: value}, actual)

    console.print(Panel(
        f"MaÃ§: {cf.match_id}\n"
        f"Senaryo: {cf.scenario}\n\n"
        f"GerÃ§ek SonuÃ§: {cf.actual_outcome}\n"
        f"KarÅŸÄ±-Olgusal: {cf.counterfactual_outcome}\n"
        f"Fark: {cf.difference}\n\n"
        f"GÃ¼ven: {cf.confidence:.0%}\n\n"
        f"{cf.explanation}",
        title="KarÅŸÄ±-Olgusal Analiz",
        border_style="cyan",
    ))


@app.command()
def topology(
    n_players: int = typer.Option(11, help="Oyuncu sayÄ±sÄ±"),
    formation: str = typer.Option("442", help="Formasyon (442, 433, 352)"),
):
    """TDA â€“ formasyon topolojik analizi."""
    from src.quant.topology_scanner import TopologyScanner
    from rich.panel import Panel
    import random

    console.rule(f"[bold cyan]TOPOLOJÄ°K FORMASYON ANALÄ°ZÄ° â€“ {formation}[/]")

    scanner = TopologyScanner()

    # Demo koordinatlar (formasyona gÃ¶re)
    coords = []
    if formation == "442":
        base = [
            (5, 34),  # GK
            (20, 10), (20, 25), (20, 43), (20, 58),  # Savunma
            (45, 12), (45, 28), (45, 40), (45, 56),  # Orta
            (75, 22), (75, 46),  # Forvet
        ]
    elif formation == "433":
        base = [
            (5, 34),
            (20, 10), (20, 25), (20, 43), (20, 58),
            (45, 18), (45, 34), (45, 50),
            (75, 15), (75, 34), (75, 53),
        ]
    else:
        base = [
            (5, 34),
            (18, 15), (18, 34), (18, 53),
            (40, 8), (40, 25), (40, 43), (40, 58), (40, 34),
            (75, 22), (75, 46),
        ]

    for x, y in base[:n_players]:
        coords.append((x + random.uniform(-3, 3), y + random.uniform(-3, 3)))

    state = scanner.analyze_formation(coords, team="Demo")
    alerts = scanner.check_integrity(coords, team="Demo", match_id="demo")

    color = "green"
    if state.formation_integrity < 0.5:
        color = "red"
    elif state.formation_integrity < 0.7:
        color = "yellow"

    console.print(Panel(
        f"Formasyon: {formation} ({n_players} oyuncu)\n"
        f"Metod: {state.method}\n\n"
        f"{'â”€' * 40}\n"
        f"BÃ¼tÃ¼nlÃ¼k: {state.formation_integrity:.0%}\n"
        f"Kaplama AlanÄ±: {state.convex_hull_area:.0f} m^2\n"
        f"SÄ±kÄ±lÄ±k: {state.compactness:.0%}\n"
        f"GeniÅŸlik: {state.width:.1f}m | Derinlik: {state.depth:.1f}m\n"
        f"Savunma HattÄ±: {state.defensive_line_height:.1f}m\n\n"
        f"{'â”€' * 40}\n"
        f"H0 (BaÄŸlÄ± BileÅŸen): {state.h0_components}\n"
        f"H1 (Delik/BoÅŸluk): {state.h1_holes}\n"
        f"Persistence Entropy: {state.persistence_entropy:.4f}\n"
        f"Max Persistence: {state.max_persistence:.4f}",
        title="Topolojik Formasyon Raporu",
        border_style=color,
    ))

    for alert in alerts:
        console.print(Panel(
            alert.description,
            title=f"UYARI: {alert.alert_type.upper()}",
            border_style="red",
        ))


# â”€â”€ Level 13: Physics of Information & Zero-Latency â”€â”€


@app.command()
def nash(
    prob_home: float = typer.Argument(0.55, help="Ev sahibi kazanma olasÄ±lÄ±ÄŸÄ±"),
    prob_draw: float = typer.Argument(0.25, help="Beraberlik olasÄ±lÄ±ÄŸÄ±"),
    prob_away: float = typer.Argument(0.20, help="Deplasman kazanma olasÄ±lÄ±ÄŸÄ±"),
    odds_home: float = typer.Option(1.80, help="Ev sahibi oranÄ±"),
    odds_draw: float = typer.Option(3.50, help="Beraberlik oranÄ±"),
    odds_away: float = typer.Option(4.50, help="Deplasman oranÄ±"),
    stake: float = typer.Option(100.0, help="Bahis miktarÄ±"),
):
    """Nash Dengesi â€“ oyun teorisi ile optimal strateji."""
    from src.quant.nash_solver import NashGameSolver
    from rich.panel import Panel
    from rich.table import Table

    console.rule("[bold cyan]NASH DENGESÄ° â€“ OYUN TEORÄ°SÄ°[/]")

    solver = NashGameSolver()
    analysis = solver.analyze_match(
        model_probs={"prob_home": prob_home, "prob_draw": prob_draw, "prob_away": prob_away},
        market_odds={"home": odds_home, "draw": odds_draw, "away": odds_away},
        stake=stake,
    )

    # Strateji tablosu
    tbl = Table(title="Bot Stratejisi (Nash Dengesi)", show_lines=True)
    tbl.add_column("Aksiyon", style="cyan")
    tbl.add_column("AÄŸÄ±rlÄ±k", justify="right")
    if analysis.equilibrium.bettor_strategy is not None:
        for name, w in zip(solver.ACTION_NAMES, analysis.equilibrium.bettor_strategy):
            style = "bold green" if w > 0.3 else "dim"
            tbl.add_row(name, f"{w:.1%}", style=style)
    console.print(tbl)

    # SonuÃ§ paneli
    margin_str = f"{analysis.bookmaker_margin:.2%}" if analysis.bookmaker_margin else "?"
    console.print(Panel(
        f"Optimal Aksiyon: {analysis.optimal_action.upper()}\n"
        f"Nash Beklenen DeÄŸer: {analysis.expected_value:+.1f}\n"
        f"Risk-Adjusted EV: {analysis.risk_adjusted_ev:+.1f}\n"
        f"SÃ¶mÃ¼rÃ¼lebilirlik: {analysis.equilibrium.exploitability:.0%}\n"
        f"BÃ¼ro MarjÄ±: {margin_str}\n"
        f"Metod: {analysis.equilibrium.method}\n\n"
        f"ğŸ’¡ {analysis.recommendation}",
        title="Oyun Teorisi Sonucu",
        border_style="green" if analysis.expected_value > 0 else "red",
    ))


@app.command()
def entropy(
    prob_home: float = typer.Argument(0.55, help="Ev sahibi olasÄ±lÄ±ÄŸÄ±"),
    prob_draw: float = typer.Argument(0.25, help="Beraberlik olasÄ±lÄ±ÄŸÄ±"),
    prob_away: float = typer.Argument(0.20, help="Deplasman olasÄ±lÄ±ÄŸÄ±"),
    odds_home: float = typer.Option(1.80, help="Ev sahibi oranÄ±"),
    odds_draw: float = typer.Option(3.50, help="Beraberlik oranÄ±"),
    odds_away: float = typer.Option(4.50, help="Deplasman oranÄ±"),
):
    """Shannon Entropy â€“ bilgi fiziÄŸi ile belirsizlik analizi."""
    from src.quant.entropy_meter import EntropyMeter
    from rich.panel import Panel

    console.rule("[bold cyan]SHANNON ENTROPÄ°SÄ° â€“ BÄ°LGÄ° FÄ°ZÄ°ÄÄ°[/]")

    meter = EntropyMeter()
    report = meter.analyze_match(
        match_id="CLI_TEST",
        model_probs={"prob_home": prob_home, "prob_draw": prob_draw, "prob_away": prob_away},
        market_odds={"home": odds_home, "draw": odds_draw, "away": odds_away},
    )

    colors = {"stable": "green", "uncertain": "yellow", "volatile": "bright_yellow", "chaotic": "red"}
    color = colors.get(report.chaos_level, "white")

    max_entropy = 1.585  # log2(3) â€“ 3 eÅŸit olasÄ±lÄ±klÄ± sonuÃ§
    bar_len = 30
    fill = int(bar_len * min(report.match_entropy / max_entropy, 1.0))
    bar = "â–ˆ" * fill + "â–‘" * (bar_len - fill)

    console.print(Panel(
        f"Shannon Entropy: {report.match_entropy:.4f} bit\n"
        f"Max Entropy (1X2): {max_entropy:.3f} bit\n"
        f"Doluluk: [{bar}] {report.match_entropy / max_entropy:.0%}\n\n"
        f"KL-Divergence (Modelâ†”Piyasa): {report.kl_divergence:.4f}\n"
        f"Cross Entropy: {report.cross_entropy:.4f}\n\n"
        f"Kaos Seviyesi: {report.chaos_level.upper()}\n"
        f"Kill Switch: {'EVET' if report.kill_switch else 'HayÄ±r'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Bilgi FiziÄŸi Raporu",
        border_style=color,
    ))


@app.command()
def jit_bench():
    """JIT Benchmark â€“ hÄ±zlandÄ±rma testi."""
    from src.core.jit_accelerator import JITAccelerator
    from rich.table import Table
    import time as _time

    console.rule("[bold cyan]JIT BENCHMARK[/]")

    acc = JITAccelerator()
    acc.warmup()

    results = []

    # Kelly
    start = _time.perf_counter_ns()
    for _ in range(100_000):
        acc.kelly(0.55, 2.10)
    elapsed = (_time.perf_counter_ns() - start) / 1_000_000
    results.append(("Kelly (100K)", f"{elapsed:.1f}ms"))

    # Poisson 1X2
    start = _time.perf_counter_ns()
    for _ in range(10_000):
        acc.poisson_1x2(1.4, 1.1)
    elapsed = (_time.perf_counter_ns() - start) / 1_000_000
    results.append(("Poisson 1X2 (10K)", f"{elapsed:.1f}ms"))

    # Monte Carlo
    start = _time.perf_counter_ns()
    acc.monte_carlo(1.4, 1.1, n=100_000)
    elapsed = (_time.perf_counter_ns() - start) / 1_000_000
    results.append(("Monte Carlo (100K sim)", f"{elapsed:.1f}ms"))

    # Entropy
    import numpy as np
    start = _time.perf_counter_ns()
    for _ in range(100_000):
        acc.entropy(np.array([0.5, 0.3, 0.2]))
    elapsed = (_time.perf_counter_ns() - start) / 1_000_000
    results.append(("Entropy (100K)", f"{elapsed:.1f}ms"))

    # Distance
    matrix = np.random.rand(10000, 32)
    query = np.random.rand(32)
    start = _time.perf_counter_ns()
    acc.distances(query, matrix)
    elapsed = (_time.perf_counter_ns() - start) / 1_000_000
    results.append(("Euclidean 10KÃ—32", f"{elapsed:.1f}ms"))

    tbl = Table(title="JIT Benchmark SonuÃ§larÄ±", show_lines=True)
    tbl.add_column("Test", style="cyan")
    tbl.add_column("SÃ¼re", justify="right", style="green")
    for name, dur in results:
        tbl.add_row(name, dur)
    console.print(tbl)

    console.print(f"\n[bold]Numba JIT: {'Aktif âœ“' if acc.is_jit_available else 'Devre DÄ±ÅŸÄ± âœ—'}[/]")


@app.command()
def psycho(
    period: str = typer.Option("weekly", help="DÃ¶nem: weekly | monthly"),
):
    """Psikoloji Profili â€“ yatÄ±rÄ±mcÄ± davranÄ±ÅŸ analizi."""
    from src.utils.psycho_profiler import PsychoProfiler
    from rich.panel import Panel

    console.rule("[bold cyan]YATIRIMCI PSÄ°KOLOJÄ°SÄ° PROFÄ°LÄ°[/]")

    profiler = PsychoProfiler()
    report = profiler.weekly_report() if period == "weekly" else profiler.monthly_report()

    if report.total_decisions == 0:
        console.print("[yellow]HenÃ¼z karar verisi yok. Bot Ã¶nerilere tepki verdikÃ§e profil oluÅŸacak.[/]")
        return

    console.print(Panel(
        f"Toplam Karar: {report.total_decisions}\n"
        f"  Onaylanan: {report.approved}\n"
        f"  Reddedilen: {report.rejected}\n\n"
        f"{'â”€' * 40}\n"
        f"Omission (kaÃ§Ä±rÄ±lan): {report.omission_errors} maÃ§ â†’ {report.omission_cost:,.0f} TL\n"
        f"Commission (gereksiz): {report.commission_errors} maÃ§ â†’ {report.commission_cost:,.0f} TL\n\n"
        f"{'â”€' * 40}\n"
        f"Risk Ä°ÅŸtahÄ±: {report.risk_aversion_score:.0%}\n"
        f"KayÄ±p Korkusu: {report.loss_aversion:.0%}\n"
        f"AÅŸÄ±rÄ± GÃ¼ven: {report.overconfidence:.0%}\n"
        f"Recency Bias: {report.recency_bias:.0%}\n"
        f"Duygusal Karar: {report.emotional_decision_rate:.0%}\n\n"
        f"{'â”€' * 40}\n"
        f"Bot ROI: {report.bot_alone_roi:.1%}\n"
        f"Ä°nsan+Bot ROI: {report.human_filter_roi:.1%}\n"
        f"Optimal Skor: {report.optimal_score:.0%}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title=f"{'HaftalÄ±k' if period == 'weekly' else 'AylÄ±k'} Psikoloji Raporu",
        border_style="magenta",
    ))


# â”€â”€ Level 14: Digital Twin & Extreme Value Theory â”€â”€


@app.command()
def twin(
    home_team: str = typer.Argument("Galatasaray", help="Ev sahibi takÄ±m"),
    away_team: str = typer.Argument("FenerbahÃ§e", help="Deplasman takÄ±mÄ±"),
    home_quality: float = typer.Option(78, help="Ev sahibi kalite (0-100)"),
    away_quality: float = typer.Option(76, help="Deplasman kalite (0-100)"),
    n_sims: int = typer.Option(1000, help="SimÃ¼lasyon sayÄ±sÄ±"),
    formation: str = typer.Option("442", help="Formasyon"),
):
    """Digital Twin â€“ maÃ§Ä±n sayÄ±sal ikizi (ABM simÃ¼lasyon)."""
    from src.quant.digital_twin_sim import DigitalTwinSimulator
    from rich.panel import Panel
    from rich.table import Table

    console.rule(f"[bold cyan]SAYISAL Ä°KÄ°Z â€“ {home_team} vs {away_team}[/]")
    console.print(f"[dim]{n_sims} simÃ¼lasyon baÅŸlatÄ±lÄ±yorâ€¦[/]")

    sim = DigitalTwinSimulator()
    home_squad = sim.generate_team(home_team, quality=home_quality, formation=formation)
    away_squad = sim.generate_team(away_team, quality=away_quality, formation=formation)

    report = sim.simulate_match(
        f"{home_team}_vs_{away_team}", home_squad, away_squad, n_sims=n_sims,
    )

    # Skor daÄŸÄ±lÄ±mÄ±
    tbl = Table(title="En SÄ±k Skorlar", show_lines=True)
    tbl.add_column("Skor", style="cyan")
    tbl.add_column("OlasÄ±lÄ±k", justify="right", style="green")
    for score, prob in sorted(report.score_distribution.items(),
                               key=lambda x: -x[1])[:8]:
        tbl.add_row(score, f"{prob:.1%}")
    console.print(tbl)

    console.print(Panel(
        f"SimÃ¼lasyon: {report.n_simulations} maÃ§ ({report.method})\n"
        f"Formasyon: {formation}\n\n"
        f"{'â”€' * 40}\n"
        f"Ev Sahibi KazanÄ±r: {report.prob_home:.1%}\n"
        f"Beraberlik: {report.prob_draw:.1%}\n"
        f"Deplasman KazanÄ±r: {report.prob_away:.1%}\n\n"
        f"{'â”€' * 40}\n"
        f"Ort. Gol (Ev): {report.avg_home_goals:.2f}\n"
        f"Ort. Gol (Dep): {report.avg_away_goals:.2f}\n"
        f"Ort. Toplam Gol: {report.avg_total_goals:.2f}\n"
        f"Ãœst 2.5: {report.prob_over25:.1%}\n"
        f"KG: {report.prob_btts:.1%}\n\n"
        f"{'â”€' * 40}\n"
        f"xG (Ev): {report.avg_home_xg:.2f}\n"
        f"xG (Dep): {report.avg_away_xg:.2f}\n\n"
        f"{report.fatigue_impact or 'Yorgunluk verisi yok'}",
        title=f"SayÄ±sal Ä°kiz Raporu â€“ {home_team} vs {away_team}",
        border_style="blue",
    ))


@app.command()
def evt(
    n_samples: int = typer.Option(500, help="GeÃ§miÅŸ gÃ¶zlem sayÄ±sÄ± (demo)"),
    stake: float = typer.Option(100, help="Bahis miktarÄ±"),
):
    """EVT â€“ Extreme Value Theory risk analizi."""
    from src.quant.evt_risk_manager import EVTRiskManager
    from rich.panel import Panel
    import numpy as np

    console.rule("[bold cyan]EXTREME VALUE THEORY â€“ KUYRUK RÄ°SKÄ°[/]")

    mgr = EVTRiskManager()

    # Demo: rastgele kayÄ±p daÄŸÄ±lÄ±mÄ± (kalÄ±n kuyruklu)
    np.random.seed(42)
    normal_losses = np.random.exponential(2.0, n_samples)
    # Fat tail ekle
    tail_events = np.random.pareto(1.5, n_samples // 20) * 10
    losses = np.concatenate([normal_losses, tail_events])

    mgr.add_observations(losses.tolist())
    params = mgr.fit()

    report = mgr.analyze_match_risk(
        match_id="DEMO",
        model_probs={"prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20},
        odds={"home": 1.80, "draw": 3.50, "away": 4.50},
        stake=stake,
    )

    color = {
        "normal": "green", "moderate": "yellow",
        "heavy": "bright_yellow", "extreme": "red",
    }.get(report.tail_severity, "white")

    console.print(Panel(
        f"GPD Parametreleri:\n"
        f"  Î¾ (shape): {report.shape_xi:.4f}\n"
        f"  Ïƒ (scale): {report.scale_sigma:.4f}\n"
        f"  EÅŸik (POT): {report.threshold:.2f}\n"
        f"  AÅŸÄ±m sayÄ±sÄ±: {report.n_exceedances}\n"
        f"  Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"VaR (%95): {report.var_95:.1f} TL\n"
        f"VaR (%99): {report.var_99:.1f} TL\n"
        f"CVaR (%95): {report.cvar_95:.1f} TL\n"
        f"CVaR (%99): {report.cvar_99:.1f} TL\n\n"
        f"{'â”€' * 40}\n"
        f"Return Level (10 yÄ±l): {report.return_level_10:.1f}\n"
        f"Return Level (50 yÄ±l): {report.return_level_50:.1f}\n"
        f"Siyah KuÄŸu OlasÄ±lÄ±ÄŸÄ±: {report.black_swan_prob:.2%}\n\n"
        f"{'â”€' * 40}\n"
        f"Kuyruk: {report.tail_severity.upper()}\n"
        f"Kelly Ã‡arpanÄ±: {report.kelly_adjustment:.0%}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="EVT Risk Raporu",
        border_style=color,
    ))


@app.command()
def ask(
    question: str = typer.Argument(..., help="DoÄŸal dilde soru (TÃ¼rkÃ§e)"),
):
    """Text-to-SQL â€“ doÄŸal dilde veritabanÄ± sorgulama."""
    from src.utils.query_assistant import QueryAssistant
    from rich.panel import Panel

    console.rule("[bold cyan]DOÄAL DÄ°L SORGU[/]")

    assistant = QueryAssistant()
    result = assistant.ask(question)

    if result.error:
        console.print(Panel(
            f"Soru: {result.question}\n\nâŒ Hata: {result.error}",
            title="Sorgu HatasÄ±",
            border_style="red",
        ))
    else:
        console.print(Panel(
            f"Soru: {result.question}\n\n"
            f"SQL:\n{result.sql}\n\n"
            f"{'â”€' * 40}\n"
            f"SonuÃ§:\n{result.formatted_answer}\n\n"
            f"â± {result.execution_time_ms:.0f}ms ({result.method}, "
            f"gÃ¼ven: {result.confidence:.0%})",
            title="Sorgu Sonucu",
            border_style="green",
        ))


@app.command()
def what_if(
    home_team: str = typer.Argument("Galatasaray", help="Ev sahibi"),
    away_team: str = typer.Argument("FenerbahÃ§e", help="Deplasman"),
    scenario: str = typer.Option("key_player_injured", help="Senaryo: key_player_injured | fatigue_boost | red_card_60"),
    n_sims: int = typer.Option(500, help="SimÃ¼lasyon sayÄ±sÄ±"),
):
    """What-If â€“ senaryo simÃ¼lasyonu (Digital Twin)."""
    from src.quant.digital_twin_sim import DigitalTwinSimulator
    from rich.panel import Panel

    console.rule(f"[bold cyan]WHAT-IF â€“ {scenario.upper()}[/]")

    sim = DigitalTwinSimulator()
    home_squad = sim.generate_team(home_team, quality=78)
    away_squad = sim.generate_team(away_team, quality=76)

    result = sim.what_if_scenario(
        f"{home_team}_vs_{away_team}",
        home_squad, away_squad,
        scenario=scenario, n_sims=n_sims,
    )

    normal = result["normal"]
    modified = result["modified"]
    impact = result["impact"]

    console.print(Panel(
        f"Senaryo: {result['scenario']}\n"
        f"Hedef: {result['target_player'] or 'Otomatik'}\n\n"
        f"{'â”€' * 40}\n"
        f"{'':>15} {'NORMAL':>10} {'SENARYO':>10} {'FARK':>10}\n"
        f"{'Ev Sahibi':>15} {normal['prob_home']:>9.1%} {modified['prob_home']:>9.1%} {impact['prob_home_change']:>+9.1%}\n"
        f"{'Deplasman':>15} {normal['prob_away']:>9.1%} {modified['prob_away']:>9.1%} {impact['prob_away_change']:>+9.1%}\n"
        f"{'Toplam Gol':>15} {normal['avg_goals']:>9.2f} {modified['avg_goals']:>9.2f} {impact['goals_change']:>+9.2f}",
        title=f"What-If Raporu â€“ {home_team} vs {away_team}",
        border_style="cyan",
    ))


# â”€â”€ Level 15: Distributed Computing & Fractal Geometry â”€â”€


@app.command()
def fractal_hurst(
    n_points: int = typer.Option(200, help="Veri noktasÄ± sayÄ±sÄ± (demo)"),
    regime: str = typer.Option("trending", help="Demo rejim: trending | mean_reverting | random"),
):
    """Fractal â€“ Hurst Exponent analizi."""
    from src.quant.fractal_analyzer import FractalAnalyzer
    from rich.panel import Panel
    import numpy as np

    console.rule("[bold cyan]FRAKTAL ANALÄ°Z â€“ HURST EXPONENT[/]")

    np.random.seed(42)
    if regime == "trending":
        # Trending: H > 0.5 (kalÄ±cÄ± seri)
        data = np.cumsum(np.random.randn(n_points) * 0.3 + 0.05)
    elif regime == "mean_reverting":
        # Mean reverting: H < 0.5 (ortalamaya dÃ¶nen)
        data = np.zeros(n_points)
        for i in range(1, n_points):
            data[i] = data[i - 1] * 0.3 + np.random.randn() * 0.5
    else:
        data = np.random.randn(n_points)

    analyzer = FractalAnalyzer()
    report = analyzer.analyze_entity(data, entity=f"Demo ({regime})", entity_type="demo")
    h = report.hurst_result

    bar_len = 30
    fill = int(bar_len * h.hurst)
    bar = "â–ˆ" * fill + "â–‘" * (bar_len - fill)

    colors = {
        "trending": "green", "weak_trending": "green",
        "random": "yellow", "weak_mean_reverting": "cyan",
        "mean_reverting": "blue",
    }
    color = colors.get(h.regime, "white")

    console.print(Panel(
        f"Veri: {n_points} nokta ({regime} demo)\n"
        f"Metod: {h.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Hurst Exponent: {h.hurst:.4f}\n"
        f"[{bar}]\n"
        f"0.0 â†â”€â”€ Mean Revert â”€â”€â”€â”€ Random â”€â”€â”€â”€ Trending â”€â”€â†’ 1.0\n\n"
        f"{'â”€' * 40}\n"
        f"Rejim: {h.regime.upper()}\n"
        f"Fraktal Boyut: {h.fractal_dimension:.4f}\n"
        f"KalÄ±p GÃ¼cÃ¼: {h.pattern_strength:.4f}\n"
        f"Kelly Ã‡arpanÄ±: {h.kelly_multiplier:.2f}x\n"
        f"GÃ¼venilirlik: {h.confidence:.2f}\n\n"
        f"DFA Î±: {report.dfa_alpha:.4f}\n"
        f"AC(1): {report.autocorrelation_lag1:.4f}\n"
        f"AC(5): {report.autocorrelation_lag5:.4f}\n"
        f"KararlÄ±: {'Evet' if report.is_stable else 'HayÄ±r'}\n\n"
        f"{'â”€' * 40}\n"
        f"ğŸ’¡ {h.recommended_strategy}",
        title="Hurst Exponent Raporu",
        border_style=color,
    ))


@app.command()
def bsts(
    n_points: int = typer.Option(60, help="Veri noktasÄ± sayÄ±sÄ±"),
    break_at: int = typer.Option(30, help="KÄ±rÄ±lma noktasÄ± indeksi"),
    effect_size: float = typer.Option(1.5, help="MÃ¼dahale etki bÃ¼yÃ¼klÃ¼ÄŸÃ¼"),
):
    """BSTS â€“ Bayesian Structural Time Series mÃ¼dahale analizi."""
    from src.quant.bsts_impact import BSTSImpactAnalyzer
    from rich.panel import Panel
    import numpy as np

    console.rule("[bold cyan]BSTS MÃœDAHALEAnalizi[/]")

    np.random.seed(42)
    pre = np.random.randn(break_at) * 0.5 + 1.5
    post = np.random.randn(n_points - break_at) * 0.5 + 1.5 + effect_size
    data = np.concatenate([pre, post])

    analyzer = BSTSImpactAnalyzer()
    effect = analyzer.analyze_intervention(
        data=data,
        intervention_idx=break_at,
        intervention_name="Hoca DeÄŸiÅŸimi (Demo)",
        metric="xG",
    )

    color = "green" if effect.is_significant else "yellow"

    console.print(Panel(
        f"MÃ¼dahale: {effect.intervention}\n"
        f"Metrik: {effect.target_metric}\n"
        f"Metod: {effect.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Ã–nceki Ortalama: {effect.predicted_without:.3f}\n"
        f"Sonraki Ortalama: {effect.actual:.3f}\n"
        f"Etki: {effect.avg_effect:+.3f} ({effect.relative_effect_pct:+.1f}%)\n"
        f"KÃ¼mÃ¼latif Etki: {effect.cumulative_effect:+.3f}\n\n"
        f"{'â”€' * 40}\n"
        f"AnlamlÄ±lÄ±k: {'EVET' if effect.is_significant else 'HAYIR'} "
        f"(p={effect.posterior_prob:.4f})\n"
        f"KalÄ±cÄ±lÄ±k: {'KALICI' if effect.is_permanent else 'GEÃ‡Ä°CÄ°'}\n"
        f"YarÄ± Ã–mÃ¼r: {effect.half_life:.0f} maÃ§\n\n"
        f"ğŸ’¡ {effect.recommendation}",
        title="MÃ¼dahale Etki Raporu",
        border_style=color,
    ))

    # KÄ±rÄ±lma tespiti
    breaks = analyzer.detect_breaks(data)
    if breaks:
        for bp in breaks:
            console.print(Panel(
                f"KÄ±rÄ±lma NoktasÄ±: {bp.index}\n"
                f"Ã–nceki: {bp.pre_mean:.3f} â†’ Sonraki: {bp.post_mean:.3f}\n"
                f"DeÄŸiÅŸim: {bp.change_pct:+.1f}%\n"
                f"AnlamlÄ±lÄ±k: {'EVET' if bp.is_significant else 'HAYIR'} "
                f"(p={bp.p_value:.4f})",
                title="YapÄ±sal KÄ±rÄ±lma",
                border_style="red" if bp.is_significant else "dim",
            ))


@app.command()
def cluster_status():
    """DaÄŸÄ±tÄ±k kÃ¼me durumu (Ray / ProcessPool)."""
    from src.core.distributed_core import DistributedCore
    from rich.panel import Panel

    console.rule("[bold cyan]DAÄITIK KÃœME DURUMU[/]")

    dist = DistributedCore()
    dist.start()
    st = dist.status()

    runtime = "Ray" if st.is_ray else "ProcessPool (Fallback)"
    color = "green" if st.is_ray else "yellow"

    console.print(Panel(
        f"Runtime: {runtime}\n"
        f"CPU: {st.num_cpus}\n"
        f"GPU: {st.num_gpus}\n"
        f"Workers: {st.num_workers}\n"
        f"Object Store: {st.object_store_mb:.0f} MB\n\n"
        f"Tamamlanan GÃ¶rev: {st.tasks_completed}\n"
        f"Havadaki GÃ¶rev: {st.tasks_in_flight}\n"
        f"Uptime: {st.uptime_seconds:.0f}s",
        title="KÃ¼me Durumu",
        border_style=color,
    ))

    dist.shutdown()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 16 CLI â€“ Transfer Learning, Transport Metric, Mimic
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def transfer(
    source: str = typer.Option("europe_base", help="Kaynak model adÄ±"),
    target_league: str = typer.Option("super_lig", help="Hedef lig"),
    epochs: int = typer.Option(20, help="Fine-tune epoch sayÄ±sÄ±"),
):
    """Transfer Learning â€“ lig arasÄ± bilgi transferi."""
    from src.quant.transfer_learner import TransferLearner
    from rich.panel import Panel

    learner = TransferLearner(input_dim=20, n_classes=3)
    loaded = learner.load(source)

    if loaded:
        console.print(f"[green]âœ“ Model yÃ¼klendi:[/] {source}")
    else:
        console.print(f"[yellow]Kaynak model bulunamadÄ±: {source} â€“ sÄ±fÄ±rdan baÅŸlanacak.[/]")

    # Demo: rastgele veri ile fine-tune (gerÃ§ek kullanÄ±mda DB'den gelir)
    import numpy as np
    X_demo = np.random.randn(200, 20).astype(np.float32)
    y_demo = np.random.randint(0, 3, size=200)

    report = learner.fine_tune(
        X_demo, y_demo, epochs=epochs, target_name=target_league,
    )

    color = "green" if report.transfer_beneficial else "yellow"
    console.print(Panel(
        f"Source: {report.source_domain or source}\n"
        f"Target: {report.target_domain}\n"
        f"Ã–rnekler: {report.target_samples}\n"
        f"Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Ã–nceki DoÄŸruluk: {report.target_accuracy_before:.1%}\n"
        f"Sonraki DoÄŸruluk: {report.target_accuracy_after:.1%}\n"
        f"Ä°yileÅŸme: {report.improvement_pct:+.1f}%\n"
        f"DondurulmuÅŸ Katman: {report.frozen_layers}\n"
        f"EÄŸitilebilir Param: {report.trainable_params:,}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Transfer Learning Raporu",
        border_style=color,
    ))


@app.command()
def drift(
    n_ref: int = typer.Option(500, help="Referans Ã¶rnek sayÄ±sÄ±"),
    n_live: int = typer.Option(100, help="CanlÄ± Ã¶rnek sayÄ±sÄ±"),
    inject_drift: bool = typer.Option(False, help="Yapay drift enjekte et"),
):
    """Optimal Transport â€“ model drift tespiti."""
    from src.quant.transport_metric import TransportMetric
    from rich.panel import Panel
    import numpy as np

    tm = TransportMetric()

    # Referans (eÄŸitim) daÄŸÄ±lÄ±mÄ±
    X_ref = np.random.randn(n_ref, 10)
    tm.set_reference(X_ref)

    # CanlÄ± daÄŸÄ±lÄ±m
    if inject_drift:
        X_live = np.random.randn(n_live, 10) + 0.8  # Mean shift
        console.print("[yellow]Yapay drift enjekte edildi (mean shift +0.8)[/]")
    else:
        X_live = np.random.randn(n_live, 10) + 0.05  # Hafif

    report = tm.check_drift(X_live, name="cli_test")

    if report.kill_betting:
        color = "red"
    elif report.is_drifted:
        color = "yellow"
    else:
        color = "green"

    console.print(Panel(
        f"Wasserstein-1D: {report.wasserstein_1:.6f}\n"
        f"Wasserstein-ND: {report.wasserstein_2:.6f}\n"
        f"Sinkhorn: {report.sinkhorn:.6f}\n"
        f"MMD: {report.max_mean_discrepancy:.6f}\n\n"
        f"{'â”€' * 40}\n"
        f"Drift Seviyesi: {report.drift_severity.upper()}\n"
        f"Model GÃ¼venilir: {'EVET' if report.model_reliable else 'HAYIR'}\n"
        f"Kill Betting: {'EVET' if report.kill_betting else 'HAYIR'}\n"
        f"Kayma BoyutlarÄ±: {report.drift_dimensions[:5]}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Optimal Transport â€“ Drift Raporu",
        border_style=color,
    ))


@app.command()
def mimic_test(
    persona: str = typer.Option("random", help="Persona tipi (tired/excited/casual/professional/random)"),
    steps: int = typer.Option(5, help="SimÃ¼le edilecek aksiyon sayÄ±sÄ±"),
):
    """Mimic Engine â€“ insansÄ± davranÄ±ÅŸ simÃ¼lasyonu."""
    import asyncio as aio
    from src.core.mimic_engine import MimicEngine
    from rich.panel import Panel

    engine = MimicEngine(persona=persona)
    fp = engine.session_fingerprint()

    console.print(Panel(
        f"Persona: {engine.profile.name}\n"
        f"Circadian: {fp['circadian']:.2f}x\n"
        f"Viewport: {fp['viewport']}\n"
        f"Platform: {fp['platform']}\n"
        f"Session: {fp['session_id']}",
        title="Mimic Session",
        border_style="blue",
    ))

    # Fare hareketi simÃ¼lasyonu
    path = engine.mouse_path((100, 200), (800, 500))
    console.print(f"[dim]Fare yolu: {len(path)} nokta[/]")

    # Yazma gecikmesi
    text = "Galatasaray vs FenerbahÃ§e"
    delays = engine.typing_delays(text)
    avg_d = sum(delays) / max(len(delays), 1)
    console.print(f"[dim]Yazma: '{text}' â†’ ort. {avg_d*1000:.0f}ms/tuÅŸ[/]")

    # Aksiyon simÃ¼lasyonu
    async def _run():
        for i in range(steps):
            if engine.should_hesitate():
                h = await engine.hesitate()
                console.print(f"  [{i+1}] TereddÃ¼t: {h*1000:.0f}ms")
            d = await engine.human_delay(action=f"step_{i+1}")
            console.print(f"  [{i+1}] Aksiyon gecikmesi: {d*1000:.0f}ms")

    aio.run(_run())
    console.print(f"[green]âœ“ {steps} aksiyon simÃ¼le edildi.[/]")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 17 CLI â€“ Federated, Hypergraph, Annealer, RLHF
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def federated(
    n_rounds: int = typer.Option(5, help="Federasyon round sayÄ±sÄ±"),
    local_epochs: int = typer.Option(5, help="Yerel eÄŸitim epoch"),
):
    """Federated Learning â€“ daÄŸÄ±tÄ±k sÃ¼rÃ¼ eÄŸitimi."""
    from src.quant.federated_trainer import FederatedTrainer
    from rich.panel import Panel
    import numpy as np

    ft = FederatedTrainer(
        leagues=["super_lig", "premier_league", "bundesliga"],
        input_dim=20, n_classes=3,
    )

    # Demo veri (gerÃ§ek kullanÄ±mda DB'den gelir)
    for league in ["super_lig", "premier_league", "bundesliga"]:
        n = np.random.randint(200, 500)
        X = np.random.randn(n, 20).astype(np.float32)
        y = np.random.randint(0, 3, size=n)
        ft.load_league_data(league, X, y)

    report = ft.train(n_rounds=n_rounds, local_epochs=local_epochs)

    console.print(Panel(
        f"Ligler: {', '.join(report.leagues)}\n"
        f"Client SayÄ±sÄ±: {report.num_clients}\n"
        f"Toplam Round: {report.total_rounds}\n"
        f"Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Global DoÄŸruluk: {report.global_accuracy:.1%}\n"
        f"YakÄ±nsama Roundu: {report.convergence_round}\n\n"
        + "\n".join(
            f"  â€¢ {cr.league}: acc={cr.accuracy:.1%}, loss={cr.loss:.4f} "
            f"({cr.samples} Ã¶rnek)"
            for cr in report.client_reports
        )
        + f"\n\nğŸ’¡ {report.recommendation}",
        title="Federated Learning Raporu",
        border_style="green" if report.global_accuracy > 0.5 else "yellow",
    ))


@app.command()
def hypergraph_test(
    team: str = typer.Option("Galatasaray", help="TakÄ±m adÄ±"),
    missing: str = typer.Option("", help="Eksik oyuncu indeksleri (virgÃ¼lle)"),
):
    """Hypergraph â€“ taktiksel birim kÄ±rÄ±lganlÄ±k analizi."""
    from src.quant.hypergraph_unit import HypergraphUnitAnalyzer, TacticalUnit
    from rich.panel import Panel
    import numpy as np

    hg = HypergraphUnitAnalyzer()

    ratings = np.array([75, 80, 72, 78, 85, 70, 82, 90, 65, 77, 88], dtype=np.float64)
    units = [
        TacticalUnit("Defans HattÄ±", "defense", [0, 1, 2, 3],
                      ["GK", "CB1", "CB2", "RB"], weight=1.5),
        TacticalUnit("Orta Saha ÃœÃ§lÃ¼sÃ¼", "midfield", [4, 5, 6],
                      ["CM1", "CM2", "CAM"], weight=1.2),
        TacticalUnit("HÃ¼cum BloÄŸu", "attack", [7, 8, 9, 10],
                      ["LW", "ST", "RW", "SS"], weight=1.0),
    ]

    missing_list = [int(x.strip()) for x in missing.split(",") if x.strip().isdigit()]

    report = hg.analyze_team(team, units, ratings, missing_players=missing_list)

    lines = []
    for ua in report.unit_analyses:
        alert = "ğŸ”´" if ua.failure_prob > 0.35 else "ğŸŸ¢"
        lines.append(
            f"{alert} {ua.unit.name}: Ã§Ã¶kÃ¼ÅŸ={ua.failure_prob:.1%}, "
            f"uyum={ua.cohesion:.2f}, yedeklilik={ua.redundancy:.2f}"
            + (f" âš  zayÄ±f={ua.weakest_link}" if ua.weakest_link else "")
        )

    color = "red" if report.defense_alert else ("yellow" if report.vulnerability_index > 0.3 else "green")
    console.print(Panel(
        f"TakÄ±m: {report.team}\n"
        f"Oyuncu: {report.n_players} | Birim: {report.n_units}\n"
        f"Eksik: {missing_list or 'Yok'}\n\n"
        f"{'â”€' * 40}\n"
        + "\n".join(lines) + "\n\n"
        f"{'â”€' * 40}\n"
        f"TakÄ±m Uyumu: {report.team_cohesion:.2f}\n"
        f"KÄ±rÄ±lganlÄ±k: {report.vulnerability_index:.2f}\n"
        f"YapÄ±sal Entropi: {report.structural_entropy:.3f}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Hypergraph Birim Analizi",
        border_style=color,
    ))


@app.command()
def anneal(
    n_candidates: int = typer.Option(30, help="Aday bahis sayÄ±sÄ±"),
    max_bets: int = typer.Option(8, help="Maksimum kupon boyutu"),
):
    """Quantum Annealer â€“ optimal kupon sepeti optimizasyonu."""
    from src.core.quantum_annealer import QuantumAnnealer, BetCandidate
    from rich.panel import Panel
    import numpy as np

    qa = QuantumAnnealer(bankroll=10000.0, max_bets=max_bets, max_iter=10000)

    # Demo adaylar
    candidates = []
    leagues = ["EPL", "SL", "BL", "LL", "SA"]
    for i in range(n_candidates):
        prob = np.random.uniform(0.35, 0.70)
        odds = round(1 / prob + np.random.uniform(-0.2, 0.5), 2)
        edge = round(prob * odds - 1, 4)
        candidates.append(BetCandidate(
            match_id=f"match_{i+1}",
            selection=np.random.choice(["home", "draw", "away", "over25"]),
            odds=odds,
            prob=prob,
            value_edge=max(0, edge),
            risk=round(np.random.uniform(0.1, 0.5), 3),
            expected_return=round(max(0, edge), 4),
            league=np.random.choice(leagues),
        ))

    sol = qa.optimize(candidates, max_bets=max_bets)

    color = "green" if sol.sharpe_ratio > 1.0 else ("yellow" if sol.sharpe_ratio > 0 else "red")
    console.print(Panel(
        f"Aday Havuzu: {n_candidates} maÃ§\n"
        f"SeÃ§ilen: {sol.n_bets} bahis\n"
        f"Metod: {sol.method}\n"
        f"SÃ¼re: {sol.elapsed_ms:.0f}ms\n\n"
        f"{'â”€' * 40}\n"
        f"Sharpe Ratio: {sol.sharpe_ratio:.2f}\n"
        f"Beklenen Kar: {sol.expected_profit:.2f} TL\n"
        f"Beklenen Risk: {sol.expected_risk:.2f}\n"
        f"Stake: {sol.total_stake:.0f} TL\n"
        f"Ã‡eÅŸitlilik: {sol.diversification:.0%}\n\n"
        f"SeÃ§ilen MaÃ§lar: {sol.selected_matches[:10]}\n\n"
        f"ğŸ’¡ {sol.recommendation}",
        title="Simulated Annealing â€“ Optimal PortfÃ¶y",
        border_style=color,
    ))


@app.command()
def rlhf(
    days: int = typer.Option(30, help="Rapor sÃ¼resi (gÃ¼n)"),
):
    """RLHF â€“ Ä°nsan geri bildirim istatistikleri."""
    from src.utils.human_feedback_loop import HumanFeedbackLoop
    from rich.panel import Panel

    hfl = HumanFeedbackLoop()
    stats = hfl.get_stats(days=days)

    console.print(Panel(
        f"Toplam Geri Bildirim: {stats.total_feedback}\n"
        f"  âœ… Onay: {stats.approvals}\n"
        f"  âŒ Ret: {stats.rejections}\n\n"
        f"{'â”€' * 40}\n"
        f"Ä°nsan DoÄŸruluÄŸu: {stats.human_accuracy:.1%}\n"
        f"Model DoÄŸruluÄŸu: {stats.model_accuracy:.1%}\n"
        f"Ä°nsan-Model Uyumu: {stats.agreement_rate:.1%}\n"
        f"Ort. Ã–dÃ¼l: {stats.avg_reward:+.3f}\n"
        f"Toplam Ã–dÃ¼l: {stats.total_reward:+.2f}\n\n"
        f"{'â”€' * 40}\n"
        f"Ret Sebepleri: {dict(stats.reason_distribution) or 'HenÃ¼z yok'}",
        title="RLHF Geri Bildirim Raporu",
        border_style="blue",
    ))

    # Feature ayarlamalarÄ±
    adjustments = hfl.get_feature_adjustments()
    if adjustments:
        console.print(Panel(
            "\n".join(f"  {k}: {v:+.4f}" for k, v in adjustments.items()),
            title="Feature AÄŸÄ±rlÄ±k AyarlamalarÄ±",
            border_style="dim",
        ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 18 CLI â€“ Self-Healing, Fluid Dynamics, Ricci Curvature
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def fluid(
    n_players: int = typer.Option(22, help="Oyuncu sayÄ±sÄ±"),
):
    """Fluid Dynamics â€“ saha kontrolÃ¼ ve akÄ±ÅŸ analizi."""
    from src.quant.fluid_pitch import FluidPitchAnalyzer, PlayerState
    from rich.panel import Panel
    import numpy as np

    fpa = FluidPitchAnalyzer()

    # Demo: rastgele 22 oyuncu konumu
    players = []
    for i in range(11):
        players.append(PlayerState(
            player_id=f"h{i}", team="home",
            x=float(np.random.uniform(5, 100)),
            y=float(np.random.uniform(5, 63)),
            vx=float(np.random.uniform(-2, 3)),
            vy=float(np.random.uniform(-2, 2)),
            has_ball=(i == 9),
            rating=float(np.random.uniform(65, 90)),
        ))
    for i in range(11):
        players.append(PlayerState(
            player_id=f"a{i}", team="away",
            x=float(np.random.uniform(5, 100)),
            y=float(np.random.uniform(5, 63)),
            vx=float(np.random.uniform(-3, 2)),
            vy=float(np.random.uniform(-2, 2)),
            rating=float(np.random.uniform(65, 90)),
        ))

    report = fpa.analyze(players)

    color = "green" if report.home_control_pct > 55 else ("red" if report.away_control_pct > 55 else "yellow")
    console.print(Panel(
        f"Saha KontrolÃ¼: Ev={report.home_control_pct:.1f}% / Dep={report.away_control_pct:.1f}%\n"
        f"xT (Ev): {report.home_xt:.4f} | xT (Dep): {report.away_xt:.4f}\n"
        f"BaskÄ±: Ev={report.home_pressure_index:.3f} / Dep={report.away_pressure_index:.3f}\n\n"
        f"{'â”€' * 40}\n"
        f"Momentum: {report.momentum_score:+.3f} ({report.flow_direction})\n"
        f"AÃ§Ä±k Kanallar: {report.open_channels} ({report.dominant_channel or '-'})\n"
        f"Savunma KompaktlÄ±k: {report.defensive_compactness:.3f}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="AkÄ±ÅŸkanlar DinamiÄŸi â€“ Saha KontrolÃ¼",
        border_style=color,
    ))


@app.command()
def ricci(
    n_teams: int = typer.Option(8, help="TakÄ±m sayÄ±sÄ±"),
    stress: bool = typer.Option(False, help="Stres senaryosu simÃ¼le et"),
):
    """Ricci Curvature â€“ piyasa sistemik risk analizi."""
    from src.quant.ricci_flow import RicciFlowAnalyzer
    from rich.panel import Panel
    import networkx as nx
    import numpy as np

    rfa = RicciFlowAnalyzer()

    # Demo: takÄ±m iliÅŸki aÄŸÄ±
    teams = ["GS", "FB", "BJK", "TS", "BAS", "ADN", "SIV", "ANT"][:n_teams]
    G = nx.Graph()
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            w = float(np.random.uniform(0.3, 1.0))
            if stress:
                w *= 0.3  # Stres: dÃ¼ÅŸÃ¼k aÄŸÄ±rlÄ±klar â†’ negatif eÄŸrilik
            G.add_edge(teams[i], teams[j], weight=w)

    report = rfa.analyze(G, name="cli_test")

    if report.kill_betting:
        color = "red"
    elif report.stress_level == "high":
        color = "yellow"
    else:
        color = "green"

    edges_str = "\n".join(
        f"  {u} â†” {v}: Îº={c:+.4f}" for u, v, c in report.critical_edges
    )

    console.print(Panel(
        f"AÄŸ: {n_teams} takÄ±m, {report.n_edges} kenar\n"
        f"Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Ort. EÄŸrilik (Îº): {report.avg_curvature:+.4f}\n"
        f"Min/Max: {report.min_curvature:+.4f} / {report.max_curvature:+.4f}\n"
        f"Std: {report.std_curvature:.4f}\n"
        f"Negatif Kenarlar: {report.n_negative}/{report.n_edges}\n\n"
        f"{'â”€' * 40}\n"
        f"Stres: {report.stress_level.upper()}\n"
        f"Sistemik Risk: {report.systemic_risk:.0%}\n"
        f"Kriz OlasÄ±lÄ±ÄŸÄ±: {report.crisis_probability:.0%}\n"
        f"Stake Ã‡arpanÄ±: x{report.stake_multiplier:.1f}\n"
        f"Kill Betting: {'EVET' if report.kill_betting else 'HAYIR'}\n\n"
        f"Kritik Kenarlar:\n{edges_str}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Ricci Curvature â€“ Sistemik Risk",
        border_style=color,
    ))


@app.command()
def healing_report(
    days: int = typer.Option(30, help="Rapor sÃ¼resi (gÃ¼n)"),
):
    """Self-Healing â€“ otomatik kod iyileÅŸtirme raporu."""
    from src.core.auto_healer import SelfHealingEngine
    from rich.panel import Panel

    engine = SelfHealingEngine()
    report = engine.get_report(days=days)

    console.print(Panel(
        f"Toplam Deneme: {report.total_attempts}\n"
        f"  âœ… BaÅŸarÄ±lÄ±: {report.successful_heals}\n"
        f"  âŒ BaÅŸarÄ±sÄ±z: {report.failed_heals}\n"
        f"  ğŸ”„ Rollback: {report.rollbacks}\n"
        f"  ğŸš« BloklanmÄ±ÅŸ: {report.blocked_patches}\n\n"
        f"{'â”€' * 40}\n"
        f"Benzersiz Hata: {report.unique_errors}\n\n"
        f"En SÄ±k Hatalar:\n"
        + "\n".join(
            f"  â€¢ {etype}: {count}x" for etype, count in report.top_errors
        ) if report.top_errors else "  HenÃ¼z hata kaydÄ± yok.",
        title="Self-Healing Raporu",
        border_style="blue",
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 19 CLI â€“ Telemetry, Regime Switcher, SDE Pricer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def regime(
    n_periods: int = typer.Option(18, help="Zaman periyodu sayÄ±sÄ± (~5dk her biri)"),
    team: str = typer.Option("Galatasaray", help="TakÄ±m adÄ±"),
):
    """Hidden Markov â€“ gizli rejim tespiti (BaskÄ±n/Dengeli/Pasif)."""
    from src.quant.regime_switcher import RegimeSwitcher
    from rich.panel import Panel
    import numpy as np

    rs = RegimeSwitcher(n_regimes=3)

    # Demo: maÃ§ istatistikleri [ÅŸut, pas, top_kontrolÃ¼]
    data = np.column_stack([
        np.random.poisson(3, n_periods) + np.array([2]*6 + [0]*6 + [1]*6)[:n_periods],
        np.random.normal(45, 8, n_periods),
        np.random.normal(50, 10, n_periods),
    ])

    report = rs.analyze_match(data, team=team, match_id="cli_test")

    regime_seq = " â†’ ".join(
        f"{c}{n}" for c, n in zip(
            [{"BaskÄ±n": "ğŸŸ¢", "Dengeli": "ğŸŸ¡", "Pasif": "ğŸ”´"}.get(n, "âšª") for n in report.regime_names],
            report.regime_names,
        )
    )

    color = "red" if report.momentum_break else ("green" if report.current.regime_id == 0 else "yellow")
    console.print(Panel(
        f"TakÄ±m: {report.team}\n"
        f"Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Mevcut Rejim: {report.current.regime_name} "
        f"(gÃ¼ven={report.current.confidence:.0%})\n"
        f"Gol Ã‡arpanÄ±: x{report.current.goal_multiplier:.2f}\n\n"
        f"BaskÄ±nlÄ±k: {report.dominance_pct:.0f}% | Pasiflik: {report.passivity_pct:.0f}%\n"
        f"Stabilite: {report.stability:.2f}\n"
        f"Momentum KÄ±rÄ±lmasÄ±: {'EVET' if report.momentum_break else 'HAYIR'}\n\n"
        f"{'â”€' * 40}\n"
        f"Rejim Dizisi:\n{regime_seq}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Hidden Markov â€“ Rejim Analizi",
        border_style=color,
    ))


@app.command()
def sde(
    current: float = typer.Option(1.85, help="Mevcut oran"),
    horizon: int = typer.Option(10, help="Tahmin ufku (dakika)"),
    n_history: int = typer.Option(30, help="GeÃ§miÅŸ veri noktasÄ± sayÄ±sÄ±"),
):
    """SDE Pricer â€“ Ornstein-Uhlenbeck oran tahmini."""
    from src.quant.sde_pricer import SDEPricer
    from rich.panel import Panel
    import numpy as np

    pricer = SDEPricer()

    # Demo: oran geÃ§miÅŸi (mean-reverting)
    fair = current + np.random.uniform(-0.1, 0.1)
    noise = np.cumsum(np.random.randn(n_history) * 0.02)
    history = fair + noise * 0.5
    history = np.clip(history, 1.05, 10.0)
    history[-1] = current

    fc = pricer.forecast(history.tolist(), match_id="cli_test", horizon_min=horizon)

    # MC simÃ¼lasyon
    sim = pricer.simulate(
        current, fc.params.theta, fc.fair_value,
        fc.params.sigma, horizon_min=horizon,
    )

    if fc.value_signal == "BUY":
        color = "green"
    elif fc.value_signal == "SELL":
        color = "red"
    else:
        color = "yellow"

    console.print(Panel(
        f"Mevcut Oran: {fc.current_odds}\n"
        f"Fair Value (Î¼): {fc.fair_value:.4f}\n"
        f"Beklenen ({horizon}dk): {fc.predicted_odds:.4f} "
        f"({fc.expected_change_pct:+.2f}%)\n"
        f"GÃ¼ven AralÄ±ÄŸÄ±: [{fc.ci_lower:.4f}, {fc.ci_upper:.4f}]\n\n"
        f"{'â”€' * 40}\n"
        f"OU Parametreleri:\n"
        f"  Î¸ (mean-reversion): {fc.params.theta:.4f}\n"
        f"  Î¼ (fair value): {fc.params.mu:.4f}\n"
        f"  Ïƒ (volatilite): {fc.params.sigma:.4f}\n"
        f"  YarÄ± Ã–mÃ¼r: {fc.params.half_life:.1f} dk\n"
        f"  RÂ²: {fc.params.r_squared:.4f}\n\n"
        f"{'â”€' * 40}\n"
        f"Monte Carlo ({pricer._n_paths} yol):\n"
        f"  Ort: {sim['mean']:.4f} | Medyan: {sim['median']:.4f}\n"
        f"  P(yukarÄ±): {sim['prob_up']:.0%} | P(aÅŸaÄŸÄ±): {sim['prob_down']:.0%}\n"
        f"  [p5, p95]: [{sim['p5']:.4f}, {sim['p95']:.4f}]\n\n"
        f"Sinyal: {fc.value_signal} | Edge: {fc.edge_pct:+.1f}%\n"
        f"Volatilite: {fc.volatility_rank}\n\n"
        f"ğŸ’¡ {fc.recommendation}",
        title="SDE Pricer â€“ Ornstein-Uhlenbeck",
        border_style=color,
    ))


@app.command()
def telemetry_report():
    """Telemetry â€“ sistem darboÄŸaz raporu."""
    from src.core.telemetry_tracer import TelemetryTracer
    from rich.panel import Panel

    tracer = TelemetryTracer()

    # Demo span'ler
    import time as _time
    for module in ["neo4j", "vision", "rl_trader", "scraper", "ensemble"]:
        with tracer.span(f"{module}_query", module=module):
            _time.sleep(0.001 * (hash(module) % 50 + 5))

    report = tracer.get_bottleneck_report()

    lines = []
    for mod, stats in sorted(
        report.module_stats.items(),
        key=lambda x: -x[1]["avg_ms"],
    ):
        lines.append(
            f"  {'ğŸ”´' if stats['avg_ms'] > 100 else 'ğŸŸ¢'} {mod}: "
            f"ort={stats['avg_ms']:.1f}ms, "
            f"max={stats['max_ms']:.1f}ms, "
            f"n={stats['count']}"
        )

    console.print(Panel(
        f"Toplam Span: {report.total_spans}\n"
        f"Toplam SÃ¼re: {report.total_duration_ms:.1f}ms\n"
        f"Hata OranÄ±: {report.error_rate:.1%}\n\n"
        f"{'â”€' * 40}\n"
        f"ModÃ¼l PerformanslarÄ±:\n"
        + "\n".join(lines) + "\n\n"
        f"{'â”€' * 40}\n"
        f"DarboÄŸaz: {report.bottleneck_module} "
        f"({report.bottleneck_avg_ms:.1f}ms)\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Telemetry â€“ DarboÄŸaz Raporu",
        border_style="blue",
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 20 CLI â€“ Quantum Brain, Hawkes, Blind Strategy, War Room
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def quantum(
    n_samples: int = typer.Option(200, help="EÄŸitim Ã¶rneÄŸi sayÄ±sÄ±"),
    n_qubits: int = typer.Option(4, help="Qubit sayÄ±sÄ±"),
    epochs: int = typer.Option(20, help="EÄŸitim epok sayÄ±sÄ±"),
):
    """Quantum Brain â€“ VQC/QSVM ile maÃ§ tahmini."""
    from src.quant.quantum_brain import QuantumBrain
    from rich.panel import Panel
    import numpy as np

    qb = QuantumBrain(n_qubits=n_qubits, n_layers=2, lr=0.01)

    # Demo veri: [xG, ÅŸut, pas%, top_kontrolÃ¼]
    X = np.random.randn(n_samples, n_qubits) * 0.5
    y = np.random.randint(0, 3, n_samples)

    report = qb.train(X, y, epochs=epochs)

    # Tek maÃ§ tahmini
    test_feat = np.random.randn(n_qubits) * 0.3
    pred = qb.predict_match(test_feat, match_id="cli_test")

    labels = {0: "Ev Sahibi", 1: "Beraberlik", 2: "Deplasman"}
    color = "green" if pred.confidence > 0.5 else "yellow"
    console.print(Panel(
        f"Metod: {report.method}\n"
        f"Qubit: {report.n_qubits} | Katman: {report.n_layers}\n"
        f"EÄŸitim DoÄŸruluÄŸu: {report.accuracy:.1%}\n\n"
        f"{'â”€' * 40}\n"
        f"Test Tahmin: {labels.get(pred.prediction, '?')}\n"
        f"OlasÄ±lÄ±klar: Ev={pred.probabilities[0]:.0%}, "
        f"Ber={pred.probabilities[1]:.0%}, "
        f"Dep={pred.probabilities[2]:.0%}\n"
        f"GÃ¼ven: {pred.confidence:.0%}\n"
        f"Devre DerinliÄŸi: {pred.circuit_depth}\n"
        f"Hesaplama: {pred.compute_time_ms:.1f}ms\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Quantum Brain â€“ VQC/QSVM",
        border_style=color,
    ))


@app.command()
def hawkes_cmd(
    goals: str = typer.Option("12,35,37,78", help="Gol dakikalarÄ± (virgÃ¼lle)"),
    current_min: float = typer.Option(85.0, help="Mevcut dakika"),
):
    """Hawkes Momentum â€“ gol bulaÅŸÄ±cÄ±lÄ±ÄŸÄ± analizi."""
    from src.quant.hawkes_momentum import HawkesMomentumAnalyzer
    from rich.panel import Panel

    hma = HawkesMomentumAnalyzer(match_duration=90.0)

    event_times = [float(x.strip()) for x in goals.split(",") if x.strip()]
    report = hma.analyze_match(event_times, match_id="cli_test", current_min=current_min)

    # Gelecek tahmin
    next_pred = hma.predict_next_event(event_times, current_min, horizon_min=10)

    if report.criticality == "supercritical":
        color = "red"
    elif report.criticality == "critical":
        color = "yellow"
    else:
        color = "green"

    console.print(Panel(
        f"Metod: {report.method}\n"
        f"Olay SayÄ±sÄ±: {report.n_events}\n"
        f"Son Olay: {report.last_event_min}. dk "
        f"({report.time_since_last:.0f} dk Ã¶nce)\n\n"
        f"{'â”€' * 40}\n"
        f"Hawkes Parametreleri:\n"
        f"  Î¼ (base rate): {report.params.mu:.4f}\n"
        f"  Î± (excitement): {report.params.alpha:.4f}\n"
        f"  Î² (decay): {report.params.beta:.4f}\n"
        f"  BR (branching): {report.params.branching_ratio:.4f}\n"
        f"  YarÄ± Ã–mÃ¼r: {report.params.half_life:.1f} dk\n\n"
        f"{'â”€' * 40}\n"
        f"AnlÄ±k YoÄŸunluk: {report.current_intensity:.4f} "
        f"(x{report.excitement_ratio:.1f} baseline)\n"
        f"5dk OlasÄ±lÄ±k: {report.next_event_prob_5min:.0%}\n"
        f"10dk OlasÄ±lÄ±k: {report.next_event_prob_10min:.0%}\n\n"
        f"Kritiklik: {report.criticality}\n"
        f"Momentum: {report.momentum_level}\n"
        f"Ãœst Sinyali: {'EVET' if report.over_signal else 'HAYIR'}\n"
        f"Gol PatlamasÄ±: {'EVET' if report.goal_burst_alert else 'HAYIR'}\n\n"
        f"Pik Dakika (+{next_pred['peak_minute']}dk): "
        f"{next_pred['peak_prob']:.0%}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Hawkes Momentum â€“ Self-Exciting Process",
        border_style=color,
    ))


@app.command()
def blind(
    n_bets: int = typer.Option(5, help="Bahis sayÄ±sÄ±"),
):
    """Blind Strategy â€“ ÅŸifreli hesaplama raporu."""
    from src.core.blind_strategy import BlindStrategyEngine
    from rich.panel import Panel
    import numpy as np

    bse = BlindStrategyEngine()

    # Demo: ÅŸifreli Kelly
    probs = np.random.uniform(0.3, 0.7, n_bets).tolist()
    odds = [round(1 / max(p, 0.01) + np.random.uniform(-0.2, 0.5), 2) for p in probs]

    kelly_result = bse.blind_kelly(
        bse.encrypt(probs, "demo"),
        probs, odds,
    )

    # Demo: ÅŸifreli skor
    features = np.random.randn(8).tolist()
    weights = np.random.randn(8).tolist()
    score_result = bse.blind_score(features, weights)

    report = bse.get_report()

    console.print(Panel(
        f"Metod: {report.method}\n"
        f"Toplam Ä°ÅŸlem: {report.total_operations}\n"
        f"Åifreli Ä°ÅŸlem: {report.encrypted_ops}\n"
        f"Plaintext Ä°ÅŸlem: {report.plaintext_ops}\n\n"
        f"{'â”€' * 40}\n"
        f"Åifreli Kelly SonuÃ§larÄ±:\n"
        + "\n".join(
            f"  Bahis {i+1}: odds={odds[i]:.2f}, p={probs[i]:.2f} â†’ "
            f"f*={kelly_result.plaintext_result[i]:.4f}"
            for i in range(min(n_bets, len(kelly_result.plaintext_result)))
        ) + "\n\n"
        f"{'â”€' * 40}\n"
        f"Åifreli Skor: {score_result.plaintext_result[0]:.4f}\n"
        f"Hesaplama: {kelly_result.compute_time_ms:.1f}ms (Kelly), "
        f"{score_result.compute_time_ms:.1f}ms (Score)\n"
        f"Åifreleme: {kelly_result.encryption_scheme or 'masked'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Blind Strategy â€“ Homomorphic Encryption",
        border_style="blue",
    ))


@app.command()
def warroom(
    home: str = typer.Option("Galatasaray", help="Ev sahibi"),
    away: str = typer.Option("FenerbahÃ§e", help="Deplasman"),
    odds: float = typer.Option(2.10, help="Oran"),
    prob: float = typer.Option(0.55, help="Model olasÄ±lÄ±ÄŸÄ±"),
):
    """War Room â€“ 3 ajan tartÄ±ÅŸmasÄ±."""
    from src.utils.war_room import WarRoom
    from rich.panel import Panel

    wr = WarRoom(llm_backend="auto")

    ev = prob * odds - 1
    kelly = max(0, (prob * odds - 1) / (odds - 1)) if odds > 1 else 0

    result = wr.debate(
        match_info={
            "home": home,
            "away": away,
            "odds": odds,
            "prob": prob,
            "ev": ev,
            "kelly": kelly,
            "confidence": prob,
        },
        match_id="cli_warroom",
    )

    if result.majority_verdict == "BET":
        color = "green"
    elif result.majority_verdict == "SKIP":
        color = "red"
    else:
        color = "yellow"

    lines = [f"MaÃ§: {home} vs {away}\n"]
    for op in result.opinions:
        lines.append(f"{op.agent_name}:")
        lines.append(f"  \"{op.opinion}\"")
        lines.append(f"  Karar: {op.verdict} (gÃ¼ven={op.confidence:.0%})\n")

    lines.append("â”€" * 40)
    lines.append(
        f"OYLAMA: âœ…BET={result.bet_count} â¸ï¸HOLD={result.hold_count} "
        f"âŒSKIP={result.skip_count}"
    )
    lines.append(
        f"KARAR: {result.majority_verdict}"
        + (" (OYBÄ°RLÄ°ÄÄ°)" if result.consensus else "")
    )

    console.print(Panel(
        "\n".join(lines),
        title="War Room â€“ Multi-Agent Debate",
        border_style=color,
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 20+ CLI â€“ Survival Estimator, Fatigue Engine
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def survival_cmd(
    n_matches: int = typer.Option(50, help="GeÃ§miÅŸ maÃ§ sayÄ±sÄ±"),
    current_min: float = typer.Option(65.0, help="Mevcut dakika"),
    last_goal_min: float = typer.Option(40.0, help="Son gol dakikasÄ±"),
    team: str = typer.Option("Galatasaray", help="TakÄ±m"),
):
    """Survival Analysis â€“ savunma saÄŸkalÄ±m analizi."""
    from src.quant.survival_estimator import SurvivalEstimator
    from rich.panel import Panel
    import numpy as np

    se = SurvivalEstimator()

    # Demo: gol yemeye kadar geÃ§en sÃ¼reler
    durations = np.concatenate([
        np.random.exponential(35, n_matches // 2),
        np.full(n_matches - n_matches // 2, 90.0),
    ])
    durations = np.clip(durations, 1, 90)
    observed = (durations < 90).astype(int)

    se.fit(durations, observed)
    report = se.analyze(
        current_minute=current_min,
        last_goal_minute=last_goal_min,
        team=team,
        match_id="cli_test",
    )

    if report.dam_breaking:
        color = "red"
    elif report.fortress_mode:
        color = "green"
    else:
        color = "yellow"

    console.print(Panel(
        f"TakÄ±m: {report.team}\n"
        f"Metod: {report.method}\n"
        f"Mevcut Dakika: {report.current_minute:.0f}\n"
        f"Son Golden Beri: {report.minutes_since_last_goal:.0f} dk\n\n"
        f"{'â”€' * 40}\n"
        f"SaÄŸkalÄ±m Parametreleri:\n"
        f"  S(t): {report.params.survival_prob:.0%} "
        f"(saÄŸkalÄ±m olasÄ±lÄ±ÄŸÄ±)\n"
        f"  H(t): {report.params.cumulative_hazard:.4f} "
        f"(kÃ¼mÃ¼latif tehlike)\n"
        f"  h(t): {report.params.current_hazard:.6f} "
        f"(anlÄ±k tehlike)\n"
        f"  Medyan: {report.params.median_survival:.1f} dk\n\n"
        f"{'â”€' * 40}\n"
        f"Tahminler:\n"
        f"  5dk gol yeme: {report.prob_concede_5min:.0%}\n"
        f"  10dk gol yeme: {report.prob_concede_10min:.0%}\n"
        f"  15dk gol yeme: {report.prob_concede_15min:.0%}\n"
        f"  Beklenen gol: {report.expected_time_to_goal:.1f} dk\n\n"
        f"{'â”€' * 40}\n"
        f"Risk: {report.risk_level}\n"
        f"Baraj YÄ±kÄ±lÄ±yor: {'EVET' if report.dam_breaking else 'HAYIR'}\n"
        f"Kale SaÄŸlam: {'EVET' if report.fortress_mode else 'HAYIR'}\n"
        f"Tehlike Sivrilmesi: {'EVET' if report.hazard_spike else 'HAYIR'}\n"
        f"Ãœst Sinyali: {'EVET' if report.over_signal else 'HAYIR'}\n"
        f"Alt Sinyali: {'EVET' if report.under_signal else 'HAYIR'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Survival Analysis â€“ SaÄŸkalÄ±m",
        border_style=color,
    ))


@app.command()
def fatigue_cmd(
    n_players: int = typer.Option(11, help="Oyuncu sayÄ±sÄ±"),
    current_min: float = typer.Option(75.0, help="Mevcut dakika"),
    team: str = typer.Option("Galatasaray", help="TakÄ±m"),
):
    """Fatigue Engine â€“ biyomekanik yorgunluk analizi."""
    from src.quant.fatigue_engine import FatigueEngine
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    fe = FatigueEngine(base_intensity=0.5)

    positions = ["GK"] + ["DEF"] * 4 + ["MID"] * 3 + ["FWD"] * 3
    names = [
        "Muslera", "Boey", "Davinson", "Nelsson", "Kerem A.",
        "Torreira", "Mertens", "Ziyech",
        "Icardi", "BarÄ±ÅŸ A.", "Kerem D.",
    ]

    players = []
    for i in range(min(n_players, len(names))):
        pos = positions[i]
        base_dist = {"GK": 4.5, "DEF": 9.5, "MID": 11.5, "FWD": 9.0}[pos]
        dist = base_dist * (current_min / 90) + np.random.uniform(-0.5, 0.5)
        sprints = int({"GK": 2, "DEF": 10, "MID": 18, "FWD": 15}[pos]
                     * (current_min / 90) + np.random.randint(-2, 3))
        hi_mins = float({"GK": 5, "DEF": 15, "MID": 22, "FWD": 18}[pos]
                       * (current_min / 90) + np.random.uniform(-3, 3))
        players.append({
            "id": f"p{i+1}", "name": names[i], "position": pos,
            "distance_km": round(dist, 1),
            "sprints": max(0, sprints),
            "hi_mins": round(max(0, hi_mins), 1),
        })

    report = fe.analyze_team(
        players, current_minute=current_min,
        team=team, match_id="cli_test",
    )

    # Oyuncu tablosu
    table = Table(title="Oyuncu Yorgunluk DetayÄ±")
    table.add_column("Oyuncu", style="bold")
    table.add_column("Pozisyon")
    table.add_column("Enerji", justify="right")
    table.add_column("Laktik", justify="right")
    table.add_column("BiliÅŸsel", justify="right")
    table.add_column("Hata%", justify="right")
    table.add_column("Durum")

    for p in sorted(report.players, key=lambda x: x.stamina):
        status = "ğŸ”´ KRÄ°TÄ°K" if p.is_critical else ("ğŸŸ¡ Yorgun" if p.stamina < 40 else "ğŸŸ¢ Ä°yi")
        table.add_row(
            p.player_name, p.position,
            f"{p.stamina:.0f}%",
            f"{p.lactic_acid:.1f}",
            f"{p.cognitive_factor:.2f}",
            f"{p.error_probability:.1%}",
            status,
        )

    console.print(table)

    if report.defense_collapse_risk:
        color = "red"
    elif report.late_goal_signal:
        color = "yellow"
    else:
        color = "green"

    console.print(Panel(
        f"TakÄ±m: {report.team} (dk {report.current_minute:.0f})\n\n"
        f"{'â”€' * 40}\n"
        f"TakÄ±m Ã–zeti:\n"
        f"  Ort. Enerji: {report.avg_stamina:.0f}%\n"
        f"  Min Enerji: {report.min_stamina:.0f}% ({report.weakest_player})\n"
        f"  Kritik Oyuncu: {report.critical_count}\n\n"
        f"{'â”€' * 40}\n"
        f"Savunma:\n"
        f"  Ort. Enerji: {report.defense_avg_stamina:.0f}%\n"
        f"  KÄ±rÄ±lganlÄ±k: {report.defense_vulnerability:.0%}\n"
        f"  Ã‡Ã¶kÃ¼ÅŸ Riski: {'EVET' if report.defense_collapse_risk else 'HAYIR'}\n\n"
        f"HÃ¼cum:\n"
        f"  Ort. Enerji: {report.attack_avg_stamina:.0f}%\n"
        f"  Etkinlik: {report.attack_effectiveness:.0%}\n\n"
        f"{'â”€' * 40}\n"
        f"Kontra Atak Riski: {'EVET' if report.counter_attack_risk else 'HAYIR'}\n"
        f"GeÃ§ Gol Sinyali: {'EVET' if report.late_goal_signal else 'HAYIR'}\n"
        f"Oyuncu DeÄŸiÅŸikliÄŸi Etkisi: {report.substitution_impact:.0f}%\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Fatigue Engine â€“ Biyomekanik Yorgunluk",
        border_style=color,
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 23 CLI â€“ Chaos Filter, Homology Scanner, Rust Engine
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def chaos(
    n_points: int = typer.Option(100, help="Zaman serisi uzunluÄŸu"),
    regime: str = typer.Option("mixed", help="Rejim: stable|chaotic|mixed"),
):
    """Chaos Filter â€“ Lyapunov Ã¼ssÃ¼ ile kaos tespiti."""
    from src.quant.chaos_filter import ChaosFilter
    from rich.panel import Panel
    import numpy as np

    cf = ChaosFilter(emb_dim=3, lag=1)

    # Demo zaman serisi
    if regime == "stable":
        data = 1.85 + np.cumsum(np.random.randn(n_points) * 0.005)
    elif regime == "chaotic":
        # Logistic map (kaotik)
        data = np.zeros(n_points)
        data[0] = 0.5
        r = 3.9  # Kaotik rejim
        for i in range(1, n_points):
            data[i] = r * data[i - 1] * (1 - data[i - 1])
    else:
        # KarÄ±ÅŸÄ±k: ilk yarÄ± stabil, ikinci yarÄ± kaotik
        stable = 1.85 + np.cumsum(np.random.randn(n_points // 2) * 0.003)
        chaotic = np.zeros(n_points - n_points // 2)
        chaotic[0] = 0.5
        for i in range(1, len(chaotic)):
            chaotic[i] = 3.9 * chaotic[i - 1] * (1 - chaotic[i - 1])
        data = np.concatenate([stable, chaotic + 1.5])

    report = cf.analyze(data, match_id="cli_test", market="home_odds")

    if report.kill_betting:
        color = "red"
    elif report.reduce_stake:
        color = "yellow"
    elif report.boost_confidence:
        color = "green"
    else:
        color = "blue"

    console.print(Panel(
        f"Metod: {report.method}\n"
        f"Veri NoktasÄ±: {report.n_observations}\n\n"
        f"{'â”€' * 40}\n"
        f"Kaos Parametreleri:\n"
        f"  Î» (Lyapunov): {report.params.max_lyapunov:.6f}\n"
        f"  Kor. Boyut: {report.params.correlation_dim:.4f}\n"
        f"  Sample Entropi: {report.params.sample_entropy:.4f}\n"
        f"  Hurst: {report.params.hurst_exponent:.4f}\n"
        f"  DFA: {report.params.dfa:.4f}\n\n"
        f"{'â”€' * 40}\n"
        f"Rejim: {report.regime}\n"
        f"Kaos Skoru: {report.chaos_score:.0%}\n"
        f"Tahmin Edilebilirlik: {report.predictability:.0%}\n\n"
        f"Bahis Ä°ptal: {'EVET' if report.kill_betting else 'HAYIR'}\n"
        f"Stake DÃ¼ÅŸÃ¼r: {'EVET' if report.reduce_stake else 'HAYIR'}\n"
        f"GÃ¼ven ArtÄ±r: {'EVET' if report.boost_confidence else 'HAYIR'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Chaos Filter â€“ Lyapunov Exponents",
        border_style=color,
    ))


@app.command()
def homology_cmd(
    n_players: int = typer.Option(10, help="Oyuncu sayÄ±sÄ±"),
    formation: str = typer.Option("442", help="Formasyon: 442|433|352"),
):
    """Homology Scanner â€“ topolojik organizasyon analizi."""
    from src.quant.homology_scanner import HomologyScanner
    from rich.panel import Panel
    import numpy as np

    hs = HomologyScanner(max_dim=1, max_edge=50.0)

    # Demo pozisyonlar (formasyona gÃ¶re)
    if formation == "433":
        positions = np.array([
            [5, 34],                         # GK
            [20, 10], [20, 25], [20, 43], [20, 58],  # DEF
            [45, 20], [45, 34], [45, 48],   # MID
            [70, 15], [70, 34], [70, 53],   # FWD
        ])[:n_players]
    elif formation == "352":
        positions = np.array([
            [5, 34],
            [20, 15], [20, 34], [20, 53],   # 3 DEF
            [40, 5], [40, 20], [40, 34], [40, 48], [40, 63],  # 5 MID
            [70, 25], [70, 43],             # 2 FWD
        ])[:n_players]
    else:  # 442
        positions = np.array([
            [5, 34],
            [20, 10], [20, 25], [20, 43], [20, 58],
            [45, 10], [45, 25], [45, 43], [45, 58],
            [70, 25], [70, 43],
        ])[:n_players]

    # GÃ¼rÃ¼ltÃ¼ ekle
    positions = positions.astype(float) + np.random.randn(*positions.shape) * 3

    report = hs.analyze_team(
        positions, team="Demo TakÄ±m", match_id="cli_test",
    )

    if report.team_panicking:
        color = "red"
    elif report.formation_broken:
        color = "yellow"
    elif report.organized_play:
        color = "green"
    else:
        color = "blue"

    # KalÄ±cÄ±lÄ±k daÄŸÄ±lÄ±mÄ±
    finite = [
        p for p in report.persistence_pairs
        if p.persistence < float("inf")
    ]
    dim0 = [p for p in finite if p.dimension == 0]
    dim1 = [p for p in finite if p.dimension == 1]

    console.print(Panel(
        f"TakÄ±m: {report.team}\n"
        f"Metod: {report.method}\n"
        f"Oyuncu SayÄ±sÄ±: {report.n_players}\n\n"
        f"{'â”€' * 40}\n"
        f"Betti SayÄ±larÄ±:\n"
        f"  Î²â‚€ (BaÄŸlÄ± BileÅŸen): {report.betti_0}\n"
        f"  Î²â‚ (DÃ¶ngÃ¼/Delik): {report.betti_1}\n\n"
        f"KalÄ±cÄ±lÄ±k:\n"
        f"  Toplam Ã–zellik: {report.n_features}\n"
        f"  Hâ‚€ (sonlu): {len(dim0)} | Hâ‚: {len(dim1)}\n"
        f"  Ort. KalÄ±cÄ±lÄ±k: {report.avg_persistence:.2f}\n"
        f"  Maks KalÄ±cÄ±lÄ±k: {report.max_persistence:.2f}\n"
        f"  GÃ¼rÃ¼ltÃ¼ OranÄ±: {report.noise_ratio:.0%}\n\n"
        f"{'â”€' * 40}\n"
        f"Skorlar:\n"
        f"  Organizasyon: {report.organization_score:.0%}\n"
        f"  KompaktlÄ±k: {report.compactness_score:.0%}\n"
        f"  BaÄŸlantÄ±sallÄ±k: {report.connectivity_score:.0%}\n\n"
        f"Panik: {'EVET' if report.team_panicking else 'HAYIR'}\n"
        f"Organize: {'EVET' if report.organized_play else 'HAYIR'}\n"
        f"Kopuk Gruplar: {'EVET' if report.isolated_groups else 'HAYIR'}\n"
        f"Pas DÃ¶ngÃ¼leri: {'EVET' if report.passing_cycles else 'HAYIR'}\n"
        f"Formasyon Bozuk: {'EVET' if report.formation_broken else 'HAYIR'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Homology Scanner â€“ Persistent Homology",
        border_style=color,
    ))


@app.command()
def rust_bench():
    """Rust Engine â€“ benchmark raporu."""
    from src.core.rust_engine import RustEngine
    from rich.panel import Panel
    from rich.table import Table

    engine = RustEngine()
    report = engine.benchmark(n_sims=100_000)

    table = Table(title="Benchmark SonuÃ§larÄ±")
    table.add_column("Fonksiyon", style="bold")
    table.add_column("Motor")
    table.add_column("SÃ¼re (ms)", justify="right")

    for b in report.benchmarks:
        ms = b.rust_ms if b.engine == "rust" else b.python_ms
        table.add_row(b.function, b.engine, f"{ms:.2f}")

    console.print(table)

    if report.engine == "rust":
        color = "green"
    elif report.engine == "numba":
        color = "yellow"
    else:
        color = "blue"

    console.print(Panel(
        f"Motor: {report.engine}\n"
        f"Toplam Ã‡aÄŸrÄ±: {report.total_calls}\n"
        f"Toplam SÃ¼re: {report.total_time_ms:.2f}ms\n"
        f"Ort. SÃ¼re: {report.avg_time_ms:.2f}ms\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Rust Engine â€“ Demir Ã‡ekirdek",
        border_style=color,
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 24 CLI â€“ AutoML, Synthetic, Fuzzy, Briefing
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def automl_cmd(
    n_samples: int = typer.Option(500, help="EÄŸitim Ã¶rneÄŸi sayÄ±sÄ±"),
    n_features: int = typer.Option(10, help="Ã–zellik sayÄ±sÄ±"),
    time_min: int = typer.Option(2, help="Arama sÃ¼resi (dakika)"),
):
    """AutoML â€“ en iyi modeli otomatik bul."""
    from src.quant.automl_engine import AutoMLEngine
    from rich.panel import Panel
    import numpy as np

    aml = AutoMLEngine(generations=3, population_size=30)

    # Demo veri
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    result = aml.search(X, y, task="classify", time_budget_min=time_min)

    color = "green" if result.best_score > 0.6 else ("yellow" if result.best_score > 0.45 else "red")
    console.print(Panel(
        f"Metod: {result.method}\n"
        f"En Ä°yi Model: {result.best_model_name}\n"
        f"Skor: {result.best_score:.1%}\n"
        f"Denenen Model: {result.n_models_tried}\n"
        f"Arama SÃ¼resi: {result.search_time_sec:.1f}s\n\n"
        f"{'â”€' * 40}\n"
        f"Parametreler:\n"
        + "\n".join(f"  {k}: {v}" for k, v in result.best_params.items())
        + f"\n\nModel Yolu: {result.model_path}\n"
        f"Pipeline Kodu: {result.pipeline_code}\n\n"
        f"ğŸ’¡ {result.recommendation}",
        title="AutoML â€“ TPOT / RandomSearch",
        border_style=color,
    ))


@app.command()
def synthetic_cmd(
    n_real: int = typer.Option(200, help="GerÃ§ek veri satÄ±r sayÄ±sÄ±"),
    n_synthetic: int = typer.Option(5000, help="Ãœretilecek sentetik sayÄ±sÄ±"),
    n_features: int = typer.Option(8, help="Ã–zellik sayÄ±sÄ±"),
):
    """Synthetic Trainer â€“ sentetik veri Ã¼retimi ve kalite kontrolÃ¼."""
    from src.quant.synthetic_trainer import SyntheticTrainer
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    st = SyntheticTrainer(noise_scale=0.03)

    # Demo gerÃ§ek veri
    real = np.random.randn(n_real, n_features)
    real[:, 0] = np.random.poisson(1.5, n_real)  # xG
    real[:, 1] = np.random.poisson(12, n_real)    # Åut
    real[:, 2] = np.random.normal(55, 10, n_real)  # Pas%

    cols = ["xG", "Åut", "Pas%"] + [f"f{i}" for i in range(3, n_features)]
    synthetic = st.generate(real, n_samples=n_synthetic, column_names=cols)
    report = st.quality_check(real, synthetic, column_names=cols)

    # Kalite tablosu
    table = Table(title="SÃ¼tun BazlÄ± Kalite KontrolÃ¼")
    table.add_column("SÃ¼tun", style="bold")
    table.add_column("KS Stat", justify="right")
    table.add_column("KS p-val", justify="right")
    table.add_column("Ort. Fark%", justify="right")
    table.add_column("Durum")

    for m in report.quality_metrics[:8]:
        status = "âœ…" if m.passed else "âŒ"
        table.add_row(
            m.column,
            f"{m.ks_statistic:.4f}",
            f"{m.ks_pvalue:.4f}",
            f"{m.mean_diff_pct:.1f}%",
            status,
        )

    console.print(table)

    colors = {"excellent": "green", "good": "yellow", "poor": "red"}
    console.print(Panel(
        f"GerÃ§ek: {report.n_real} satÄ±r | Sentetik: {report.n_synthetic} satÄ±r\n"
        f"Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Genel Kalite: {report.overall_quality}\n"
        f"Ort. KS p-deÄŸeri: {report.avg_ks_pvalue:.4f}\n"
        f"Korelasyon FarkÄ±: {report.correlation_diff:.4f}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Synthetic Data Vault",
        border_style=colors.get(report.overall_quality, "blue"),
    ))


@app.command()
def fuzzy_cmd(
    weather: float = typer.Option(0.7, help="Hava (0=gÃ¼neÅŸ, 1=yaÄŸmur)"),
    fatigue_val: float = typer.Option(0.6, help="Yorgunluk (0-1)"),
    travel: float = typer.Option(0.4, help="Deplasman mesafesi (0-1)"),
    injury: float = typer.Option(0.3, help="SakatlÄ±k (0-1)"),
    motivation: float = typer.Option(0.5, help="Motivasyon (0-1)"),
):
    """Fuzzy Logic â€“ bulanÄ±k mantÄ±k risk deÄŸerlendirmesi."""
    from src.quant.fuzzy_reasoning import FuzzyReasoningEngine, FuzzyInput
    from rich.panel import Panel

    fre = FuzzyReasoningEngine()

    inputs = FuzzyInput(
        weather=weather,
        fatigue=fatigue_val,
        travel_distance=travel,
        injury_count=injury,
        motivation=motivation,
        form=0.6,
        crowd_factor=0.5,
    )

    output = fre.evaluate(inputs)

    colors = {
        "dÃ¼ÅŸÃ¼k": "green", "orta": "yellow",
        "yÃ¼ksek": "red", "Ã§ok_yÃ¼ksek": "red",
    }

    rules_text = "\n".join(f"  â€¢ {r}" for r in output.active_rules) if output.active_rules else "  (Yok)"

    console.print(Panel(
        f"Metod: {output.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Girdiler:\n"
        f"  Hava: {weather:.1f} | Yorgunluk: {fatigue_val:.1f}\n"
        f"  Deplasman: {travel:.1f} | SakatlÄ±k: {injury:.1f}\n"
        f"  Motivasyon: {motivation:.1f}\n\n"
        f"{'â”€' * 40}\n"
        f"Risk Skoru: {output.risk_score:.0f}/100\n"
        f"Risk Seviyesi: {output.risk_level}\n"
        f"GÃ¼ven Ã‡arpanÄ±: x{output.confidence_modifier:.1f}\n"
        f"Gol Beklentisi Ã‡arpanÄ±: x{output.goal_expectation_mod:.2f}\n\n"
        f"{'â”€' * 40}\n"
        f"Aktif Kurallar:\n{rules_text}\n\n"
        f"ğŸ’¡ {output.recommendation}",
        title="Fuzzy Logic â€“ BulanÄ±k MantÄ±k",
        border_style=colors.get(output.risk_level, "blue"),
    ))


@app.command()
def briefing(
    bankroll: float = typer.Option(12450, help="Kasa (TL)"),
    change: float = typer.Option(3.2, help="GÃ¼nlÃ¼k deÄŸiÅŸim (%)"),
    roi: float = typer.Option(8.7, help="30 gÃ¼nlÃ¼k ROI (%)"),
):
    """Daily Briefing â€“ yÃ¶netici gÃ¼nlÃ¼k Ã¶zeti."""
    from src.utils.daily_briefing import DailyBriefing, BriefingData
    from rich.panel import Panel

    db = DailyBriefing(llm_backend="auto")

    data = BriefingData(
        bankroll=bankroll,
        bankroll_change_pct=change,
        roi_30d=roi,
        sharpe_ratio=1.42,
        max_drawdown=5.3,
        win_rate_7d=0.62,
        total_bets_7d=23,
        profit_7d=850,
        top_opportunities=[
            {"home": "Galatasaray", "away": "FenerbahÃ§e", "ev": 0.08, "kelly": 0.032},
            {"home": "BeÅŸiktaÅŸ", "away": "Trabzonspor", "ev": 0.05, "kelly": 0.021},
            {"home": "Adana Demir", "away": "Gaziantep", "ev": 0.04, "kelly": 0.018},
        ],
        alerts=[
            "Model drift tespit (Wasserstein: 0.32)",
            "GS defansÄ± yorgun (stamina: 28%)",
        ],
        uptime_pct=99.7,
        error_rate=0.3,
        model_accuracy=0.62,
    )

    report = db.generate(data)

    console.print(Panel(
        report.telegram_text,
        title="Executive Daily Briefing",
        border_style="cyan",
    ))

    if report.summary:
        console.print(f"\n[dim]Ã–zet metodu: {report.method} "
                      f"({report.generation_time_ms:.0f}ms)[/]")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 25 CLI â€“ Uncertainty, Topology, GraphRAG
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def uncertainty_cmd(
    n_samples: int = typer.Option(300, help="EÄŸitim Ã¶rneÄŸi sayÄ±sÄ±"),
    n_features: int = typer.Option(6, help="Ã–zellik sayÄ±sÄ±"),
):
    """Epistemic vs Aleatoric â€“ belirsizlik ayrÄ±mÄ± demo."""
    from src.quant.uncertainty_separator import UncertaintySeparator
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    us = UncertaintySeparator(n_models=10, n_classes=3)

    # Demo veri
    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] * 0.5 > 0).astype(int)
    y_train[X_train[:, 2] > 1.5] = 2

    us.fit(X_train, y_train)

    # Test maÃ§larÄ±
    test_cases = [
        ("Favori MaÃ§ (Bilinen)", np.array([2.0, 1.5, 0.3, 0.1, 0.5, 0.7])),
        ("Belirsiz MaÃ§ (Veri Yok)", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Kaotik MaÃ§ (YÃ¼ksek Risk)", np.array([0.5, -0.5, 1.8, -1.2, 2.0, -0.8])),
    ]

    table = Table(title="Epistemic vs Aleatoric Belirsizlik AyrÄ±mÄ±")
    table.add_column("MaÃ§", style="bold")
    table.add_column("Tahmin", justify="center")
    table.add_column("Epistemik", justify="right")
    table.add_column("Aleatorik", justify="right")
    table.add_column("Toplam", justify="right")
    table.add_column("Karar")

    for name, features in test_cases:
        report = us.analyze(features, match_id=name, team=name)
        labels = {0: "Ev", 1: "Ber", 2: "Dep"}
        pred = labels.get(report.predicted_class, "?")

        color = {"BET": "green", "HALF_KELLY": "yellow", "ABSTAIN": "red"}.get(
            report.decision, "white",
        )
        table.add_row(
            name,
            f"{pred} ({report.predicted_prob:.0%})",
            f"{report.epistemic:.3f}",
            f"{report.aleatoric:.3f}",
            f"{report.total_uncertainty:.3f}",
            f"[{color}]{report.decision}[/]",
        )

    console.print(table)

    # DetaylÄ± son rapor
    last = us.analyze(test_cases[-1][1], match_id="demo")
    console.print(Panel(
        f"Metod: {last.method}\n"
        f"Model SayÄ±sÄ±: {last.n_samples}\n\n"
        f"{'â”€' * 40}\n"
        f"Epistemik OranÄ±: {last.epistemic_ratio:.0%} (bilgisizlik)\n"
        f"Aleatorik OranÄ±: {last.aleatoric_ratio:.0%} (ÅŸans)\n"
        f"GÃ¼ven Ã‡arpanÄ±: x{last.confidence_modifier:.1f}\n\n"
        f"ğŸ’¡ {last.recommendation}",
        title="Belirsizlik DetayÄ±",
        border_style="cyan",
    ))


@app.command()
def topology_cmd(
    n_matches: int = typer.Option(500, help="Tarihsel maÃ§ sayÄ±sÄ±"),
    n_features: int = typer.Option(7, help="Ã–zellik sayÄ±sÄ±"),
):
    """Topology Mapper â€“ topolojik kÃ¼me haritasÄ± demo."""
    from src.quant.topology_mapper import TopologyMapper
    from rich.panel import Panel
    import numpy as np

    tm = TopologyMapper(n_cubes=8, overlap=0.3)

    # Demo tarihsel veri
    X = np.random.randn(n_matches, n_features)
    X[:50, :] *= 3.0   # Anomali grubu
    labels = np.random.randint(0, 3, n_matches)
    labels[:50] = 2     # Anomali grubu deplasman

    cols = ["xG", "Åut", "Pas%", "Form", "OddH", "OddD", "OddA"]
    tm.fit(X, labels=labels, column_names=cols)

    # BugÃ¼nkÃ¼ maÃ§lar
    test_cases = [
        ("Normal MaÃ§", np.random.randn(n_features) * 0.5),
        ("Anomali MaÃ§", np.random.randn(n_features) * 3.5),
    ]

    for name, features in test_cases:
        report = tm.analyze_match(features, match_id=name, team=name)
        color = "red" if report.is_anomalous else "green"

        console.print(Panel(
            f"Metod: {report.method}\n\n"
            f"{'â”€' * 40}\n"
            f"KÃ¼me: #{report.assigned_cluster} ({report.cluster_size} maÃ§)\n"
            f"KÃ¼me Etiketi: {report.cluster_label}\n"
            f"KÃ¼me Ort. SonuÃ§: {report.cluster_avg_outcome:.2f}\n"
            f"Anomali Skoru: {report.anomaly_score:.0%}\n"
            f"Anomali: {'EVET' if report.is_anomalous else 'HAYIR'}\n\n"
            f"Graf: {report.n_nodes} dÃ¼ÄŸÃ¼m, {report.n_edges} baÄŸ\n\n"
            f"ğŸ’¡ {report.recommendation}",
            title=f"Topology Mapper â€“ {name}",
            border_style=color,
        ))

    # HTML oluÅŸtur
    html_path = tm.visualize("demo_mapper.html")
    console.print(f"\n[dim]HTML: {html_path}[/]")


@app.command()
def graphrag_cmd(
    team: str = typer.Option("Galatasaray", help="TakÄ±m adÄ±"),
):
    """GraphRAG â€“ bilgi grafiÄŸi kriz analizi demo."""
    from src.memory.graph_rag import GraphRAG
    from rich.panel import Panel
    from rich.table import Table

    grag = GraphRAG(llm_backend="auto")

    # Demo haberler
    news = [
        {"title": f"{team} kaptanÄ± sakatlÄ±k geÃ§irdi, 3 hafta yok", "source": "spor_gazetesi"},
        {"title": f"{team} teknik direktÃ¶rÃ¼ istifa sinyali verdi", "source": "tv_haberi"},
        {"title": f"{team} taraftarÄ± protesto dÃ¼zenledi", "source": "sosyal_medya"},
        {"title": f"{team} yeni transfer bombasÄ±nÄ± patlattÄ±", "source": "transfer_haberi"},
        {"title": f"{team} son maÃ§ta muhteÅŸem galibiyet aldÄ±", "source": "spor_gazetesi"},
        {"title": f"{team} defans oyuncusu kadro dÄ±ÅŸÄ± bÄ±rakÄ±ldÄ±", "source": "kulÃ¼p_aÃ§Ä±klama"},
    ]

    grag.ingest_news(news, team=team)

    # Kriz analizi
    crisis = grag.analyze_crisis(team, match_id="demo")

    colors = {
        "stable": "green", "tension": "yellow",
        "crisis": "red", "meltdown": "red",
    }

    events = "\n".join(f"  â€¢ {e}" for e in crisis.connected_events) if crisis.connected_events else "  (Yok)"
    hidden = "\n".join(f"  â€¢ {h}" for h in crisis.hidden_connections) if crisis.hidden_connections else "  (Yok)"

    console.print(Panel(
        f"Metod: {crisis.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Kriz Skoru: {crisis.crisis_score:.0%}\n"
        f"Kriz Seviyesi: {crisis.crisis_level.upper()}\n"
        f"Negatif Haber: {crisis.negative_news_count}\n\n"
        f"{'â”€' * 40}\n"
        f"BaÄŸlÄ± Olaylar:\n{events}\n\n"
        f"Kilit VarlÄ±klar: {', '.join(crisis.key_entities[:5])}\n\n"
        f"Gizli BaÄŸlantÄ±lar:\n{hidden}\n\n"
        + (f"{'â”€' * 40}\nLLM Analizi: {crisis.llm_analysis}\n\n" if crisis.llm_analysis else "")
        + f"ğŸ’¡ {crisis.recommendation}",
        title=f"GraphRAG â€“ {team} Kriz Analizi",
        border_style=colors.get(crisis.crisis_level, "blue"),
    ))

    # Grafik istatistikleri
    stats = grag.get_stats()
    console.print(f"\n[dim]Grafik: {stats}[/]")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 26 CLI â€“ Probabilistic, Active Inference, MF-DFA
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def probabilistic_cmd(
    n_matches: int = typer.Option(200, help="EÄŸitim maÃ§ sayÄ±sÄ±"),
    n_teams: int = typer.Option(6, help="TakÄ±m sayÄ±sÄ±"),
):
    """Probabilistic Engine â€“ olasÄ±lÄ±ksal Bayesian maÃ§ tahmini."""
    from src.quant.probabilistic_engine import ProbabilisticEngine
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    pe = ProbabilisticEngine(n_samples=1500, n_tune=500)

    # Demo veri
    teams = [f"TakÄ±m_{chr(65 + i)}" for i in range(n_teams)]
    home_teams, away_teams, hg_list, ag_list = [], [], [], []
    for _ in range(n_matches):
        h, a = np.random.choice(len(teams), 2, replace=False)
        home_teams.append(teams[h])
        away_teams.append(teams[a])
        hg_list.append(int(np.random.poisson(1.5)))
        ag_list.append(int(np.random.poisson(1.1)))

    report = pe.fit(hg_list, ag_list, home_teams, away_teams)

    console.print(Panel(
        f"Metod: {report.method}\n"
        f"TakÄ±m: {report.n_teams} | MaÃ§: {report.n_matches_trained}\n"
        f"EÄŸitim: {report.fit_time_sec:.1f}s\n\n"
        f"{'â”€' * 40}\n"
        f"Ev AvantajÄ±: {report.home_advantage.mean:.3f} "
        f"Â± {report.home_advantage.std:.3f}\n"
        f"  HDI: [{report.home_advantage.hdi_low:.3f}, "
        f"{report.home_advantage.hdi_high:.3f}]",
        title="Probabilistic Engine â€“ Bayesian Fit",
        border_style="cyan",
    ))

    # Tahmin tablosu
    table = Table(title="OlasÄ±lÄ±ksal MaÃ§ Tahminleri")
    table.add_column("MaÃ§", style="bold")
    table.add_column("Ev", justify="right")
    table.add_column("Ber", justify="right")
    table.add_column("Dep", justify="right")
    table.add_column("Ãœ2.5", justify="right")
    table.add_column("KG", justify="right")
    table.add_column("En OlasÄ± Skor")

    for _ in range(3):
        h, a = np.random.choice(len(teams), 2, replace=False)
        pred = pe.predict(teams[h], teams[a])
        top = pred.most_likely_scores[0] if pred.most_likely_scores else (0, 0, 0)
        table.add_row(
            f"{teams[h]} vs {teams[a]}",
            f"{pred.p_home:.0%}", f"{pred.p_draw:.0%}", f"{pred.p_away:.0%}",
            f"{pred.p_over25:.0%}", f"{pred.p_btts:.0%}",
            f"{top[0]}-{top[1]} ({top[2]:.0%})",
        )

    console.print(table)


@app.command()
def active_inf_cmd(
    n_obs: int = typer.Option(100, help="SimÃ¼le edilecek gÃ¶zlem sayÄ±sÄ±"),
):
    """Active Inference â€“ serbest enerji ajanÄ± demo."""
    from src.core.active_inference_agent import ActiveInferenceAgent
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    aia = ActiveInferenceAgent(modules=[
        "poisson", "lightgbm", "lstm", "ensemble", "sentiment",
    ])

    # SimÃ¼le edilmiÅŸ gÃ¶zlemler
    for i in range(n_obs):
        observed = np.random.choice([0, 1, 2], p=[0.45, 0.27, 0.28])
        for mod in ["poisson", "lightgbm", "lstm", "ensemble", "sentiment"]:
            # Her modÃ¼lÃ¼n farklÄ± doÄŸruluk seviyesi
            noise = {"poisson": 0.3, "lightgbm": 0.15, "lstm": 0.25,
                     "ensemble": 0.1, "sentiment": 0.5}[mod]
            base = np.array([0.45, 0.27, 0.28])
            probs = base + np.random.randn(3) * noise
            probs = np.clip(probs, 0.05, 1.0)
            probs /= probs.sum()
            aia.observe(mod, probs.tolist(), int(observed))

    report = aia.get_report()
    prec = aia.get_precision_weights()
    alloc = aia.get_resource_allocation()

    # ModÃ¼l tablosu
    table = Table(title="ModÃ¼l DurumlarÄ± (Active Inference)")
    table.add_column("ModÃ¼l", style="bold")
    table.add_column("DoÄŸruluk", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Ort. Surprisal", justify="right")
    table.add_column("Ensemble AÄŸÄ±rlÄ±k", justify="right")
    table.add_column("Kaynak", justify="right")
    table.add_column("Durum")

    for name, state in report.module_states.items():
        status = "ğŸ”´ YENÄ°DEN EÄÄ°T" if state.needs_retraining else "ğŸŸ¢ STABIL"
        table.add_row(
            name,
            f"{state.accuracy:.0%}",
            f"{state.precision:.2f}",
            f"{state.avg_surprisal:.3f}",
            f"{prec.get(name, 0):.1%}",
            f"{alloc.get(name, 0):.1%}",
            status,
        )

    console.print(table)

    color = "red" if report.retrain_targets else "green"
    console.print(Panel(
        f"Metod: {report.method}\n\n"
        f"{'â”€' * 40}\n"
        f"Ort. Surprisal: {report.avg_surprisal:.4f}\n"
        f"Serbest Enerji: {report.total_free_energy:.4f}\n\n"
        f"Yeniden EÄŸitim: {', '.join(report.retrain_targets) or 'Yok'}\n"
        f"Aktif Ã–rnekleme: {', '.join(report.active_sampling_targets) or 'Yok'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Active Inference â€“ Serbest Enerji Raporu",
        border_style=color,
    ))


@app.command()
def multifractal_cmd(
    n_points: int = typer.Option(500, help="Zaman serisi uzunluÄŸu"),
    regime: str = typer.Option("mixed", help="Rejim: stable/chaotic/mixed"),
):
    """MF-DFA â€“ Ã§oklu fraktal piyasa analizi demo."""
    from src.quant.multifractal_logic import MultifractalAnalyzer
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    mfa = MultifractalAnalyzer(q_min=-5, q_max=5, q_step=1.0)

    # Demo zaman serisi
    if regime == "stable":
        data = np.cumsum(np.random.randn(n_points) * 0.01) + 2.0
    elif regime == "chaotic":
        data = np.cumsum(np.random.randn(n_points) * 0.1) + 2.0
        data[200:250] += np.random.randn(50) * 0.5  # Åok
    else:
        data = np.cumsum(np.random.randn(n_points) * 0.01) + 2.0
        data[300:350] += np.random.randn(50) * 0.3  # Hafif ÅŸok

    report = mfa.analyze(data, match_id=f"demo_{regime}", market="odds")
    p = report.params

    # h(q) tablosu
    if p.h_values:
        table = Table(title="h(q) â€“ GenelleÅŸtirilmiÅŸ Hurst ÃœssÃ¼")
        table.add_column("q", justify="right")
        table.add_column("h(q)", justify="right")
        for q_val in sorted(p.h_values.keys()):
            table.add_row(f"{q_val:.1f}", f"{p.h_values[q_val]:.4f}")
        console.print(table)

    colors = {
        "monofractal": "green",
        "weak_multifractal": "yellow",
        "strong_multifractal": "red",
    }

    console.print(Panel(
        f"Metod: {report.method}\n"
        f"Veri: {report.n_points} nokta\n\n"
        f"{'â”€' * 40}\n"
        f"h(2) â€“ Standart Hurst: {p.hurst_q2:.4f}\n"
        f"Î”h â€“ Ã‡oklu fraktallik: {p.delta_h:.4f}\n"
        f"Î± geniÅŸliÄŸi: {p.alpha_width:.4f}\n\n"
        f"{'â”€' * 40}\n"
        f"Rejim: {report.regime}\n"
        f"Trend: {'EVET' if report.is_trending else 'HAYIR'}\n"
        f"Ortalamaya DÃ¶nÃ¼ÅŸ: {'EVET' if report.is_mean_reverting else 'HAYIR'}\n"
        f"Rejim DeÄŸiÅŸikliÄŸi: {'âš ï¸ EVET' if report.regime_change_signal else 'HayÄ±r'}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title=f"MF-DFA â€“ Ã‡oklu Fraktal Analiz ({regime})",
        border_style=colors.get(report.regime, "blue"),
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Level 27 CLI â€“ Symbolic, Wavelet, Decision Flow
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def symbolic_cmd(
    n_samples: int = typer.Option(300, help="EÄŸitim Ã¶rneÄŸi sayÄ±sÄ±"),
):
    """Symbolic Discovery â€“ sembolik regresyon formÃ¼l keÅŸfi."""
    from src.quant.symbolic_discovery import SymbolicDiscovery
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    sd = SymbolicDiscovery(max_complexity=15, n_iterations=30)

    # Demo veri: gol = sqrt(xG) * form^1.1 + noise
    xg = np.random.exponential(1.2, n_samples)
    shots = np.random.poisson(12, n_samples).astype(float)
    form = np.random.uniform(0.3, 1.0, n_samples)
    possession = np.random.normal(55, 10, n_samples)
    goals = np.sqrt(xg) * form ** 1.1 + np.random.randn(n_samples) * 0.3

    X = np.column_stack([xg, shots, form, possession])
    report = sd.discover(X, goals, feature_names=["xG", "Åut", "Form", "Pas%"], target="gol")

    # Pareto front tablosu
    if report.pareto_front:
        table = Table(title="Pareto Front â€“ KeÅŸfedilen FormÃ¼ller")
        table.add_column("#", justify="right")
        table.add_column("FormÃ¼l", style="bold")
        table.add_column("RÂ²", justify="right")
        table.add_column("KarmaÅŸÄ±klÄ±k", justify="right")
        table.add_column("Skor", justify="right")

        for i, f in enumerate(report.pareto_front[:8], 1):
            color = "green" if f.r2 > 0.5 else ("yellow" if f.r2 > 0.2 else "red")
            table.add_row(
                str(i),
                f.equation[:60],
                f"[{color}]{f.r2:.1%}[/]",
                str(f.complexity),
                f"{f.score:.4f}",
            )
        console.print(table)

    color = "green" if report.best_formula.r2 > 0.5 else "yellow"
    console.print(Panel(
        f"Metod: {report.method}\n"
        f"Veri: {report.n_samples} maÃ§, {report.n_features} Ã¶zellik\n"
        f"Arama: {report.n_generations} nesil, {report.search_time_sec:.1f}s\n\n"
        f"{'â”€' * 40}\n"
        f"EN Ä°YÄ° FORMÃœL:\n"
        f"  {report.best_formula.equation}\n\n"
        f"RÂ²: {report.best_formula.r2:.1%} | "
        f"KarmaÅŸÄ±klÄ±k: {report.best_formula.complexity}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Symbolic Discovery â€“ FormÃ¼l KeÅŸfi",
        border_style=color,
    ))


@app.command()
def wavelet_cmd(
    n_points: int = typer.Option(200, help="Zaman serisi uzunluÄŸu"),
    noise_level: float = typer.Option(0.1, help="GÃ¼rÃ¼ltÃ¼ seviyesi"),
):
    """Wavelet Denoiser â€“ sinyal temizleme demo."""
    from src.quant.wavelet_denoiser import WaveletDenoiser
    from rich.panel import Panel
    import numpy as np

    wd = WaveletDenoiser(wavelet="db4", level=4)

    # Demo: trend + gÃ¼rÃ¼ltÃ¼ + fake moves
    t = np.linspace(0, 10, n_points)
    trend = 2.0 + 0.3 * np.sin(t * 0.5)  # GerÃ§ek trend
    noise = np.random.randn(n_points) * noise_level
    signal = trend + noise
    # Fake moves (3 adet ani spike)
    signal[50] += 0.8
    signal[120] -= 0.7
    signal[170] += 0.9

    report = wd.analyze(signal, match_id="demo", market="odds")
    r = report.result

    energy_text = "\n".join(
        f"  Seviye {k}: {v:.1f}%"
        for k, v in sorted(report.energy_by_level.items())
    ) if report.energy_by_level else "  (hesaplanamadÄ±)"

    color = "red" if report.fake_move_detected else "green"
    console.print(Panel(
        f"Metod: {r.method} (wavelet: {r.wavelet})\n"
        f"Veri: {len(r.original)} nokta, seviye: {r.level}\n\n"
        f"{'â”€' * 40}\n"
        f"GÃ¼rÃ¼ltÃ¼: %{r.noise_pct:.1f}\n"
        f"SNR: {r.snr_before:.1f} â†’ {r.snr_after:.1f} dB\n"
        f"Threshold: {r.threshold:.6f}\n\n"
        f"{'â”€' * 40}\n"
        f"Trend: {r.trend_direction} ({r.trend_slope:+.6f})\n"
        f"Fake Move: {'EVET (' + str(len(report.fake_move_times)) + ')' if report.fake_move_detected else 'HAYIR'}\n"
        f"{'  t = ' + str(report.fake_move_times[:5]) if report.fake_move_times else ''}\n\n"
        f"Enerji DaÄŸÄ±lÄ±mÄ±:\n{energy_text}\n"
        f"BaskÄ±n Frekans: {report.dominant_frequency}\n\n"
        f"ğŸ’¡ {report.recommendation}",
        title="Wavelet Denoiser â€“ Sinyal Temizleme",
        border_style=color,
    ))


@app.command()
def decision_flow_cmd(
    home: str = typer.Option("Galatasaray", help="Ev sahibi takÄ±m"),
    away: str = typer.Option("FenerbahÃ§e", help="Deplasman takÄ±mÄ±"),
):
    """Decision Flow â€“ karar akÄ±ÅŸ ÅŸemasÄ± demo."""
    from src.utils.decision_flow_gen import (
        DecisionFlowGenerator, DecisionFlow, DecisionStep,
    )
    from rich.panel import Panel

    dfg = DecisionFlowGenerator()

    flow = DecisionFlow(
        match_id="derbi_2026",
        home_team=home,
        away_team=away,
        steps=[
            DecisionStep("Kadro KontrolÃ¼", "TAMAM", "11/11 fit", "pass", "lineup"),
            DecisionStep("xG Analizi", "YÃœKSEK", "xG=1.82", "pass", "poisson"),
            DecisionStep("Oran DeÄŸeri (EV)", "+8.5%", "EV=+8.5%", "pass", "fair_value"),
            DecisionStep("Form Trendi", "YÃœKSELÄ°Å", "Mom=0.78", "pass", "lstm"),
            DecisionStep("Hava Durumu", "YAÄMURLU", "Riskâ†‘", "warn", "fuzzy"),
            DecisionStep("Yorgunluk", "YÃœKSEK", "Stamina=35%", "warn", "fatigue"),
            DecisionStep("Kaos Filtresi", "STABÄ°L", "Î»=-0.03", "pass", "chaos"),
            DecisionStep("Belirsizlik", "DÃœÅÃœK", "Îµ=0.12", "pass", "uncertainty"),
        ],
        final_decision="OYNA",
        confidence=0.75,
        ev_pct=8.5,
    )

    # Metin
    text = dfg.generate_text(flow)
    console.print(Panel(text, title="Karar AkÄ±ÅŸ ÅemasÄ±", border_style="cyan"))

    # Mermaid
    mermaid = dfg.generate_mermaid(flow)
    console.print(f"\n[dim]Mermaid.js:[/]\n{mermaid}")

    # GÃ¶rsel
    path = dfg.generate_image(flow)
    console.print(f"\n[dim]GÃ¶rsel: {path}[/]")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LEVEL 28: Volatility & Stream CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def volatility_cmd(
    n_points: int = typer.Option(200, help="Zaman serisi uzunluÄŸu"),
    team: str = typer.Option("Galatasaray", help="TakÄ±m adÄ±"),
):
    """GARCH Volatility Analyzer â€“ oynaklÄ±k kÃ¼melenmesi tespiti."""
    from src.quant.volatility_analyzer import VolatilityAnalyzer
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    va = VolatilityAnalyzer(model_type="GARCH", p=1, q=1)

    # Sentetik oran verisi: trend + kÃ¼meleÅŸen oynaklÄ±k
    np.random.seed(42)
    base = 1.85
    prices = [base]
    vol_state = 0.01
    for i in range(n_points - 1):
        if np.random.random() < 0.05:  # %5 rejim deÄŸiÅŸimi
            vol_state = np.random.choice([0.005, 0.02, 0.05, 0.10])
        shock = np.random.normal(0, vol_state)
        prices.append(max(prices[-1] + shock, 1.01))

    odds_arr = np.array(prices)
    log_ret = np.diff(np.log(odds_arr))

    report = va.analyze(log_ret, match_id="demo", team=team, market="odds")

    table = Table(title=f"GARCH Volatility â€“ {team}")
    table.add_column("Metrik", style="cyan")
    table.add_column("DeÄŸer", style="bold")

    table.add_row("Metod", report.method)
    table.add_row("GÃ¶zlem SayÄ±sÄ±", str(report.n_observations))
    table.add_row("AnlÄ±k Ïƒ", f"{report.current_volatility:.6f}")
    table.add_row("Ortalama Ïƒ", f"{report.avg_volatility:.6f}")
    table.add_row("Ïƒ YÃ¼zdeliÄŸi", f"{report.volatility_percentile:.1f}%")
    table.add_row("Rejim", f"{'ğŸ”´' if report.regime == 'crisis' else 'ğŸŸ¡' if report.regime == 'storm' else 'ğŸŸ¢'} {report.regime}")
    table.add_row("Rejim DeÄŸiÅŸimi", "âœ… Evet" if report.regime_change else "âŒ HayÄ±r")
    table.add_row("VaR 95%", f"{report.var_95:.6f}")
    table.add_row("VaR 99%", f"{report.var_99:.6f}")
    table.add_row("Kelly Ã‡arpanÄ±", f"x{report.kelly_multiplier:.2f}")
    table.add_row("1 GÃ¼n Ïƒ Tahmini", f"{report.forecast_1d:.6f}")
    table.add_row("5 GÃ¼n Ïƒ Tahmini", f"{report.forecast_5d:.6f}")

    p = report.params
    table.add_row("Ï‰ (omega)", f"{p.omega:.8f}")
    table.add_row("Î± (ARCH)", f"{p.alpha:.6f}")
    table.add_row("Î² (GARCH)", f"{p.beta:.6f}")
    table.add_row("Persistence (Î±+Î²)", f"{p.persistence:.6f}")
    table.add_row("YarÄ± Ã–mÃ¼r", f"{p.half_life:.1f} periyot")

    console.print(table)
    console.print(Panel(report.recommendation, title="Ã–neri", border_style="yellow"))


@app.command()
def stream_cmd():
    """Stream Processor â€“ canlÄ± veri akÄ±ÅŸÄ± demo."""
    from src.core.stream_processor import StreamProcessor, StreamEvent
    from rich.panel import Panel
    from rich.table import Table
    import asyncio
    import numpy as np

    async def _demo():
        sp = StreamProcessor(max_queue=1000, window_sec=10.0, n_workers=2)

        events_received = []

        def log_handler(event: StreamEvent):
            events_received.append(event)

        sp.register_consumer("logger", log_handler)
        await sp.start()

        # 50 olay gÃ¶nder
        np.random.seed(42)
        for i in range(50):
            await sp.emit(StreamEvent(
                event_type="odds_update",
                match_id=f"match_{i % 5}",
                data={"odds": round(float(1.5 + np.random.normal(0, 0.1)), 3)},
                source="demo",
            ))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.5)  # Ä°ÅŸleme zamanÄ±

        stats = sp.get_stats()
        await sp.stop()
        return stats, events_received, sp

    stats, events, sp = asyncio.run(_demo())

    table = Table(title="Stream Processor Stats")
    table.add_column("Metrik", style="cyan")
    table.add_column("DeÄŸer", style="bold")

    table.add_row("Toplam Olay", str(stats.total_events))
    table.add_row("Olay/sn", f"{stats.events_per_sec:.1f}")
    table.add_row("Aktif Consumer", str(stats.active_consumers))
    table.add_row("Kuyruk Boyutu", str(stats.queue_size))
    table.add_row("Ort. Gecikme", f"{stats.avg_latency_ms:.3f} ms")
    table.add_row("Hatalar", str(stats.errors))
    table.add_row("Uptime", f"{stats.uptime_sec:.1f} sn")
    table.add_row("Ä°ÅŸlenen", str(len(events)))

    console.print(table)

    # Windowed aggregate
    for mid in sp.get_active_matches():
        agg = sp.get_window(mid)
        if agg:
            console.print(Panel(
                f"Count: {agg.count}, "
                f"Avg: {agg.avg_value:.3f}, "
                f"Std: {agg.std_value:.4f}, "
                f"Range: [{agg.min_value:.3f}, {agg.max_value:.3f}]",
                title=f"Window â€“ {mid}",
                border_style="blue",
            ))


@app.command()
def orchestrator_cmd(
    stage: str = typer.Option("", help="Tek stage Ã§alÄ±ÅŸtÄ±r (ingestion/memory/quant/risk/utils)"),
):
    """Workflow Orchestrator â€“ Prefect pipeline durum raporu."""
    from src.core.workflow_orchestrator import (
        WorkflowOrchestrator,
        INGESTION_TASKS, MEMORY_TASKS, QUANT_TASKS, RISK_TASKS, UTILS_TASKS,
    )
    from rich.panel import Panel
    from rich.table import Table

    orch = WorkflowOrchestrator(max_retries=3)

    # Stage bazlÄ± task sayÄ±larÄ±
    stages = {
        "ingestion": INGESTION_TASKS,
        "memory": MEMORY_TASKS,
        "quant": QUANT_TASKS,
        "risk": RISK_TASKS,
        "utils": UTILS_TASKS,
    }

    table = Table(title="Workflow Orchestrator â€“ Pipeline YapÄ±sÄ±")
    table.add_column("Stage", style="cyan")
    table.add_column("Task SayÄ±sÄ±", style="bold")
    table.add_column("Ã‡alÄ±ÅŸma Modu", style="green")
    table.add_column("Ã–rnek Task'lar", style="dim")

    total = 0
    for stage_name, tasks in stages.items():
        total += len(tasks)
        mode = "Paralel" if stage_name in ("ingestion", "quant", "utils") else "SÄ±ralÄ±"
        examples = ", ".join(t.name for t in tasks[:3]) + ("â€¦" if len(tasks) > 3 else "")
        table.add_row(stage_name.upper(), str(len(tasks)), mode, examples)

    table.add_row("â”€" * 10, "â”€" * 5, "â”€" * 10, "â”€" * 20)
    table.add_row("TOPLAM", str(total), "", "")
    console.print(table)

    console.print(Panel(
        "Pipeline AkÄ±ÅŸÄ±:\n"
        "  1. INGESTION (Paralel) â†’ Veri toplama\n"
        "  2. MEMORY (SÄ±ralÄ±) â†’ Veri depolama & baÄŸlam\n"
        "  3. QUANT (Paralel) â†’ Kantitatif analiz\n"
        "  4. RISK (SÄ±ralÄ±) â†’ Risk yÃ¶netimi & karar\n"
        "  5. UTILS (Paralel) â†’ Raporlama & bildirim\n\n"
        "Retry: Her task 3 kez tekrar dener (exp. backoff)\n"
        "Circuit Breaker: 3 ardÄ±ÅŸÄ±k hata â†’ task devre dÄ±ÅŸÄ±",
        title="Mimari", border_style="blue",
    ))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LEVEL 29: Particle Filter & Deep Logging & Causal Discovery CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def particle_cmd(
    n_minutes: int = typer.Option(45, help="SimÃ¼lasyon dakika sayÄ±sÄ±"),
):
    """Particle Strength Tracker â€“ dinamik gÃ¼Ã§ takibi demo."""
    from src.quant.particle_strength_tracker import (
        ParticleStrengthTracker, MatchObservation,
    )
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    pst = ParticleStrengthTracker(n_particles=1000)
    pst.initialize(home_prior=0.55, away_prior=0.45)

    np.random.seed(42)
    reports = []
    for m in range(1, n_minutes + 1):
        obs = MatchObservation(
            minute=m,
            home_shots=int(np.random.poisson(0.15 * m)),
            away_shots=int(np.random.poisson(0.10 * m)),
            home_possession=float(np.clip(52 + np.random.normal(0, 5), 30, 70)),
            away_possession=0,
            home_dangerous_attacks=int(np.random.poisson(0.2 * m)),
            away_dangerous_attacks=int(np.random.poisson(0.15 * m)),
        )
        obs.away_possession = 100 - obs.home_possession
        r = pst.update(obs, match_id="demo")
        reports.append(r)

    last = reports[-1]
    table = Table(title=f"Particle Filter â€“ dk.{last.minute}")
    table.add_column("Metrik", style="cyan")
    table.add_column("DeÄŸer", style="bold")

    s = last.state
    table.add_row("Ev Sahibi GÃ¼cÃ¼", f"{s.home_power:.4f}")
    table.add_row("Deplasman GÃ¼cÃ¼", f"{s.away_power:.4f}")
    table.add_row("GÃ¼Ã§ FarkÄ±", f"{s.power_diff:+.4f}")
    table.add_row("Momentum", f"{s.momentum:+.4f}")
    table.add_row("Yorgunluk (Ev)", f"{s.fatigue_home:.4f}")
    table.add_row("Yorgunluk (Dep)", f"{s.fatigue_away:.4f}")
    table.add_row("ESS OranÄ±", f"{last.ess_ratio:.1%}")
    table.add_row("P(Ev KazanÄ±r)", f"{last.home_win_prob:.1%}")
    table.add_row("P(Beraberlik)", f"{last.draw_prob:.1%}")
    table.add_row("P(Dep KazanÄ±r)", f"{last.away_win_prob:.1%}")

    shifts = [r for r in reports if r.momentum_shift.detected]
    table.add_row("Momentum KaymalarÄ±", str(len(shifts)))

    console.print(table)

    if shifts:
        for sh in shifts[:3]:
            ms = sh.momentum_shift
            console.print(Panel(
                f"dk.{ms.minute}: {ms.direction} "
                f"(Î”={ms.magnitude:.3f}, "
                f"Ã¶nceki={ms.previous_power_diff:+.3f} â†’ "
                f"ÅŸimdi={ms.current_power_diff:+.3f})",
                title="Momentum Shift", border_style="red",
            ))


@app.command()
def causal_cmd(
    n_samples: int = typer.Option(200, help="Veri Ã¶rneÄŸi sayÄ±sÄ±"),
):
    """Causal Discovery â€“ nedensellik DAG keÅŸfi demo."""
    from src.quant.causal_discovery import CausalDiscovery
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np

    cd = CausalDiscovery(significance=0.05, max_cond_size=2)

    # Sentetik veri: xG â†’ goals, shots â†’ xG, poss â†’ shots (nedensellik zinciri)
    np.random.seed(42)
    possession = np.random.normal(55, 10, n_samples)
    shots = 0.3 * possession + np.random.normal(0, 2, n_samples)
    xg = 0.15 * shots + np.random.normal(0, 0.3, n_samples)
    goals = 0.8 * xg + np.random.normal(0, 0.5, n_samples)
    corners = 0.1 * possession + np.random.normal(5, 2, n_samples)  # Sahte korelasyon
    fouls = np.random.normal(12, 3, n_samples)  # BaÄŸÄ±msÄ±z

    data = np.column_stack([possession, shots, xg, goals, corners, fouls])
    names = ["possession", "shots", "xG", "goals", "corners", "fouls"]

    report = cd.analyze_match(data, names, target="goals", match_id="demo")

    table = Table(title="Causal Discovery â€“ DAG")
    table.add_column("Neden â†’ SonuÃ§", style="cyan")
    table.add_column("GÃ¼Ã§", style="bold")
    table.add_column("Tip", style="green")
    table.add_column("GÃ¼ven", style="yellow")

    for edge in report.dag.edges[:10]:
        table.add_row(
            f"{edge.source} â†’ {edge.target}",
            f"{edge.strength:.3f}",
            edge.edge_type,
            f"{edge.direction_confidence:.0%}",
        )

    console.print(table)
    console.print(f"\n[cyan]Metod:[/] {report.method}")
    console.print(f"[cyan]KÃ¶k Nedenler:[/] {', '.join(report.dag.root_causes) or 'Yok'}")
    console.print(f"[cyan]Yaprak Etkiler:[/] {', '.join(report.dag.leaf_effects) or 'Yok'}")
    console.print(f"[cyan]Gol KÃ¶k Nedenleri:[/] {', '.join(report.goal_root_causes) or 'Yok'}")
    console.print(f"[cyan]Sahte Korelasyonlar:[/] {len(report.spurious_correlations)}")
    console.print(f"[cyan]Ters Nedensellik UyarÄ±larÄ±:[/] {len(report.reverse_causation_warnings)}")
    console.print(Panel(report.recommendation, title="Ã–neri", border_style="yellow"))


@app.command()
def super_log_cmd():
    """Super Logger â€“ yapÄ±landÄ±rÄ±lmÄ±ÅŸ log sistemi demo."""
    from src.utils.super_logger import SuperLogger
    from rich.panel import Panel
    from rich.table import Table

    sl = SuperLogger(log_dir="data/logs")

    # ModÃ¼l loglarÄ±
    lg_poisson = sl.get_module_logger("quant.poisson")
    lg_lstm = sl.get_module_logger("quant.lstm")
    lg_risk = sl.get_module_logger("core.risk")

    lg_poisson.bind(match_id="gs_fb", duration_ms=12.5).info("Tahmin Ã¼retildi")
    lg_lstm.bind(match_id="gs_fb", duration_ms=45.8).info("Momentum hesaplandÄ±")
    lg_risk.bind(match_id="gs_fb", duration_ms=3.2).info("Risk deÄŸerlendirmesi")

    # Timed context
    with sl.timed("quant.poisson", match_id="gs_ts"):
        import time
        time.sleep(0.01)  # SimÃ¼lasyon

    # Karar logu
    sl.log_decision(
        module="ensemble", match_id="gs_fb",
        decision="BET", confidence=0.82,
        reason="EV=+8.5%, Kelly=3.2%",
        inputs={"xG": 1.82, "form": 0.78},
        outputs={"prob_home": 0.55, "fair_odds": 1.82},
    )

    table = Table(title="Super Logger â€“ YapÄ±landÄ±rÄ±lmÄ±ÅŸ Log Sistemi")
    table.add_column("Ã–zellik", style="cyan")
    table.add_column("DeÄŸer", style="bold")

    table.add_row("Log Dizini", str(sl.get_log_dir()))
    table.add_row("Log DosyalarÄ±", str(len(sl.get_log_files())))
    table.add_row("Rotation", "100 MB")
    table.add_row("Retention", "30 gÃ¼n")
    table.add_row("Compression", "gzip")
    table.add_row("Format", "JSON (JSONL)")

    stats = sl.get_all_stats()
    for module, s in stats.items():
        table.add_row(
            f"  {module}",
            f"entries={s.total_entries}, avg={s.avg_duration_ms:.1f}ms, "
            f"errors={s.errors}",
        )

    console.print(table)

    files = sl.get_log_files()
    if files:
        console.print(Panel(
            "\n".join(files[:10]),
            title="Log DosyalarÄ±", border_style="blue",
        ))


@app.command()
def agent_poll_cmd(
    home: str = typer.Option("Galatasaray", help="Ev sahibi takÄ±m"),
    away: str = typer.Option("FenerbahÃ§e", help="Deplasman takÄ±mÄ±"),
    odds: float = typer.Option(2.10, help="Bahis oranÄ±"),
    ev: float = typer.Option(0.08, help="Beklenen deÄŸer"),
):
    """Agent Poll â€“ Ajan Konseyi oylama demo."""
    from src.utils.war_room import WarRoom
    from src.utils.agent_poll_system import AgentPollSystem
    from rich.panel import Panel
    from rich.table import Table

    wr = WarRoom(llm_backend="auto")
    poll_sys = AgentPollSystem(notifier=None)

    match_info = {
        "home": home, "away": away,
        "odds": odds, "prob": 0.55, "ev": ev,
        "kelly": 0.04, "confidence": 0.75,
    }

    debate = wr.debate(match_info, match_id="demo_derbi")
    council = poll_sys.create_council_decision(debate, match_info)

    # Ajan oylarÄ± tablosu
    table = Table(title=f"Karar Konseyi: {home} vs {away}")
    table.add_column("Ajan", style="cyan")
    table.add_column("Oy", style="bold")
    table.add_column("GÃ¼ven", style="yellow")
    table.add_column("Anahtar Metrik", style="green")
    table.add_column("GerekÃ§e", style="dim")

    for v in council.votes:
        vote_emoji = "âœ…" if v.vote == "EVET" else "âŒ" if v.vote == "HAYIR" else "ğŸ¤”"
        table.add_row(
            f"{v.agent_emoji} {v.agent_name}",
            f"{vote_emoji} {v.vote}",
            f"{v.confidence:.0%}",
            v.key_metric,
            v.reasoning[:60] + "â€¦" if len(v.reasoning) > 60 else v.reasoning,
        )

    console.print(table)

    # KonsensÃ¼s
    _, desc = poll_sys.CONSENSUS_MAP.get(council.consensus_type, ("â“", "?"))
    console.print(Panel(
        f"{council.consensus_emoji} {desc}\n\n"
        f"EVET: {council.yes_count} | HAYIR: {council.no_count} | "
        f"KARASIZ: {council.undecided_count}\n\n"
        f"Konsey KararÄ±: {council.council_verdict}",
        title="Oylama Sonucu", border_style="green",
    ))

    # Telegram mesaj Ã¶nizleme
    msg = poll_sys.format_council_message(council)
    console.print(Panel(msg, title="Telegram Mesaj Ã–nizleme", border_style="blue"))

    # Anket seÃ§enekleri
    console.print("\n[bold]Anket SeÃ§enekleri:[/]")
    for i, opt in enumerate(poll_sys.get_poll_options(), 1):
        console.print(f"  {i}. {opt}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  YENÄ° MODÃœL CLI KOMUTLARI (v2)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@app.command()
def regime_kelly_cmd(
    bankroll: float = typer.Option(10000.0, help="BaÅŸlangÄ±Ã§ kasasÄ±"),
    prob: float = typer.Option(0.55, help="Kazanma olasÄ±lÄ±ÄŸÄ±"),
    odds: float = typer.Option(2.10, help="Oran"),
):
    """Regime-Aware Kelly Criterion ile stake hesapla."""
    from src.core.regime_kelly import RegimeKelly, RegimeState
    rk = RegimeKelly(bankroll=bankroll)
    decision = rk.calculate(probability=prob, odds=odds, match_id="cli_test")
    console.rule("[bold cyan]REGIME KELLY[/]")
    console.print(f"  Edge: {decision.edge:.2%}")
    console.print(f"  Raw Kelly: {decision.raw_kelly:.4f}")
    console.print(f"  Regime Mult: {decision.regime_multiplier:.2f}")
    console.print(f"  Final Kelly: {decision.final_kelly:.4f}")
    console.print(f"  Stake: {decision.stake_amount:.2f}")
    console.print(f"  Onay: {'âœ…' if decision.approved else 'âŒ ' + decision.rejection_reason}")
    if decision.adjustments:
        for adj in decision.adjustments:
            console.print(f"  â””â”€ {adj}")


@app.command()
def fisher_cmd(
    ref_mean: float = typer.Option(2.0, help="Referans daÄŸÄ±lÄ±m ortalamasÄ±"),
    ref_std: float = typer.Option(0.3, help="Referans std sapma"),
    cur_mean: float = typer.Option(2.5, help="Mevcut oran ortalamasÄ±"),
    cur_std: float = typer.Option(0.4, help="Mevcut std sapma"),
):
    """Fisher-Rao Information Geometry ile daÄŸÄ±lÄ±m mesafesi Ã¶lÃ§."""
    from src.quant.fisher_geometry import FisherGeometry
    import numpy as np
    fg = FisherGeometry()
    ref = np.random.normal(ref_mean, ref_std, 200)
    cur = np.random.normal(cur_mean, cur_std, 50)
    report = fg.compare_distributions(ref, cur, match_id="cli_test")
    console.rule("[bold cyan]FISHER GEOMETRY[/]")
    console.print(f"  Fisher-Rao Distance: {report.fisher_rao_distance:.4f}")
    console.print(f"  KL Divergence: {report.kl_divergence:.4f}")
    console.print(f"  Hellinger: {report.hellinger_distance:.4f}")
    console.print(f"  det(I): {report.fim_determinant:.4f}")
    console.print(f"  Anomali: {'âš ï¸ EVET' if report.is_anomaly else 'âœ… HayÄ±r'}")
    console.print(f"  Rejim DeÄŸiÅŸimi: {'ğŸ”´ EVET' if report.regime_shift else 'ğŸŸ¢ HayÄ±r'}")
    console.print(f"  Tavsiye: {report.recommendation}")


@app.command()
def philo_cmd(
    prob: float = typer.Option(0.72, help="Model olasÄ±lÄ±ÄŸÄ±"),
    conf: float = typer.Option(0.85, help="Model gÃ¼veni"),
    sample_size: int = typer.Option(200, help="Ã–rneklem bÃ¼yÃ¼klÃ¼ÄŸÃ¼"),
):
    """Philosophical Engine â€“ epistemik felsefi analiz."""
    from src.quant.philosophical_engine import PhilosophicalEngine
    phi = PhilosophicalEngine()
    report = phi.evaluate(
        probability=prob, confidence=conf,
        sample_size=sample_size, strategy_age_days=30,
        model_count=5, match_id="cli_test",
    )
    console.rule("[bold cyan]EPÄ°STEMÄ°K ANALÄ°Z[/]")
    console.print(f"  Epistemik Skor: {report.epistemic_score:.2f}")
    console.print(f"  Dunning-Kruger: {report.dunning_kruger_score:.2f}")
    console.print(f"  Black Swan Risk: {report.black_swan_risk:.2f}")
    console.print(f"  Antifragility: {report.antifragility:.2f}")
    console.print(f"  Lindy Score: {report.lindy_score:.2f}")
    console.print(f"  Falsifiability: {report.falsifiability:.2f}")
    console.print(f"  Meta-Uncertainty: {report.meta_uncertainty:.2f}")
    console.print(f"  Onay: {'âœ…' if report.epistemic_approved else 'âŒ'}")
    if report.rejection_reasons:
        for reason in report.rejection_reasons:
            console.print(f"  â””â”€ âŒ {reason}")
    for ref in report.reflections:
        console.print(f"  â””â”€ ğŸ’­ {ref}")


@app.command()
def evolver_cmd(
    pop: int = typer.Option(50, help="PopÃ¼lasyon bÃ¼yÃ¼klÃ¼ÄŸÃ¼"),
    generations: int = typer.Option(5, help="Nesil sayÄ±sÄ±"),
):
    """Strategy Evolver â€“ otonom strateji evrimi simÃ¼lasyonu."""
    import numpy as np
    from src.core.strategy_evolver import StrategyEvolver
    evolver = StrategyEvolver(population_size=pop)
    console.rule("[bold cyan]STRATEJÄ° EVRÄ°MÄ°[/]")
    for gen in range(1, generations + 1):
        mock_results = [
            {
                "won": np.random.random() > 0.45,
                "pnl": np.random.normal(5, 30),
                "ev": np.random.uniform(-0.05, 0.15),
                "odds": np.random.uniform(1.5, 4.0),
                "prob": np.random.uniform(0.3, 0.7),
            }
            for _ in range(100)
        ]
        report = evolver.evolve(mock_results)
        console.print(
            f"  Gen #{report.generation}: "
            f"best={report.best_fitness:.4f}, "
            f"avg={report.avg_fitness:.4f}, "
            f"worst={report.worst_fitness:.4f}"
        )
    best = evolver.get_best_dna()
    console.print("\n[bold]En Ä°yi DNA Parametreleri:[/]")
    for k, v in best.to_dict().items():
        console.print(f"  {k}: {v:.4f}")


@app.command()
def guardian_cmd():
    """Guardian saÄŸlÄ±k raporu â€“ hata istatistikleri."""
    from src.core.exception_guardian import ExceptionGuardian
    guardian = ExceptionGuardian()
    # Demo hatalar Ã¼ret
    import random
    modules = ["poisson", "lightgbm", "lstm", "ensemble", "kelly"]
    for mod in modules:
        for _ in range(random.randint(0, 5)):
            with guardian.protect(mod):
                if random.random() < 0.3:
                    raise ValueError(f"Demo hata in {mod}")
    report = guardian.health_report()
    console.rule("[bold cyan]GUARDIAN SAÄLIK RAPORU[/]")
    console.print(f"  Toplam Hata: {report['total_errors']}")
    console.print(f"  AÃ§Ä±k Devreler: {report['open_circuits'] or 'Yok'}")
    for mod, stats in report["modules"].items():
        console.print(
            f"  {mod}: {stats['total_errors']} hata, "
            f"circuit={'ğŸ”´' if stats['circuit_open'] else 'ğŸŸ¢'}"
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if __name__ == "__main__":
    app()


