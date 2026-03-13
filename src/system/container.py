from typing import Optional, Any
from loguru import logger
from src.system.config import settings

# Placeholder imports for core services to avoid circular deps during init
# In a real DI framework, we'd bind these dynamically.
# For now, we use a simple singleton pattern.

class DependencyContainer:
    """Simple Dependency Injection Container."""

    _instance = None

    def __init__(self):
        self._services: dict[str, Any] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DependencyContainer()
        return cls._instance

    def get(self, service_name: str) -> Any:
        """Get a service by name, initializing it if necessary."""
        if service_name not in self._services:
            self._initialize_service(service_name)
        return self._services.get(service_name)

    def register(self, service_name: str, instance: Any):
        """Register a service instance manually."""
        self._services[service_name] = instance

    def _initialize_service(self, name: str):
        """Lazy initialization of services."""
        logger.debug(f"Initializing service: {name}")

        try:
            if name == "db":
                from src.memory.db_manager import DBManager
                self._services["db"] = DBManager()

            elif name == "cache":
                from src.memory.feature_cache import FeatureCache
                self._services["cache"] = FeatureCache()

            elif name == "smart_cache":
                from src.memory.smart_cache import SmartCache
                self._services["smart_cache"] = SmartCache(ttl_l2=3600.0)

            elif name == "graph_rag":
                from src.memory.graph_rag import GraphRAG
                self._services["graph_rag"] = GraphRAG(
                    neo4j_uri=settings.NEO4J_URI,
                    neo4j_user=settings.NEO4J_USER,
                    neo4j_password=settings.NEO4J_PASSWORD,
                    llm_backend=settings.LLM_BACKEND
                )

            elif name == "notifier":
                # Mock or Real Telegram Notifier
                try:
                    from src.ui.telegram_mini_app import TelegramNotifier
                    self._services["notifier"] = TelegramNotifier()
                except ImportError:
                    logger.warning("TelegramNotifier not found, using Mock.")
                    class MockNotifier:
                        async def send(self, msg): logger.info(f"[MockTelegram] {msg}")
                        async def send_anomaly_alert(self, alert): logger.info(f"[MockAnomaly] {alert}")
                    self._services["notifier"] = MockNotifier()

            elif name == "event_bus":
                 from src.core.event_bus import EventBus, EventStore
                 self._services["event_bus"] = EventBus(store=EventStore())

            # --- Quant Engines ---
            elif name == "prob_engine":
                from src.quant.analysis.probabilistic_engine import ProbabilisticEngine
                self._services["prob_engine"] = ProbabilisticEngine()

            elif name == "regime_kelly":
                from src.core.regime_kelly import RegimeKelly
                self._services["regime_kelly"] = RegimeKelly()

            elif name == "portfolio_opt":
                 from src.core.portfolio_optimizer import PortfolioOptimizer
                 self._services["portfolio_opt"] = PortfolioOptimizer()

            elif name == "arb_executor":
                from src.quant.finance.arbitrage_execution import ArbitrageExecutionManager
                self._services["arb_executor"] = ArbitrageExecutionManager()

            elif name == "smart_money":
                from src.extensions.smart_money import SmartMoneyDetector
                self._services["smart_money"] = SmartMoneyDetector()

            elif name == "active_agent":
                from src.core.active_inference_agent import ActiveInferenceAgent
                self._services["active_agent"] = ActiveInferenceAgent()

            elif name == "treasury":
                from src.quant.finance.treasury import TreasuryEngine
                self._services["treasury"] = TreasuryEngine()

            elif name == "market_god":
                try:
                    from src.extensions.market_god import MarketGod
                    self._services["market_god"] = MarketGod()
                except ImportError:
                    self._services["market_god"] = None  # type: ignore

            elif name == "behavioral_arb":
                try:
                    from src.extensions.behavioral_arbitrage import BehavioralArbitrage
                    self._services["behavioral_arb"] = BehavioralArbitrage()
                except ImportError:
                    self._services["behavioral_arb"] = None  # type: ignore

            elif name == "alpha_generator":
                from src.extensions.alpha_generator import AlphaGenerator
                self._services["alpha_generator"] = AlphaGenerator(self.get("event_bus"))

            elif name == "auto_tuner":
                from src.extensions.auto_tuner import AutoTuner
                self._services["auto_tuner"] = AutoTuner()

            elif name == "sentiment_alpha":
                from src.extensions.sentiment_alpha import SentimentAlphaEngine
                self._services["sentiment_alpha"] = SentimentAlphaEngine()

            elif name == "ceo_dashboard":
                from src.extensions.ceo_dashboard import CEODashboard
                self._services["ceo_dashboard"] = CEODashboard()

            elif name == "kelly_benter":
                from src.extensions.kelly_benter_optimizer import KellyBenterOptimizer
                self._services["kelly_benter"] = KellyBenterOptimizer()

            elif name == "philosophical_risk":
                from src.extensions.philosophical_risk import PhilosophicalRiskEngine
                self._services["philosophical_risk"] = PhilosophicalRiskEngine()

            elif name == "quantum_pricing":
                from src.extensions.quantum_pricing_model import QuantumPricingModel
                self._services["quantum_pricing"] = QuantumPricingModel()

            elif name == "dynamic_hedging":
                from src.extensions.dynamic_hedging import DynamicHedgingEngine
                # Dependencies injected manually in Sentinel
                self._services["dynamic_hedging"] = DynamicHedgingEngine(None, None)

            elif name == "opportunity_scanner":
                from src.extensions.opportunity_scanner import OpportunityScanner
                self._services["opportunity_scanner"] = OpportunityScanner(self.get("event_bus"))

            elif name == "bayesian_updater":
                from src.extensions.bayesian_updater import BayesianOddsUpdater
                self._services["bayesian_updater"] = BayesianOddsUpdater()

            elif name == "macro_correlation":
                from src.extensions.macro_correlation import MacroCorrelationEngine
                self._services["macro_correlation"] = MacroCorrelationEngine()

            elif name == "rl_agent":
                try:
                    from src.quant.models.rl_trader import RLBettingAgent
                    agent = RLBettingAgent()
                    agent.load() # Try to load pre-trained if exists
                    self._services["rl_agent"] = agent
                except ImportError:
                    logger.warning("RLBettingAgent missing dependencies. RL overlay disabled.")
                    self._services["rl_agent"] = None

            elif name == "regime_hmm":
                from src.extensions.regime_hmm import MarketRegimeHMM
                self._services["regime_hmm"] = MarketRegimeHMM()

            elif name == "boardroom":
                from src.core.boardroom import Boardroom
                self._services["boardroom"] = Boardroom()

            # Add more services as needed...


            else:
                raise ValueError(f"Unknown service: {name}")

        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            raise

container = DependencyContainer.get_instance()
