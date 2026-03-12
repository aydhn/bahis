import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict
from loguru import logger
from src.system.container import container
from src.system.lifecycle import lifecycle
from src.core.event_bus import EventBus, Event
from src.core.auto_healer import SelfHealingEngine


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the stage logic."""
        pass

class PipelineEngine:
    """Orchestrates the execution of pipeline stages."""

    def __init__(self, bus: EventBus = None):
        self.stages: list[PipelineStage] = []
        self.context: Dict[str, Any] = {}
        self.running = False
        self.bus = bus
        self.healer = SelfHealingEngine(llm_backend="auto")

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)

    async def run(self):
        """Run the pipeline loop until shutdown signal."""
        self.running = True
        cycle = 0

        # Heartbeat setup
        from pathlib import Path
        import time
        heartbeat_file = Path("data/heartbeat.txt")
        if not heartbeat_file.parent.exists():
            heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        while not lifecycle.shutdown_event.is_set():
            # Update Heartbeat
            try:
                heartbeat_file.write_text(str(time.time()))
            except Exception as e:
                logger.error(f"Heartbeat update failed: {e}")

            cycle += 1
            logger.info(f"═══ Pipeline Cycle #{cycle} ═══")


            # Reset context for each cycle, but keep some persistent state if needed
            # For now, we clear it or just pass the previous one?
            # Usually pipelines share state. Let's keep persistent objects in self.context
            # and ephemeral data in a local dict.

            if self.bus:
                await self.bus.emit(Event(event_type="pipeline_cycle_start", cycle=cycle))

            cycle_context = self.context.copy()
            cycle_context["cycle"] = cycle
            cycle_context["bus"] = self.bus  # Inject EventBus into context

            # Initialize BettingContext (Enterprise Grade State)
            try:
                from src.pipeline.context import BettingContext
                # Create fresh context for this cycle
                ctx = BettingContext(cycle_id=cycle)
                # Inject persistent data if any (e.g. from self.context)
                cycle_context["ctx"] = ctx
            except ImportError as e:
                logger.warning(f"Could not initialize BettingContext: {e}")

            try:
                for stage in self.stages:
                    if lifecycle.shutdown_event.is_set():
                        break

                    logger.debug(f"Starting stage: {stage.name}")
                    try:
                        # Execute stage
                        result = await stage.execute(cycle_context)
                        # Update context with results
                        if result:
                            cycle_context.update(result)
                    except Exception as e:
                        logger.error(f"Error in stage {stage.name}: {e}")
                        # Autonomous Healing Attempt
                        try:
                            # We need the module path for the stage.
                            # Best guess: src/pipeline/stages/{stage.name}.py or verify via inspect
                            import inspect
                            stage_file = inspect.getfile(stage.__class__)

                            healing_attempt = await self.healer.attempt_heal(e, module_path=stage_file)

                            if healing_attempt.patch_applied and healing_attempt.test_passed:
                                logger.success(f"AutoHealer fixed {stage.name}! Retrying next cycle.")
                                if self.bus:
                                    await self.bus.emit(Event("system_healed", data={"stage": stage.name, "error": str(e)}))
                            else:
                                logger.warning(f"AutoHealer failed for {stage.name}.")

                        except Exception as heal_err:
                            logger.error(f"AutoHealer crashed: {heal_err}")

                if self.bus:
                    await self.bus.emit(Event(event_type="pipeline_cycle_end", cycle=cycle))

            except Exception as e:
                logger.critical(f"Pipeline crashed: {e}")
                if self.bus:
                    await self.bus.emit(Event(event_type="pipeline_crash", data={"error": str(e)}))
                await asyncio.sleep(5) # Backoff

            # Wait for next cycle (e.g. cron schedule or simple sleep)
            # In bahis.py it was just a loop with sleep if no matches.
            # We'll let the stages decide if they need to sleep or if the engine should.
            await asyncio.sleep(1)

    async def run_once(self, initial_context: Dict[str, Any] = None):
        """Run a single pass of the pipeline (for CLI/Testing)."""
        cycle_context = self.context.copy()
        if initial_context:
            cycle_context.update(initial_context)

        cycle_context["bus"] = self.bus

        # Initialize BettingContext
        try:
            from src.pipeline.context import BettingContext
            ctx = BettingContext(cycle_id=1)
            cycle_context["ctx"] = ctx
        except ImportError:
            pass

        for stage in self.stages:
            logger.info(f"Running stage: {stage.name}")
            res = await stage.execute(cycle_context)
            if res:
                cycle_context.update(res)
        return cycle_context

def create_default_pipeline(bot_instance: Any = None, bus: EventBus = None) -> PipelineEngine:
    """Create a pipeline with standard stages."""
    # Lazy imports to avoid circular dependencies
    from src.pipeline.stages.ingestion import IngestionStage
    from src.pipeline.stages.features import FeatureStage
    from src.pipeline.stages.validator import DataValidatorStage
    from src.pipeline.stages.physics import PhysicsStage
    from src.pipeline.stages.inference import InferenceStage
    from src.pipeline.stages.ensemble import EnsembleStage
    from src.pipeline.stages.risk import RiskStage
    from src.pipeline.stages.execution import ExecutionStage
    from src.pipeline.stages.reporting import ReportingStage

    # New Bridge Stage
    from src.pipeline.stages.ingestion_bridge import IngestionBridgeStage

    engine = PipelineEngine(bus=bus)
    engine.add_stage(IngestionStage())
    engine.add_stage(DataValidatorStage())
    engine.add_stage(FeatureStage())

    # Add Zero-Copy Bridge after features are generated but before inference
    engine.add_stage(IngestionBridgeStage())

    engine.add_stage(PhysicsStage())
    engine.add_stage(InferenceStage())
    engine.add_stage(EnsembleStage())
    engine.add_stage(RiskStage())
    engine.add_stage(ExecutionStage())
    engine.add_stage(ReportingStage(bot_instance=bot_instance))

    return engine


def create_shadow_pipeline(bus: EventBus = None) -> PipelineEngine:
    """Create a lightweight shadow pipeline for simulation/dream mode."""
    from src.pipeline.stages.features import FeatureStage
    from src.pipeline.stages.physics import PhysicsStage
    from src.pipeline.stages.inference import InferenceStage
    from src.pipeline.stages.ensemble import EnsembleStage
    from src.pipeline.stages.risk import RiskStage

    engine = PipelineEngine(bus=bus)
    engine.add_stage(FeatureStage())
    engine.add_stage(PhysicsStage())
    engine.add_stage(InferenceStage())
    engine.add_stage(EnsembleStage())
    engine.add_stage(RiskStage())

    return engine

class LightweightInferenceStage(PipelineStage):
    """Lightweight inference simulation for Digital Twin."""
    def __init__(self):
        super().__init__("lightweight_inference")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        match_data = context.get("match_data", {})
        home_odds = match_data.get("home_odds", 2.0)

        # Simulate a model prediction based on odds (favorites usually win)
        implied_prob = 1.0 / home_odds
        model_prob = implied_prob * 1.05  # Mild edge

        decision = "HOME" if model_prob > 0.5 else "SKIP"

        return {"decision": decision}

class LightweightRiskStage(PipelineStage):
    """Lightweight risk simulation for Digital Twin."""
    def __init__(self):
        super().__init__("lightweight_risk")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        match_data = context.get("match_data", {})
        decision = context.get("decision", "SKIP")

        home_score = match_data.get("home_score", 1)  # Default mock
        away_score = match_data.get("away_score", 0)

        winner = "HOME"
        if away_score > home_score:
            winner = "AWAY"
        elif away_score == home_score:
            winner = "DRAW"

        stake = 100.0  # Fixed unit for sim
        pnl = 0.0
        won = False

        if decision == "HOME":
            if winner == "HOME":
                pnl = stake * (match_data.get("home_odds", 2.0) - 1)
                won = True
            else:
                pnl = -stake

        return {"result": winner, "pnl": pnl, "won": won}

def create_lightweight_pipeline(bus: EventBus = None) -> PipelineEngine:
    """Create a specialized lightweight pipeline for speed."""
    engine = PipelineEngine(bus=bus)
    engine.add_stage(LightweightInferenceStage())
    engine.add_stage(LightweightRiskStage())
    return engine
