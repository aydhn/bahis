import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict
from loguru import logger
from src.system.lifecycle import lifecycle

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

    def __init__(self):
        self.stages: list[PipelineStage] = []
        self.context: Dict[str, Any] = {}
        self.running = False

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)

    async def run(self):
        """Run the pipeline loop until shutdown signal."""
        self.running = True
        cycle = 0

        while not lifecycle.shutdown_event.is_set():
            cycle += 1
            logger.info(f"═══ Pipeline Cycle #{cycle} ═══")

            # Reset context for each cycle, but keep some persistent state if needed
            # For now, we clear it or just pass the previous one?
            # Usually pipelines share state. Let's keep persistent objects in self.context
            # and ephemeral data in a local dict.

            cycle_context = self.context.copy()
            cycle_context["cycle"] = cycle

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
                        # Depending on severity, we might continue or break
                        # For now, log and continue to next stage (or next cycle?)
                        # Ideally, use a circuit breaker here.

            except Exception as e:
                logger.critical(f"Pipeline crashed: {e}")
                await asyncio.sleep(5) # Backoff

            # Wait for next cycle (e.g. cron schedule or simple sleep)
            # In bahis.py it was just a loop with sleep if no matches.
            # We'll let the stages decide if they need to sleep or if the engine should.
            await asyncio.sleep(1)

    async def run_once(self):
        """Run a single pass of the pipeline (for CLI/Testing)."""
        cycle_context = self.context.copy()

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

def create_default_pipeline() -> PipelineEngine:
    """Create a pipeline with standard stages."""
    # Lazy imports to avoid circular dependencies
    from src.pipeline.stages.ingestion import IngestionStage
    from src.pipeline.stages.features import FeatureStage
    from src.pipeline.stages.inference import InferenceStage
    from src.pipeline.stages.ensemble import EnsembleStage
    from src.pipeline.stages.risk import RiskStage
    from src.pipeline.stages.execution import ExecutionStage
    from src.pipeline.stages.reporting import ReportingStage

    engine = PipelineEngine()
    engine.add_stage(IngestionStage())
    engine.add_stage(FeatureStage())
    engine.add_stage(InferenceStage())
    engine.add_stage(EnsembleStage())
    engine.add_stage(RiskStage())
    engine.add_stage(ExecutionStage())
    engine.add_stage(ReportingStage())

    return engine
