
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFIER")

try:
    logger.info("Importing ActiveInferenceAgent...")
    from src.core.active_inference_agent import ActiveInferenceAgent
    agent = ActiveInferenceAgent()
    logger.info("ActiveInferenceAgent instantiated.")
except Exception as e:
    logger.error(f"Failed to load ActiveInferenceAgent: {e}")
    sys.exit(1)

try:
    logger.info("Importing GameTheoryEngine...")
    from src.quant.analysis.game_theory_engine import GameTheoryEngine
    gt = GameTheoryEngine()
    logger.info("GameTheoryEngine instantiated.")
except Exception as e:
    logger.error(f"Failed to load GameTheoryEngine: {e}")
    sys.exit(1)

try:
    logger.info("Importing RustEngine...")
    from src.core.rust_engine import RustEngine
    rust = RustEngine()
    logger.info(f"RustEngine instantiated (Mode: {rust.engine_name}).")
except Exception as e:
    logger.error(f"Failed to load RustEngine: {e}")
    sys.exit(1)

logger.info("All new modules verified successfully.")
