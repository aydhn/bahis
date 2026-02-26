import asyncio
from typing import Dict, Any, List
from loguru import logger
import polars as pl
import numpy as np
import hashlib

from src.pipeline.core import PipelineStage

# Defensive Imports for Physics Engines
try:
    from src.quant.physics.chaos_filter import ChaosFilter
except ImportError:
    ChaosFilter = None

try:
    from src.quant.physics.quantum_brain import QuantumBrain
except ImportError:
    QuantumBrain = None

try:
    from src.quant.physics.ricci_flow import RicciFlowAnalyzer
except ImportError:
    RicciFlowAnalyzer = None

try:
    from src.quant.physics.geometric_intelligence import GeometricIntelligence
except ImportError:
    GeometricIntelligence = None

try:
    from src.quant.physics.particle_strength_tracker import ParticleStrengthTracker, MatchObservation
except ImportError:
    ParticleStrengthTracker = None
    MatchObservation = None

class PhysicsStage(PipelineStage):
    """
    Pipeline stage for running advanced Physics Engines:
    - ChaosFilter: Detects chaotic regimes (Lyapunov Exponents).
    - QuantumBrain: Quantum Machine Learning predictions.
    - RicciFlowAnalyzer: Systemic risk via curvature analysis.
    - GeometricIntelligence: Spatial dominance metrics.
    - ParticleStrengthTracker: Live momentum tracking.
    """

    def __init__(self):
        super().__init__("physics")

        # Initialize Physics Engines with verified arguments
        if ChaosFilter:
            try:
                self.chaos_filter = ChaosFilter(emb_dim=3, lag=1)
            except Exception as e:
                logger.error(f"Failed to init ChaosFilter: {e}")
                self.chaos_filter = None
        else:
            logger.warning("ChaosFilter module missing.")
            self.chaos_filter = None

        if QuantumBrain:
            try:
                self.quantum_brain = QuantumBrain(n_qubits=4, n_layers=2)
            except Exception as e:
                logger.error(f"Failed to init QuantumBrain: {e}")
                self.quantum_brain = None
        else:
            logger.warning("QuantumBrain module missing.")
            self.quantum_brain = None

        if RicciFlowAnalyzer:
            try:
                self.ricci_analyzer = RicciFlowAnalyzer(alpha=0.5)
            except Exception as e:
                logger.error(f"Failed to init RicciFlowAnalyzer: {e}")
                self.ricci_analyzer = None
        else:
            logger.warning("RicciFlowAnalyzer module missing.")
            self.ricci_analyzer = None

        if GeometricIntelligence:
            try:
                self.geometric_intel = GeometricIntelligence()
            except Exception as e:
                logger.error(f"Failed to init GeometricIntelligence: {e}")
                self.geometric_intel = None
        else:
            logger.warning("GeometricIntelligence module missing.")
            self.geometric_intel = None

        if ParticleStrengthTracker:
            try:
                self.particle_tracker = ParticleStrengthTracker(n_particles=1000)
            except Exception as e:
                logger.error(f"Failed to init ParticleStrengthTracker: {e}")
                self.particle_tracker = None
        else:
            logger.warning("ParticleStrengthTracker module missing.")
            self.particle_tracker = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run physics analysis on the current batch of matches."""
        matches = context.get("matches", pl.DataFrame())
        features = context.get("features", pl.DataFrame())

        results = {
            "chaos_reports": {},
            "quantum_predictions": {},
            "ricci_report": None,
            "geometric_potentials": {},
            "particle_reports": {}
        }

        # 1. Global Systemic Risk (Ricci Flow)
        if self.ricci_analyzer and not matches.is_empty():
            try:
                match_list = matches.to_dicts()
                G = self.ricci_analyzer.build_market_graph(match_list)
                if G:
                    ricci_rep = self.ricci_analyzer.analyze(G, name=f"cycle_{context.get('cycle', 0)}")
                    results["ricci_report"] = ricci_rep
                    if ricci_rep.stress_level in ["high", "critical"]:
                        logger.warning(f"Ricci Flow Alert: {ricci_rep.stress_level} (Risk: {ricci_rep.systemic_risk:.2f})")
            except Exception as e:
                logger.error(f"Ricci Flow analysis failed: {e}")

        if matches.is_empty():
            return results

        # Pre-compute feature map for quick lookup
        feat_map = {}
        if not features.is_empty():
             feat_map = {row["match_id"]: row for row in features.iter_rows(named=True)}

        # 2. Per-Match Analysis
        for row in matches.iter_rows(named=True):
            match_id = row.get("match_id")
            if not match_id: continue

            # A. Chaos Filter (Needs Odds History)
            if self.chaos_filter:
                try:
                    # In a real scenario, fetch history from DB. Here we simulate for consistency.
                    odds_history = self._simulate_odds_history(match_id)
                    chaos_rep = self.chaos_filter.analyze(odds_history, match_id=match_id)
                    results["chaos_reports"][match_id] = chaos_rep
                except Exception as e:
                    logger.warning(f"Chaos analysis failed for {match_id}: {e}")

            # B. Quantum Brain (Needs Features)
            if self.quantum_brain:
                try:
                    # Construct feature vector
                    feat_row = feat_map.get(match_id, {})
                    # Select relevant numeric features. Fallback to basic odds if features missing.
                    if feat_row:
                        # Extract numerical values only
                        vec = [v for k, v in feat_row.items() if isinstance(v, (int, float)) and k != "match_id"]
                        # Pad or truncate to match expected input size if needed (simple heuristic)
                        vec = vec[:10] if len(vec) > 10 else vec + [0.0]*(10-len(vec))
                    else:
                        vec = [row.get("home_odds", 2.0), row.get("draw_odds", 3.0), row.get("away_odds", 3.5)]

                    q_pred = self.quantum_brain.predict_match(vec, match_id=match_id)
                    results["quantum_predictions"][match_id] = q_pred
                except Exception as e:
                    logger.warning(f"Quantum prediction failed for {match_id}: {e}")

            # C. Geometric Intelligence (Needs Features)
            if self.geometric_intel:
                try:
                    feat_row = feat_map.get(match_id, row) # Fallback to match row if features missing
                    # GeometricIntelligence expects a specific structure or just a dict/row
                    # We wrap it in a DataFrame as per the tool definition usually, or pass dict if method supports
                    # Looking at file trace: compute_potential takes DataFrame.
                    # We can construct a mini DF for this row.
                    mini_df = pl.DataFrame([feat_row])
                    pot_df = self.geometric_intel.compute_potential(mini_df)
                    if not pot_df.is_empty():
                        results["geometric_potentials"][match_id] = pot_df.to_dicts()[0]
                except Exception as e:
                    logger.warning(f"Geometric analysis failed for {match_id}: {e}")

            # D. Particle Strength Tracker (Live/Mock)
            if self.particle_tracker:
                try:
                    # Check if match is 'live' (mock logic: if it has a specific flag or just run for all in demo)
                    # For demo purposes, we run for all or a subset
                    obs = self._create_mock_observation(match_id, context.get("cycle", 0))
                    p_rep = self.particle_tracker.update(obs, match_id=match_id)
                    results["particle_reports"][match_id] = p_rep
                except Exception as e:
                    logger.warning(f"Particle tracking failed for {match_id}: {e}")

        return results

    def _simulate_odds_history(self, match_id: str) -> np.ndarray:
        """
        Simulates odds history for Chaos Analysis (Deterministic based on match_id).
        """
        seed = int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        n = 50
        # Random walk around 2.0
        odds = np.cumprod(1 + np.random.normal(0, 0.02, n)) * 2.0
        return odds

    def _create_mock_observation(self, match_id: str, cycle: int) -> MatchObservation:
        """
        Creates a simulated observation for Particle Strength Tracker.
        Evolution based on cycle count to simulate match progression.
        """
        seed = int(hashlib.md5(f"{match_id}_{cycle}".encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)

        minute = (cycle * 5) % 90  # Simulate 5 mins per cycle
        if minute == 0: minute = 1

        # Simulate dynamic stats
        return MatchObservation(
            minute=minute,
            home_shots=np.random.poisson(minute/10) + 1,
            away_shots=np.random.poisson(minute/15),
            home_possession=50 + np.random.normal(0, 5),
            away_possession=50 - np.random.normal(0, 5),
            home_dangerous_attacks=np.random.poisson(minute/5),
            away_dangerous_attacks=np.random.poisson(minute/7),
            home_corners=np.random.poisson(minute/20),
            away_corners=np.random.poisson(minute/25),
            score_home=0, # Simplified
            score_away=0
        )
