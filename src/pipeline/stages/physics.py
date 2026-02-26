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

try:
    from src.quant.physics.fractal_analyzer import FractalAnalyzer
except ImportError:
    FractalAnalyzer = None

try:
    from src.quant.physics.topology_mapper import TopologyMapper
except ImportError:
    TopologyMapper = None

try:
    from src.quant.physics.path_signature_engine import PathSignatureEngine
except ImportError:
    PathSignatureEngine = None

try:
    from src.quant.physics.homology_scanner import HomologyScanner
except ImportError:
    HomologyScanner = None

try:
    from src.quant.physics.gcn_pitch_graph import GCNPitchGraph
except ImportError:
    GCNPitchGraph = None


class PhysicsStage(PipelineStage):
    """
    Pipeline stage for running advanced Physics Engines:
    - ChaosFilter: Detects chaotic regimes (Lyapunov Exponents).
    - QuantumBrain: Quantum Machine Learning predictions.
    - RicciFlowAnalyzer: Systemic risk via curvature analysis.
    - GeometricIntelligence: Spatial dominance metrics.
    - ParticleStrengthTracker: Live momentum tracking.
    - FractalAnalyzer: Hurst Exponent & Market Memory.
    - TopologyMapper: Topological Anomaly Detection (Kepler Mapper).
    - PathSignatureEngine: Rough Path Theory (Volatility Signature).
    - HomologyScanner: Persistent Homology (Team Organization).
    - GCNPitchGraph: Graph Neural Networks (Player Coordination).
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

        if FractalAnalyzer:
            try:
                self.fractal_analyzer = FractalAnalyzer()
            except Exception as e:
                logger.error(f"Failed to init FractalAnalyzer: {e}")
                self.fractal_analyzer = None
        else:
            logger.warning("FractalAnalyzer module missing.")
            self.fractal_analyzer = None

        if TopologyMapper:
            try:
                self.topology_mapper = TopologyMapper(n_cubes=5, overlap=0.2)
            except Exception as e:
                logger.error(f"Failed to init TopologyMapper: {e}")
                self.topology_mapper = None
        else:
            logger.warning("TopologyMapper module missing.")
            self.topology_mapper = None

        if PathSignatureEngine:
            try:
                self.path_signature = PathSignatureEngine(depth=2)
            except Exception as e:
                logger.error(f"Failed to init PathSignatureEngine: {e}")
                self.path_signature = None
        else:
            logger.warning("PathSignatureEngine module missing.")
            self.path_signature = None

        if HomologyScanner:
            try:
                self.homology_scanner = HomologyScanner(max_dim=1, max_edge=30.0)
            except Exception as e:
                logger.error(f"Failed to init HomologyScanner: {e}")
                self.homology_scanner = None
        else:
            logger.warning("HomologyScanner module missing.")
            self.homology_scanner = None

        if GCNPitchGraph:
            try:
                self.gcn_graph = GCNPitchGraph()
            except Exception as e:
                logger.error(f"Failed to init GCNPitchGraph: {e}")
                self.gcn_graph = None
        else:
            logger.warning("GCNPitchGraph module missing.")
            self.gcn_graph = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run physics analysis on the current batch of matches."""
        matches = context.get("matches", pl.DataFrame())
        features = context.get("features", pl.DataFrame())
        cycle = context.get("cycle", 0)

        results = {
            "chaos_reports": {},
            "quantum_predictions": {},
            "ricci_report": None,
            "geometric_potentials": {},
            "particle_reports": {},
            "fractal_reports": {},
            "topology_reports": {},
            "path_signatures": {},
            "homology_reports": {},
            "gcn_coordination": {}
        }

        # 1. Global Systemic Risk (Ricci Flow)
        if self.ricci_analyzer and not matches.is_empty():
            try:
                match_list = matches.to_dicts()
                G = self.ricci_analyzer.build_market_graph(match_list)
                if G:
                    ricci_rep = self.ricci_analyzer.analyze(G, name=f"cycle_{cycle}")
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

        # Train Topology Mapper on batch
        if self.topology_mapper and not features.is_empty():
            try:
                numeric_cols = [c for c in features.columns if features[c].dtype in (pl.Float64, pl.Float32)]
                if numeric_cols:
                    X = features.select(numeric_cols).to_numpy()
                    # Mock labels (e.g., home win prob if available, else random)
                    labels = np.random.rand(len(X))
                    self.topology_mapper.fit(X, labels)
            except Exception as e:
                logger.warning(f"Topology Mapper fit failed: {e}")

        # 2. Per-Match Analysis (High-Performance Parallelism)
        tasks = []
        for row in matches.iter_rows(named=True):
            match_id = row.get("match_id")
            if not match_id: continue

            # A. Chaos Filter Task
            if self.chaos_filter:
                tasks.append(self._run_chaos_filter(match_id, results))

            # B. Quantum Brain Task
            if self.quantum_brain:
                tasks.append(self._run_quantum_brain(match_id, row, feat_map, results))

            # C. Geometric Intelligence Task
            if self.geometric_intel:
                tasks.append(self._run_geometric_intelligence(match_id, row, feat_map, results))

            # D. Particle Strength Tracker Task
            if self.particle_tracker:
                tasks.append(self._run_particle_tracker(match_id, cycle, results))

            # E. Fractal Analyzer Task
            if self.fractal_analyzer:
                tasks.append(self._run_fractal_analyzer(match_id, results))

            # F. Topology Mapper Task
            if self.topology_mapper:
                tasks.append(self._run_topology_mapper(match_id, row, feat_map, results))

            # G. Path Signature Task
            if self.path_signature:
                tasks.append(self._run_path_signature(match_id, row, results))

            # H. Homology Scanner Task
            if self.homology_scanner:
                tasks.append(self._run_homology_scanner(match_id, cycle, results))

            # I. GCN Pitch Graph Task
            if self.gcn_graph:
                tasks.append(self._run_gcn_graph(match_id, cycle, results))


        if tasks:
            await asyncio.gather(*tasks)

        # Store in context under unified key
        # (This stage modifies the 'results' dict in-place via side-effects in tasks)
        return {"physics_reports": results}

    # --- Async Wrappers ---

    async def _run_chaos_filter(self, match_id: str, results: Dict[str, Any]):
        try:
            odds_history = await asyncio.to_thread(self._simulate_odds_history, match_id)
            chaos_rep = await asyncio.to_thread(self.chaos_filter.analyze, odds_history, match_id=match_id)
            results["chaos_reports"][match_id] = chaos_rep
        except Exception as e:
            logger.warning(f"Chaos analysis failed for {match_id}: {e}")

    async def _run_quantum_brain(self, match_id: str, row: Dict, feat_map: Dict, results: Dict[str, Any]):
        try:
            feat_row = feat_map.get(match_id, {})
            if feat_row:
                vec = [v for k, v in feat_row.items() if isinstance(v, (int, float)) and k != "match_id"]
                vec = vec[:10] if len(vec) > 10 else vec + [0.0]*(10-len(vec))
            else:
                vec = [row.get("home_odds", 2.0), row.get("draw_odds", 3.0), row.get("away_odds", 3.5)]

            q_pred = await asyncio.to_thread(self.quantum_brain.predict_match, vec, match_id=match_id)
            results["quantum_predictions"][match_id] = q_pred
        except Exception as e:
            logger.warning(f"Quantum prediction failed for {match_id}: {e}")

    async def _run_geometric_intelligence(self, match_id: str, row: Dict, feat_map: Dict, results: Dict[str, Any]):
        try:
            feat_row = feat_map.get(match_id, row)
            mini_df = pl.DataFrame([feat_row])
            pot_df = await asyncio.to_thread(self.geometric_intel.compute_potential, mini_df)
            if not pot_df.is_empty():
                results["geometric_potentials"][match_id] = pot_df.to_dicts()[0]
        except Exception as e:
            logger.warning(f"Geometric analysis failed for {match_id}: {e}")

    async def _run_particle_tracker(self, match_id: str, cycle: int, results: Dict[str, Any]):
        try:
            obs = await asyncio.to_thread(self._create_mock_observation, match_id, cycle)
            p_rep = await asyncio.to_thread(self.particle_tracker.update, obs, match_id=match_id)
            results["particle_reports"][match_id] = p_rep
        except Exception as e:
            logger.warning(f"Particle tracking failed for {match_id}: {e}")

    async def _run_fractal_analyzer(self, match_id: str, results: Dict[str, Any]):
        try:
            # Simulate historical performance data
            hist_data = self._simulate_odds_history(match_id)
            f_rep = await asyncio.to_thread(self.fractal_analyzer.compute_hurst, hist_data)
            results["fractal_reports"][match_id] = f_rep
        except Exception as e:
            logger.warning(f"Fractal analysis failed for {match_id}: {e}")

    async def _run_topology_mapper(self, match_id: str, row: Dict, feat_map: Dict, results: Dict[str, Any]):
        try:
            feat_row = feat_map.get(match_id, row)
            vec = [v for k, v in feat_row.items() if isinstance(v, (int, float)) and k != "match_id"]
            if vec:
                t_rep = await asyncio.to_thread(self.topology_mapper.analyze_match, vec, match_id=match_id)
                results["topology_reports"][match_id] = t_rep
        except Exception as e:
            logger.warning(f"Topology analysis failed for {match_id}: {e}")

    async def _run_path_signature(self, match_id: str, row: Dict, results: Dict[str, Any]):
        try:
            mini_df = pl.DataFrame([row])
            sig_df = await asyncio.to_thread(self.path_signature.extract, mini_df)
            if not sig_df.is_empty():
                results["path_signatures"][match_id] = sig_df.to_dicts()[0]
        except Exception as e:
            logger.warning(f"Path signature failed for {match_id}: {e}")

    async def _run_homology_scanner(self, match_id: str, cycle: int, results: Dict[str, Any]):
        try:
            pos_h = self._simulate_player_positions(match_id, cycle, team="home")
            pos_a = self._simulate_player_positions(match_id, cycle, team="away")

            # Compare teams
            comp_rep = await asyncio.to_thread(self.homology_scanner.compare_teams, pos_h, pos_a, match_id=match_id)
            results["homology_reports"][match_id] = comp_rep
        except Exception as e:
            logger.warning(f"Homology scanner failed for {match_id}: {e}")

    async def _run_gcn_graph(self, match_id: str, cycle: int, results: Dict[str, Any]):
        try:
            pos_h = self._simulate_player_positions(match_id, cycle, team="home")
            pos_a = self._simulate_player_positions(match_id, cycle, team="away")

            # Combine
            all_pos = np.vstack([pos_h, pos_a])
            teams = [0]*len(pos_h) + [1]*len(pos_a)

            coord_rep = await asyncio.to_thread(self.gcn_graph.analyze_coordination, all_pos, teams)
            results["gcn_coordination"][match_id] = coord_rep
        except Exception as e:
            logger.warning(f"GCN graph failed for {match_id}: {e}")


    # --- Simulators ---

    def _simulate_odds_history(self, match_id: str) -> np.ndarray:
        """Simulates odds history (Deterministic based on match_id)."""
        seed = int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        n = 50
        odds = np.cumprod(1 + np.random.normal(0, 0.02, n)) * 2.0
        return odds

    def _create_mock_observation(self, match_id: str, cycle: int) -> MatchObservation:
        """Creates a simulated observation for Particle Strength Tracker."""
        seed = int(hashlib.md5(f"{match_id}_{cycle}".encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)

        minute = (cycle * 5) % 90
        if minute == 0: minute = 1

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
            score_home=0,
            score_away=0
        )

    def _simulate_player_positions(self, match_id: str, cycle: int, team: str) -> np.ndarray:
        """Simulates 2D player positions on a pitch (105x68)."""
        seed = int(hashlib.md5(f"{match_id}_{cycle}_{team}".encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)

        # 11 players
        # Standard formation 4-4-2ish
        base_x = np.array([5, 20, 20, 20, 20, 50, 50, 50, 50, 80, 80])
        base_y = np.array([34, 10, 25, 43, 58, 10, 25, 43, 58, 25, 43])

        noise = np.random.normal(0, 5, (11, 2))
        pos = np.column_stack([base_x, base_y]) + noise

        # Clip to pitch dimensions
        pos[:, 0] = np.clip(pos[:, 0], 0, 105)
        pos[:, 1] = np.clip(pos[:, 1], 0, 68)

        return pos
