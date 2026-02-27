import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger
import polars as pl
import numpy as np
import hashlib

from src.pipeline.core import PipelineStage

# Defensive Imports for Physics Engines
# We attempt to import each engine. If it fails (e.g. missing dependencies), we set it to None.

try:
    from src.quant.physics.chaos_filter import ChaosFilter
except ImportError:
    ChaosFilter = None
    logger.warning("ChaosFilter could not be imported.")

try:
    from src.quant.physics.quantum_brain import QuantumBrain
except ImportError:
    QuantumBrain = None
    logger.warning("QuantumBrain could not be imported.")

try:
    from src.quant.physics.ricci_flow import RicciFlowAnalyzer
except ImportError:
    RicciFlowAnalyzer = None
    logger.warning("RicciFlowAnalyzer could not be imported.")

try:
    from src.quant.physics.geometric_intelligence import GeometricIntelligence
except ImportError:
    GeometricIntelligence = None
    logger.warning("GeometricIntelligence could not be imported.")

try:
    from src.quant.physics.particle_strength_tracker import ParticleStrengthTracker, MatchObservation
except ImportError:
    ParticleStrengthTracker = None
    MatchObservation = None
    logger.warning("ParticleStrengthTracker could not be imported.")

try:
    from src.quant.physics.fractal_analyzer import FractalAnalyzer
except ImportError:
    FractalAnalyzer = None
    logger.warning("FractalAnalyzer could not be imported.")

try:
    from src.quant.physics.topology_mapper import TopologyMapper
except ImportError:
    TopologyMapper = None
    logger.warning("TopologyMapper could not be imported.")

try:
    from src.quant.physics.path_signature_engine import PathSignatureEngine
except ImportError:
    PathSignatureEngine = None
    logger.warning("PathSignatureEngine could not be imported.")

try:
    from src.quant.physics.homology_scanner import HomologyScanner
except ImportError:
    HomologyScanner = None
    logger.warning("HomologyScanner could not be imported.")

try:
    from src.quant.physics.gcn_pitch_graph import GCNPitchGraph
except ImportError:
    GCNPitchGraph = None
    logger.warning("GCNPitchGraph could not be imported.")

try:
    from src.quant.physics.renormalization import RenormalizationGroup
except ImportError:
    RenormalizationGroup = None
    logger.warning("RenormalizationGroup could not be imported.")

try:
    from src.quant.analysis.hypergraph_unit import HypergraphUnitAnalyzer, TacticalUnit
except ImportError:
    HypergraphUnitAnalyzer = None
    TacticalUnit = None
    logger.warning("HypergraphUnitAnalyzer could not be imported.")

try:
    from src.quant.physics.fisher_geometry import FisherGeometry
except ImportError:
    FisherGeometry = None
    logger.warning("FisherGeometry could not be imported.")

try:
    from src.quant.physics.multifractal_logic import MultifractalAnalyzer
except ImportError:
    MultifractalAnalyzer = None
    logger.warning("MultifractalAnalyzer could not be imported.")


class PhysicsStage(PipelineStage):
    """
    The 'Quantum Cortex' of the pipeline.

    This stage unifies multiple advanced physics-based engines to analyze market and match dynamics
    from a complexity science perspective. It moves beyond simple statistics into:
    - Chaos Theory (Lyapunov Exponents)
    - Quantum Probability
    - Differential Geometry (Ricci Flow, Fisher Information)
    - Topological Data Analysis (Persistent Homology, Kepler Mapper)
    - Statistical Mechanics (Renormalization Group, Particle Tracking)
    - Graph Theory (GCN, Hypergraphs)

    Outputs:
        context['physics_reports']: Detailed objects/reports from each engine.
        context['physics_context']: Simplified metrics (float/str) for downstream ML models.
    """

    def __init__(self):
        super().__init__("physics")

        # --- Initialize Engines ---

        # 1. Chaos & Fractals
        if ChaosFilter:
            try:
                self.chaos_filter = ChaosFilter(emb_dim=3, lag=1)
            except Exception as e:
                logger.error(f"Failed to init ChaosFilter: {e}")
                self.chaos_filter = None
        else:
            self.chaos_filter = None

        if FractalAnalyzer:
            try:
                self.fractal_analyzer = FractalAnalyzer()
            except Exception as e:
                logger.error(f"Failed to init FractalAnalyzer: {e}")
                self.fractal_analyzer = None
        else:
            self.fractal_analyzer = None

        if MultifractalAnalyzer:
            try:
                self.multifractal = MultifractalAnalyzer()
            except Exception as e:
                logger.error(f"Failed to init MultifractalAnalyzer: {e}")
                self.multifractal = None
        else:
            self.multifractal = None

        # 2. Quantum & Geometry
        if QuantumBrain:
            try:
                self.quantum_brain = QuantumBrain(n_qubits=4, n_layers=2)
            except Exception as e:
                logger.error(f"Failed to init QuantumBrain: {e}")
                self.quantum_brain = None
        else:
            self.quantum_brain = None

        if RicciFlowAnalyzer:
            try:
                self.ricci_analyzer = RicciFlowAnalyzer(alpha=0.5)
            except Exception as e:
                logger.error(f"Failed to init RicciFlowAnalyzer: {e}")
                self.ricci_analyzer = None
        else:
            self.ricci_analyzer = None

        if FisherGeometry:
            try:
                self.fisher_geo = FisherGeometry(anomaly_threshold=2.0)
            except Exception as e:
                logger.error(f"Failed to init FisherGeometry: {e}")
                self.fisher_geo = None
        else:
            self.fisher_geo = None

        if GeometricIntelligence:
            try:
                self.geometric_intel = GeometricIntelligence()
            except Exception as e:
                logger.error(f"Failed to init GeometricIntelligence: {e}")
                self.geometric_intel = None
        else:
            self.geometric_intel = None

        # 3. Topology & Graph
        if TopologyMapper:
            try:
                self.topology_mapper = TopologyMapper(n_cubes=5, overlap=0.2)
            except Exception as e:
                logger.error(f"Failed to init TopologyMapper: {e}")
                self.topology_mapper = None
        else:
            self.topology_mapper = None

        if HomologyScanner:
            try:
                self.homology_scanner = HomologyScanner(max_dim=1, max_edge=30.0)
            except Exception as e:
                logger.error(f"Failed to init HomologyScanner: {e}")
                self.homology_scanner = None
        else:
            self.homology_scanner = None

        if GCNPitchGraph:
            try:
                self.gcn_graph = GCNPitchGraph()
            except Exception as e:
                logger.error(f"Failed to init GCNPitchGraph: {e}")
                self.gcn_graph = None
        else:
            self.gcn_graph = None

        if HypergraphUnitAnalyzer:
            try:
                self.hypergraph = HypergraphUnitAnalyzer()
            except Exception as e:
                logger.error(f"Failed to init HypergraphUnitAnalyzer: {e}")
                self.hypergraph = None
        else:
            self.hypergraph = None

        # 4. Dynamics & Flow
        if ParticleStrengthTracker:
            try:
                self.particle_tracker = ParticleStrengthTracker(n_particles=1000)
            except Exception as e:
                logger.error(f"Failed to init ParticleStrengthTracker: {e}")
                self.particle_tracker = None
        else:
            self.particle_tracker = None

        if PathSignatureEngine:
            try:
                self.path_signature = PathSignatureEngine(depth=2)
            except Exception as e:
                logger.error(f"Failed to init PathSignatureEngine: {e}")
                self.path_signature = None
        else:
            self.path_signature = None

        if RenormalizationGroup:
            try:
                self.rg_flow = RenormalizationGroup()
            except Exception as e:
                logger.error(f"Failed to init RenormalizationGroup: {e}")
                self.rg_flow = None
        else:
            self.rg_flow = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run unified physics analysis on the current batch of matches."""
        matches = context.get("matches", pl.DataFrame())
        features = context.get("features", pl.DataFrame())
        cycle = context.get("cycle", 0)

        # Initialize Results Container
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
            "gcn_coordination": {},
            "rg_flow_reports": {},
            "hypergraph_reports": {},
            "fisher_reports": {},
            "multifractal_reports": {}
        }

        # Initialize Simplified Context Map (for ML models)
        physics_context_map = {}

        # --- 1. Global / Systemic Analysis ---

        # Ricci Flow (Systemic Risk)
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
            return {"physics_reports": results, "physics_context": {}}

        # --- 2. Batch Training (if applicable) ---

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

        # --- 3. Per-Match Parallel Execution ---

        tasks = []
        for row in matches.iter_rows(named=True):
            match_id = row.get("match_id")
            if not match_id: continue

            # Ensure entry exists
            physics_context_map[match_id] = {}

            # Append tasks
            if self.chaos_filter:
                tasks.append(self._run_chaos_filter(match_id, results, physics_context_map))

            if self.quantum_brain:
                tasks.append(self._run_quantum_brain(match_id, row, feat_map, results, physics_context_map))

            if self.geometric_intel:
                tasks.append(self._run_geometric_intelligence(match_id, row, feat_map, results, physics_context_map))

            if self.particle_tracker:
                tasks.append(self._run_particle_tracker(match_id, cycle, results, physics_context_map))

            if self.fractal_analyzer:
                tasks.append(self._run_fractal_analyzer(match_id, results, physics_context_map))

            if self.topology_mapper:
                tasks.append(self._run_topology_mapper(match_id, row, feat_map, results, physics_context_map))

            if self.path_signature:
                tasks.append(self._run_path_signature(match_id, row, results, physics_context_map))

            if self.homology_scanner:
                tasks.append(self._run_homology_scanner(match_id, cycle, results, physics_context_map))

            if self.gcn_graph:
                tasks.append(self._run_gcn_graph(match_id, cycle, results, physics_context_map))

            if self.rg_flow:
                tasks.append(self._run_rg_flow(match_id, results, physics_context_map))

            if self.hypergraph:
                tasks.append(self._run_hypergraph_analysis(match_id, cycle, results, physics_context_map))

            if self.fisher_geo:
                tasks.append(self._run_fisher_geometry(match_id, results, physics_context_map))

            if self.multifractal:
                tasks.append(self._run_multifractal(match_id, results, physics_context_map))

        # Run all tasks concurrently
        if tasks:
            await asyncio.gather(*tasks)

        # Return results
        return {
            "physics_reports": results,
            "physics_context": physics_context_map
        }

    # --- Async Worker Methods ---

    async def _run_chaos_filter(self, match_id: str, results: Dict[str, Any], ctx_map: Dict):
        try:
            # Simulation: In prod, fetch real odds history
            odds_history = await asyncio.to_thread(self._simulate_odds_history, match_id)
            chaos_rep = await asyncio.to_thread(self.chaos_filter.analyze, odds_history, match_id=match_id)
            results["chaos_reports"][match_id] = chaos_rep
            ctx_map[match_id]["chaos_regime"] = chaos_rep.regime
            ctx_map[match_id]["lyapunov"] = chaos_rep.params.max_lyapunov
            ctx_map[match_id]["chaos_kill"] = chaos_rep.kill_betting
        except Exception as e:
            logger.warning(f"Chaos analysis failed for {match_id}: {e}")

    async def _run_quantum_brain(self, match_id: str, row: Dict, feat_map: Dict, results: Dict[str, Any], ctx_map: Dict):
        try:
            feat_row = feat_map.get(match_id, {})
            # Use features if available, else simplified odds
            if feat_row:
                vec = [v for k, v in feat_row.items() if isinstance(v, (int, float)) and k != "match_id"]
                # Quantum brain expects fixed size, e.g. 10
                vec = vec[:10] if len(vec) > 10 else vec + [0.0]*(10-len(vec))
            else:
                vec = [row.get("home_odds", 2.0), row.get("draw_odds", 3.0), row.get("away_odds", 3.5)]
                vec = vec + [0.0] * (10 - len(vec))

            q_pred = await asyncio.to_thread(self.quantum_brain.predict_match, vec, match_id=match_id)
            results["quantum_predictions"][match_id] = q_pred
            ctx_map[match_id]["quantum_conf"] = q_pred.confidence
            ctx_map[match_id]["quantum_prob_home"] = q_pred.probabilities[0]
        except Exception as e:
            logger.warning(f"Quantum prediction failed for {match_id}: {e}")

    async def _run_geometric_intelligence(self, match_id: str, row: Dict, feat_map: Dict, results: Dict[str, Any], ctx_map: Dict):
        try:
            feat_row = feat_map.get(match_id, row)
            mini_df = pl.DataFrame([feat_row])
            pot_df = await asyncio.to_thread(self.geometric_intel.compute_potential, mini_df)
            if not pot_df.is_empty():
                pot_dict = pot_df.to_dicts()[0]
                results["geometric_potentials"][match_id] = pot_dict
                ctx_map[match_id]["geo_dominance"] = pot_dict.get("dominance", 0.0)
                ctx_map[match_id]["geo_momentum"] = pot_dict.get("momentum", 0.0)
        except Exception as e:
            logger.warning(f"Geometric analysis failed for {match_id}: {e}")

    async def _run_particle_tracker(self, match_id: str, cycle: int, results: Dict[str, Any], ctx_map: Dict):
        try:
            # Simulation: In prod, fetch real match events/stats
            obs = await asyncio.to_thread(self._create_mock_observation, match_id, cycle)
            p_rep = await asyncio.to_thread(self.particle_tracker.update, obs, match_id=match_id)
            results["particle_reports"][match_id] = p_rep
            ctx_map[match_id]["particle_home_power"] = p_rep.state.home_power
            ctx_map[match_id]["particle_momentum"] = p_rep.state.momentum
        except Exception as e:
            logger.warning(f"Particle tracking failed for {match_id}: {e}")

    async def _run_fractal_analyzer(self, match_id: str, results: Dict[str, Any], ctx_map: Dict):
        try:
            hist_data = self._simulate_odds_history(match_id)
            f_rep = await asyncio.to_thread(self.fractal_analyzer.compute_hurst, hist_data)
            results["fractal_reports"][match_id] = f_rep
            ctx_map[match_id]["fractal_dim"] = f_rep.fractal_dimension
            ctx_map[match_id]["hurst"] = f_rep.hurst
            ctx_map[match_id]["fractal_regime"] = f_rep.regime
            ctx_map[match_id]["fractal_mult"] = f_rep.kelly_multiplier
        except Exception as e:
            logger.warning(f"Fractal analysis failed for {match_id}: {e}")

    async def _run_topology_mapper(self, match_id: str, row: Dict, feat_map: Dict, results: Dict[str, Any], ctx_map: Dict):
        try:
            feat_row = feat_map.get(match_id, row)
            vec = [v for k, v in feat_row.items() if isinstance(v, (int, float)) and k != "match_id"]
            if vec:
                t_rep = await asyncio.to_thread(self.topology_mapper.analyze_match, vec, match_id=match_id)
                results["topology_reports"][match_id] = t_rep
                ctx_map[match_id]["topology_cluster"] = t_rep.assigned_cluster
                ctx_map[match_id]["topology_anomaly"] = 1 if t_rep.is_anomalous else 0
        except Exception as e:
            logger.warning(f"Topology analysis failed for {match_id}: {e}")

    async def _run_path_signature(self, match_id: str, row: Dict, results: Dict[str, Any], ctx_map: Dict):
        try:
            mini_df = pl.DataFrame([row])
            sig_df = await asyncio.to_thread(self.path_signature.extract, mini_df)
            if not sig_df.is_empty():
                sig_dict = sig_df.to_dicts()[0]
                results["path_signatures"][match_id] = sig_dict
                ctx_map[match_id]["roughness"] = sig_dict.get("sig_roughness", 0.0)
        except Exception as e:
            logger.warning(f"Path signature failed for {match_id}: {e}")

    async def _run_homology_scanner(self, match_id: str, cycle: int, results: Dict[str, Any], ctx_map: Dict):
        try:
            # Simulation: In prod, fetch player coordinates
            pos_h = self._simulate_player_positions(match_id, cycle, team="home")
            pos_a = self._simulate_player_positions(match_id, cycle, team="away")

            comp_rep = await asyncio.to_thread(self.homology_scanner.compare_teams, pos_h, pos_a, match_id=match_id)
            results["homology_reports"][match_id] = comp_rep
            ctx_map[match_id]["homology_org_diff"] = comp_rep.get("org_advantage", 0.0)
            ctx_map[match_id]["home_org"] = comp_rep.get("home_org", 0.0)
            ctx_map[match_id]["away_panicking"] = 1 if comp_rep.get("away_panicking", False) else 0
        except Exception as e:
            logger.warning(f"Homology scanner failed for {match_id}: {e}")

    async def _run_gcn_graph(self, match_id: str, cycle: int, results: Dict[str, Any], ctx_map: Dict):
        try:
            pos_h = self._simulate_player_positions(match_id, cycle, team="home")
            pos_a = self._simulate_player_positions(match_id, cycle, team="away")
            all_pos = np.vstack([pos_h, pos_a])
            teams = [0]*len(pos_h) + [1]*len(pos_a)

            coord_rep = await asyncio.to_thread(self.gcn_graph.analyze_coordination, all_pos, teams)
            results["gcn_coordination"][match_id] = coord_rep
            ctx_map[match_id]["gcn_home_coord"] = coord_rep.get("home_coordination", 0.5)
        except Exception as e:
            logger.warning(f"GCN graph failed for {match_id}: {e}")

    async def _run_rg_flow(self, match_id: str, results: Dict[str, Any], ctx_map: Dict):
        try:
            momentum = self._simulate_momentum_series(match_id)
            rg_rep = await asyncio.to_thread(self.rg_flow.analyze_flow, momentum)
            results["rg_flow_reports"][match_id] = rg_rep
            ctx_map[match_id]["rg_criticality"] = rg_rep.get("criticality", 0.0)
            ctx_map[match_id]["rg_beta"] = rg_rep.get("beta", 0.0)
        except Exception as e:
            logger.warning(f"RG Flow failed for {match_id}: {e}")

    async def _run_hypergraph_analysis(self, match_id: str, cycle: int, results: Dict[str, Any], ctx_map: Dict):
        try:
            # We mock ratings/units since we don't have deep player db access here
            if not TacticalUnit: raise ImportError("TacticalUnit not available")

            units_h = [
                TacticalUnit("Defense", "defense", [0, 1, 2, 3], weight=1.5),
                TacticalUnit("Midfield", "midfield", [4, 5, 6, 7], weight=1.2),
                TacticalUnit("Attack", "attack", [8, 9, 10], weight=1.0)
            ]
            units_a = [
                TacticalUnit("Defense", "defense", [0, 1, 2, 3], weight=1.5),
                TacticalUnit("Midfield", "midfield", [4, 5, 6, 7], weight=1.2),
                TacticalUnit("Attack", "attack", [8, 9, 10], weight=1.0)
            ]
            ratings_h = np.clip(np.random.normal(70, 10, 11), 40, 99)
            ratings_a = np.clip(np.random.normal(70, 10, 11), 40, 99)

            report_h = await asyncio.to_thread(self.hypergraph.analyze_team, "Home", units_h, ratings_h)
            report_a = await asyncio.to_thread(self.hypergraph.analyze_team, "Away", units_a, ratings_a)

            results["hypergraph_reports"][match_id] = {"home": report_h, "away": report_a}
            ctx_map[match_id]["hyper_vuln_home"] = report_h.vulnerability_index
            ctx_map[match_id]["hyper_vuln_away"] = report_a.vulnerability_index

        except Exception as e:
            logger.warning(f"Hypergraph analysis failed for {match_id}: {e}")

    async def _run_fisher_geometry(self, match_id: str, results: Dict[str, Any], ctx_map: Dict):
        try:
            ref_data = self._simulate_odds_history(match_id + "_ref")
            curr_data = self._simulate_odds_history(match_id)

            fish_rep = await asyncio.to_thread(self.fisher_geo.compare_distributions, ref_data, curr_data, match_id=match_id)
            results["fisher_reports"][match_id] = fish_rep
            ctx_map[match_id]["fisher_distance"] = fish_rep.fisher_rao_distance
            ctx_map[match_id]["fisher_regime_shift"] = 1 if fish_rep.regime_shift else 0
        except Exception as e:
            logger.warning(f"Fisher Geometry failed for {match_id}: {e}")

    async def _run_multifractal(self, match_id: str, results: Dict[str, Any], ctx_map: Dict):
        try:
            hist_data = self._simulate_odds_history(match_id)
            mf_rep = await asyncio.to_thread(self.multifractal.analyze, hist_data, match_id=match_id)
            results["multifractal_reports"][match_id] = mf_rep
            ctx_map[match_id]["mf_regime"] = mf_rep.regime
            ctx_map[match_id]["mf_delta_h"] = mf_rep.params.delta_h
            ctx_map[match_id]["mf_crash_signal"] = 1 if mf_rep.regime_change_signal else 0
        except Exception as e:
             logger.warning(f"Multifractal analysis failed for {match_id}: {e}")

    # --- Simulation Helpers (Mocking Data Providers) ---

    def _simulate_odds_history(self, match_id: str) -> np.ndarray:
        """Simulate odds history for analysis (if DB history missing)."""
        seed = int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        n = 60
        # Random walk with drift
        returns = np.random.normal(0.001, 0.02, n)
        odds = np.cumprod(1 + returns) * 2.0
        return odds

    def _create_mock_observation(self, match_id: str, cycle: int) -> Optional[Any]:
        """Creates a simulated observation for Particle Strength Tracker."""
        if not MatchObservation: return None
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
        """Simulate 2D player positions on a pitch (105x68)."""
        seed = int(hashlib.md5(f"{match_id}_{cycle}_{team}".encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        # 4-4-2 Formation bases
        base_x = np.array([5, 20, 20, 20, 20, 50, 50, 50, 50, 80, 80])
        base_y = np.array([34, 10, 25, 43, 58, 10, 25, 43, 58, 25, 43])
        # Add noise
        noise = np.random.normal(0, 5, (11, 2))
        pos = np.column_stack([base_x, base_y]) + noise
        # Clip
        pos[:, 0] = np.clip(pos[:, 0], 0, 105)
        pos[:, 1] = np.clip(pos[:, 1], 0, 68)
        return pos

    def _simulate_momentum_series(self, match_id: str) -> List[float]:
        """Simulate momentum time series."""
        seed = int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        n = 60
        white_noise = np.random.normal(0, 1, n)
        momentum = np.cumsum(white_noise)
        momentum = (momentum - momentum.mean()) / (momentum.std() + 1e-9)
        return momentum.tolist()
