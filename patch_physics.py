import re

# Read the original file
with open("src/pipeline/stages/physics.py", "r") as f:
    content = f.read()

execute_replacement = '''
    def _init_results_container(self) -> Dict[str, Any]:
        return {
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
            "multifractal_reports": {},
            "hawkes_momentum": {}
        }

    def _run_systemic_analysis(self, matches: pl.DataFrame, results: Dict[str, Any], cycle: int):
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

    def _run_batch_training(self, features: pl.DataFrame):
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

    def _create_match_tasks(self, row: Dict[str, Any], feat_map: Dict[str, Any], cycle: int, results: Dict[str, Any], physics_context_map: Dict[str, Any]) -> List[asyncio.Task]:
        match_id = row.get("match_id")
        if not match_id:
            return []

        # Ensure entry exists
        physics_context_map[match_id] = {}

        tasks = []
        if self.chaos_filter:
            tasks.append(asyncio.create_task(self._run_chaos_filter(match_id, results, physics_context_map)))
        if self.quantum_brain:
            tasks.append(asyncio.create_task(self._run_quantum_brain(match_id, row, feat_map, results, physics_context_map)))
        if self.geometric_intel:
            tasks.append(asyncio.create_task(self._run_geometric_intelligence(match_id, row, feat_map, results, physics_context_map)))
        if self.particle_tracker:
            tasks.append(asyncio.create_task(self._run_particle_tracker(match_id, cycle, results, physics_context_map)))
        if self.fractal_analyzer:
            tasks.append(asyncio.create_task(self._run_fractal_analyzer(match_id, results, physics_context_map)))
        if self.topology_mapper:
            tasks.append(asyncio.create_task(self._run_topology_mapper(match_id, row, feat_map, results, physics_context_map)))
        if self.path_signature:
            tasks.append(asyncio.create_task(self._run_path_signature(match_id, row, results, physics_context_map)))
        if self.homology_scanner:
            tasks.append(asyncio.create_task(self._run_homology_scanner(match_id, cycle, results, physics_context_map)))
        if self.gcn_graph:
            tasks.append(asyncio.create_task(self._run_gcn_graph(match_id, cycle, results, physics_context_map)))
        if self.rg_flow:
            tasks.append(asyncio.create_task(self._run_rg_flow(match_id, results, physics_context_map)))
        if self.hypergraph:
            tasks.append(asyncio.create_task(self._run_hypergraph_analysis(match_id, cycle, results, physics_context_map)))
        if self.fisher_geo:
            tasks.append(asyncio.create_task(self._run_fisher_geometry(match_id, results, physics_context_map)))
        if self.multifractal:
            tasks.append(asyncio.create_task(self._run_multifractal(match_id, results, physics_context_map)))
        if self.hawkes_momentum:
            tasks.append(asyncio.create_task(self._run_hawkes_momentum(match_id, cycle, results, physics_context_map)))

        return tasks

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run unified physics analysis on the current batch of matches."""
        matches = context.get("matches", pl.DataFrame())
        features = context.get("features", pl.DataFrame())
        cycle = context.get("cycle", 0)

        # Initialize Results Container
        results = self._init_results_container()

        # Initialize Simplified Context Map (for ML models)
        physics_context_map = {}

        # --- 1. Global / Systemic Analysis ---
        self._run_systemic_analysis(matches, results, cycle)

        if matches.is_empty():
            return {"physics_reports": results, "physics_context": {}}

        # --- 2. Batch Training (if applicable) ---
        # Pre-compute feature map for quick lookup
        feat_map = {}
        if not features.is_empty():
             feat_map = {row["match_id"]: row for row in features.iter_rows(named=True)}

        self._run_batch_training(features)

        # --- 3. Per-Match Parallel Execution ---
        tasks = []
        for row in matches.iter_rows(named=True):
            tasks.extend(self._create_match_tasks(row, feat_map, cycle, results, physics_context_map))

        # Run all tasks concurrently
        if tasks:
            await asyncio.gather(*tasks)

        # Compute Holistic Confidence for each match
        for match_id, ctx in physics_context_map.items():
            ctx["holistic_confidence"] = self._calculate_holistic_confidence(match_id, results)

        # Return results
        return {
            "physics_reports": results,
            "physics_context": physics_context_map
        }
'''

# find the execute block inside PhysicsStage
match = re.search(r'    async def execute\(self, context: Dict\[str, Any\]\) -> Dict\[str, Any\]:(.*?)    def _calculate_holistic_confidence', content, re.DOTALL)

if match:
    new_content = content[:match.start()] + execute_replacement + "\n    def _calculate_holistic_confidence" + content[match.end():]
    with open("src/pipeline/stages/physics.py", "w") as f:
        f.write(new_content)
    print("Successfully patched physics.py")
else:
    print("Could not find the execute method to patch.")
