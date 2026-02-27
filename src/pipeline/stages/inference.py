from typing import Any, Dict, List
import asyncio
from loguru import logger
import polars as pl
import numpy as np

from src.pipeline.core import PipelineStage
from src.system.container import container
from src.core.model_registry import ModelRegistry
from src.ingestion.news_rag import NewsRAGAnalyzer
from src.quant.models.ensemble import EnsembleModel
from src.quant.risk.entropy_kelly import EntropyKelly
from src.quant.analysis.similarity import SimilarityEngine
from src.quant.analysis.market_sentiment import MarketSentiment
from src.quant.risk.volatility_modulator import VolatilityModulator
from src.quant.meta_labeling import MetaLabeler
from src.quant.uncertainty.conformal import ConformalPredictor
from src.quant.analysis.teleology import TeleologicalEngine
from src.core.zero_copy_bridge import ZeroCopyBridge

# New Imports
try:
    from src.quant.analysis.multi_task_backbone import MultiTaskBackbone
except ImportError:
    MultiTaskBackbone = None

try:
    from src.quant.analysis.transport_metric import TransportMetric
except ImportError:
    TransportMetric = None

from src.quant.analysis.game_theory_engine import GameTheoryEngine
try:
    from src.extensions.market_god import MarketGod
except ImportError:
    MarketGod = None

class InferenceStage(PipelineStage):
    """
    Quant Models & AI Analysis Engine.
    Uses EnsembleModel for robust predictions and EntropyKelly for uncertainty metrics.
    Integrated with SimilarityEngine (Pattern Matching), MarketSentiment (Odds Analysis),
    MetaLabeler (Error Correction), and TeleologicalEngine (Purpose & Narrative).
    """

    def __init__(self):
        super().__init__("inference")

        # Initialize Ensemble Model (Aggregates Benter, Dixon-Coles, LSTM)
        self.ensemble = EnsembleModel()

        # Initialize Entropy Calculator
        self.entropy_calc = EntropyKelly()

        # Initialize Advanced Quant Engines
        self.similarity_engine = SimilarityEngine()
        logger.info("Loading history for Similarity Engine...")
        self.similarity_engine.load_history()

        self.market_sentiment = MarketSentiment()

        self.teleology_engine = TeleologicalEngine()

        self.meta_labeler = MetaLabeler()
        logger.info("Training Meta-Labeler on DB...")
        self.meta_labeler.train_on_db()

        # Initialize Conformal Predictor
        self.conformal = ConformalPredictor(alpha=0.1) # 90% confidence
        # Ideally, calibrate here if historical data is available in context or DB
        # self.conformal.calibrate(X_cal, y_cal)

        # Risk / Regime Context
        self.volatility_modulator = VolatilityModulator()

        # Game Theory Engine (NEW)
        self.game_theory = GameTheoryEngine()

        # Market God (The Omniscient Strategist)
        self.market_god = MarketGod() if MarketGod else None

        # RAG Analyzer (Optional)
        try:
            self.rag = NewsRAGAnalyzer()
        except Exception:
            self.rag = None

        # Transport Metric (Drift Detection)
        if TransportMetric:
            self.transport = TransportMetric()
        else:
            self.transport = None
            logger.warning("TransportMetric not found.")

        # Multi-Task Learning Backbone
        if MultiTaskBackbone:
            try:
                self.mtl_backbone = MultiTaskBackbone()
                logger.info("Multi-Task Backbone initialized.")
            except Exception as e:
                logger.error(f"Failed to init MultiTaskBackbone: {e}")
                self.mtl_backbone = None
        else:
            self.mtl_backbone = None
            logger.warning("MultiTaskBackbone module missing.")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel analysis for all matches."""
        matches = context.get("matches", pl.DataFrame())

        # Get Global Market Regime once per cycle
        regime_status = self.volatility_modulator.get_status()
        kelly_fraction = self.volatility_modulator.get_kelly_fraction()

        logger.info(f"Inference Cycle Start. Regime: {regime_status}")

        # Merge mocked features if available (Autonomous Mode)
        features = context.get("features", pl.DataFrame())
        mock_features = context.get("mock_features")

        if mock_features is not None:
            if features.is_empty():
                features = mock_features
                logger.info("Using Mock Features for Inference.")
            else:
                pass

        if matches.is_empty():
            logger.info("No matches to analyze.")
            return {"ensemble_results": []}

        # --- Speed Upgrade: Zero-Copy Inference ---
        shm_info = context.get("shm_info")
        mtl_predictions = {}

        # Use Zero-Copy Path if available AND MTL is active
        if shm_info and self.mtl_backbone:
            try:
                logger.info(f"Using Zero-Copy Bridge: {shm_info['name']}")
                bridge = ZeroCopyBridge(
                    name=shm_info["name"],
                    shape=shm_info["shape"],
                    create=False
                )

                # Zero-copy read (returns a copy for safety but initial access is fast)
                # In true zero-copy, we'd process directly from buffer but safety first.
                raw_data = bridge.read()
                bridge.close()

                # Slice to valid rows
                valid_rows = shm_info.get("valid_rows", raw_data.shape[0])
                feature_matrix = raw_data[:valid_rows]

                # Vectorized Inference via MTL Backbone
                # Assuming MTL accepts numpy array directly for speed
                # Note: mtl_backbone.predict typically expects Polars or Dict,
                # so we might need an adapter method `predict_batch_numpy`

                # Retrieve match IDs from context or features
                # If only SHM is passed, we rely on context['match_ids'] if available
                match_ids = context.get("match_ids")

                if match_ids is None and not features.is_empty():
                    match_ids = features["match_id"].to_list()

                # Safety check: Ensure alignment
                if match_ids and len(match_ids) != feature_matrix.shape[0]:
                    # Mismatch handling: slice to min length
                    min_len = min(len(match_ids), feature_matrix.shape[0])
                    match_ids = match_ids[:min_len]
                    feature_matrix = feature_matrix[:min_len]
                    logger.warning(f"SHM shape mismatch corrected. New len: {min_len}")

                # Optimized Path: Bypass DataFrame construction if backbone supports numpy
                if hasattr(self.mtl_backbone, "predict_batch_numpy"):
                    mtl_df = await asyncio.to_thread(self.mtl_backbone.predict_batch_numpy, feature_matrix, match_ids)
                else:
                    # Fallback to standard predict (requires DF reconstruction if features empty)
                    if features.is_empty():
                        # Construct minimal DF from numpy array just for prediction (slower)
                        # Assuming we know column names or backbone can handle unnamed
                        features = pl.DataFrame(feature_matrix)
                        if match_ids:
                            features = features.with_columns(pl.Series("match_id", match_ids))

                    mtl_df = await asyncio.to_thread(self.mtl_backbone.predict, features)

                for row in mtl_df.iter_rows(named=True):
                    mtl_predictions[row["match_id"]] = row

                logger.info(f"Zero-Copy MTL inference complete for {len(mtl_predictions)} matches.")

            except Exception as e:
                logger.error(f"Zero-Copy Inference failed: {e}")
                # Fallback to standard DF path
                if not features.is_empty():
                    try:
                        mtl_df = await asyncio.to_thread(self.mtl_backbone.predict, features)
                        for row in mtl_df.iter_rows(named=True):
                            mtl_predictions[row["match_id"]] = row
                    except Exception as ex:
                        logger.error(f"Fallback MTL failed: {ex}")

        # Standard Path if SHM not used/failed
        elif self.mtl_backbone and not features.is_empty() and not mtl_predictions:
            try:
                mtl_df = await asyncio.to_thread(self.mtl_backbone.predict, features)
                # Convert to dict for fast lookup
                for row in mtl_df.iter_rows(named=True):
                    mtl_predictions[row["match_id"]] = row
                logger.info(f"MTL inference complete for {len(mtl_predictions)} matches.")
            except Exception as e:
                logger.error(f"MTL inference failed: {e}")

        # --- Data Drift Check (Transport Metric) ---
        if self.transport and not features.is_empty():
            try:
                # Select only numeric features for drift detection
                numeric_features = features.select(pl.col(pl.Float64, pl.Int64)).to_numpy()

                # If reference is not set (first run), set it
                if self.transport._monitor.reference_dist is None:
                    self.transport.set_reference(numeric_features)
                    logger.info("TransportMetric: Reference distribution set.")
                else:
                    drift_report = self.transport.check_drift(numeric_features)
                    if drift_report.is_drifted:
                        logger.warning(f"TransportMetric: DRIFT DETECTED ({drift_report.drift_severity}) W={drift_report.wasserstein_2:.4f}")
                        # Inject drift info into context for RiskStage
                        context["data_drift_report"] = drift_report
            except Exception as e:
                logger.error(f"TransportMetric check failed: {e}")


        # Physics Metrics from Context
        quantum_predictions = context.get("quantum_predictions", {})
        geometric_potentials = context.get("geometric_potentials", {})

        tasks = []
        # Optimize: Create feature map
        feat_map = {row["match_id"]: row for row in features.iter_rows(named=True)}

        for row in matches.iter_rows(named=True):
            match_id = row["match_id"]
            match_feat = feat_map.get(match_id, {})
            full_context = {**row, **match_feat}

            # Inject global regime info
            full_context["_regime_status"] = regime_status
            full_context["_kelly_fraction"] = kelly_fraction

            # Inject Physics Metrics into Context for Ensemble
            if match_id in quantum_predictions:
                q_pred = quantum_predictions[match_id]
                full_context["quantum_conf"] = q_pred.confidence
                full_context["quantum_prob"] = q_pred.probabilities[0] # Home prob

            if match_id in geometric_potentials:
                geo_pot = geometric_potentials[match_id]
                full_context["geometric_dominance"] = geo_pot.get("dominance", 0.0)

            # Inject MTL Predictions
            if match_id in mtl_predictions:
                mtl_res = mtl_predictions[match_id]
                full_context["mtl_prob_home"] = mtl_res.get("mtl_prob_home", 0.0)
                full_context["mtl_expected_goals"] = mtl_res.get("mtl_expected_goals", 0.0)
                full_context["mtl_expected_corners"] = mtl_res.get("mtl_expected_corners", 0.0)

            tasks.append(self._analyze_single_match(full_context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for res in results:
            if isinstance(res, dict):
                valid_results.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Match Analysis Error: {res}")

        return {"ensemble_results": valid_results}

    async def _analyze_single_match(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single match using Ensemble, RAG, Similarity, Meta-Labeling, and Teleology."""
        match_id = context.get("match_id", "Unknown")

        # 1. Ensemble Prediction (CPU Bound -> Thread)
        prediction = await asyncio.to_thread(self.ensemble.predict, context)

        # 2. Entropy Calculation
        probs = [
            prediction.get("prob_home", 0.0),
            prediction.get("prob_draw", 0.0),
            prediction.get("prob_away", 0.0)
        ]
        entropy = self.entropy_calc.calculate_entropy(probs)
        prediction["entropy"] = entropy

        # 3. Market Sentiment (Odds Movement)
        try:
            sentiment = self.market_sentiment.analyze_sentiment(match_id)
            prediction["market_sentiment"] = sentiment
        except Exception:
            prediction["market_sentiment"] = {}

        # 4. Teleological Analysis (Narrative & Motivation)
        # Assuming context has some rank/points data (from Features stage)
        try:
            teleology = self.teleology_engine.analyze(context)
            prediction["teleology_score"] = teleology["teleology_score"]
            prediction["teleology_narrative"] = teleology["narrative"]
            prediction["is_biscuit"] = teleology["is_biscuit"]
        except Exception as e:
            logger.warning(f"Teleology failed for {match_id}: {e}")
            prediction["teleology_score"] = 0.5

        # 5. Pattern Matching (Similarity Engine)
        try:
            # Construct feature vector based on ODDS (Market View)
            # [home_odds, draw_odds, away_odds]
            h_odd = context.get("home_odds", 2.0)
            d_odd = context.get("draw_odds", 3.0)
            a_odd = context.get("away_odds", 3.5)

            feat_vec = np.array([[h_odd, d_odd, a_odd]])

            similar_res = self.similarity_engine.find_similar(feat_vec)
            prediction["similar_matches"] = similar_res

            # Smart Adjustment: If history says 80% Home Win, but model says 40%, flag it.
            hist_prob = similar_res.get("historical_probs", {}).get("prob_home", 0.0)
            prediction["historical_prob_home"] = hist_prob

        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")

        # 6. Meta-Labeling (Quality Check)
        try:
            # We need Odds to assess quality properly.
            meta_features = {
                "confidence": max(probs),
                "entropy": entropy,
                "odds": context.get("home_odds", 2.0), # Assuming Home bet for score calculation, simplified
                "ev": (max(probs) * context.get("home_odds", 2.0)) - 1.0
            }
            quality_score = self.meta_labeler.predict_score(meta_features)
            prediction["meta_quality_score"] = quality_score
        except Exception as e:
            logger.warning(f"Meta-labeling failed: {e}")
            prediction["meta_quality_score"] = 0.5

        # 7. Conformal Prediction (Certainty Set)
        try:
            # Create a mock prob vector for conformal
            prob_vec = np.array([prediction.get("prob_home", 0.33),
                                 prediction.get("prob_draw", 0.33),
                                 prediction.get("prob_away", 0.33)])
            # Since conformal.predict expects batch (N, 3), we wrap and unwrap
            pred_set = self.conformal.predict(prob_vec.reshape(1, -1))[0]
            prediction["conformal_set"] = pred_set # Set of indices {0, 1, ...}
            prediction["conformal_certainty"] = self.conformal.check_certainty(pred_set)
        except Exception as e:
            logger.warning(f"Conformal prediction failed: {e}")
            prediction["conformal_certainty"] = "UNKNOWN"

        # 8. Game Theory Check (Strategic Defense)
        # If market sentiment is extremely skewed (everyone betting home),
        # check if we should randomize our decision to avoid exploitation?
        # Or simply check Nash Equilibrium of the odds.
        # Simple use case: If implied prob (1/odds) is exactly equal to model prob,
        # it's a Nash Equilibrium (Efficient Market). Edge is 0.
        # We can flag "Perfectly Priced" markets.
        try:
            # Mock payoff matrix for simulation: Bettor vs Market
            # 2x2: [Bet Home, Bet Away] vs [Home Wins, Away Wins] (Simplified)
            # This is illustrative. Real application would be more complex.
            # Here we just check if EV is near zero, which is a sign of efficiency.
            implied_home = 1.0 / max(context.get("home_odds", 2.0), 1.01)
            model_home = prediction.get("prob_home", 0.33)

            if abs(implied_home - model_home) < 0.01:
                prediction["game_theory_status"] = "NASH_EQUILIBRIUM"
                prediction["reasoning"] = "Market is efficient. No edge."
            else:
                prediction["game_theory_status"] = "EXPLOITABLE"
        except Exception:
            pass

        # 9. Market God Consultation
        if self.market_god:
            try:
                # Construct odds data from context
                odds_data = {
                    "home": context.get("home_odds", 2.0),
                    "draw": context.get("draw_odds", 3.0),
                    "away": context.get("away_odds", 4.0)
                }
                # Volatility history ideally passed in context, using dummy or empty list
                vol_hist = context.get("volatility_history", [])

                god_sig = self.market_god.consult(match_id, odds_data, vol_hist)

                prediction["god_signal"] = god_sig.signal_type
                prediction["god_conviction"] = god_sig.conviction
                prediction["god_narrative"] = god_sig.narrative
                prediction["god_multiplier"] = god_sig.suggested_multiplier

                # Apply God Multiplier to Confidence?
                # If God says BULLISH, we boost model confidence.
                if god_sig.signal_type == "BULLISH":
                    prediction["confidence"] = min(prediction.get("confidence", 0.5) * 1.2, 1.0)
                elif god_sig.signal_type == "BEARISH":
                    prediction["confidence"] *= 0.8
                elif god_sig.signal_type in ["BLACK_SWAN", "FIX_DETECTED"]:
                    # Severe warning
                    prediction["god_veto"] = True

            except Exception as e:
                logger.warning(f"Market God is silent for {match_id}: {e}")

        # 10. Inject Risk Context
        prediction["regime_status"] = context.get("_regime_status", "NORMAL")
        prediction["kelly_fraction"] = context.get("_kelly_fraction", 1.0)

        # Add basic identification
        prediction["match_id"] = match_id
        prediction["home_team"] = context.get("home_team")
        prediction["away_team"] = context.get("away_team")

        return prediction
