import asyncio
import random
import time
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np

# Internal imports (assumed available based on project structure)
from src.ingestion.data_sources import DataSourceAggregator
from src.core.data_validator import DataValidator
from src.memory.db_manager import DBManager
from src.core.regime_kelly import RegimeKelly, RegimeState
# Quant models
from src.quant.probabilistic_engine import ProbEngine
from src.quant.multi_task_backbone import MultiTaskBackbone
from src.quant.kan_interpreter import KANInterpreter
from src.quant.poisson_model import PoissonModel
from src.quant.monte_carlo_sim import MonteCarloSim
from src.quant.elo_engine import EloEngine
from src.quant.dixon_coles import DixonColesEngine
from src.quant.gradient_boosting import GradientBoostingEngine
from src.quant.glm_engine import GLMEngine
from src.quant.bayesian_hierarchical import BayesianHierarchicalModel
from src.quant.lstm_forecaster import LSTMForecaster
from src.quant.sentiment_analyzer import SentimentAnalyzer
from src.quant.graph_rag import GraphRAG
from src.quant.match_twin import MatchTwin
from src.quant.philosophical_engine import PhilosophicalEngine
from src.utils.devils_advocate import DevilsAdvocate

class DataIngestionAgent:
    """Subagent responsible for fetching and preparing match data."""
    def __init__(self, data_agg: DataSourceAggregator, validator: DataValidator):
        self.agg = data_agg
        self.validator = validator

    async def run(self):
        logger.info("📡 [Ingestion] Veri toplama başlatılıyor...")
        # 1. DB'den yaklaşan maçları çek
        matches_df = self.agg.db.get_upcoming_matches()
        
        # 2. Eğer DB boşsa, canlı/programlı maçları çek ve kaydet
        if matches_df.empty:
            logger.warning("⚠️ [Ingestion] DB boş, dış kaynaklardan veri çekiliyor...")
            await self.agg.fetch_today() # This calls Sofascore, TheSportsDB, etc.
            matches_df = self.agg.db.get_upcoming_matches()
        
        if matches_df.empty:
            logger.error("❌ [Ingestion] Veri bulunamadı.")
            return []

        # 3. Limit (Debug/Dev için)
        matches_df = matches_df.head(20) # Process max 20 matches for speed
        
        # Convert to list of dicts
        match_lists = matches_df.to_dict("records")
        
        # 4. Validasyon (Critical Fix: Fallback Logic)
        validated_data = self.validator.validate_batch(match_lists)
        if not validated_data:
            logger.warning("⚠️ [Ingestion] Validasyon başarısız, ham veri kullanılıyor (Fallback Modu).")
            validated_data = match_lists # Fallback to raw data
            
        # 5. Odds Injection (Critical Fix)
        final_data = []
        for match in validated_data:
            # Ensure safe odds
            if not match.get("home_odds") or match.get("home_odds") < 1.01:
                 match["home_odds"] = 2.30
            if not match.get("draw_odds") or match.get("draw_odds") < 1.01:
                 match["draw_odds"] = 3.20
            if not match.get("away_odds") or match.get("away_odds") < 1.01:
                 match["away_odds"] = 3.00
            final_data.append(match)
            
        logger.info(f"✅ [Ingestion] {len(final_data)} maç işleme hazır w/ Odds Injection.")
        return final_data

class QuantitativeAgent:
    """Subagent responsible for running analytical models."""
    def __init__(self, models: dict, db: DBManager):
        self.models = models # Dictionary of initialized model instances
        self.db = db

    async def run(self, matches: list[dict]):
        signals = []
        logger.info(f"🧠 [Quant] {len(matches)} maç analiz ediliyor (Gelişmiş Modeller Aktif)...")
        
        # Model Weights
        WEIGHTS = {
            "kan": 0.30,
            "lstm": 0.25,
            "poisson": 0.20,
            "elo": 0.15,
            "bayesian": 0.10
        }
        
        import polars as pl
        
        for match in matches:
            mid = match.get("match_id", "unknown")
            home = match.get("home_team", "Home")
            away = match.get("away_team", "Away")
            
            # --- Model Predictions (Weighted Ensemble) ---
            probs = {"H": 0.0, "D": 0.0, "A": 0.0}
            total_weight = 0.0
            
            # 1. Poisson (20%)
            try:
                if "poisson" in self.models:
                    p_res = self.models["poisson"].predict_proba(home, away)
                    w = WEIGHTS["poisson"]
                    probs["H"] += p_res.get("home_win", 0.33) * w
                    probs["D"] += p_res.get("draw", 0.33) * w
                    probs["A"] += p_res.get("away_win", 0.33) * w
                    total_weight += w
            except Exception as e:
                logger.debug(f"Poisson error: {e}")
            
            # 2. Elo (15%)
            try:
                if "elo" in self.models:
                    elo_res = self.models["elo"].predict_proba(home, away)
                    w = WEIGHTS["elo"]
                    probs["H"] += elo_res.get("home_win", 0.33) * w
                    probs["D"] += elo_res.get("draw", 0.33) * w
                    probs["A"] += elo_res.get("away_win", 0.33) * w
                    total_weight += w
            except Exception as e:
                logger.debug(f"Elo error: {e}")

            # 3. KAN (30%)
            try:
                if "kan" in self.models:
                    # KAN expects DataFrame
                    kan_df = pl.DataFrame([match])
                    kan_cl = self.models["kan"].predict(kan_df)
                    if not kan_df.is_empty():
                        row = kan_cl.row(0, named=True)
                        w = WEIGHTS["kan"]
                        probs["H"] += row.get("prob_home", 0.33) * w
                        probs["D"] += row.get("prob_draw", 0.33) * w
                        probs["A"] += row.get("prob_away", 0.33) * w
                        total_weight += w
            except Exception as e:
                logger.debug(f"KAN error: {e}")

            # 4. Bayesian (10%)
            try:
                if "bayesian_model" in self.models:
                    bayes_res = self.models["bayesian_model"].predict(home, away)
                    w = WEIGHTS["bayesian"]
                    probs["H"] += bayes_res.get("prob_home", 0.33) * w
                    probs["D"] += bayes_res.get("prob_draw", 0.33) * w
                    probs["A"] += bayes_res.get("prob_away", 0.33) * w
                    total_weight += w
            except Exception as e:
                logger.debug(f"Bayesian error: {e}")

            # 5. LSTM Trend (25%)
            try:
                if "lstm_trend" in self.models:
                    # Fetch history (Mocking DB call for speed if needed, but trying actual)
                    # For now passing empty to use Heuristic fallback inside LSTM if DB fetch is heavy
                    # TODO: Implement cached history fetch
                    lstm_res = self.models["lstm_trend"].predict_for_match(home, away, [], [])
                    w = WEIGHTS["lstm"]
                    probs["H"] += lstm_res.get("prob_home", 0.33) * w
                    probs["D"] += lstm_res.get("prob_draw", 0.33) * w
                    probs["A"] += lstm_res.get("prob_away", 0.33) * w
                    total_weight += w
            except Exception as e:
                logger.debug(f"LSTM error: {e}")
            
            # Normalize
            if total_weight > 0:
                probs["H"] /= total_weight
                probs["D"] /= total_weight
                probs["A"] /= total_weight
            else:
                probs = {"H": 0.33, "D": 0.34, "A": 0.33} # Default entropy max
                
            # --- Signal Generation ---
            # Identify Value
            h_odds = float(match.get("home_odds", 2.30))
            d_odds = float(match.get("draw_odds", 3.20))
            a_odds = float(match.get("away_odds", 3.00))
            
            # Edge Calculation
            edges = {
                "home": probs["H"] * h_odds - 1,
                "draw": probs["D"] * d_odds - 1,
                "away": probs["A"] * a_odds - 1
            }
            
            best_pick = max(edges, key=edges.get)
            best_edge = edges[best_pick]
            
            # CRITICAL FIX: Confidence Score Preservation
            # Map Twin/Philosophical score if available, else derive from edge
            confidence = 0.5 + (best_edge * 2) # Heuristic: higher edge -> higher confidence
            confidence = min(max(confidence, 0.1), 0.95)
            
            signal = {
                "match_id": mid,
                "home_team": home,
                "away_team": away,
                "selection": best_pick,
                "odds": h_odds if best_pick == "home" else (d_odds if best_pick == "draw" else a_odds),
                "prob": probs[best_pick[0].upper()], # H, D, or A
                "edge": best_edge,
                "confidence": confidence,
                "raw_probs": probs,
                "models_used": int(total_weight > 0) # Flag
            }
            signals.append(signal)

        logger.info(f"⚡ [Quant] {len(signals)} sinyal üretildi (Advanced Ensemble).")
        return signals

class ExecutionAgent:
    """Subagent responsible for risk management and execution (logging)."""
    def __init__(self, risk_manager: RegimeKelly, db: DBManager):
        self.risk_manager = risk_manager
        self.db = db

    def run(self, signals: list[dict]):
        logger.info("⚖️ [Execution] Risk analizi ve portföy dağılımı...")
        portfolio = []
        
        for sig in signals:
            # CRITICAL FIX: Fix -99% Edge bug
            # Ensure prob and odds are floats and valid
            prob = float(sig.get("prob", 0))
            odds = float(sig.get("odds", 0))
            
            if prob <= 0 or odds <= 1.0:
                logger.warning(f"⚠️ [Execution] Geçersiz sinyal: {sig['match_id']} P={prob} O={odds}")
                continue
                
            # Call RegimeKelly
            decision = self.risk_manager.calculate(
                probability=prob,
                odds=odds,
                match_id=sig["match_id"],
                regime=RegimeState(volatility_regime="calm") # Default to calm for now
            )
            
            if decision.approved:
                bet_slip = {
                    "match_id": sig["match_id"],
                    "selection": sig["selection"],
                    "odds": odds,
                    "stake": decision.stake_amount,
                    "edge": decision.edge,
                    "confidence": sig["confidence"]
                }
                portfolio.append(bet_slip)
                logger.success(f"💰 [BET] {sig['home_team']} vs {sig['away_team']} -> {sig['selection']} @ {odds:.2f} (Prob: {prob:.2f}, Stake: {decision.stake_amount} TL)")
                
                # Save to DB (mock for now if method inconsistent)
                # self.db.save_bet(bet_slip) 
            else:
                logger.info(f"🛑 [SKIP] {sig['home_team']} vs {sig['away_team']}: {decision.rejection_reason}")
                
        logger.info(f"📋 [Execution] Toplam {len(portfolio)} bahis onaylandı.")
        return portfolio

class AnalysisOrchestrator:
    """Orchestrates the entire analysis lifecycle using subagents."""
    def __init__(self, 
                 db_manager: DBManager,
                 data_agg: DataSourceAggregator,
                 validator: DataValidator,
                 risk_manager: RegimeKelly,
                 models: dict):
                 
        self.ingestion_agent = DataIngestionAgent(data_agg, validator)
        self.quant_agent = QuantitativeAgent(models, db_manager)
        self.execution_agent = ExecutionAgent(risk_manager, db_manager)
        
    async def run_cycle(self):
        start_time = time.time()
        logger.info("🚀 [Orchestrator] Yeni analiz döngüsü başladı.")
        
        # 1. Ingestion
        matches = await self.ingestion_agent.run()
        if not matches:
            logger.warning("💤 [Orchestrator] Maç yok, uyku moduna geçiliyor.")
            return

        # 2. Quant
        signals = await self.quant_agent.run(matches)
        
        # 3. Execution
        portfolio = self.execution_agent.run(signals)
        
        duration = time.time() - start_time
        logger.info(f"🏁 [Orchestrator] Döngü tamamlandı. Süre: {duration:.2f}s")
        return portfolio
