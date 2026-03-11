"""
alpha_generator.py – Autonomous Alpha Signal Generator.

Scans the database and external context for statistical anomalies
and multi-dimensional divergences. Emits "alpha_signal" events
when a robust, statistically significant edge is found.

Enhanced with scipy.stats for Hypothesis Testing (e.g. Welch's t-test
for odds compression vs. historical distributions).
"""
import asyncio
import numpy as np
from scipy import stats
from loguru import logger
from src.core.event_bus import EventBus, Event
from src.system.container import container

class AlphaGenerator:
    """Autonomous Alpha Signal Generator with Statistical Rigor."""
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.running = False
        self.p_value_threshold = 0.05 # 95% Confidence Level for Alpha

    async def start(self):
        self.running = True
        logger.info("AlphaGenerator (Stat-Engine) started.")
        while self.running:
            try:
                loop = asyncio.get_running_loop()
                await asyncio.to_thread(self._check_market_anomalies, loop)
            except Exception as e:
                logger.error(f"AlphaGenerator failed during check: {e}")
            await asyncio.sleep(300)

    def _check_market_anomalies(self, loop: asyncio.AbstractEventLoop):
        """Scans the database for statistical anomalies and emits alpha signals."""
        db = container.get("db")
        if not db:
            return

        try:
            # Detect multi-dimensional anomalies (Goals, xG divergence, Volatility)
            query = """
            SELECT
                home_team, away_team,
                (home_score + away_score) as total_goals,
                home_odds, away_odds
            FROM matches
            WHERE status = 'FINISHED'
            ORDER BY date DESC LIMIT 500
            """
            recent_matches = db.query(query)
            if recent_matches is None or (hasattr(recent_matches, 'is_empty') and recent_matches.is_empty()) or len(recent_matches) == 0:
                 return

            # Convert to lists/numpy arrays safely
            if hasattr(recent_matches, 'columns'):
                 goals_array = recent_matches["total_goals"].to_numpy()
                 home_odds_array = recent_matches["home_odds"].to_numpy() if "home_odds" in recent_matches.columns else np.array([])
            else:
                 goals_array = np.array([r.get("total_goals", 0) for r in recent_matches])
                 home_odds_array = np.array([r.get("home_odds", 2.0) for r in recent_matches])

            if len(goals_array) < 30:
                 return # Not enough data for statistical significance

            # 1. Goal Anomaly Check (T-Test)
            # Compare the last 30 matches vs the historical baseline
            recent_goals = goals_array[:30]
            historical_goals = goals_array[30:]

            if len(historical_goals) > 10:
                t_stat, p_val = stats.ttest_ind(recent_goals, historical_goals, equal_var=False)

                if p_val < self.p_value_threshold:
                    mean_diff = np.mean(recent_goals) - np.mean(historical_goals)
                    if mean_diff > 0.5:
                        # Recent matches are significantly higher scoring
                        self._emit_signal(loop, "Global_Over_Anomaly", 1.0 - p_val, np.mean(recent_goals), "Statistically significant goal surge. Buy OVERs.")
                    elif mean_diff < -0.5:
                        # Recent matches are significantly lower scoring
                        self._emit_signal(loop, "Global_Under_Anomaly", 1.0 - p_val, np.mean(recent_goals), "Statistically significant goal drought. Buy UNDERs.")

            # 2. Odds Compression Check
            if len(home_odds_array) > 30:
                 recent_odds = home_odds_array[:30]
                 hist_odds = home_odds_array[30:]
                 if len(hist_odds) > 10:
                     # Check if recent home odds are unusually low (Market pricing Home stronger than historically)
                     t_stat_odds, p_val_odds = stats.ttest_ind(recent_odds, hist_odds, equal_var=False)
                     if p_val_odds < self.p_value_threshold and np.mean(recent_odds) < np.mean(hist_odds) - 0.2:
                          self._emit_signal(loop, "Home_Bias_Compression", 1.0 - p_val_odds, np.mean(recent_odds), "Home odds are statistically compressed. Seek Away Value (Contrarian).")

        except Exception as e:
            logger.error(f"AlphaGenerator DB/Stats error: {e}")

    def _emit_signal(self, loop: asyncio.AbstractEventLoop, sig_type: str, confidence: float, metric: float, action: str):
         if self.bus:
             logger.info(f"AlphaGenerator: Emitting {sig_type} (Conf: {confidence:.2f})")
             asyncio.run_coroutine_threadsafe(
                 self.bus.emit(Event("alpha_signal", {
                     "type": sig_type,
                     "confidence": float(confidence),
                     "metric_value": float(metric),
                     "action": action
                 })),
                 loop
             )

    def stop(self):
        self.running = False
