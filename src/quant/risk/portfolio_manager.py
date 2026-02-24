"""
portfolio_manager.py - Modern Portfolio Theory & Adaptive Kelly Optimization.

This module treats individual bets as a "Portfolio" and maximizes the Sharpe Ratio
while minimizing total risk, respecting Kelly Criterion constraints.
"""
from typing import Dict, Any, List, Optional
from src.core.event_bus import EventBus, Event
from src.quant.risk.kelly import AdaptiveKelly
from loguru import logger
import numpy as np
from scipy.optimize import minimize

class PortfolioManager:
    """
    Optimizes the betting portfolio.
    Predict -> Optimize -> Bet flow.
    Integrates Markowitz MVO with Adaptive Kelly constraints.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.current_opportunities: List[Dict[str, Any]] = []
        self.kelly_manager = AdaptiveKelly(base_fraction=0.20, window_size=50)

        # Listen for events
        if self.bus:
            self.bus.subscribe("prediction_ready", self.on_prediction)
            self.bus.subscribe("pipeline_cycle_end", self.on_cycle_end)
            self.bus.subscribe("bet_resolved", self.on_bet_resolved)

    def on_prediction(self, event: Event):
        """Add new prediction to the pool."""
        prediction = event.data
        if prediction.get("confidence", 0) < 0.5:
            return

        self.current_opportunities.append(prediction)
        logger.debug(f"PortfolioManager: Opportunity added -> {prediction.get('match_id')}")

    def on_bet_resolved(self, event: Event):
        """Update Kelly calibration with resolved bet."""
        # expected format: {"predicted_prob": ..., "odds": ..., "won": ...}
        self.kelly_manager.update_outcome(event.data)

    async def on_cycle_end(self, event: Event):
        """Run optimization at cycle end."""
        if not self.current_opportunities:
            return

        logger.info(f"PortfolioManager: Optimizing {len(self.current_opportunities)} opportunities...")

        allocations = self.optimize_portfolio(self.current_opportunities)

        bet_count = 0
        for opp in self.current_opportunities:
            opp_id = opp.get("match_id")
            stake_pct = allocations.get(opp_id, 0.0)

            if stake_pct > 0.001:  # Min bet 0.1%
                bet_order = {
                    "match_id": opp_id,
                    "selection": opp.get("selection"),
                    "odds": opp.get("odds"),
                    "stake_pct": round(stake_pct, 4),
                    "reason": "Portfolio Optimization (Sharpe Max + Kelly Constraint)",
                    "kelly_fraction": opp.get("kelly_cap", 0.0)
                }
                if self.bus:
                    await self.bus.emit(Event(
                        event_type="bet_placed",
                        source="PortfolioManager",
                        match_id=opp_id,
                        data=bet_order
                    ))
                bet_count += 1

        if bet_count > 0:
            logger.success(f"PortfolioManager: {bet_count} orders generated.")

        self.current_opportunities = []

    def optimize_portfolio(self, opportunities: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Markowitz Mean-Variance Optimization with Correlation Awareness.
        Subject to 0 <= weight <= AdaptiveKelly(i).
        """
        n = len(opportunities)
        if n == 0:
            return {}

        returns = []
        stds = []
        kelly_caps = []

        for opp in opportunities:
            p = opp.get("prob_win", 0.5)
            o = opp.get("odds", 2.0)
            conf = opp.get("confidence", 0.5)

            # E[R] = p*o - 1
            exp_ret = (p * o) - 1

            # Variance (Bernoulli): p*(1-p)*(o-1-(-1))^2 ? No, simpler:
            # Var = E[X^2] - (E[X])^2
            # E[X^2] = p*(o-1)^2 + (1-p)*(-1)^2
            var = (p * (o - 1)**2 + (1 - p) * 1) - (exp_ret**2)
            std = np.sqrt(var) if var > 0 else 1.0

            returns.append(exp_ret)
            stds.append(std)

            # Calculate Adaptive Kelly limit for this bet
            k_cap = self.kelly_manager.calculate_fraction(p, o, confidence=conf)
            kelly_caps.append(k_cap)
            opp["kelly_cap"] = k_cap # Store for logging

        returns = np.array(returns)
        stds = np.array(stds)

        # Build Correlation Matrix
        cov_matrix = self._build_covariance_matrix(opportunities, stds)

        # Optimization Function (Negative Sharpe)
        def neg_sharpe(weights):
            p_ret = np.sum(returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol <= 1e-6:
                return 0.0
            return -(p_ret / p_vol)

        # Constraints
        # Total weight <= 1.0 (No leverage)
        constraints = ({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x)})

        # Bounds: 0 <= weight <= KellyCap
        bounds = tuple((0.0, cap) for cap in kelly_caps)

        # Initial Guess
        init_guess = np.array([min(cap, 0.01) for cap in kelly_caps])

        try:
            result = minimize(
                neg_sharpe,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            optimal_weights = result.x
        except Exception as e:
            logger.error(f"Optimization error: {e}. Fallback to flat Kelly.")
            optimal_weights = np.array(kelly_caps)

        # Map results
        allocations = {}
        for i, opp in enumerate(opportunities):
            allocations[opp.get("match_id")] = float(optimal_weights[i])

        return allocations

    def _build_covariance_matrix(self, opportunities: List[Dict], stds: np.ndarray) -> np.ndarray:
        """
        Construct a covariance matrix with heuristic correlations.
        - Same League: 0.2 correlation
        - Same Home Team: 0.5 correlation (rare in single batch but possible)
        """
        n = len(opportunities)
        corr_matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                rho = 0.0
                opp_i = opportunities[i]
                opp_j = opportunities[j]

                # Check League
                if opp_i.get("league") == opp_j.get("league") and opp_i.get("league"):
                    rho += 0.2

                # Check Teams (if data available)
                # Assuming 'teams' key exists as list [home, away] or similar
                # This is a placeholder for deeper logic

                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

        # Convert Correlation to Covariance: Cov_ij = rho_ij * std_i * std_j
        D = np.diag(stds)
        cov_matrix = np.dot(D, np.dot(corr_matrix, D))
        return cov_matrix
