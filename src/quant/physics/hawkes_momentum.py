"""
hawkes_momentum.py – Hawkes Process for Self-Exciting Match Momentum.

In football, events (shots, corners, dangerous attacks) are not independent.
A shot often leads to a corner, which leads to another shot. This is a "self-exciting" process.
We model this using a Hawkes Process to calculate the true "Momentum Intensity" of a team.

Concepts:
  - Base Rate (mu): The team's inherent rate of generating events.
  - Excitation (alpha): How much one event increases the probability of another.
  - Decay (beta): How fast the excitement fades away.
  - Intensity (lambda_t): The real-time probability of an event happening NOW.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from dataclasses import dataclass
from typing import List

@dataclass
class HawkesParams:
    mu: float = 0.05      # Base event rate (events per minute)
    alpha: float = 0.5    # Jump in intensity after an event
    beta: float = 0.1     # Exponential decay rate

@dataclass
class HawkesReport:
    match_id: str = ""
    home_intensity: float = 0.0
    away_intensity: float = 0.0
    home_base_rate: float = 0.0
    away_base_rate: float = 0.0
    momentum_imbalance: float = 0.0  # (Home - Away) / (Home + Away)
    is_home_surging: bool = False
    is_away_surging: bool = False
    expected_goals_next_10m_home: float = 0.0
    expected_goals_next_10m_away: float = 0.0

class HawkesMomentum:
    """
    Calculates the real-time intensity of match events using a Univariate Hawkes Process.
    """

    def __init__(self, params_home: HawkesParams = None, params_away: HawkesParams = None):
        self.params_home = params_home or HawkesParams()
        self.params_away = params_away or HawkesParams()
        logger.debug("HawkesMomentum Engine initialized.")

    def calculate_intensity(self, current_time: float, event_times: List[float], params: HawkesParams) -> float:
        """
        Calculates the Hawkes intensity lambda(t) at current_time.
        lambda(t) = mu + alpha * Sum_{t_i < t} exp(-beta * (t - t_i))
        """
        intensity = params.mu
        if not event_times:
            return intensity

        # Filter events that happened before current_time
        past_events = np.array([t for t in event_times if t < current_time])
        if len(past_events) == 0:
            return intensity

        # Calculate decays
        time_diffs = current_time - past_events
        decay_sum = np.sum(np.exp(-params.beta * time_diffs))

        intensity += params.alpha * decay_sum
        return intensity

    def analyze_match(self, match_id: str, current_minute: int,
                      home_event_minutes: List[int],
                      away_event_minutes: List[int]) -> HawkesReport:
        """
        Analyzes the current momentum state of the match.
        """
        report = HawkesReport(match_id=match_id)

        # Calculate current intensities
        h_int = self.calculate_intensity(current_minute, home_event_minutes, self.params_home)
        a_int = self.calculate_intensity(current_minute, away_event_minutes, self.params_away)

        report.home_intensity = float(h_int)
        report.away_intensity = float(a_int)
        report.home_base_rate = self.params_home.mu
        report.away_base_rate = self.params_away.mu

        total_int = h_int + a_int
        if total_int > 0:
            report.momentum_imbalance = float((h_int - a_int) / total_int)

        # Surges: If intensity is > 3x the base rate
        if h_int > self.params_home.mu * 3:
            report.is_home_surging = True
        if a_int > self.params_away.mu * 3:
            report.is_away_surging = True

        # Expected events (proxy for xG if events are dangerous attacks)
        # Integral of lambda(t) from t to t+10
        # Roughly: 10 * mu + (alpha/beta) * Sum(exp(-beta*(t-ti)) * (1 - exp(-beta*10)))

        def expected_events(params, current_time, events, horizon=10):
            past = np.array([t for t in events if t < current_time])
            base = params.mu * horizon
            if len(past) == 0: return base

            diffs = current_time - past
            decay_factor = np.exp(-params.beta * diffs)
            integral_factor = (1.0 - np.exp(-params.beta * horizon)) / params.beta

            excitement = params.alpha * np.sum(decay_factor) * integral_factor
            return base + excitement

        # Assuming 10% of these 'events' turn into goals
        conversion_rate = 0.10
        report.expected_goals_next_10m_home = float(expected_events(self.params_home, current_minute, home_event_minutes) * conversion_rate)
        report.expected_goals_next_10m_away = float(expected_events(self.params_away, current_minute, away_event_minutes) * conversion_rate)

        return report
