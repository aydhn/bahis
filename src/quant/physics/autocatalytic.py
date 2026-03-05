"""
autocatalytic.py - Autocatalytic Edge Detection

Measures momentum feedback loops (e.g. scoring momentum,
possession dominance leading to shots). Uses an ODE-inspired
approach to detect if a team's edge is self-reinforcing.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class AutocatalyticReport:
    home_momentum_score: float = 0.0
    away_momentum_score: float = 0.0
    is_autocatalytic: bool = False
    dominant_team: str = "NONE"

class AutocatalyticDetector:
    """
    Detects if a team's recent actions (shots, corners, goals)
    create a positive feedback loop.
    """
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate

    def detect(self, match_events: list[dict]) -> AutocatalyticReport:
        """
        Analyzes a time-series of events to find autocatalytic momentum.
        Events should have: 'team', 'minute', 'type' (goal, shot, corner)
        """
        if not match_events:
            return AutocatalyticReport()

        h_score = 0.0
        a_score = 0.0

        weights = {"goal": 1.0, "shot": 0.2, "corner": 0.1}

        # Simple exponential decay integration
        # dp/dt = Event_Impact - decay * p
        # If p grows non-linearly, it's autocatalytic.

        for event in sorted(match_events, key=lambda x: x.get("minute", 0)):
            team = event.get("team", "")
            e_type = event.get("type", "")
            weight = weights.get(e_type, 0.0)

            if team == "HOME":
                h_score = h_score * (1 - self.decay_rate) + weight
                a_score = a_score * (1 - self.decay_rate)
            elif team == "AWAY":
                a_score = a_score * (1 - self.decay_rate) + weight
                h_score = h_score * (1 - self.decay_rate)

        report = AutocatalyticReport(
            home_momentum_score=h_score,
            away_momentum_score=a_score
        )

        # Determine if it's autocatalytic (score > threshold indicates self-reinforcement)
        threshold = 1.5
        if h_score > threshold and h_score > a_score * 2:
            report.is_autocatalytic = True
            report.dominant_team = "HOME"
        elif a_score > threshold and a_score > h_score * 2:
            report.is_autocatalytic = True
            report.dominant_team = "AWAY"

        return report
