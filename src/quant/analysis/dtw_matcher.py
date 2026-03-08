"""
dtw_matcher.py - Dynamic Time Warping (DTW) Pattern Matcher.

This module detects anomalies like flash crashes or "fix matches" by comparing
the current odds trajectory against historical templates using DTW.
DTW is robust to time shifts (e.g., an odds drop happening 10 minutes earlier than usual).
"""

import numpy as np
from loguru import logger
from typing import List
from dataclasses import dataclass

@dataclass
class DTWReport:
    anomaly_score: float  # Range [0, 1] - Higher is more anomalous compared to 'normal' baseline
    is_crash: bool        # True if match trajectory closely matches a known crash template
    closest_template: str # Name or ID of the closest matching template ("NORMAL", "FLASH_CRASH", "FIX")
    distance: float       # Raw DTW distance to the closest template

class DTWMatcher:
    """
    Detects flash crashes and anomalous patterns in odds time-series using Dynamic Time Warping.
    """
    def __init__(self, crash_threshold: float = 0.5):
        self.crash_threshold = crash_threshold
        # Simple mock templates for baseline behavior (in a real system, these are learned from DB)
        self.templates = {
            "NORMAL_DRIFT": np.array([2.0, 2.01, 2.02, 2.0, 1.99, 2.0]),
            "FLASH_CRASH": np.array([2.0, 1.95, 1.8, 1.5, 1.2, 1.1]),
            "SUSPICIOUS_BUMP": np.array([2.0, 2.0, 2.0, 2.5, 3.0, 2.8]),
            "STEADY_DECLINE": np.array([2.5, 2.4, 2.3, 2.2, 2.1, 2.0])
        }
        logger.debug(f"DTWMatcher initialized with {len(self.templates)} templates and threshold {self.crash_threshold}.")

    def compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Computes the Dynamic Time Warping distance between two sequences.
        Uses a standard dynamic programming approach.

        Args:
            seq1, seq2: NumPy arrays representing 1D time series.
        Returns:
            DTW distance (float).
        """
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return float('inf')

        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i - 1] - seq2[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                              dtw_matrix[i, j - 1],    # deletion
                                              dtw_matrix[i - 1, j - 1]) # match

        return dtw_matrix[n, m]

    def detect_flash_crash(self, odds_history: List[float]) -> DTWReport:
        """
        Compares the current odds history against known templates to identify anomalous patterns.

        Args:
            odds_history: List of recent odds values (time series).

        Returns:
            DTWReport containing match analysis.
        """
        if not odds_history or len(odds_history) < 2:
            return DTWReport(anomaly_score=0.0, is_crash=False, closest_template="INSUFFICIENT_DATA", distance=0.0)

        current_seq = np.array(odds_history)

        # Normalize the sequence relative to the starting point to focus on shape
        if current_seq[0] > 0:
            current_seq = current_seq / current_seq[0] * 2.0 # Scale to match template baseline (~2.0)

        best_match = None
        min_distance = float('inf')

        distances = {}
        for name, template in self.templates.items():
            dist = self.compute_dtw_distance(current_seq, template)
            distances[name] = dist
            if dist < min_distance:
                min_distance = dist
                best_match = name

        # Calculate anomaly score based on distance to NORMAL_DRIFT vs FLASH_CRASH
        normal_dist = distances.get("NORMAL_DRIFT", float('inf'))
        crash_dist = distances.get("FLASH_CRASH", float('inf'))

        # If it's closer to crash than normal, it's anomalous
        if normal_dist + crash_dist > 0:
            # Score approaches 1.0 as it gets closer to a crash pattern relative to normal
            anomaly_score = 1.0 - (crash_dist / (normal_dist + crash_dist))
        else:
            anomaly_score = 0.0

        is_crash = best_match == "FLASH_CRASH" and anomaly_score > self.crash_threshold

        return DTWReport(
            anomaly_score=anomaly_score,
            is_crash=is_crash,
            closest_template=best_match or "UNKNOWN",
            distance=min_distance
        )
