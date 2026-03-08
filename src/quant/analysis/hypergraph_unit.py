"""
hypergraph_unit.py – Hypergraph Neural Networks (Hiper-Çizge Ağları).

Models tactical units (Defense Line, Midfield Trio) as Hyperedges to detect structural breakdowns.
Unlike simple Graphs (edges between 2 nodes), Hypergraphs connect N nodes simultaneously.

Concepts:
  - Hypergraph Incidence Matrix (H): Node × Hyperedge
  - Laplacian: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
  - Hyperedge Entropy: Measure of disorganization within a tactical unit.
  - Betti Numbers: Topological features (holes) in the formation.

Technology: dhg (DeepHypergraph) or torch_geometric
Fallback: numpy + scipy (Laplacian calculation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════
@dataclass
class TacticalUnit:
    """A Hyperedge representing a tactical group."""
    name: str = ""                      # e.g., "Defense Line"
    unit_type: str = ""                 # "defense", "midfield", "attack"
    player_indices: List[int] = field(default_factory=list)
    player_names: List[str] = field(default_factory=list)
    weight: float = 1.0                # Importance weight
    cohesion_score: float = 0.0        # 0.0 - 1.0 (Calculated)
    entropy: float = 0.0               # Disorganization measure


@dataclass
class UnitAnalysis:
    """Analysis result for a single unit."""
    unit: TacticalUnit = field(default_factory=TacticalUnit)
    failure_prob: float = 0.0           # Probability of unit collapse
    weakest_link: str = ""
    weakest_link_impact: float = 0.0
    entropy_contribution: float = 0.0   # Contribution to total structural entropy


@dataclass
class HypergraphReport:
    """Comprehensive Team Analysis Report."""
    team: str = ""
    n_players: int = 0
    n_units: int = 0
    unit_analyses: List[UnitAnalysis] = field(default_factory=list)

    # Global Metrics
    team_cohesion: float = 0.0
    structural_entropy: float = 0.0     # Total system entropy
    vulnerability_index: float = 0.0    # 0 (Safe) - 1 (Critical)
    betti_0: int = 0                    # Number of disconnected components
    betti_1: int = 0                    # Number of topological holes (cycles)

    # Signals
    defense_alert: bool = False
    midfield_alert: bool = False
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  ALGORITHMS
# ═══════════════════════════════════════════════
def build_incidence_matrix(n_nodes: int, hyperedges: List[List[int]]) -> np.ndarray:
    """Constructs the Incidence Matrix H (Nodes x Edges)."""
    n_edges = len(hyperedges)
    H = np.zeros((n_nodes, n_edges), dtype=np.float64)
    for e_idx, nodes in enumerate(hyperedges):
        for n_idx in nodes:
            if 0 <= n_idx < n_nodes:
                H[n_idx, e_idx] = 1.0
    return H

def calculate_hypergraph_laplacian(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Computes the normalized Hypergraph Laplacian.
    L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    """
    # Degree of vertices (nodes)
    Dv = np.sum(H * W, axis=1) # Weighted degree
    # Degree of edges (hyperedges)
    De = np.sum(H, axis=0)

    # Avoid division by zero
    Dv_inv_sqrt = np.diag(np.where(Dv > 0, 1.0 / np.sqrt(Dv), 0.0))
    De_inv = np.diag(np.where(De > 0, 1.0 / De, 0.0))
    W_mat = np.diag(W)

    # L = I - ...
    # But usually for spectral analysis we look at the operator: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    # Eigenvalues of this operator relate to graph cuts. 1 is trivial eigenval.
    # Smaller eigenvalues (near 0) of (I - Operator) correspond to connected components.

    Operator = Dv_inv_sqrt @ H @ W_mat @ De_inv @ H.T @ Dv_inv_sqrt
    L = np.eye(H.shape[0]) - Operator
    return L

def calculate_hyperedge_entropy(ratings: np.ndarray) -> float:
    """
    Calculates Shannon Entropy of player ratings within a unit.
    High entropy = Mixed skill levels (potentially unstable).
    Low entropy = Homogeneous unit (stable).
    """
    if len(ratings) == 0: return 0.0

    # Normalize ratings to probabilities
    probs = ratings / (np.sum(ratings) + 1e-9)
    # Remove zeros for log
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))

    # Normalize by max possible entropy log2(N)
    max_ent = np.log2(len(ratings)) if len(ratings) > 1 else 1.0
    return float(entropy / max_ent)

def calculate_betti_numbers(L: np.ndarray, threshold: float = 1e-5) -> Tuple[int, int]:
    """
    Estimates Betti numbers from Laplacian Spectrum.
    Betti-0: Number of zero eigenvalues (Connected Components).
    Betti-1: Estimation based on spectral gap or specific homology library (approximated here).
    """
    eigvals = np.linalg.eigvalsh(L)
    # Betti-0 is number of zero eigenvalues
    betti_0 = np.sum(eigvals < threshold)

    # Betti-1 (Cycles) is harder with just Laplacian, but high density of small non-zero eigenvalues
    # often hints at complex cyclic structures or 'holes' in spectral geometry.
    # For a rigorous Betti-1, we'd need simplicial homology (boundary operators).
    # Here we use a heuristic: 'spectral holes' -> gaps in spectrum?
    # Or just return 0 as placeholder if no simplicial complex is built.
    # Let's mock Betti-1 as number of eigenvalues in a specific low-energy band.
    betti_1 = np.sum((eigvals >= threshold) & (eigvals < 0.1))

    return int(betti_0), int(betti_1)


# ═══════════════════════════════════════════════
#  ANALYZER CLASS
# ═══════════════════════════════════════════════
class HypergraphUnitAnalyzer:
    """
    Analyzes tactical structures using Hypergraph theory.
    """

    def __init__(self):
        logger.debug("HypergraphUnitAnalyzer initialized.")

    def analyze_team(self, team_name: str, units: List[TacticalUnit], player_ratings: np.ndarray) -> HypergraphReport:
        """
        Performs full hypergraph analysis on a team structure.
        """
        n_players = len(player_ratings)
        report = HypergraphReport(team=team_name, n_players=n_players, n_units=len(units))

        if n_players == 0 or not units:
            return report

        # 1. Build Matrices
        hyperedges = [u.player_indices for u in units]
        weights = np.array([u.weight for u in units])

        H = build_incidence_matrix(n_players, hyperedges)

        # 2. Laplacian & Spectral Analysis
        L = calculate_hypergraph_laplacian(H, weights)
        b0, b1 = calculate_betti_numbers(L)
        report.betti_0 = b0
        report.betti_1 = b1

        # 3. Structural Entropy
        # Entropy of the normalized Laplacian spectrum
        eigvals = np.linalg.eigvalsh(L)
        eig_probs = eigvals / (np.sum(eigvals) + 1e-9)
        eig_probs = eig_probs[eig_probs > 0]
        report.structural_entropy = float(-np.sum(eig_probs * np.log2(eig_probs)))

        # 4. Unit-Level Analysis
        total_vulnerability = 0.0

        for unit in units:
            unit_ratings = []
            valid_indices = [i for i in unit.player_indices if 0 <= i < n_players]

            if valid_indices:
                unit_ratings = player_ratings[valid_indices]
            else:
                unit_ratings = np.array([50.0]) # Default if empty

            # Calculate Unit Metrics
            unit.entropy = calculate_hyperedge_entropy(unit_ratings)

            # Cohesion: Inverse of variance (Homogeneity) + Mean Rating impact
            # Higher mean -> Better. Lower Variance -> Better.
            mean_rating = np.mean(unit_ratings)
            std_rating = np.std(unit_ratings)
            unit.cohesion_score = (mean_rating / 100.0) * (1.0 / (1.0 + std_rating/10.0))

            # Failure Probability (Sigmoid of inverted cohesion)
            # Low cohesion -> High failure probability
            fail_prob = 1.0 / (1.0 + np.exp(10 * (unit.cohesion_score - 0.4)))

            total_vulnerability += fail_prob * unit.weight

            # Create Analysis Record
            analysis = UnitAnalysis(
                unit=unit,
                failure_prob=float(fail_prob),
                entropy_contribution=unit.entropy * unit.weight,
                weakest_link=f"Player_{np.argmin(unit_ratings)}" if len(unit_ratings)>0 else "None"
            )
            report.unit_analyses.append(analysis)

            # Alerts
            if unit.unit_type == "defense" and fail_prob > 0.6:
                report.defense_alert = True
            if unit.unit_type == "midfield" and fail_prob > 0.5:
                report.midfield_alert = True

        # 5. Aggregation
        total_weight = np.sum(weights)
        report.vulnerability_index = float(total_vulnerability / total_weight) if total_weight > 0 else 0.0
        report.team_cohesion = float(np.mean([u.cohesion_score for u in units]))

        # Recommendation Logic
        if report.defense_alert:
            report.recommendation = "HIGH RISK: Defensive Structure Critical."
        elif report.betti_0 > 2:
            report.recommendation = "CAUTION: Disconnected Team Structure (High Betti-0)."
        elif report.structural_entropy > 2.5:
            report.recommendation = "WARNING: High Structural Disorder."
        else:
            report.recommendation = "Structure Stable."

        return report
