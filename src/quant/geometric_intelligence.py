"""
geometric_intelligence.py – Clifford Algebra ile uzamsal analiz.
Oyuncuların hareket potansiyelini multivektörlerle hesaplar.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    from clifford import Cl
    CLIFFORD_AVAILABLE = True
except ImportError:
    CLIFFORD_AVAILABLE = False
    logger.warning("clifford yüklü değil – GeometricIntelligence basit modda.")


class GeometricIntelligence:
    """Clifford Algebra tabanlı uzamsal hareket ve potansiyel analizi."""

    def __init__(self):
        if CLIFFORD_AVAILABLE:
            self._layout, self._blades = Cl(3)  # 3D uzay
            self._e1, self._e2, self._e3 = self._blades["e1"], self._blades["e2"], self._blades["e3"]
        logger.debug("GeometricIntelligence başlatıldı.")

    def compute_potential(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için geometrik potansiyel metrikleri hesaplar."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")

            # Özellik uzayını geometrik olarak yorumla
            if CLIFFORD_AVAILABLE:
                potential = self._clifford_analysis(row)
            else:
                potential = self._vector_analysis(row)

            results.append({
                "match_id": mid,
                "geometric_edge": potential["edge"],
                "momentum_magnitude": potential["momentum"],
                "rotation_index": potential["rotation"],
                "spatial_dominance": potential["dominance"],
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _clifford_analysis(self, row: dict) -> dict:
        """Clifford Algebra ile tam geometrik analiz."""
        # Takım güçlerini multivektörlere kodla
        home_power = (
            row.get("home_xg", 1.3) * self._e1
            + row.get("home_win_rate", 0.4) * self._e2
            + row.get("home_possession", 50) / 100 * self._e3
        )
        away_power = (
            row.get("away_xg", 1.1) * self._e1
            + row.get("away_win_rate", 0.3) * self._e2
            + row.get("away_possession", 50) / 100 * self._e3
        )

        # Geometric product → üstünlük ölçütü
        product = home_power * away_power

        # Scalar part = iç çarpım (benzerlik)
        scalar = float(product.value[0])

        # Bivector part = dış çarpım (fark/dönme)
        bivector_norm = float(np.sqrt(sum(v**2 for v in product.value[4:7])))

        # Momentum: ev sahibi vektör büyüklüğü / deplasman
        home_mag = float(np.sqrt(sum(v**2 for v in home_power.value[1:4])))
        away_mag = float(np.sqrt(sum(v**2 for v in away_power.value[1:4])))

        edge = (home_mag - away_mag) / (home_mag + away_mag + 1e-8)
        momentum = home_mag - away_mag
        rotation = bivector_norm / (scalar + 1e-8)
        dominance = float(np.tanh(edge * 2))

        return {
            "edge": float(np.clip(edge, -1, 1)),
            "momentum": float(momentum),
            "rotation": float(np.clip(rotation, 0, 5)),
            "dominance": float(np.clip(dominance, -1, 1)),
        }

    def _vector_analysis(self, row: dict) -> dict:
        """Clifford olmadan basit vektörel analiz."""
        home_vec = np.array([
            row.get("home_xg", 1.3) or 1.3,
            row.get("home_win_rate", 0.4) or 0.4,
            (row.get("home_possession", 50) or 50) / 100,
        ])
        away_vec = np.array([
            row.get("away_xg", 1.1) or 1.1,
            row.get("away_win_rate", 0.3) or 0.3,
            (row.get("away_possession", 50) or 50) / 100,
        ])

        home_mag = np.linalg.norm(home_vec)
        away_mag = np.linalg.norm(away_vec)

        # Cosine similarity
        cos_sim = np.dot(home_vec, away_vec) / (home_mag * away_mag + 1e-8)

        # Cross product magnitude (2D rotation analogue)
        cross = np.cross(home_vec, away_vec)
        cross_mag = np.linalg.norm(cross)

        edge = (home_mag - away_mag) / (home_mag + away_mag + 1e-8)
        momentum = home_mag - away_mag
        rotation = cross_mag / (np.dot(home_vec, away_vec) + 1e-8)
        dominance = float(np.tanh(edge * 2))

        return {
            "edge": float(np.clip(edge, -1, 1)),
            "momentum": float(momentum),
            "rotation": float(np.clip(abs(rotation), 0, 5)),
            "dominance": float(np.clip(dominance, -1, 1)),
        }
