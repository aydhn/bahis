"""
path_signature_engine.py – iisignature ile oran hareketlerinin geometrik imzası.
Rough Path teorisi: zaman serisinin "parmak izini" çıkarır.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    import iisignature
    IIS_AVAILABLE = True
except ImportError:
    IIS_AVAILABLE = False
    logger.warning("iisignature yüklü değil – PathSignature basit modda.")


class PathSignatureEngine:
    """Oran hareketlerinin path signature özelliklerini çıkarır."""

    def __init__(self, depth: int = 3):
        self._depth = depth
        logger.debug("PathSignatureEngine başlatıldı.")

    def extract(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için path signature feature'ları çıkarır."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")

            # Oran verisini yol (path) olarak kodla
            path = self._build_path(row)
            sig_features = self._compute_signature(path)

            results.append({
                "match_id": mid,
                "sig_length": float(sig_features.get("path_length", 0)),
                "sig_roughness": float(sig_features.get("roughness", 0)),
                "sig_trend": float(sig_features.get("trend", 0)),
                "sig_mean_return": float(sig_features.get("mean_return", 0)),
                "sig_volatility_sig": float(sig_features.get("vol_signature", 0)),
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _build_path(self, row: dict) -> np.ndarray:
        """Oran verisini çok boyutlu yola dönüştürür."""
        ho = row.get("home_odds", 2.5) or 2.5
        do_ = row.get("draw_odds", 3.3) or 3.3
        ao = row.get("away_odds", 3.0) or 3.0
        o25 = row.get("over25_odds", 1.9) or 1.9
        u25 = row.get("under25_odds", 1.9) or 1.9

        # İmplied probabilities ile normalizasyon
        probs = np.array([1/ho, 1/do_, 1/ao])
        probs /= probs.sum()

        # 2D yol: (implied_home, implied_away) üzerinden
        # Basit sentetik yol (gerçek zaman serisi varsa genişler)
        t = np.linspace(0, 1, 10)
        path = np.column_stack([
            probs[0] + 0.02 * np.sin(2 * np.pi * t),
            probs[2] + 0.015 * np.cos(2 * np.pi * t),
            np.linspace(1/o25, 1/u25, 10),
        ])

        return path

    def _compute_signature(self, path: np.ndarray) -> dict:
        """Path signature hesaplar."""
        if path.shape[0] < 2:
            return {"path_length": 0, "roughness": 0, "trend": 0, "mean_return": 0, "vol_signature": 0}

        if IIS_AVAILABLE:
            return self._compute_iisig(path)
        else:
            return self._compute_manual(path)

    def _compute_iisig(self, path: np.ndarray) -> dict:
        """iisignature kütüphanesi ile tam signature."""
        try:
            sig = iisignature.sig(path, self._depth)

            # Signature boyutundan feature'lar çıkar
            path_length = float(np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))))
            roughness = float(np.std(sig[:path.shape[1]]))
            trend = float(np.mean(sig[:path.shape[1]]))
            vol_signature = float(np.std(sig))

            # Log-signature (daha kompakt)
            logsig = iisignature.logsig(path, iisignature.prepare(path.shape[1], self._depth))
            mean_return = float(np.mean(logsig[:path.shape[1]]))

            return {
                "path_length": path_length,
                "roughness": roughness,
                "trend": trend,
                "mean_return": mean_return,
                "vol_signature": vol_signature,
            }
        except Exception as e:
            logger.debug(f"iisignature hesaplama hatası: {e}")
            return self._compute_manual(path)

    def _compute_manual(self, path: np.ndarray) -> dict:
        """iisignature olmadan temel geometrik özellikler."""
        diffs = np.diff(path, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        path_length = float(np.sum(lengths))
        roughness = float(np.std(lengths))
        trend = float(path[-1, 0] - path[0, 0])
        returns = diffs[:, 0] / (np.abs(path[:-1, 0]) + 1e-8)
        mean_return = float(np.mean(returns))
        vol_signature = float(np.std(returns))

        return {
            "path_length": path_length,
            "roughness": roughness,
            "trend": trend,
            "mean_return": mean_return,
            "vol_signature": vol_signature,
        }
