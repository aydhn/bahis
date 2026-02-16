"""
jump_diffusion_model.py – Merton Jump-Diffusion modeli.
Oranlardaki ani şokları (jumps) matematiksel olarak ayrıştırır.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats, optimize
from loguru import logger


class JumpDiffusionModel:
    """Merton modeli: diffusion (normal hareket) + jump (şok) ayrıştırması."""

    def __init__(self, dt: float = 1.0 / 252):
        self._dt = dt  # Zaman adımı
        logger.debug("JumpDiffusionModel başlatıldı.")

    def detect(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için oran hareketlerinde jump tespiti yapar."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")

            # Oran hareketlerini simüle et (gerçek veri varsa replace edilir)
            odds_series = self._extract_odds_series(row)
            jump_analysis = self._analyze_jumps(odds_series)

            results.append({
                "match_id": mid,
                "has_jump": jump_analysis["has_jump"],
                "jump_intensity": jump_analysis["lambda"],
                "jump_mean": jump_analysis["mu_j"],
                "jump_vol": jump_analysis["sigma_j"],
                "diffusion_vol": jump_analysis["sigma"],
                "jump_probability": jump_analysis["jump_prob"],
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _extract_odds_series(self, row: dict) -> np.ndarray:
        """Oran verisinden zaman serisi oluşturur."""
        ho = row.get("home_odds", 2.5) or 2.5
        vol = row.get("odds_volatility", 0.05) or 0.05

        # Sentetik oran geçmişi (gerçek veri ile değiştirilecek)
        np.random.seed(hash(row.get("match_id", "")) % 2**31)
        n = 50
        base = np.log(ho)
        returns = np.random.normal(0, vol, n)

        # Rastgele jump ekle
        jump_times = np.random.poisson(0.3, n)
        jump_sizes = np.random.normal(-0.05, 0.1, n) * jump_times

        log_prices = base + np.cumsum(returns + jump_sizes)
        return np.exp(log_prices)

    def _analyze_jumps(self, prices: np.ndarray) -> dict:
        """Fiyat serisinde jump-diffusion parametrelerini tahmin eder."""
        if len(prices) < 5:
            return self._default_params()

        log_returns = np.diff(np.log(prices + 1e-8))

        if len(log_returns) < 5 or np.std(log_returns) < 1e-10:
            return self._default_params()

        # Jump tespiti: returns'ün normal dağılımdan sapması
        _, pval_normal = stats.normaltest(log_returns)
        kurtosis = stats.kurtosis(log_returns)

        # Yüksek kurtosis = fat tails = muhtemel jumplar
        has_jump = (pval_normal < 0.05) or (abs(kurtosis) > 3)

        # MLE ile parametre tahmini
        params = self._mle_fit(log_returns)

        # Jump olasılığı (anlık)
        jump_prob = 1 - np.exp(-params["lambda"] * self._dt) if params["lambda"] > 0 else 0

        return {
            "has_jump": has_jump,
            "lambda": params["lambda"],
            "mu_j": params["mu_j"],
            "sigma_j": params["sigma_j"],
            "sigma": params["sigma"],
            "jump_prob": float(np.clip(jump_prob, 0, 1)),
        }

    def _mle_fit(self, returns: np.ndarray) -> dict:
        """Maximum Likelihood ile Merton parametrelerini tahmin eder."""
        try:
            sigma = np.std(returns)
            mu = np.mean(returns)

            # Basit ayrıştırma: büyük hareketler = jump
            threshold = 2 * sigma
            jumps = returns[np.abs(returns) > threshold]
            diffusion = returns[np.abs(returns) <= threshold]

            lam = len(jumps) / max(len(returns), 1)  # Jump yoğunluğu
            mu_j = float(np.mean(jumps)) if len(jumps) > 0 else 0.0
            sigma_j = float(np.std(jumps)) if len(jumps) > 1 else sigma * 2
            sigma_d = float(np.std(diffusion)) if len(diffusion) > 1 else sigma

            return {
                "lambda": float(np.clip(lam, 0, 5)),
                "mu_j": mu_j,
                "sigma_j": sigma_j,
                "sigma": sigma_d,
            }
        except Exception:
            return {"lambda": 0.1, "mu_j": 0.0, "sigma_j": 0.05, "sigma": 0.02}

    def _default_params(self) -> dict:
        return {
            "has_jump": False,
            "lambda": 0.0,
            "mu_j": 0.0,
            "sigma_j": 0.0,
            "sigma": 0.0,
            "jump_prob": 0.0,
        }

    def simulate_path(self, s0: float, params: dict, n_steps: int = 100, n_paths: int = 1000) -> np.ndarray:
        """Monte Carlo: jump-diffusion fiyat yolları simülasyonu."""
        lam = params.get("lambda", 0.1)
        mu_j = params.get("mu_j", 0.0)
        sigma_j = params.get("sigma_j", 0.05)
        sigma = params.get("sigma", 0.02)
        mu = 0  # Risk-neutral

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = s0

        for t in range(n_steps):
            z = np.random.standard_normal(n_paths)
            jumps = np.random.poisson(lam * self._dt, n_paths)
            jump_sizes = np.random.normal(mu_j, sigma_j, n_paths) * jumps

            drift = (mu - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * self._dt
            diffusion = sigma * np.sqrt(self._dt) * z

            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion + jump_sizes)

        return paths
