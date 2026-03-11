"""
rust_engine.py – Rust-Powered Core (Demir Çekirdek).

Python'daki en ağır hesaplamaları Rust'a devreder.
Derlenmiş Rust modülü varsa onu kullanır, yoksa
Numba JIT ile hızlandırılmış Python fallback çalışır.

Kapsam:
  - Monte Carlo simülasyon (Digital Twin)
  - Fluid Dynamics saha hesaplaması (Pitch Control)
  - Matris çarpımları (Kelly, Portföy)
  - Poisson / Dixon-Coles olasılık hesaplaması

Teknoloji:
  - Rust + PyO3 + Maturin (derleme aracı)
  - Fallback: Numba @njit + NumPy vectorized

Rust Modülü Yapısı (Cargo.toml):
  [lib]
  name = "quant_core"
  crate-type = ["cdylib"]
  [dependencies]
  pyo3 = { version = "0.21", features = ["extension-module"] }
  ndarray = "0.15"
  rand = "0.8"
  rayon = "1.10"

Rust derleme (terminal):
  cd src/core/rust_engine && maturin develop --release

Python kullanım:
  from src.core.rust_engine import RustEngine
  engine = RustEngine()
  result = engine.monte_carlo_sim(n_sims=100_000, ...)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

# Rust modülünü dene
RUST_OK = False
_rust_mod: Any = None
try:
    import quant_core as _rust_mod  # type: ignore[import-untyped]
    RUST_OK = True
    logger.info("[Rust] quant_core Rust modülü yüklendi! 🦀")
except ImportError:
    logger.debug("[Rust] quant_core bulunamadı – Numba/NumPy fallback.")

# Numba JIT fallback
NUMBA_OK = False
try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    pass


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class BenchmarkResult:
    """Benchmark sonucu."""
    function: str = ""
    rust_ms: float = 0.0
    python_ms: float = 0.0
    speedup: float = 1.0
    engine: str = ""   # "rust" | "numba" | "numpy"


@dataclass
class RustReport:
    """Rust motor raporu."""
    engine: str = ""
    total_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  NUMBA JIT KERNELS (Fallback)
# ═══════════════════════════════════════════════
if NUMBA_OK:
    @njit(cache=True)
    def _mc_sim_numba(n_sims: int, home_xg: float, away_xg: float,
                        rho: float = 0.0) -> tuple:
        """Numba-hızlandırılmış Monte Carlo simülasyonu."""
        home_wins = 0
        draws = 0
        away_wins = 0
        over25 = 0
        total_goals = np.zeros(n_sims, dtype=np.float64)

        for i in prange(n_sims):
            # Poisson örnekleme (inverse CDF)
            h_goals = 0
            p = np.exp(-home_xg)
            s = p
            u = np.random.random()
            while u > s:
                h_goals += 1
                p *= home_xg / h_goals
                s += p

            a_goals = 0
            p = np.exp(-away_xg)
            s = p
            u = np.random.random()
            while u > s:
                a_goals += 1
                p *= away_xg / a_goals
                s += p

            total_goals[i] = h_goals + a_goals

            if h_goals > a_goals:
                home_wins += 1
            elif h_goals == a_goals:
                draws += 1
            else:
                away_wins += 1

            if h_goals + a_goals > 2:
                over25 += 1

        return (
            home_wins / n_sims,
            draws / n_sims,
            away_wins / n_sims,
            over25 / n_sims,
            np.mean(total_goals),
            np.std(total_goals),
        )

    @njit(cache=True)
    def _kelly_batch_numba(probs: np.ndarray,
                              odds: np.ndarray) -> np.ndarray:
        """Numba-hızlandırılmış toplu Kelly hesaplama."""
        n = len(probs)
        stakes = np.zeros(n, dtype=np.float64)
        for i in range(n):
            b = odds[i]
            p = probs[i]
            if b > 1 and p > 0:
                f = (p * b - 1) / (b - 1)
                stakes[i] = max(0.0, min(f, 0.25))
        return stakes

    @njit(cache=True)
    def _influence_field_numba(grid_h: int, grid_w: int,
                                  px: float, py: float,
                                  vx: float, vy: float,
                                  sigma: float,
                                  intensity: float) -> np.ndarray:
        """Numba-hızlandırılmış oyuncu etki alanı hesaplama."""
        field = np.zeros((grid_h, grid_w), dtype=np.float64)
        speed = np.sqrt(vx**2 + vy**2)

        for r in range(grid_h):
            for c in range(grid_w):
                dx = c - px
                dy = r - py
                # Anizotropik sigma (hız yönünde genişler)
                if speed > 0.5:
                    cos_a = vx / speed
                    sin_a = vy / speed
                    dx_rot = dx * cos_a + dy * sin_a
                    dy_rot = -dx * sin_a + dy * cos_a
                    sx = sigma * (1 + speed * 0.3)
                    sy = sigma
                    dist_sq = (dx_rot / sx) ** 2 + (dy_rot / sy) ** 2
                else:
                    dist_sq = (dx**2 + dy**2) / (sigma**2)

                field[r, c] = intensity * np.exp(-0.5 * dist_sq)

        return field


# ═══════════════════════════════════════════════
#  NUMPY FALLBACK KERNELS
# ═══════════════════════════════════════════════
def _mc_sim_numpy(n_sims: int, home_xg: float, away_xg: float,
                    rho: float = 0.0) -> tuple:
    """NumPy vectorized Monte Carlo."""
    h = np.random.poisson(home_xg, n_sims)
    a = np.random.poisson(away_xg, n_sims)
    total = h + a
    return (
        float(np.mean(h > a)),
        float(np.mean(h == a)),
        float(np.mean(h < a)),
        float(np.mean(total > 2)),
        float(np.mean(total)),
        float(np.std(total)),
    )


def _kelly_batch_numpy(probs: np.ndarray,
                          odds: np.ndarray) -> np.ndarray:
    """NumPy vectorized Kelly."""
    b = odds
    p = probs
    f = np.where(b > 1, (p * b - 1) / (b - 1), 0.0)
    return np.clip(f, 0, 0.25)


def _influence_field_numpy(grid_h: int, grid_w: int,
                              px: float, py: float,
                              vx: float, vy: float,
                              sigma: float,
                              intensity: float) -> np.ndarray:
    """NumPy vectorized etki alanı."""
    y, x = np.mgrid[0:grid_h, 0:grid_w]
    dx = x.astype(np.float64) - px
    dy = y.astype(np.float64) - py
    dist_sq = (dx**2 + dy**2) / max(sigma**2, 1e-6)
    return intensity * np.exp(-0.5 * dist_sq)


# ═══════════════════════════════════════════════
#  RUST ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class RustEngine:
    """Rust/Numba/NumPy hızlandırılmış hesaplama motoru.

    Kullanım:
        engine = RustEngine()

        # Monte Carlo simülasyon
        result = engine.monte_carlo_sim(100_000, 1.5, 0.8)

        # Toplu Kelly
        stakes = engine.kelly_batch(probs, odds)

        # Pitch control (etki alanı)
        field = engine.player_influence_field(68, 105, 50, 34, 3.0, 1.5, 5.0)

        # Benchmark
        report = engine.benchmark()
    """

    def __init__(self):
        if RUST_OK:
            self._engine = "rust"
        elif NUMBA_OK:
            self._engine = "numba"
        else:
            self._engine = "numpy"

        self._call_count = 0
        self._total_ms = 0.0

        logger.debug(f"[Rust] Motor: {self._engine}")

    @property
    def engine_name(self) -> str:
        return self._engine

    # ── Monte Carlo Simülasyon ──
    def monte_carlo_sim(self, n_sims: int = 100_000,
                          home_xg: float = 1.5,
                          away_xg: float = 0.8,
                          rho: float = 0.0) -> dict:
        """Hızlandırılmış Monte Carlo maç simülasyonu."""
        t0 = time.perf_counter()
        self._call_count += 1

        if self._engine == "rust" and _rust_mod:
            try:
                result = _rust_mod.monte_carlo_sim(n_sims, home_xg, away_xg, rho)
                elapsed = (time.perf_counter() - t0) * 1000
                self._total_ms += elapsed
                return {**result, "engine": "rust", "elapsed_ms": elapsed}
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        if self._engine == "numba" and NUMBA_OK:
            hw, dw, aw, o25, mean_g, std_g = _mc_sim_numba(
                n_sims, home_xg, away_xg, rho,
            )
        else:
            hw, dw, aw, o25, mean_g, std_g = _mc_sim_numpy(
                n_sims, home_xg, away_xg, rho,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        self._total_ms += elapsed

        return {
            "home_win": round(float(hw), 4),
            "draw": round(float(dw), 4),
            "away_win": round(float(aw), 4),
            "over_2_5": round(float(o25), 4),
            "mean_goals": round(float(mean_g), 3),
            "std_goals": round(float(std_g), 3),
            "n_sims": n_sims,
            "engine": self._engine,
            "elapsed_ms": round(elapsed, 2),
        }

    # ── Toplu Kelly Hesaplama ──
    def kelly_batch(self, probs: np.ndarray | list,
                      odds: np.ndarray | list) -> np.ndarray:
        """Hızlandırılmış toplu Kelly Kriteri."""
        t0 = time.perf_counter()
        self._call_count += 1
        p = np.asarray(probs, dtype=np.float64)
        o = np.asarray(odds, dtype=np.float64)

        if self._engine == "rust" and _rust_mod:
            try:
                result = np.array(_rust_mod.kelly_batch(p.tolist(), o.tolist()))
                self._total_ms += (time.perf_counter() - t0) * 1000
                return result
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        if self._engine == "numba" and NUMBA_OK:
            result = _kelly_batch_numba(p, o)
        else:
            result = _kelly_batch_numpy(p, o)

        self._total_ms += (time.perf_counter() - t0) * 1000
        return result

    # ── Oyuncu Etki Alanı (Pitch Control) ──
    def player_influence_field(self, grid_h: int, grid_w: int,
                                  px: float, py: float,
                                  vx: float = 0.0, vy: float = 0.0,
                                  sigma: float = 5.0,
                                  intensity: float = 1.0) -> np.ndarray:
        """Hızlandırılmış oyuncu etki alanı hesaplama."""
        t0 = time.perf_counter()
        self._call_count += 1

        if self._engine == "rust" and _rust_mod:
            try:
                result = np.array(_rust_mod.influence_field(
                    grid_h, grid_w, px, py, vx, vy, sigma, intensity,
                ))
                self._total_ms += (time.perf_counter() - t0) * 1000
                return result
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        if self._engine == "numba" and NUMBA_OK:
            result = _influence_field_numba(
                grid_h, grid_w, px, py, vx, vy, sigma, intensity,
            )
        else:
            result = _influence_field_numpy(
                grid_h, grid_w, px, py, vx, vy, sigma, intensity,
            )

        self._total_ms += (time.perf_counter() - t0) * 1000
        return result

    # ── Matris Çarpım (Portföy) ──
    def mat_mul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Hızlandırılmış matris çarpımı."""
        t0 = time.perf_counter()
        self._call_count += 1

        if self._engine == "rust" and _rust_mod:
            try:
                result = np.array(_rust_mod.mat_mul(
                    A.tolist(), B.tolist(),
                ))
                self._total_ms += (time.perf_counter() - t0) * 1000
                return result
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        result = A @ B
        self._total_ms += (time.perf_counter() - t0) * 1000
        return result

    # ── Benchmark ──
    def benchmark(self, n_sims: int = 100_000) -> RustReport:
        """Motor benchmark raporu."""
        report = RustReport(engine=self._engine)
        benchmarks = []

        # Monte Carlo benchmark
        for label, n in [("MC_10K", 10_000), ("MC_100K", 100_000)]:
            t0 = time.perf_counter()
            self.monte_carlo_sim(n, 1.5, 0.8)
            elapsed = (time.perf_counter() - t0) * 1000
            benchmarks.append(BenchmarkResult(
                function=label,
                python_ms=elapsed if self._engine == "numpy" else 0,
                rust_ms=elapsed if self._engine == "rust" else 0,
                engine=self._engine,
            ))

        # Kelly benchmark
        p = np.random.uniform(0.3, 0.7, 1000)
        o = np.random.uniform(1.2, 5.0, 1000)
        t0 = time.perf_counter()
        self.kelly_batch(p, o)
        elapsed = (time.perf_counter() - t0) * 1000
        benchmarks.append(BenchmarkResult(
            function="Kelly_1K", engine=self._engine,
            python_ms=elapsed if self._engine == "numpy" else 0,
            rust_ms=elapsed if self._engine == "rust" else 0,
        ))

        # Influence field benchmark
        t0 = time.perf_counter()
        self.player_influence_field(68, 105, 50, 34, 3.0, 1.5, 5.0)
        elapsed = (time.perf_counter() - t0) * 1000
        benchmarks.append(BenchmarkResult(
            function="PitchField", engine=self._engine,
            python_ms=elapsed if self._engine == "numpy" else 0,
            rust_ms=elapsed if self._engine == "rust" else 0,
        ))

        report.benchmarks = benchmarks
        report.total_calls = self._call_count
        report.total_time_ms = round(self._total_ms, 2)
        report.avg_time_ms = round(
            self._total_ms / max(self._call_count, 1), 2,
        )

        if self._engine == "rust":
            report.recommendation = (
                "Rust çekirdeği aktif – maksimum performans. 🦀"
            )
        elif self._engine == "numba":
            report.recommendation = (
                "Numba JIT aktif – iyi performans. "
                "Rust için: cd src/core/rust_engine && maturin develop --release"
            )
        else:
            report.recommendation = (
                "Saf NumPy – temel performans. "
                "Numba veya Rust yükleyin: pip install numba"
            )

        return report
