"""
quantum_annealer.py – Simulated Annealing (Kuantum Benzeri Optimizasyon).

100 tane "Value Maç" var, kasa sınırlı. Hangi kombinasyon (Kupon)
riski en aza indirip karı maksimize eder? Bu NP-Hard bir problem.

Simulated Annealing: Metallerin ısıtılıp yavaşça soğutulması
(Tavlama) mantığını taklit eder.
  1. Yüksek sıcaklık → rastgele çözümler kabul eder
  2. Sıcaklık düştükçe → sadece iyi çözümleri kabul eder
  3. Global minimum'a ulaşır, yerel tuzaklara düşmez

Amaç: Sharpe Oranını maksimize eden "En Mükemmel Kupon Sepeti"
bulmak.

Teknoloji: simanneal veya scipy.optimize.dual_annealing
Fallback: Manuel Metropolis-Hastings implementasyonu
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from loguru import logger

try:
    from scipy.optimize import dual_annealing
    SCIPY_ANNEAL_OK = True
except ImportError:
    SCIPY_ANNEAL_OK = False

try:
    from simanneal import Annealer
    SIMANNEAL_OK = True
except ImportError:
    SIMANNEAL_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class BetCandidate:
    """Bahis adayı."""
    match_id: str = ""
    selection: str = ""
    odds: float = 1.5
    prob: float = 0.5
    value_edge: float = 0.0
    risk: float = 0.0              # 0–1 arası risk skoru
    expected_return: float = 0.0   # Beklenen getiri
    league: str = ""
    correlation_group: str = ""    # Korelasyon grubu (aynı lig vs.)


@dataclass
class PortfolioSolution:
    """Optimizasyon çözümü."""
    selected_indices: list[int] = field(default_factory=list)
    selected_matches: list[str] = field(default_factory=list)
    total_stake: float = 0.0
    expected_profit: float = 0.0
    expected_risk: float = 0.0
    sharpe_ratio: float = 0.0
    diversification: float = 0.0     # 0–1 (1=çok çeşitli)
    n_bets: int = 0
    # Meta
    iterations: int = 0
    temperature_final: float = 0.0
    method: str = ""
    elapsed_ms: float = 0.0
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  AMAÇ FONKSİYONU (Objective)
# ═══════════════════════════════════════════════
def portfolio_objective(selection: np.ndarray,
                         candidates: list[BetCandidate],
                         bankroll: float = 10000.0,
                         max_bets: int = 10,
                         max_risk: float = 0.15,
                         corr_penalty: float = 0.3) -> float:
    """Portföy amaç fonksiyonu (minimize edilecek = negatif Sharpe).

    selection: binary array (0/1), her aday için seçim kararı
    """
    selected_idx = np.where(selection > 0.5)[0]

    if len(selected_idx) == 0:
        return 1e6  # Boş portföy cezası

    if len(selected_idx) > max_bets:
        return 1e6 + len(selected_idx)  # Fazla bahis cezası

    # Beklenen getiri
    total_return = 0.0
    total_risk = 0.0
    groups: dict[str, int] = {}

    for idx in selected_idx:
        c = candidates[idx]
        stake = bankroll * 0.02  # %2 sabit stake
        expected = stake * c.expected_return
        total_return += expected
        total_risk += c.risk * stake / bankroll

        # Korelasyon grubu kontrolü
        grp = c.correlation_group or c.league
        groups[grp] = groups.get(grp, 0) + 1

    # Risk cezası
    if total_risk > max_risk:
        risk_penalty = (total_risk - max_risk) * 100
    else:
        risk_penalty = 0.0

    # Korelasyon cezası (aynı gruptan çok bahis)
    corr_pen = 0.0
    for grp, count in groups.items():
        if count > 2:
            corr_pen += (count - 2) * corr_penalty

    # Çeşitlilik bonusu
    n_groups = len(groups)
    diversity_bonus = n_groups * 0.05

    # Sharpe benzeri metrik
    if total_risk > 0:
        sharpe = (total_return - risk_penalty - corr_pen + diversity_bonus) / total_risk
    else:
        sharpe = total_return

    # Minimize: negatif Sharpe
    return -sharpe


# ═══════════════════════════════════════════════
#  MANUAL SIMULATED ANNEALING
# ═══════════════════════════════════════════════
def manual_annealing(candidates: list[BetCandidate],
                      bankroll: float = 10000.0,
                      max_bets: int = 10,
                      max_risk: float = 0.15,
                      T_start: float = 100.0,
                      T_min: float = 0.01,
                      alpha: float = 0.995,
                      max_iter: int = 10000) -> tuple[np.ndarray, float]:
    """Manuel Metropolis-Hastings Simulated Annealing.

    Returns: (best_solution, best_energy)
    """
    n = len(candidates)
    if n == 0:
        return np.array([]), 0.0

    # Başlangıç: rastgele seçim
    current = np.zeros(n)
    n_initial = min(max_bets, n, max(1, n // 3))
    initial_idx = random.sample(range(n), n_initial)
    for idx in initial_idx:
        current[idx] = 1

    obj_fn = lambda s: portfolio_objective(
        s, candidates, bankroll, max_bets, max_risk,
    )

    current_energy = obj_fn(current)
    best = current.copy()
    best_energy = current_energy
    T = T_start

    for iteration in range(max_iter):
        # Komşu çözüm: rastgele bir biti çevir
        neighbor = current.copy()
        flip_idx = random.randint(0, n - 1)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]

        neighbor_energy = obj_fn(neighbor)
        delta = neighbor_energy - current_energy

        # Kabul kriteri (Metropolis)
        if delta < 0:
            # Daha iyi → her zaman kabul
            current = neighbor
            current_energy = neighbor_energy
        else:
            # Daha kötü → olasılıkla kabul (yüksek T'de daha olası)
            prob = math.exp(-delta / max(T, 1e-15))
            if random.random() < prob:
                current = neighbor
                current_energy = neighbor_energy

        # En iyi güncelle
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy

        # Soğutma
        T *= alpha

        if T < T_min:
            break

    return best, best_energy


# ═══════════════════════════════════════════════
#  SIMANNEAL WRAPPER
# ═══════════════════════════════════════════════
if SIMANNEAL_OK:
    class _BetPortfolioAnnealer(Annealer):
        """simanneal kütüphanesi wrapper."""

        def __init__(self, candidates, bankroll, max_bets, max_risk):
            n = len(candidates)
            initial = [0] * n
            for i in random.sample(range(n), min(max_bets, n)):
                initial[i] = 1
            super().__init__(initial)
            self.candidates = candidates
            self.bankroll = bankroll
            self.max_bets = max_bets
            self.max_risk = max_risk

        def move(self):
            idx = random.randint(0, len(self.state) - 1)
            self.state[idx] = 1 - self.state[idx]

        def energy(self):
            return portfolio_objective(
                np.array(self.state),
                self.candidates,
                self.bankroll,
                self.max_bets,
                self.max_risk,
            )


# ═══════════════════════════════════════════════
#  QUANTUM ANNEALER (Ana Sınıf)
# ═══════════════════════════════════════════════
class QuantumAnnealer:
    """Simulated Annealing ile optimal portföy seçimi.

    Kullanım:
        qa = QuantumAnnealer(bankroll=10000)

        candidates = [
            BetCandidate("m1", "home", 1.8, 0.6, 0.08, 0.3, 0.12, "EPL"),
            BetCandidate("m2", "over25", 1.9, 0.55, 0.05, 0.2, 0.10, "SL"),
            ...
        ]

        solution = qa.optimize(candidates, max_bets=8)
        print(solution.sharpe_ratio, solution.selected_matches)
    """

    def __init__(self, bankroll: float = 10000.0, max_bets: int = 10,
                 max_risk: float = 0.15, max_iter: int = 10000):
        self._bankroll = bankroll
        self._max_bets = max_bets
        self._max_risk = max_risk
        self._max_iter = max_iter
        logger.debug(
            f"[Annealer] Başlatıldı: bankroll={bankroll}, "
            f"max_bets={max_bets}, max_risk={max_risk}"
        )

    def optimize(self, candidates: list[BetCandidate],
                  max_bets: int | None = None,
                  max_risk: float | None = None) -> PortfolioSolution:
        """Optimal kupon sepetini bul."""
        t0 = time.perf_counter()
        max_bets = max_bets or self._max_bets
        max_risk = max_risk or self._max_risk
        sol = PortfolioSolution()

        if not candidates:
            sol.recommendation = "Aday bahis yok."
            return sol

        # Yöntem seç
        if SIMANNEAL_OK and len(candidates) <= 200:
            sol = self._optimize_simanneal(candidates, max_bets, max_risk)
        elif SCIPY_ANNEAL_OK and len(candidates) <= 30:
            sol = self._optimize_scipy(candidates, max_bets, max_risk)
        else:
            sol = self._optimize_manual(candidates, max_bets, max_risk)

        sol.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        sol.recommendation = self._advice(sol)
        return sol

    def _optimize_manual(self, candidates: list[BetCandidate],
                           max_bets: int, max_risk: float
                           ) -> PortfolioSolution:
        """Manuel Simulated Annealing."""
        best, best_energy = manual_annealing(
            candidates, self._bankroll, max_bets, max_risk,
            max_iter=self._max_iter,
        )

        return self._build_solution(
            best, candidates, best_energy, "manual_annealing",
        )

    def _optimize_simanneal(self, candidates: list[BetCandidate],
                               max_bets: int, max_risk: float
                               ) -> PortfolioSolution:
        """simanneal kütüphanesi ile."""
        try:
            annealer = _BetPortfolioAnnealer(
                candidates, self._bankroll, max_bets, max_risk,
            )
            annealer.steps = self._max_iter
            annealer.Tmax = 100.0
            annealer.Tmin = 0.01
            annealer.updates = 0  # Çıktıyı sustur
            state, energy = annealer.anneal()
            return self._build_solution(
                np.array(state), candidates, energy, "simanneal",
            )
        except Exception:
            return self._optimize_manual(candidates, max_bets, max_risk)

    def _optimize_scipy(self, candidates: list[BetCandidate],
                           max_bets: int, max_risk: float
                           ) -> PortfolioSolution:
        """scipy.optimize.dual_annealing ile."""
        n = len(candidates)
        bounds = [(0, 1)] * n

        def obj(x):
            binary = (x > 0.5).astype(float)
            return portfolio_objective(
                binary, candidates, self._bankroll, max_bets, max_risk,
            )

        try:
            result = dual_annealing(
                obj, bounds, maxiter=min(self._max_iter, 1000),
                seed=42,
            )
            best = (result.x > 0.5).astype(float)
            return self._build_solution(
                best, candidates, result.fun, "scipy_dual_annealing",
            )
        except Exception:
            return self._optimize_manual(candidates, max_bets, max_risk)

    def _build_solution(self, selection: np.ndarray,
                          candidates: list[BetCandidate],
                          energy: float,
                          method: str) -> PortfolioSolution:
        """Çözümü yapılandır."""
        sol = PortfolioSolution(method=method)
        selected_idx = np.where(selection > 0.5)[0].tolist()

        sol.selected_indices = selected_idx
        sol.selected_matches = [
            candidates[i].match_id for i in selected_idx
        ]
        sol.n_bets = len(selected_idx)

        # Metrikleri hesapla
        total_return = 0.0
        total_risk = 0.0
        groups: set[str] = set()

        for idx in selected_idx:
            c = candidates[idx]
            stake = self._bankroll * 0.02
            sol.total_stake += stake
            total_return += stake * c.expected_return
            total_risk += c.risk
            groups.add(c.correlation_group or c.league or "default")

        sol.expected_profit = round(total_return, 2)
        sol.expected_risk = round(total_risk / max(len(selected_idx), 1), 4)
        sol.sharpe_ratio = round(-energy, 4) if energy != 0 else 0.0
        sol.diversification = round(
            len(groups) / max(sol.n_bets, 1), 4,
        )

        return sol

    def optimize_from_bets(self, bets: list[dict]) -> PortfolioSolution:
        """Dict formatındaki bahislerden optimize et."""
        candidates = []
        for b in bets:
            if not isinstance(b, dict):
                continue
            candidates.append(BetCandidate(
                match_id=b.get("match_id", ""),
                selection=b.get("selection", ""),
                odds=b.get("odds", 1.5),
                prob=b.get("confidence", 0.5),
                value_edge=b.get("value_edge", 0.0),
                risk=b.get("risk", 0.3),
                expected_return=b.get("ev", 0.0),
                league=b.get("league", ""),
                correlation_group=b.get("league", ""),
            ))
        return self.optimize(candidates)

    def _advice(self, sol: PortfolioSolution) -> str:
        if sol.n_bets == 0:
            return "Uygun kombinasyon bulunamadı. Aday havuzunu genişletin."
        if sol.sharpe_ratio > 2.0:
            return (
                f"Mükemmel portföy: {sol.n_bets} bahis, "
                f"Sharpe={sol.sharpe_ratio:.2f}, "
                f"çeşitlilik={sol.diversification:.0%}."
            )
        if sol.sharpe_ratio > 1.0:
            return (
                f"İyi portföy: {sol.n_bets} bahis, "
                f"Sharpe={sol.sharpe_ratio:.2f}."
            )
        return (
            f"Zayıf portföy: Sharpe={sol.sharpe_ratio:.2f}. "
            f"Daha yüksek value-edge'li maçlar bekleyin."
        )
