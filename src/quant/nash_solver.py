"""
nash_solver.py – Nash Dengesi ve Oyun Teorisi ile Bahis Stratejisi.

Bahis tek taraflı bir tahmin oyunu değil, çift taraflı bir
strateji oyunudur. Siz ve Bahis Bürosu (Bookmaker) iki oyuncusunuz.

Nash Dengesi: Öyle bir strateji bulun ki, bahis bürosu oranları
nasıl değiştirirse değiştirsin, uzun vadede "sömürülemez"
(Unexploitable) olun.

Oyuncular:
  - Oyuncu A: Bahis Bürosu (Bookmaker) → Oran ayarlama stratejisi
  - Oyuncu B: Bot (Bettor) → Bahis seçme ve stake stratejisi

Kavramlar:
  - Minimax: Kaybetme riskinin minimum olduğu strateji
  - Regret Minimization: Pişmanlığı minimuma indirgeme
  - Mixed Strategy: Sabit değil, olasılıksal strateji
  - Exploitability: Büronun bizi sömürebilme derecesi

Teknoloji: nashpy (Python Game Theory Library)
Fallback: scipy.optimize (linear programming)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    import nashpy as nash
    NASH_OK = True
except ImportError:
    NASH_OK = False
    logger.info("nashpy yüklü değil – linear programming fallback.")

try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


@dataclass
class NashEquilibrium:
    """Nash dengesi sonucu."""
    bettor_strategy: np.ndarray | None = None   # Bot'un optimal karışık stratejisi
    bookmaker_strategy: np.ndarray | None = None  # Büronun optimal karışık stratejisi
    game_value: float = 0.0                      # Oyunun değeri (bot için)
    exploitability: float = 0.0                   # Sömürülebilirlik (0 = mükemmel)
    method: str = ""
    action_names: list[str] = field(default_factory=list)
    interpretation: str = ""


@dataclass
class RegretState:
    """Regret Minimization durumu."""
    cumulative_regret: np.ndarray | None = None
    strategy_sum: np.ndarray | None = None
    current_strategy: np.ndarray | None = None
    iterations: int = 0
    avg_game_value: float = 0.0


@dataclass
class BettingGameAnalysis:
    """Bir maç için oyun teorisi analizi."""
    match_id: str = ""
    equilibrium: NashEquilibrium = field(default_factory=NashEquilibrium)
    optimal_action: str = ""         # "bet_home" | "bet_draw" | "bet_away" | "pass"
    optimal_stake_pct: float = 0.0
    expected_value: float = 0.0
    risk_adjusted_ev: float = 0.0
    is_exploitable: bool = False
    bookmaker_margin: float = 0.0
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  ÖDEME MATRİSİ OLUŞTURUCU
# ═══════════════════════════════════════════════
def build_payoff_matrix(model_probs: dict,
                         market_odds: dict,
                         stake: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Bahis oyunu ödeme matrisini oluştur.

    Bot aksiyonları (satırlar):
        0: Pas Geç
        1: Ev Sahibi Bahis
        2: Beraberlik Bahis
        3: Deplasman Bahis

    Doğa durumları / Büro tepkileri (sütunlar):
        0: Ev Sahibi Kazanır
        1: Beraberlik
        2: Deplasman Kazanır

    Returns:
        (bettor_payoff, bookmaker_payoff) matrisleri
    """
    ph = model_probs.get("prob_home", 0.33)
    pd = model_probs.get("prob_draw", 0.33)
    pa = model_probs.get("prob_away", 0.34)

    oh = market_odds.get("home", 2.0)
    od = market_odds.get("draw", 3.5)
    oa = market_odds.get("away", 4.0)

    # Bot'un ödeme matrisi (4 aksiyon x 3 durum)
    bettor = np.array([
        [0, 0, 0],                                         # Pas
        [stake * (oh - 1), -stake, -stake],                # Ev bahis
        [-stake, stake * (od - 1), -stake],                # Ber bahis
        [-stake, -stake, stake * (oa - 1)],                # Dep bahis
    ], dtype=np.float64)

    # Büronun ödeme matrisi (sıfır toplamlı oyun: büro = -bot)
    bookmaker = -bettor

    return bettor, bookmaker


# ═══════════════════════════════════════════════
#  NASH SOLVER
# ═══════════════════════════════════════════════
class NashGameSolver:
    """Nash Dengesi ile optimal bahis stratejisi.

    Kullanım:
        solver = NashGameSolver()
        # Maç analizi
        analysis = solver.analyze_match(
            model_probs={"prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20},
            market_odds={"home": 1.80, "draw": 3.50, "away": 4.50},
        )
        # Regret Minimization (iteratif öğrenme)
        strategy = solver.regret_minimization(payoff_matrix, n_iter=10000)
    """

    ACTION_NAMES = ["PAS", "EV BAHİS", "BER BAHİS", "DEP BAHİS"]

    def __init__(self):
        self._regret_states: dict[str, RegretState] = {}
        logger.debug("NashGameSolver başlatıldı.")

    # ═══════════════════════════════════════════
    #  NASH DENGESİ
    # ═══════════════════════════════════════════
    def find_equilibrium(self, bettor_payoff: np.ndarray,
                          bookmaker_payoff: np.ndarray
                          ) -> NashEquilibrium:
        """Nash dengesini bul."""
        result = NashEquilibrium(action_names=self.ACTION_NAMES)

        if NASH_OK:
            result = self._nashpy_solve(bettor_payoff, bookmaker_payoff)
        elif SCIPY_OK:
            result = self._linprog_solve(bettor_payoff)
        else:
            result = self._minimax_heuristic(bettor_payoff)

        return result

    def _nashpy_solve(self, bettor_pay: np.ndarray,
                       bookmaker_pay: np.ndarray) -> NashEquilibrium:
        """nashpy ile Nash dengesi."""
        result = NashEquilibrium(
            action_names=self.ACTION_NAMES,
            method="nashpy",
        )

        try:
            game = nash.Game(bettor_pay, bookmaker_pay)

            # Support enumeration
            with warnings.catch_warnings():
                # Degenerate oyunlarda nashpy/numpy RuntimeWarning spam üretebiliyor.
                # Bu blokta uyarıları bastırıp sonuç yoksa fallback'e gidiyoruz.
                warnings.simplefilter("ignore", RuntimeWarning)
                equilibria = list(game.support_enumeration())

            if equilibria:
                best_eq = max(equilibria,
                              key=lambda eq: eq[0] @ bettor_pay @ eq[1])
                result.bettor_strategy = best_eq[0]
                result.bookmaker_strategy = best_eq[1]
                result.game_value = float(
                    best_eq[0] @ bettor_pay @ best_eq[1]
                )
            else:
                # Vertex enumeration fallback
                try:
                    equilibria = list(game.vertex_enumeration())
                    if equilibria:
                        best_eq = equilibria[0]
                        result.bettor_strategy = best_eq[0]
                        result.bookmaker_strategy = best_eq[1]
                        result.game_value = float(
                            best_eq[0] @ bettor_pay @ best_eq[1]
                        )
                except Exception:
                    return self._minimax_heuristic(bettor_pay)

            # Exploitability hesapla
            if result.bettor_strategy is not None:
                result.exploitability = self._calc_exploitability(
                    bettor_pay, result.bettor_strategy,
                )

        except Exception as e:
            logger.debug(f"[Nash] nashpy hatası: {e}")
            return self._minimax_heuristic(bettor_pay)

        return result

    def _linprog_solve(self, bettor_pay: np.ndarray) -> NashEquilibrium:
        """Linear programming ile minimax."""
        result = NashEquilibrium(
            action_names=self.ACTION_NAMES,
            method="linprog",
        )

        try:
            n_actions = bettor_pay.shape[0]
            n_states = bettor_pay.shape[1]

            # Maximize minimum payoff: max v s.t. A'x >= v, sum(x)=1, x>=0
            c = np.zeros(n_actions + 1)
            c[-1] = -1  # maximize v

            A_ub = np.zeros((n_states, n_actions + 1))
            for j in range(n_states):
                for i in range(n_actions):
                    A_ub[j, i] = -bettor_pay[i, j]
                A_ub[j, -1] = 1
            b_ub = np.zeros(n_states)

            A_eq = np.zeros((1, n_actions + 1))
            A_eq[0, :n_actions] = 1
            b_eq = np.array([1.0])

            bounds = [(0, 1)] * n_actions + [(None, None)]

            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          A_eq=A_eq, b_eq=b_eq, bounds=bounds)

            if res.success:
                result.bettor_strategy = res.x[:n_actions]
                result.game_value = -res.fun

        except Exception as e:
            logger.debug(f"[Nash] linprog hatası: {e}")
            return self._minimax_heuristic(bettor_pay)

        return result

    def _minimax_heuristic(self, bettor_pay: np.ndarray) -> NashEquilibrium:
        """Heuristic minimax strateji."""
        result = NashEquilibrium(
            action_names=self.ACTION_NAMES,
            method="heuristic",
        )

        # Her aksiyonun minimum getirisi
        min_payoffs = np.min(bettor_pay, axis=1)
        # En yüksek minimum getirili aksiyonu seç
        best_action = int(np.argmax(min_payoffs))

        strategy = np.zeros(bettor_pay.shape[0])
        strategy[best_action] = 1.0
        result.bettor_strategy = strategy
        result.game_value = float(min_payoffs[best_action])

        return result

    def _calc_exploitability(self, payoff: np.ndarray,
                               strategy: np.ndarray) -> float:
        """Stratejinin sömürülebilirlik derecesi."""
        expected_per_state = strategy @ payoff
        worst_case = float(np.min(expected_per_state))
        best_case = float(np.max(expected_per_state))

        if best_case == 0:
            return 0.0
        return max(0, 1.0 - worst_case / max(best_case, 1e-8))

    # ═══════════════════════════════════════════
    #  REGRET MINIMIZATION
    # ═══════════════════════════════════════════
    def regret_minimization(self, payoff: np.ndarray,
                             n_iter: int = 10000,
                             match_id: str = "") -> RegretState:
        """Regret Minimization ile iteratif strateji öğrenme.

        Her turda:
          1. Mevcut stratejiye göre oyna
          2. Her aksiyonun "pişmanlığını" hesapla
          3. Stratejiyi pişmanlığa göre güncelle

        Yeterli iterasyon sonrası → Nash dengesi yakınsar.
        """
        n_actions = payoff.shape[0]
        n_states = payoff.shape[1]

        state = self._regret_states.get(match_id, RegretState(
            cumulative_regret=np.zeros(n_actions),
            strategy_sum=np.zeros(n_actions),
            current_strategy=np.ones(n_actions) / n_actions,
        ))

        total_value = 0.0

        for _ in range(n_iter):
            # Mevcut strateji (regret-matched)
            positive_regret = np.maximum(state.cumulative_regret, 0)
            regret_sum = np.sum(positive_regret)

            if regret_sum > 0:
                strategy = positive_regret / regret_sum
            else:
                strategy = np.ones(n_actions) / n_actions

            # Rastgele durum (doğanın seçimi)
            nature = np.random.randint(0, n_states)

            # Aksiyonun getirisi
            action_values = payoff[:, nature]
            strategy_value = float(np.dot(strategy, action_values))
            total_value += strategy_value

            # Regret güncelle
            for a in range(n_actions):
                regret = action_values[a] - strategy_value
                state.cumulative_regret[a] += regret

            state.strategy_sum += strategy
            state.iterations += 1

        # Ortalama strateji
        if np.sum(state.strategy_sum) > 0:
            state.current_strategy = state.strategy_sum / np.sum(state.strategy_sum)
        state.avg_game_value = total_value / max(n_iter, 1)

        if match_id:
            self._regret_states[match_id] = state

        return state

    # ═══════════════════════════════════════════
    #  MAÇ ANALİZİ
    # ═══════════════════════════════════════════
    def analyze_match(self, model_probs: dict,
                       market_odds: dict,
                       match_id: str = "",
                       stake: float = 100.0) -> BettingGameAnalysis:
        """Tek maç için oyun teorisi analizi."""
        analysis = BettingGameAnalysis(match_id=match_id)

        # Ödeme matrisi
        bettor_pay, bookie_pay = build_payoff_matrix(
            model_probs, market_odds, stake,
        )

        # Nash dengesi
        eq = self.find_equilibrium(bettor_pay, bookie_pay)
        analysis.equilibrium = eq

        # Optimal aksiyon
        if eq.bettor_strategy is not None:
            best_action = int(np.argmax(eq.bettor_strategy))
            analysis.optimal_action = self.ACTION_NAMES[best_action].lower().replace(" ", "_")
            analysis.optimal_stake_pct = round(float(eq.bettor_strategy[best_action]), 3)
        else:
            analysis.optimal_action = "pas"

        # Beklenen değer
        analysis.expected_value = round(eq.game_value, 2)

        # Büro marjı
        oh = market_odds.get("home", 2)
        od = market_odds.get("draw", 3.5)
        oa = market_odds.get("away", 4)
        if oh > 0 and od > 0 and oa > 0:
            overround = (1 / oh) + (1 / od) + (1 / oa)
            analysis.bookmaker_margin = round(overround - 1, 4)

        # Exploitability
        analysis.is_exploitable = eq.exploitability > 0.3

        # Risk-adjusted EV
        analysis.risk_adjusted_ev = round(
            eq.game_value * (1 - eq.exploitability), 2,
        )

        # Tavsiye
        analysis.recommendation = self._recommend(analysis)

        return analysis

    def _recommend(self, analysis: BettingGameAnalysis) -> str:
        """Oyun teorisi tavsiyesi."""
        if analysis.expected_value <= 0:
            return (
                f"PAS GEÇ. Nash değeri negatif ({analysis.expected_value:.1f}). "
                f"Büronun marjı çok yüksek ({analysis.bookmaker_margin:.1%})."
            )

        if analysis.is_exploitable:
            return (
                f"UYARI: Strateji sömürülebilir "
                f"(exploitability={analysis.equilibrium.exploitability:.0%}). "
                f"Düşük stake ile oynayın."
            )

        action = analysis.optimal_action.replace("_", " ").upper()
        return (
            f"{action} ({analysis.optimal_stake_pct:.0%} güvenle). "
            f"Nash EV: {analysis.expected_value:.1f}, "
            f"Risk-Adj: {analysis.risk_adjusted_ev:.1f}"
        )

    # ═══════════════════════════════════════════
    #  YARDIMCI
    # ═══════════════════════════════════════════
    def batch_analyze(self, matches: list[dict]) -> list[BettingGameAnalysis]:
        """Birden fazla maçı analiz et."""
        results = []
        for m in matches:
            probs = {
                "prob_home": m.get("prob_home", 0.33),
                "prob_draw": m.get("prob_draw", 0.33),
                "prob_away": m.get("prob_away", 0.34),
            }
            odds = {
                "home": m.get("home_odds", 2.0),
                "draw": m.get("draw_odds", 3.5),
                "away": m.get("away_odds", 4.0),
            }
            analysis = self.analyze_match(probs, odds, m.get("match_id", ""))
            results.append(analysis)
        return results
