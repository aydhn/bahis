"""
strategy_evolver.py – Self-Evolving Strategy DNA (Otonom Strateji Evrimi).

Bot kendi stratejisini genetik algoritma + performans geri
bildirimi ile evrimleştirir. Hiçbir insan müdahalesi gerekmez.

Kavramlar:
  - Strategy DNA: Her strateji bir gen dizisi (parametre vektörü)
  - Fitness: Sharpe Ratio + CLV + Win Rate bileşik skor
  - Mutation: Parametrelerin rastgele küçük değişimleri
  - Crossover: İki başarılı stratejinin genlerini birleştirme
  - Elitism: En iyi %10 değişmeden sonraki nesle aktarılır
  - Tournament Selection: Rastgele 3 birey seç, en iyisini al
  - Self-Adaptation: Mutasyon oranı kendini ayarlar
  - Epoch: Her 100 bahis = 1 nesil (generation)
  - Hall of Fame: Tüm zamanların en iyi 5 stratejisi saklanır

Akış:
  1. Popülasyon başlatılır (50 strateji DNA'sı)
  2. Her strateji 100 bahis boyunca test edilir (walk-forward)
  3. Fitness hesaplanır (Sharpe × CLV × WinRate)
  4. En iyi stratejiler seçilir, çaprazlanır, mutasyona uğrar
  5. Yeni nesil oluşturulur → Döngü tekrarlanır
  6. En iyi DNA parametreleri sisteme uygulanır
  7. Tüm evrim loglanır (her nesil, her gen değişimi)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════
#  STRATEJİ DNA ŞEMASI
# ═══════════════════════════════════════════════
GENE_SCHEMA = {
    "kelly_fraction": (0.05, 0.50, 0.25),     # (min, max, default)
    "min_edge": (0.01, 0.10, 0.03),
    "max_stake_pct": (0.01, 0.10, 0.05),
    "ensemble_weight_poisson": (0.0, 1.0, 0.30),
    "ensemble_weight_lgbm": (0.0, 1.0, 0.35),
    "ensemble_weight_lstm": (0.0, 1.0, 0.20),
    "ensemble_weight_rl": (0.0, 1.0, 0.15),
    "entropy_kill_threshold": (1.5, 3.5, 2.5),
    "chaos_filter_lambda": (-0.1, 0.1, 0.0),
    "drawdown_limit": (0.05, 0.30, 0.15),
    "vol_target_weekly": (0.03, 0.15, 0.08),
    "min_confidence": (0.50, 0.85, 0.60),
    "correlation_max": (0.20, 0.70, 0.40),
    "time_decay_lambda": (0.001, 0.05, 0.01),
    "anti_tilt_streak": (3, 10, 5),
    # Market Selection Genes (Google Growth Engine)
    "league_multiplier_tier1": (0.5, 2.0, 1.0), # Aggression for Top Leagues
    "league_multiplier_tier2": (0.5, 1.5, 1.0), # Aggression for Mid Leagues
    "league_multiplier_tier3": (0.0, 1.2, 0.8), # Aggression for Lower Leagues
    "market_whitelist_threshold": (-0.10, 0.05, -0.05), # Min ROI to stay in a market
}

GENE_NAMES = list(GENE_SCHEMA.keys())
N_GENES = len(GENE_NAMES)


@dataclass
class StrategyDNA:
    """Bir stratejinin gen dizisi."""
    genes: np.ndarray = field(default_factory=lambda: np.zeros(N_GENES))
    fitness: float = 0.0
    sharpe: float = 0.0
    clv: float = 0.0
    win_rate: float = 0.0
    total_bets: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    mutation_rate: float = 0.1  # Self-adaptive

    def to_dict(self) -> dict:
        return {name: float(self.genes[i]) for i, name in enumerate(GENE_NAMES)}

    @classmethod
    def from_dict(cls, params: dict) -> StrategyDNA:
        genes = np.array([
            params.get(name, GENE_SCHEMA[name][2])
            for name in GENE_NAMES
        ])
        return cls(genes=genes)

    @classmethod
    def random(cls) -> StrategyDNA:
        genes = np.array([
            np.random.uniform(GENE_SCHEMA[name][0], GENE_SCHEMA[name][1])
            for name in GENE_NAMES
        ])
        return cls(genes=genes, mutation_rate=np.random.uniform(0.05, 0.20))


@dataclass
class EvolutionReport:
    """Evrim raporu."""
    generation: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    best_dna: dict = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)
    hall_of_fame: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  STRATEGY EVOLVER (Ana Sınıf)
# ═══════════════════════════════════════════════
class StrategyEvolver:
    """Otonom strateji evrimi motoru.

    Kullanım:
        evolver = StrategyEvolver(population_size=50)

        # Her 100 bahis sonunda evrimleştir
        report = evolver.evolve(
            results=[{"won": True, "pnl": 50}, ...],
        )

        # En iyi DNA'yı al ve sisteme uygula
        best = evolver.get_best_dna()
        kelly.set_params(best.to_dict())
    """

    SAVE_PATH = Path("data/evolution")

    def __init__(self, population_size: int = 50,
                 elitism_pct: float = 0.10,
                 tournament_size: int = 3,
                 crossover_rate: float = 0.70,
                 epoch_size: int = 100,
                 hall_of_fame_size: int = 5):
        self._pop_size = population_size
        self._elitism = int(population_size * elitism_pct)
        self._tournament_k = tournament_size
        self._crossover_rate = crossover_rate
        self._epoch = epoch_size
        self._hof_size = hall_of_fame_size

        # Popülasyon
        self._population: list[StrategyDNA] = [
            StrategyDNA.random() for _ in range(population_size)
        ]
        self._generation = 0
        self._hall_of_fame: list[StrategyDNA] = []
        self._history: list[EvolutionReport] = []

        self.SAVE_PATH.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[Evolver] Başlatıldı: pop={population_size}, "
            f"elitism={self._elitism}, tournament={tournament_size}"
        )

    def evolve(self, results: list[dict]) -> EvolutionReport:
        """Bir nesil evrimleştir."""
        self._generation += 1
        report = EvolutionReport(
            generation=self._generation,
            population_size=self._pop_size,
        )

        # 1) Fitness hesapla
        for dna in self._population:
            dna.fitness = self._evaluate_fitness(dna, results)
            dna.generation = self._generation

        # 2) Sırala
        self._population.sort(key=lambda d: d.fitness, reverse=True)

        # 3) İstatistikler
        fitnesses = [d.fitness for d in self._population]
        report.best_fitness = round(fitnesses[0], 6)
        report.avg_fitness = round(float(np.mean(fitnesses)), 6)
        report.worst_fitness = round(fitnesses[-1], 6)
        report.best_dna = self._population[0].to_dict()

        # 4) Hall of Fame güncelle
        self._update_hof(self._population[0])
        report.hall_of_fame = [
            {"fitness": d.fitness, **d.to_dict()}
            for d in self._hall_of_fame
        ]

        # 5) Yeni nesil oluştur
        new_pop: list[StrategyDNA] = []

        # Elitism: en iyi bireyler değişmeden geçer
        for i in range(self._elitism):
            elite = StrategyDNA(
                genes=self._population[i].genes.copy(),
                mutation_rate=self._population[i].mutation_rate,
            )
            new_pop.append(elite)

        # Crossover + Mutation
        while len(new_pop) < self._pop_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if np.random.random() < self._crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = StrategyDNA(
                    genes=parent1.genes.copy(),
                    mutation_rate=parent1.mutation_rate,
                )

            child = self._mutate(child)
            child.parent_ids = [id(parent1), id(parent2)]
            new_pop.append(child)

        self._population = new_pop[:self._pop_size]

        # 6) Log & kaydet
        prev_best = self._history[-1].best_fitness if self._history else 0
        if report.best_fitness > prev_best:
            improvement = report.best_fitness - prev_best
            report.improvements.append(
                f"Fitness artışı: {prev_best:.4f} → {report.best_fitness:.4f} "
                f"(+{improvement:.4f})"
            )

        self._history.append(report)
        self._save_checkpoint()

        logger.info(
            f"[Evolver] Gen #{self._generation}: "
            f"best={report.best_fitness:.4f}, "
            f"avg={report.avg_fitness:.4f}, "
            f"worst={report.worst_fitness:.4f}"
        )

        return report

    def _evaluate_fitness(self, dna: StrategyDNA,
                            results: list[dict]) -> float:
        """DNA'nın fitness skorunu hesapla."""
        params = dna.to_dict()
        if not results:
            return 0.0

        # Simüle et: bu DNA parametreleriyle ne olurdu?
        pnl_series = []
        wins = 0
        total = 0
        for r in results:
            ev = r.get("ev", 0)
            odds = r.get("odds", 2.0)
            won = r.get("won", False)

            # Kelly stake
            prob = r.get("prob", 0.5)
            kelly_f = params["kelly_fraction"]
            raw_k = max((prob * odds - 1) / max(odds - 1, 0.01), 0)
            stake = raw_k * kelly_f

            # Market Selection & Aggression (Growth Engine)
            tier = r.get("league_tier", 2) # Default to Mid
            if tier == 1:
                stake *= params["league_multiplier_tier1"]
            elif tier == 3:
                stake *= params["league_multiplier_tier3"]
            else:
                stake *= params["league_multiplier_tier2"]

            # Edge filtresi
            edge = prob * odds - 1
            if edge < params["min_edge"]:
                continue
            if prob < params["min_confidence"]:
                continue

            stake = min(stake, params["max_stake_pct"])
            total += 1

            if won:
                pnl_series.append(stake * (odds - 1))
                wins += 1
            else:
                pnl_series.append(-stake)

        if total < 10 or not pnl_series:
            return 0.0

        pnl = np.array(pnl_series)
        # Sharpe Ratio
        mean_r = np.mean(pnl)
        std_r = np.std(pnl) + 1e-8
        sharpe = float(mean_r / std_r * np.sqrt(total))

        # Win rate
        win_rate = wins / total

        # Total return
        total_return = float(np.sum(pnl))

        # Composite fitness
        fitness = (
            sharpe * 0.40
            + win_rate * 0.30
            + np.clip(total_return, -1, 2) * 0.30
        )

        dna.sharpe = round(sharpe, 4)
        dna.win_rate = round(win_rate, 4)
        dna.total_bets = total

        return round(float(fitness), 6)

    def _tournament_select(self) -> StrategyDNA:
        """Tournament selection."""
        candidates = np.random.choice(
            len(self._population), size=self._tournament_k, replace=False,
        )
        best_idx = max(candidates, key=lambda i: self._population[i].fitness)
        return self._population[best_idx]

    def _crossover(self, p1: StrategyDNA, p2: StrategyDNA) -> StrategyDNA:
        """Uniform crossover."""
        mask = np.random.random(N_GENES) < 0.5
        child_genes = np.where(mask, p1.genes, p2.genes)
        child_mr = (p1.mutation_rate + p2.mutation_rate) / 2
        return StrategyDNA(genes=child_genes, mutation_rate=child_mr)

    def _mutate(self, dna: StrategyDNA) -> StrategyDNA:
        """Gaussian mutation + self-adaptation."""
        # Self-adaptive mutation rate
        dna.mutation_rate *= np.exp(np.random.normal(0, 0.1))
        dna.mutation_rate = np.clip(dna.mutation_rate, 0.01, 0.50)

        for i, name in enumerate(GENE_NAMES):
            if np.random.random() < dna.mutation_rate:
                lo, hi, _ = GENE_SCHEMA[name]
                sigma = (hi - lo) * 0.1
                dna.genes[i] += np.random.normal(0, sigma)
                dna.genes[i] = np.clip(dna.genes[i], lo, hi)

        return dna

    def _update_hof(self, candidate: StrategyDNA) -> None:
        """Hall of Fame güncelle."""
        self._hall_of_fame.append(
            StrategyDNA(genes=candidate.genes.copy(), fitness=candidate.fitness),
        )
        self._hall_of_fame.sort(key=lambda d: d.fitness, reverse=True)
        self._hall_of_fame = self._hall_of_fame[:self._hof_size]

    def _save_checkpoint(self) -> None:
        """Checkpoint kaydet."""
        try:
            best = self._population[0]
            data = {
                "generation": self._generation,
                "best_fitness": best.fitness,
                "best_params": best.to_dict(),
                "timestamp": time.time(),
            }
            path = self.SAVE_PATH / "latest_dna.json"
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"[Evolver] Checkpoint kayıt hatası: {e}")

    def get_best_dna(self) -> StrategyDNA:
        self._population.sort(key=lambda d: d.fitness, reverse=True)
        return self._population[0]

    def load_checkpoint(self) -> bool:
        try:
            path = self.SAVE_PATH / "latest_dna.json"
            if path.exists():
                data = json.loads(path.read_text())
                best = StrategyDNA.from_dict(data["best_params"])
                best.fitness = data["best_fitness"]
                self._population[0] = best
                self._generation = data.get("generation", 0)
                logger.info(
                    f"[Evolver] Checkpoint yüklendi: "
                    f"gen={self._generation}, fitness={best.fitness:.4f}"
                )
                return True
        except Exception as e:
            logger.debug(f"[Evolver] Checkpoint yükleme hatası: {e}")
        return False
