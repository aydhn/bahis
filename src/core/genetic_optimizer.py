"""
genetic_optimizer.py – Genetik Algoritma ile Parametre Optimizasyonu.

Hangi modele % kaç güveneceğinize veya Kelly Kriteri'nde riskin
% kaç olacağına siz karar vermeyin. Bırakın, bot binlerce simülasyon
yaparak "en çok kazandıran ayarları" kendi bulsun.

Her parametre bir "Gen":
  - Poisson ağırlığı (0.0-0.5)
  - LightGBM ağırlığı (0.0-0.5)
  - Kelly çarpanı (0.1-1.0)
  - Decay oranı (0.001-0.02)
  - Minimum EV eşiği (0.01-0.15)
  - Maksimum stake (0.02-0.10)
  - Drawdown sınırı (0.05-0.25)
  ...

Genetik Algoritma:
  1. Rastgele popülasyon oluştur
  2. Her bireyi geçmiş 1 yıl üzerinde backtest yap
  3. Fitness = ROI * (1 - Drawdown)
  4. En iyileri seç, çaprazla, mutasyon uygula
  5. N jenerasyon sonra en iyi parametre setini config.json'a yaz
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger


@dataclass
class Gene:
    """Optimize edilecek tek bir parametre."""
    name: str
    min_val: float
    max_val: float
    value: float = 0.0
    step: float = 0.0    # 0 = sürekli, >0 = ayrık adım

    def random(self) -> float:
        val = random.uniform(self.min_val, self.max_val)
        if self.step > 0:
            val = round(val / self.step) * self.step
        return val

    def mutate(self, mutation_rate: float = 0.1) -> float:
        """Mevcut değerden küçük sapma."""
        range_size = self.max_val - self.min_val
        delta = random.gauss(0, range_size * mutation_rate)
        new_val = np.clip(self.value + delta, self.min_val, self.max_val)
        if self.step > 0:
            new_val = round(new_val / self.step) * self.step
        return float(new_val)


@dataclass
class Individual:
    """Bir parametre seti (birey)."""
    genes: dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    roi: float = 0.0
    drawdown: float = 0.0
    sharpe: float = 0.0
    total_bets: int = 0


class GeneticOptimizer:
    """Genetik Algoritma ile strateji parametrelerini optimize eder.

    Kullanım:
        optimizer = GeneticOptimizer()
        best = optimizer.evolve(backtest_fn, generations=50)
        optimizer.save_config(best)
    """

    # Optimize edilecek parametrelerin tanımı
    DEFAULT_GENES = {
        "poisson_weight":       Gene("poisson_weight",       0.05, 0.40),
        "dixon_coles_weight":   Gene("dixon_coles_weight",   0.05, 0.40),
        "lightgbm_weight":      Gene("lightgbm_weight",      0.05, 0.40),
        "elo_weight":           Gene("elo_weight",           0.03, 0.25),
        "bayesian_weight":      Gene("bayesian_weight",      0.03, 0.25),
        "lstm_weight":          Gene("lstm_weight",          0.03, 0.25),
        "sentiment_weight":     Gene("sentiment_weight",     0.01, 0.15),
        "monte_carlo_weight":   Gene("monte_carlo_weight",   0.03, 0.20),
        "kelly_fraction":       Gene("kelly_fraction",       0.10, 1.00),
        "min_ev_threshold":     Gene("min_ev_threshold",     0.01, 0.15, step=0.005),
        "min_confidence":       Gene("min_confidence",       0.50, 0.85, step=0.05),
        "max_stake_pct":        Gene("max_stake_pct",        0.02, 0.10, step=0.01),
        "decay_rate":           Gene("decay_rate",           0.001, 0.020, step=0.001),
        "drawdown_reduce":      Gene("drawdown_reduce",      0.05, 0.20, step=0.01),
        "drawdown_paper":       Gene("drawdown_paper",       0.10, 0.30, step=0.01),
        "max_daily_bets":       Gene("max_daily_bets",       3, 15, step=1),
        "max_corr_threshold":   Gene("max_corr_threshold",   0.20, 0.60, step=0.05),
    }

    def __init__(self, population_size: int = 100,
                 elite_ratio: float = 0.10,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.80,
                 genes: dict[str, Gene] | None = None):
        self._pop_size = population_size
        self._elite_ratio = elite_ratio
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._genes = genes or dict(self.DEFAULT_GENES)
        self._population: list[Individual] = []
        self._history: list[dict] = []
        self._best: Individual | None = None
        logger.debug(
            f"GeneticOptimizer: {len(self._genes)} gen, "
            f"pop={population_size}, mut={mutation_rate}"
        )

    async def run_batch(self, db: Any = None, **kwargs):
        """Pipeline toplu işleme (Optimization Mode)."""
        logger.info("[GeneticOptimizer] Genetik evrim tetikleniyor...")
        
        if db:
            from src.core.evolutionary_runner import EvolutionaryRunner
            runner = EvolutionaryRunner(db=db, optimizer=self)
            await runner.run_optimization_cycle()
        else:
            # Mock fallback
            def mock_backtest(params):
                return {
                    "roi": random.uniform(-0.1, 0.2), 
                    "max_drawdown": random.uniform(0.0, 0.3),
                    "sharpe": random.uniform(0.5, 2.0),
                    "total_bets": random.randint(50, 200)
                }
            self.evolve(mock_backtest, generations=2)
            self.save_config()


    # ═══════════════════════════════════════════
    #  EVRİM
    # ═══════════════════════════════════════════
    def evolve(self, backtest_fn: Callable[[dict], dict],
               generations: int = 50,
               early_stop_patience: int = 10) -> Individual:
        """Genetik algoritmayı çalıştır.

        Args:
            backtest_fn: Parametre seti alıp { roi, drawdown, sharpe, total_bets }
                         döndüren backtest fonksiyonu.
            generations: Jenerasyon sayısı.
            early_stop_patience: N jenerasyon iyileşme yoksa dur.

        Returns:
            En iyi birey (Individual).
        """
        logger.info(f"[GA] Evrim başlıyor: {generations} jenerasyon, pop={self._pop_size}")
        t0 = time.time()

        # İlk popülasyonu rastgele oluştur
        self._population = self._init_population()

        best_fitness = -np.inf
        no_improve = 0

        for gen in range(generations):
            # Her bireyi değerlendir
            self._evaluate(backtest_fn)

            # Sırala (fitness'a göre azalan)
            self._population.sort(key=lambda ind: ind.fitness, reverse=True)

            gen_best = self._population[0]
            gen_avg = np.mean([ind.fitness for ind in self._population])

            self._history.append({
                "generation": gen + 1,
                "best_fitness": gen_best.fitness,
                "best_roi": gen_best.roi,
                "best_drawdown": gen_best.drawdown,
                "avg_fitness": float(gen_avg),
            })

            # Early stopping
            if gen_best.fitness > best_fitness:
                best_fitness = gen_best.fitness
                self._best = gen_best
                no_improve = 0
            else:
                no_improve += 1

            if (gen + 1) % 5 == 0 or gen == 0:
                logger.info(
                    f"[GA] Gen {gen+1}/{generations} | "
                    f"best: {gen_best.fitness:.4f} (ROI={gen_best.roi:.2%}, "
                    f"DD={gen_best.drawdown:.1%}) | avg: {gen_avg:.4f}"
                )

            if no_improve >= early_stop_patience:
                logger.info(f"[GA] Early stop @ gen {gen+1} – {no_improve} jenerasyon iyileşme yok.")
                break

            # Yeni jenerasyon üret
            self._population = self._next_generation()

        elapsed = time.time() - t0
        logger.success(
            f"[GA] Evrim tamamlandı: {elapsed:.1f}s | "
            f"En iyi: ROI={self._best.roi:.2%}, DD={self._best.drawdown:.1%}, "
            f"Sharpe={self._best.sharpe:.2f}"
        )
        return self._best

    def _init_population(self) -> list[Individual]:
        """Rastgele başlangıç popülasyonu."""
        pop = []
        for _ in range(self._pop_size):
            genes = {}
            for name, gene in self._genes.items():
                genes[name] = gene.random()
            pop.append(Individual(genes=genes))
        return pop

    def _evaluate(self, backtest_fn: Callable):
        """Tüm bireyleri backtest fonksiyonuyla değerlendir."""
        for ind in self._population:
            if ind.fitness != 0:
                continue  # Zaten değerlendirilmiş

            try:
                result = backtest_fn(ind.genes)
                ind.roi = result.get("roi", 0)
                ind.drawdown = result.get("max_drawdown", 0)
                ind.sharpe = result.get("sharpe", 0)
                ind.total_bets = result.get("total_bets", 0)

                # Fitness: ROI'yi ödüllendir, drawdown'u cezalandır
                # Sharpe bonus ekle, yeterli bahis yoksa cezala
                ind.fitness = self._calculate_fitness(ind)
            except Exception as e:
                ind.fitness = -1.0
                logger.debug(f"[GA] Backtest hatası: {e}")

    @staticmethod
    def _calculate_fitness(ind: Individual) -> float:
        """Çok amaçlı fitness fonksiyonu.

        Fitness = ROI * (1 - MaxDD) * Sharpe_bonus * bet_penalty
        """
        roi_factor = ind.roi

        # Drawdown cezası (0 DD = 1.0, 50% DD = 0.5)
        dd_factor = max(1.0 - ind.drawdown, 0.1)

        # Sharpe bonus
        sharpe_bonus = 1.0 + max(ind.sharpe * 0.1, 0)

        # Yeterli bahis yoksa cezala (min 50 bahis)
        bet_factor = min(ind.total_bets / 50, 1.0) if ind.total_bets > 0 else 0.1

        return roi_factor * dd_factor * sharpe_bonus * bet_factor

    def _next_generation(self) -> list[Individual]:
        """Yeni jenerasyon: elitizm + crossover + mutasyon."""
        n_elite = max(int(self._pop_size * self._elite_ratio), 1)
        new_pop = []

        # Elitler doğrudan geçer
        elites = self._population[:n_elite]
        for e in elites:
            new_pop.append(Individual(genes=dict(e.genes)))

        # Geri kalanı crossover + mutasyon ile üret
        while len(new_pop) < self._pop_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if random.random() < self._crossover_rate:
                child_genes = self._crossover(parent1, parent2)
            else:
                child_genes = dict(parent1.genes)

            child_genes = self._mutate(child_genes)
            new_pop.append(Individual(genes=child_genes))

        return new_pop

    def _tournament_select(self, k: int = 3) -> Individual:
        """Turnuva seçimi: k rastgele bireyden en iyisini seç."""
        candidates = random.sample(self._population,
                                    min(k, len(self._population)))
        return max(candidates, key=lambda ind: ind.fitness)

    def _crossover(self, p1: Individual, p2: Individual) -> dict[str, float]:
        """Uniform crossover: her gen için rastgele ebeveyn seç."""
        child = {}
        for name in self._genes:
            if random.random() < 0.5:
                child[name] = p1.genes.get(name, 0)
            else:
                child[name] = p2.genes.get(name, 0)
        return child

    def _mutate(self, genes: dict[str, float]) -> dict[str, float]:
        """Her geni mutation_rate olasılıkla mutasyona uğrat."""
        for name, gene_def in self._genes.items():
            if random.random() < self._mutation_rate:
                gene_def.value = genes.get(name, gene_def.min_val)
                genes[name] = gene_def.mutate(self._mutation_rate)
        return genes

    # ═══════════════════════════════════════════
    #  KAYIT & YÜKLEME
    # ═══════════════════════════════════════════
    def save_config(self, individual: Individual | None = None,
                    path: str = "config.json"):
        """En iyi parametre setini config.json'a kaydet."""
        ind = individual or self._best
        if not ind:
            logger.warning("[GA] Kaydedilecek sonuç yok.")
            return

        config = {
            "optimized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fitness": ind.fitness,
            "roi": ind.roi,
            "max_drawdown": ind.drawdown,
            "sharpe": ind.sharpe,
            "total_bets": ind.total_bets,
            "parameters": ind.genes,
            "evolution_history": self._history[-10:],
        }

        Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False))
        logger.success(f"[GA] Optimum parametreler kaydedildi: {path}")

    def load_config(self, path: str = "config.json") -> dict | None:
        """Daha önce optimize edilmiş parametreleri yükle."""
        try:
            data = json.loads(Path(path).read_text())
            params = data.get("parameters", {})
            logger.info(
                f"[GA] Config yüklendi: ROI={data.get('roi', 0):.2%}, "
                f"fitness={data.get('fitness', 0):.4f}"
            )
            return params
        except FileNotFoundError:
            logger.debug("[GA] config.json bulunamadı – varsayılan parametreler.")
            return None
        except Exception as e:
            logger.error(f"[GA] Config yükleme hatası: {e}")
            return None

    @property
    def best(self) -> Individual | None:
        return self._best

    @property
    def history(self) -> list[dict]:
        return self._history
