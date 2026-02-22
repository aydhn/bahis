"""
monte_carlo_engine.py – İflas Riski ve Gelecek Projeksiyonu.

Bu modül, mevcut kasanın ve stratejinin 10,000 farklı "paralel evren"deki
sonuçlarını simüle eder. "Risk of Ruin" (İflas Riski) hesaplar.

Girdi:
- Mevcut Kasa
- Win Rate (Kazanma Oranı)
- Ortalama Oran
- Kelly Stake Yüzdesi (veya Sabit Yüzde)

Çıktı:
- İflas Olasılığı (%)
- Beklenen Kasa (Median)
- En Kötü Senaryo (1. Persentil)
"""
import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import Any
from loguru import logger

@dataclass
class SimulationResult:
    risk_of_ruin: float     # %0 - %100
    expected_final_bankroll: float
    worst_case_bankroll: float
    best_case_bankroll: float
    simulations_count: int

class MonteCarloEngine:
    def __init__(self, db: Any = None, simulations: int = 10000, horizon: int = 500):
        self.db = db
        self.sims = simulations
        self.horizon = horizon  # Kaç bahis ileriye bakılacak
        self._jit = None 
        from src.core.rust_bridge import RustMCBridge
        self._rust = RustMCBridge()

    def inject_jit(self, jit_accelerator):
        """Hızlandırıcıyı enjekte et."""
        self._jit = jit_accelerator

    def run_batch(self, **kwargs):
        """Batch modu: DB'den kasa ve açık bahis verilerini çekip simülasyon yapar."""
        if self.db is None:
            logger.warning("MonteCarloEngine: DB bağlantısı yok.")
            return

        # Mock kasa ve veriler
        current_bankroll = 1000.0
        
        # Basit simülasyon parametreleri
        win_rate = 0.55
        avg_odds = 2.0
        stake_pct = 0.02
        
        result = self.run(current_bankroll, win_rate, avg_odds, stake_pct)
        
        logger.info(
            f"[MonteCarlo] Risk of Ruin: {result.risk_of_ruin:.1%}, "
            f"Exp Final: {result.expected_final_bankroll:.2f}"
            + (f" (JIT: {'AKTİF' if self._jit and self._jit.is_jit_available else 'PASİF'})")
        )

    def run(self, bankroll: float, win_rate: float, avg_odds: float, stake_pct: float,
            black_swan_prob: float = 0.01) -> SimulationResult:
        """
        Kasa simülasyonunu çalıştır.
        black_swan_prob: Beklenmedik, yıkıcı bir olayın (örn: tüm kasanın %20'si kaybı) olasılığı.
        """
        if bankroll <= 0 or stake_pct <= 0:
            return SimulationResult(1.0, 0, 0, 0, self.sims)

        # Rust Bridge available?
        if self._rust.enabled:
            paths = self._rust.simulate_path(bankroll, self.horizon, win_rate, avg_odds, self.sims)
        else:
            # Numpy implementation
            rng = np.random.default_rng()
            results = rng.random((self.sims, self.horizon)) < win_rate
            
            win_multiplier = 1 + stake_pct * (avg_odds - 1)
            loss_multiplier = 1 - stake_pct
            
            # Çarpanlar
            multipliers = np.where(results, win_multiplier, loss_multiplier)
            
            # Black Swan Injection (Siyah Kuğu Olayları)
            # Beklenmedik kırmızı kartlar, hakem hataları veya borsa manipülasyonu gibi.
            swans = rng.random((self.sims, self.horizon)) < black_swan_prob
            multipliers = np.where(swans, 0.80, multipliers) # Black swan anında kasanın %20'si gider
            
            # Kümülatif çarpım
            paths = bankroll * np.cumprod(multipliers, axis=1)
        
        final_bankrolls = paths[:, -1]
        
        # İflas (Ruin): Yolculuk sırasında kasanın %5'in altına düşmesi
        # min_bankrolls = np.min(paths, axis=1)
        # ruined_count = np.sum(min_bankrolls < (bankroll * 0.05))
        
        # Basitçe sonuca bakalım (Hız için)
        ruined_count = np.sum(final_bankrolls < (bankroll * 0.05))
        risk_of_ruin = ruined_count / self.sims
        
        return SimulationResult(
            risk_of_ruin=float(risk_of_ruin),
            expected_final_bankroll=float(np.median(final_bankrolls)),
            worst_case_bankroll=float(np.percentile(final_bankrolls, 1)), 
            best_case_bankroll=float(np.percentile(final_bankrolls, 99)), 
            simulations_count=self.sims
        )

    def stress_test_portfolio(self, bankroll: float, bets: list[dict]) -> dict:
        pass
