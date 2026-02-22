"""
evolutionary_runner.py – Otomatik Strateji Evrimi ve Parametre Ayarlama.

Bu modül, GeneticOptimizer'ı kullanarak sistemin parametrelerini 
(ağırlıklar, eşikler, Kelly fraksiyonu) belirli aralıklarla (örn. haftalık) 
gerçek DuckDB verileri üzerinden optimize eder.
"""
import asyncio
from typing import Any, Dict
import pandas as pd
import numpy as np
from loguru import logger
from src.core.genetic_optimizer import GeneticOptimizer

class EvolutionaryRunner:
    def __init__(self, db: Any, optimizer: GeneticOptimizer = None):
        self.db = db
        self.optimizer = optimizer or GeneticOptimizer(population_size=20)
        logger.info("EvolutionaryRunner initialized.")

    async def run_optimization_cycle(self, days: int = 90):
        """Gerçek verilerle evrim döngüsünü çalıştırır."""
        logger.info(f"Evrimsel optimizasyon döngüsü başlıyor (Son {days} gün verisiyle)...")
        
        # 1. Backtest verilerini hazırla (Sinyaller vs Sonuçlar)
        history = self._get_historical_performance(days)
        if history.empty:
            logger.warning("Optimizasyon için yeterli veri bulunamadı.")
            return

        # 2. Backtest fonksiyonunu tanımla (Closure)
        def real_backtest(params: Dict[str, float]) -> Dict[str, Any]:
            # Parametreleri uygula ve ROI/DD hesapla
            # (Basit bir vektörize backtest)
            temp_history = history.copy()
            
            # Örnek: Model ağırlıklarını uygula (Poisson ve DixonColes)
            # Gerçekte daha karmaşık bir ensemble mantığı olurdu
            temp_history["weighted_prob"] = (
                temp_history["poisson_prob"] * params.get("poisson_weight", 0.5) +
                temp_history["dc_prob"] * (1 - params.get("poisson_weight", 0.5))
            )
            
            # EV hesabı
            temp_history["ev"] = temp_history["weighted_prob"] * temp_history["odds"] - 1
            
            # Bahis filtresi
            bets = temp_history[temp_history["ev"] > params.get("min_ev_threshold", 0.05)]
            if bets.empty:
                return {"roi": -1.0, "max_drawdown": 1.0, "sharpe": 0.0, "total_bets": 0}

            # ROI hesabı
            pnl = (bets["odds"] * bets["won"] - 1).sum()
            roi = pnl / len(bets)
            
            # Drawdown hesabı
            cum_pnl = (bets["odds"] * bets["won"] - 1).cumsum()
            running_max = cum_pnl.cummax()
            drawdown = (running_max - cum_pnl).max()
            
            return {
                "roi": float(roi),
                "max_drawdown": float(drawdown / len(bets)) if len(bets) > 0 else 0,
                "sharpe": float(roi / (np.std(bets["won"]) + 1e-6)),
                "total_bets": len(bets)
            }

        # 3. Evrimi başlat
        best = self.optimizer.evolve(real_backtest, generations=3) # Demo için kısa
        
        # 4. Kaydet
        self.optimizer.save_config(best)
        logger.success("Evrimsel optimizasyon tamamlandı ve config.json güncellendi.")

    def _get_historical_performance(self, days: int) -> pd.DataFrame:
        """DB'den backtest için sinyal ve maç sonuçlarını çeker."""
        if self.db is None:
            return pd.DataFrame()
            
        query = f"""
        SELECT s.odds, s.poisson_prob, s.dc_prob, 
               CASE WHEN m.home_score > m.away_score THEN 1 ELSE 0 END as won
        FROM signals s
        JOIN matches m ON s.match_id = m.match_id
        WHERE m.status = 'finished' 
        AND m.match_date >= CURRENT_DATE - INTERVAL '{days} DAY'
        """
        try:
            return self.db.query(query)
        except Exception as e:
            logger.error(f"Geçmiş veri çekme hatası: {e}")
            return pd.DataFrame()
