"""
meta_strategy_manager.py – Meta-Strateji ve Hiyerarşik Sermaye Dağıtımı (Kelly v4).

Sistem artık tek bir Kelly motoruna değil, bir "Fonların Fonu" (Fund of Funds) 
mantığına geçer. Her model (Dixon-Coles, RL, Bayesian, vb.) birer varlık 
atfedilir ve sermaye bu modellere canlı performanslarına göre dağıtılır.

Kavramlar:
  - Meta-Allocation: Toplam bankroll'ün modellere dağıtımı.
  - Performance Drift: Modelin tahmini ile gerçekleşen PnL arasındaki korelasyon.
  - Adaptive Risk Parity: Volatilitesi yüksek modelin payını otomatik azaltır.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from loguru import logger
from dataclasses import dataclass, field

@dataclass
class ModelPerformance:
    name: str
    sharpe: float = 0.0
    drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    pnl_history: List[float] = field(default_factory=list)

class MetaStrategyManager:
    def __init__(self, db: Any = None, total_bankroll: float = 10000.0):
        self.db = db
        self.total_bankroll = total_bankroll
        self.model_stats: Dict[str, ModelPerformance] = {}
        self.allocations: Dict[str, float] = {} # % bazlı dağılım

    def register_model(self, name: str):
        if name not in self.model_stats:
            self.model_stats[name] = ModelPerformance(name=name)
            logger.info(f"[MetaStrategy] Model kaydedildi: {name}")

    async def sync_performances(self):
        """DB'den her modelin gerçek PnL geçmişini çeker."""
        if not self.db: return
        
        try:
            # Örnek sorgu: Model bazlı PnL
            query = "SELECT model_name, pnl FROM bets ORDER BY timestamp ASC"
            df = self.db.query(query)
            if df.is_empty(): return

            for model_name in self.model_stats:
                model_df = df[df["model_name"] == model_name]
                if model_df.is_empty(): continue
                
                pnls = model_df["pnl"].to_numpy()
                stats = self.model_stats[model_name]
                stats.pnl_history = pnls.tolist()
                stats.total_trades = len(pnls)
                stats.win_rate = np.mean(pnls > 0)
                
                # Sharpe Ratio (basit)
                if len(pnls) > 5:
                    std = np.std(pnls)
                    stats.sharpe = np.mean(pnls) / std if std > 0 else 0
                
                # Max Drawdown
                cum_pnl = np.cumsum(pnls)
                peak = np.maximum.accumulate(cum_pnl)
                dd = (peak - cum_pnl)
                stats.drawdown = np.max(dd) if len(dd) > 0 else 0

        except Exception as e:
            logger.error(f"[MetaStrategy] Sync hatası: {e}")

    def update_allocations(self):
        """Risk Parity mantığıyla sermaye dağılımını günceller."""
        sharpes = np.array([m.sharpe for m in self.model_stats.values()])
        # Negatif Sharpe'ları engelle (minimum 0.01)
        sharpes = np.maximum(sharpes, 0.01)
        
        # Softmax veya basit ağırlıklandırma
        total_sharpe = np.sum(sharpes)
        if total_sharpe == 0:
            weight = 1.0 / len(self.model_stats)
            self.allocations = {name: weight for name in self.model_stats}
        else:
            self.allocations = {
                name: (m.sharpe / total_sharpe) 
                for name, m in self.model_stats.items()
            }
        
        logger.info(f"[MetaStrategy] Yeni Dağılım: {self.allocations}")

    def get_allocated_bankroll(self, model_name: str) -> float:
        """Belirli bir model için kullanılabilir kasayı döndürür."""
        pct = self.allocations.get(model_name, 0.0)
        return self.total_bankroll * pct

    async def run_batch(self, **kwargs):
        """Pipeline entegrasyonu."""
        await self.sync_performances()
        self.update_allocations()
