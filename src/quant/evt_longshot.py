"""
evt_longshot.py – Ekstrem Değer Teorisi (Extreme Value Theory).

Çoğu model 'normal' dağılımı (Gaussian) baz alır. Ancak bahis piyasalarında 
sürprizler (fat-tails) daha sık gerçekleşir. 
EVT, özellikle 'long-shot' (yüksek oranlı) bahislerdeki gizli değeri bulmak için 
'Block Maxima' yöntemiyle kuyruk analizi yapar.

Matematik: Generalized Extreme Value (GEV) dağılımı.
"""
from scipy.stats import genextreme
import numpy as np
from typing import List, Dict, Any
from loguru import logger

class EVTLongshotEngine:
    def __init__(self, db: Any = None):
        self.db = db
        # league_params[league_id] = (shape, loc, scale)
        self.params: Dict[str, tuple] = {}

    def fit_tail(self, data: List[float], league_id: str):
        """Geçmiş gol farkları veya skor sürprizleri üzerinden GEV fit eder."""
        if len(data) < 20: return
        
        # GEV Parameters: shape, location, scale
        shape, loc, scale = genextreme.fit(data)
        self.params[league_id] = (shape, loc, scale)
        logger.debug(f"[EVT] {league_id} için kuyruk modeli normalize edildi.")

    def calculate_longshot_value(self, odds: float, league_id: str) -> float:
        """
        GEV dağılımına göre ekstrem bir olayın (sürpriz) gerçekleşme olasılığını hesaplar.
        """
        if league_id not in self.params:
            return 0.0
            
        shape, loc, scale = self.params[league_id]
        # Oranların tersi (implied prob) ekstrem bölgede mi?
        # x: Gerekli olan 'performans sürprizi' eşiği
        x = np.log(odds) # Basit bir scaling
        
        # Survival function (1 - CDF) - Ekstrem olayın ihtimali
        prob = genextreme.sf(x, shape, loc, scale)
        
        # EV = (Prob * Odds) - 1
        return (prob * odds) - 1.0

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Sinyaller arasından EVT onaylı 'altın madenlerini' bulur."""
        evt_signals = []
        for sig in signals:
            odds = sig.get("odds", 1.0)
            if odds > 4.0: # Sadece yüksek oranlı 'long-shot'lar için aktif
                value = self.calculate_longshot_value(odds, sig.get("league_id", "global"))
                if value > 0.05:
                    sig["tags"] = sig.get("tags", []) + ["evt_longshot_alpha"]
                    sig["ev"] += value # EV skoruna EVT katkısı ekle
                
            evt_signals.append(sig)
        return evt_signals
