"""
kelly_matrix.py – Çoklu ve Korele Bahisler için Portföy Kelly Optimizasyonu.

Tekil Kelly (Regime Kelly), her bahisi bağımsız varsayar. Ancak aynı gündeki,
aynı ligdeki veya benzer özellik taşıyan bahisler arasında korelasyon vardır.
Bu modül, N adet bet için optimal stake vektörünü hesaplar.

Matematik:
  E[log(W)] = sum(p_i * log(1 + f_i * b_i))
  Burada f vektörünü maksimize ederken korelasyon matrisi (Sigma) üzerinden
  varyans kısıtı ekleriz.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from dataclasses import dataclass

@dataclass
class BetOpportunity:
    id: str
    probability: float
    odds: float
    correlation_group: str = "default" # Örn: "league_id" or "match_id"

class CorrelatedKellyMatrix:
    def __init__(self, max_total_risk: float = 0.20):
        self.max_total_risk = max_total_risk # Toplam kasanın max %20'si riskte olsun

    def optimize(self, opportunities: List[BetOpportunity], 
                 correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        N adet fırsat için optimal stake oranlarını (f_i) hesaplar.
        Basitleştirilmiş Markowitz-Kelly yaklaşımı kullanır.
        """
        if not opportunities:
            return {}

        n = len(opportunities)
        p = np.array([o.probability for o in opportunities])
        b = np.array([o.odds - 1 for o in opportunities]) # Net odds
        
        # 1. Beklenen Değer (EV)
        ev = p * b - (1 - p)
        
        # Filtrele: Sadece EV > 0 olanları al
        valid_idx = np.where(ev > 0)[0]
        if len(valid_idx) == 0:
            return {o.id: 0.0 for o in opportunities}

        # 2. Korelasyon Matrisi (Sigma) -> Copula Yaklaşımı (Kelly v3)
        # Sadece doğrusal korelasyon (Sigma) değil, kuyruk bağımlılığını (Tail Dependency) 
        # modellemek için Copula tabanlı bir ağırlıklandırma ekliyoruz.
        if correlation_matrix is None:
            sigma = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    if opportunities[i].correlation_group == opportunities[j].correlation_group:
                        # Clayton Copula benzeri bir 'eş zamanlı çöküş' (Tail Risk) faktörü
                        # Alpha > 1 ise kuyruk bağımlılığı artar
                        alpha = 2.0 
                        copula_dep = (2**(-1/alpha)) # Basit tail dependency katsayısı
                        sigma[i, j] = sigma[j, i] = 0.3 + (0.2 * copula_dep)
        else:
            sigma = correlation_matrix

        # Sadece geçerli olanlar için alt matris al
        p_sub = p[valid_idx]
        b_sub = b[valid_idx]
        ev_sub = ev[valid_idx]
        sigma_sub = sigma[np.ix_(valid_idx, valid_idx)]

        # 3. Kelly Vektörü Hesaplama
        # f* = Sigma^-1 * EV (Basit Gaussian varsayımıyla)
        try:
            # f* = C^-1 * (p - (1-p)/b)
            # Burada C varyans-kovaryans matrisidir.
            # Varyans (p*q*b^2) varsayımı yerine birim varyans ve Sigma üzerinden gidiyoruz
            inv_sigma = np.linalg.inv(sigma_sub)
            f_star = inv_sigma @ ev_sub
            
            # 4. Kısıtlar (Anti-Overbetting)
            # Negatifleri sıfırla (Shorting yok)
            f_star = np.maximum(f_star, 0)
            
            # Toplam riski normalize et
            total_f = np.sum(f_star)
            if total_f > self.max_total_risk:
                f_star = f_star * (self.max_total_risk / total_f)

            # 5. Sonuçları eşle
            results = {o.id: 0.0 for o in opportunities}
            for i, idx in enumerate(valid_idx):
                results[opportunities[idx].id] = float(f_star[i])
            
            return results

        except Exception as e:
            logger.error(f"Kelly Matrix optimizasyon hatası: {e}")
            # Hata durumunda bağımsız Kelly'e dön (Safe fallback)
            return {o.id: max(0, (o.probability * (o.odds - 1) - (1 - o.probability)) / (o.odds - 1)) * 0.25 
                   for o in opportunities}

    def run_batch(self, opportunities: List[Dict], **kwargs) -> Dict[str, float]:
        """Orchestrator uyumlu batch metodu."""
        objs = [BetOpportunity(
            id=o.get("match_id", str(i)),
            probability=o.get("probability", 0.5),
            odds=o.get("odds", 2.0),
            correlation_group=o.get("league", "default")
        ) for i, o in enumerate(opportunities)]
        
        return self.optimize(objs)
