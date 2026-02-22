"""
ergodicity_optimizer.py – Ergodicity Economics & Time-Average Growth.

"Ensemble Average" (paralel evrenler) yerine "Time Average" (tek bir gerçek yaşam)
büyümesini maksimize eder. Ole Peters'ın çalışmasına dayanır.

Mantık:
  - Klasik Kelly, sonsuz şansın olduğunu varsayar.
  - Ergodicity, tek bir iflasın (0 kasa) oyunu bitirdiğini bilir.
  - Volatilite arttıkça, bahis boyutu logaritmik olarak düşürülür.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from loguru import logger

@dataclass
class ErgodicityResult:
    """Optimizasyon sonucu."""
    optimal_f: float         # Önerilen kasa yüzdesi (0.0 - 1.0)
    expected_growth_rate: float  # Tahmini büyüme (log-wealth)
    volatility_tax: float    # Volatilite nedeniyle kaybedilen büyüme
    is_safe: bool            # Güvenli mi?

class ErgodicityOptimizer:
    """Zaman-ortalama kasa büyüme optimize edici.
    
    Portföyün toplam varyansını ve beklenen getirisini kullanarak
    'Geometric Mean'i maksimize eden oranı bulur.
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion # 1.0 = Standard Ergodicity, >1.0 = Ultra Conservative
        logger.debug(f"ErgodicityOptimizer başlatıldı (risk_aversion={risk_aversion})")

    def optimize(self, edge: float, odds: float, 
                 portfolio_variance: float = 0.0) -> ErgodicityResult:
        """
        Optimal bahis miktarını hesapla.
        
        Args:
            edge: Beklenen getiri (EV) - (P*Odds - 1)
            odds: Verilen oran
            portfolio_variance: Mevcut açık pozisyonların toplam varyansı (volatilite)
            
        Formül:
            Growth Rate g(f) = E[log(1 + f * X)]
            ≈ f * E[X] - (f^2 / 2) * Var[X]
        """
        # Klasik Kelly Payı
        # f = Edge / (Odds - 1)
        kelly_f = edge / (odds - 1) if odds > 1 else 0
        
        # Ergodicity Düzeltmesi (Volatility Tax)
        # Volatilite arttıkça büyüme 'tax' yer.
        # g = avg_return - variance/2
        
        # Basitleştirilmiş Ergodicity-based sizing:
        # f* = E[X] / (Var[X] * risk_aversion)
        # Burada Var[X] hem bahis volatilitesini hem de portföy volatilitesini içerir.
        
        # Bahis volatilitesi (Bernoulli trial): p*(1-p) * (odds)^2 gibi...
        # Pratik yaklaşım: Fractional Kelly'yi volatiliteye duyarlı hale getir.
        
        variance = portfolio_variance + (kelly_f ** 2)
        
        # Optimal f hesapla (Log-wealth maximization)
        # f_opt = E[X] / Var[X]
        if variance > 0:
            optimal_f = edge / (variance * self.risk_aversion * odds)
        else:
            optimal_f = kelly_f
            
        # Sınırla (0 - 0.20) -> %20'den fazlası tek bahse riskli
        optimal_f = max(0.0, min(optimal_f, 0.20))
        
        # Volatilite Vergisi: (f^2 / 2) * Variance
        tax = (optimal_f ** 2 / 2) * variance
        growth = (optimal_f * edge) - tax
        
        return ErgodicityResult(
            optimal_f=optimal_f,
            expected_growth_rate=growth,
            volatility_tax=tax,
            is_safe=growth > 0
        )

    def calculate_portfolio_volatility(self, open_bets: list[dict]) -> float:
        """Açık pozisyonların toplam varyansını (riskini) hesaplar."""
        if not open_bets:
            return 0.0
            
        # Basit toplamsal varyans (korelasyon yok varsayımıyla)
        # Daha gelişmiş modelde CorrelationMatrix kullanılmalı
        total_var = 0.0
        for bet in open_bets:
            f = bet.get("size_pct", 0.0)
            o = bet.get("odds", 2.0)
            # p * (1-p) * odds^2 basitleştirmesi
            p = 1.0 / o
            var = p * (1-p) * (f**2)
            total_var += var
            
        return total_var
