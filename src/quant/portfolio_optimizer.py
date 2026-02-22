"""
portfolio_optimizer.py – Bahis portföyü optimizasyonu.

Markowitz Modern Portföy Teorisi'nin (MPT) bahislere uyarlanmış hali.
Amaç: Tek bir maça/takıma aşırı yüklenmeyi önlemek, korelasyon riskini ve
 varyansı düşürmek.

Yöntem:
- Basit Çeşitlendirme: Aynı anda en fazla N bahis, her bahse Max %Kasa.
- Korelasyon Kontrolü: Aynı takıma zıt bahisler veya ilişkili marketler var mı?
- Kelly Capping: Kelly kriterini (Full Kelly) volatiliteye göre kıs (Fractional).
"""

from dataclasses import dataclass, field
from loguru import logger
from typing import Any

@dataclass
class PortfolioState:
    bankroll: float
    active_exposure: float
    max_exposure_pct: float = 0.20  # Kasanın max %20'si risk altında olabilir

class PortfolioOptimizer:
    def __init__(self, db: Any = None, max_daily_risk: float = 0.15):
        self.db = db
        self.max_daily_risk = max_daily_risk  # Günlük max %15 kayıp riski
        self.active_bets: list[dict] = []
        
    def run_batch(self, **kwargs):
        """Pipeline entegrasyonu için batch metodu."""
        logger.info("[PortfolioOptimizer] Portföy optimize ediliyor...")
        # 1. Bekleyen emirleri al
        # pending = self.db.get_pending_bets()
        
        # 2. Risk kontrolü yap
        # approved = self.allocate_capital(pending)
        
        # 3. Onaylananları işaretle
        # self.db.update_bets(approved)
        pass

    def allocate_capital(self, candidates: list[dict], current_bankroll: float) -> list[dict]:
        """Aday bahislere sermaye dağıtımı yapar."""
        approved_bets = []
        total_staked = 0.0
        
        # Basit "Naive Diversification"
        # Her bahse max %2, toplam max %15
        
        max_stake_per_bet = current_bankroll * 0.02
        remaining_budget = current_bankroll * self.max_daily_risk
        
        for bet in candidates:
            # Kelly önerisi (varsayılan stake_pct)
            suggested_pct = bet.get("stake_pct", 0.01)
            suggested_amount = current_bankroll * suggested_pct
            
            # Capping
            final_amount = min(suggested_amount, max_stake_per_bet, remaining_budget)
            
            if final_amount > 10.0: # Min bahis tutarı 10 TL
                bet["allocated_stake"] = final_amount
                approved_bets.append(bet)
                remaining_budget -= final_amount
                total_staked += final_amount
                
            if remaining_budget <= 0:
                break
                
        logger.info(f"[Portfolio] {len(approved_bets)}/{len(candidates)} bahis onaylandı. Toplam Risk: ₺{total_staked:.2f}")
        return approved_bets
    
    def check_correlation(self, new_bet: dict) -> bool:
        """Yeni bahis, mevcut açık bahislerle çakışıyor mu?"""
        # Örn: Aynı maça hem MS1 hem MS2 oynanmaz.
        # Örn: Aynı takıma hem KG Var hem ÜST oynanıyorsa korelasyon var.
        match_id = new_bet.get("match_id")
        selection = new_bet.get("selection")
        
        for active in self.active_bets:
            if active.get("match_id") == match_id:
                # Aynı maç
                if active.get("selection") != selection:
                    # Hedge mi? Çelişki mi?
                    logger.warning(f"Korelasyon Uyarısı: Aynı maça farklı bahis ({match_id})")
                    return True
        return False
