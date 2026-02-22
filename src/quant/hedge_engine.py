"""
hedge_engine.py – Hedge ve Arbitraj Hesaplayıcı.

Amacı:
1. Mevcut bahsi "Hedge" etmek (Risk azaltmak veya kârı garantilemek).
2. "Adil Cashout" değerini hesaplayıp, bookmaker teklifiyle kıyaslamak.
3. Arbitraj fırsatlarını değerlendirmek.

Matematik:
Fair Cashout = Stake * (Original Odds / Current Odds)
(Bookmaker marjı düşülmeden teorik değer)
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class HedgeResult:
    action: str  # "HOLD", "CASH_OUT", "PARTIAL_CASH_OUT", "HEDGE_BET"
    fair_value: float
    current_profit: float
    hedge_stake: float = 0.0 # Eğer risk-free yapmak için karşı bahis alınacaksa
    roi: float = 0.0
    reason: str = ""

class HedgeEngine:
    def __init__(self, commission: float = 0.0):
        self.comm = commission # Varsa borsa komisyonu

    def calculate_cashout_value(self, stake: float, original_odds: float, current_odds: float) -> HedgeResult:
        """
        Adil Cashout değeri hesaplar.
        Senaryo: Bahis aldık (Back), şimdi bozdurmak istiyoruz.
        """
        if current_odds <= 1.01:
            # Maç bitmiş veya bitmek üzere, cashout ~ max win
            fair_val = stake * original_odds
        else:
            # Adil Değer = Potansiyel Kazanç / Güncel Oran
            # Çünkü şu an bu parayı (Potansiyel Kazanç) kazanmanın maliyeti (Güncel Oran) kadar risk primi var.
            fair_val = (stake * original_odds) / current_odds

        profit = fair_val - stake
        roi = (profit / stake) * 100
        
        # Karar Mekanizması
        action = "HOLD"
        reason = "Değer artışı yeterli değil veya negatif."
        
        # Eğer ROI %20 üzerindeyse ve maç riskliyse (bunu dışarıdan almamız lazım ama basit mantık)
        if roi > 20.0:
            action = "CASH_OUT"
            reason = f"Kâr %{roi:.1f} seviyesinde, realize edilebilir."
        elif roi < -30.0:
            action = "STOP_LOSS"
            reason = f"Zarar %{roi:.1f} seviyesinde, stop-loss."
            
        return HedgeResult(
            action=action,
            fair_value=fair_val,
            current_profit=profit,
            roi=roi,
            reason=reason
        )

    def calculate_arbitrage_stake(self, back_odds: float, lay_odds: float, back_stake: float) -> HedgeResult:
        """
        Arbitraj / Green-Book hesaplayıcı.
        Back bahsi alındı, Lay (karşı) bahsi alarak kârı garantileme.
        Exchange (Borsa) mantığı.
        """
        if lay_odds <= 1.01:
             return HedgeResult("ERROR", 0, 0, 0, 0, "Lay odds too low")

        # Lay Stake = (Back Odds / Lay Odds) * Back Stake
        # Bu miktar, her iki durumda da eşit kazanç/kayıp sağlar (Sıfır risk).
        hedge_stake = (back_odds / lay_odds) * back_stake
        
        # Kâr Hesabı
        # Durum 1 (Back Kazanır): (BackStake * BackOdds) - BackStake - (HedgeStake * (LayOdds - 1))
        # Durum 2 (Lay Kazanır): HedgeStake - BackStake
        
        # Basitçe: Kâr = BackStake * (BackOdds - LayOdds) / LayOdds (Komisyonsuz)
        profit = back_stake * (back_odds - lay_odds) / lay_odds
        roi = (profit / back_stake) * 100
        
        if profit > 0:
            action = "ARBITRAGE"
            reason = f"Garantili kâr: {profit:.2f} (ROI: %{roi:.1f})"
        else:
            action = "NO_ARB"
            reason = "Arbitraj fırsatı yok (Negatif kâr)."
            
        return HedgeResult(
            action=action,
            fair_value=0, # Cashout değil, yeni bahis
            current_profit=profit,
            hedge_stake=hedge_stake,
            roi=roi,
            reason=reason
        )
