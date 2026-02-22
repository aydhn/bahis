"""
adverse_selection.py – Ters Seçilim (Adverse Selection) ve Smart Router.

Bu modül, infaz (execution) anında oran hızını (velocity) ölçer. 
Büyük bahisçiler veya 'sharp'lar pazara girdiğinde oranlar hızla düşer. 
Eğer biz bahsi alırken oran bizim aleyhimize (veya çok hızlı lehimize) 
kayıyorsa, sistem "Adverse Selection" (Ters Seçilim) riskinden dolayı 
işlemi durdurur.

Metrik: LOB (Limit Order Book) benzeri bir 'Price Velocity' analizi.
"""
import time
from typing import Dict, List, Any, Optional
from loguru import logger

class AdverseSelectionGuard:
    def __init__(self, velocity_threshold: float = 0.05, window_seconds: int = 300):
        self.velocity_threshold = velocity_threshold # %5'ten fazla değişim riskli
        self.window = window_seconds
        # price_history[match_id_selection] = [(timestamp, odds), ...]
        self.price_history: Dict[str, List[tuple]] = {}

    def update_price(self, match_id: str, selection: str, odds: float):
        """Oran geçmişini günceller."""
        key = f"{match_id}_{selection}"
        if key not in self.price_history:
            self.price_history[key] = []
        
        self.price_history[key].append((time.time(), odds))
        
        # Pencere dışındakileri temizle
        now = time.time()
        self.price_history[key] = [p for p in self.price_history[key] if now - p[0] <= self.window]

    def check_execution_safety(self, match_id: str, selection: str, current_odds: float) -> bool:
        """
        İşlemin güvenli olup olmadığını kontrol eder.
        True: Güvenli, False: TOXIC FLOW / ADVERSE SELECTION
        """
        key = f"{match_id}_{selection}"
        history = self.price_history.get(key, [])
        if len(history) < 2:
            return True # Yeterli veri yok, güvenli kabul et
            
        old_odds = history[0][1]
        change_pct = (current_odds / old_odds) - 1
        
        # Eğer oran bizim lehimize %5'ten fazla arttıysa (Garip bir durum, bilgi bizden kaçıyor olabilir)
        # Veya aleyhimize %5 kapandıysa (Vagonu kaçırdık, değer kalmadı)
        if abs(change_pct) > self.velocity_threshold:
            logger.warning(
                f"[AdverseSelection] {key} reddedildi! "
                f"Velocity: {change_pct:+.2%}. Toxic flow tespiti."
            )
            return False
            
        return True

    async def run_batch(self, signals: List[Dict], live_odds: List[Dict]) -> List[Dict]:
        """Sinyalleri infaz öncesi filtreler."""
        # Live odds ile geçmişi güncelle
        for o in live_odds:
            self.update_price(o["match_id"], o["selection"], o["odds"])
            
        safe_signals = []
        for s in signals:
            is_safe = self.check_execution_safety(
                s.get("match_id", ""), 
                s.get("selection", ""), 
                s.get("odds", 1.0)
            )
            if is_safe:
                safe_signals.append(s)
            else:
                s["rejection_reason"] = "ADVERSE_SELECTION_DETECTED"
                
        return safe_signals
