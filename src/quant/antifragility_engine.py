"""
antifragility_engine.py – Antikırılganlık ve Konveksite Analizi.

Nassim Taleb'in 'Antifragile' prensiplerine dayanır: 
  - Kırılgan (Fragile): Olaylardan zarar gören, uç değerlere karşı hassas. 
  - Dayanıklı (Robust): Olaylardan etkilenmeyen.
  - Antikırılgan (Antifragile): Olaylardan (volatiliteden, hatadan) fayda sağlayan.

Bu modül, yüksek oranlı 'Long-Shot' sürprizlerin matematiksel beklentisini (EV) 
değil, 'Asimetrisini' ölçer.
"""
import numpy as np
from typing import Dict, List, Any
from loguru import logger

class AntifragilityEngine:
    def __init__(self, db: Any = None):
        self.db = db

    def calculate_convexity(self, prob: float, odds: float) -> float:
        """
        Bir bahisin konveksite (eğrilik) skorunu hesaplar.
        Yüksek oranlı ama makul ihtimalli bahisler daha konvekstir.
        """
        if odds <= 1.0: return 0.0
        
        # Basit bir konveksite metrik: payout / probability_cost
        # Eğer oran, olasılığın tersinden çok daha yüksekse asimetri vardır.
        implied_prob = 1 / odds
        edge = prob - implied_prob
        
        if edge <= 0: return 0.0
        
        # Konveksite çarpanı: Oran yükseldikçe (outlier olma ihtimali) skor artar
        # Ancak risk de artar, bu yüzden logaritmik veya karekök ölçekleme kullanılır.
        convexity = (edge * np.sqrt(odds)) / implied_prob
        return float(np.clip(convexity, 0, 10))

    def evaluate_portfolio(self, bankroll: float, current_bets: List[Dict]) -> Dict[str, Any]:
        """Portföyün antikırılganlık seviyesini ölçer."""
        if not current_bets:
            return {"status": "STABLE", "antifragility_score": 1.0}

        avg_odds = np.mean([b["odds"] for b in current_bets])
        
        # 1. Barbell Stratejisi Kontrolü: 
        # Çoğunluk güvenli/düşük oran + Küçük bir kısım yüksek oran/yüksek risk.
        is_barbell = any(b["odds"] > 5.0 for b in current_bets) and any(b["odds"] < 1.5 for b in current_bets)
        
        score = 1.0
        if is_barbell: score += 0.5
        if avg_odds > 2.5: score += 0.3 # Pozitif asimetri eğilimi
        
        return {
            "antifragility_score": round(score, 2),
            "is_barbell": is_barbell,
            "status": "ANTIFRAGILE" if score > 1.5 else "ROBUST" if score >= 1.0 else "FRAGILE"
        }

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Sinyalleri antikırılganlık filtresinden geçirir."""
        scored_signals = []
        for sig in signals:
            prob = sig.get("probability", 0.0)
            odds = sig.get("odds", 1.0)
            convexity = self.calculate_convexity(prob, odds)
            
            sig["convexity_score"] = round(convexity, 2)
            # Eğer konveksite yüksekse sinyali 'PREMIUM' olarak işaretle
            if convexity > 2.0:
                sig["tags"] = sig.get("tags", []) + ["convex_opportunity"]
            
            scored_signals.append(sig)
        return scored_signals
