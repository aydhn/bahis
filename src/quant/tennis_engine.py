"""
tennis_engine.py – Tenis için Markov Zinciri Modelleme.

Tenis maçlarını puan-oyun-set hiyerarşisinde modeller. Her bir servis puanının 
kazanılma olasılığını (p) baz alarak tüm maç sonucunu tahmin eder.
"""
import numpy as np
from loguru import logger
from typing import Dict, Any

class TennisEngine:
    def __init__(self):
        pass

    def game_win_prob(self, p_server: float) -> float:
        """Bir oyuncunun kendi servisinde oyunu kazanma olasılığı (Markov)."""
        # p: servisi karşılayan oyuncunun puan kazanma olasılığı
        p = p_server
        q = 1 - p
        # 40-40 (Deuce) öncesi ve sonrası olasılık kombinasyonu
        prob = (p**4 * (15 - 4*p - (10*p**2 * q**2) / (1 - 2*p*q)))
        return float(np.clip(prob, 0, 1))

    def set_win_prob(self, p_game_server: float, p_game_receiver: float) -> float:
        """Bir oyuncunun seti kazanma olasılığı."""
        # Basitleştirilmiş model (Tie-break hariç)
        # Gerçekte 6 oyun kazananın seti alması (veya tie-break) modellenir
        return (p_game_server + (1 - p_game_receiver)) / 2

    def process_match(self, match_data: dict) -> dict:
        """Tenis maç verisini işler."""
        p_s1 = 0.65 # Oyuncu 1'in servis puanı kazanma olasılığı
        p_s2 = 0.62 # Oyuncu 2'nin servis puanı kazanma olasılığı
        
        prob_g1 = self.game_win_prob(p_s1)
        prob_m1 = self.set_win_prob(prob_g1, self.game_win_prob(p_s2))
        
        return {
            "match_id": match_data.get("id"),
            "sport": "tennis",
            "p_win_1": round(prob_m1, 4),
            "recommendation": "ATP Winner P1" if prob_m1 > 0.60 else "No Bet"
        }
