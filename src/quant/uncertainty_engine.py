"""
uncertainty_engine.py – Belirsizlik Kuantifikasyonu (Epistemic vs Aleatoric).

Modelin 'ne kadar bilmediğini' ölçer. 
Epistemic: Modelin veri eksikliği (Stake düşürülmeli).
Aleatoric: Maçın doğal şansı (Stake Kelly'ye göre ayarlanmalı).
"""
import numpy as np
from loguru import logger
from typing import Dict, Any

class UncertaintyEngine:
    def __init__(self):
        pass

    def calculate_uncertainty(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Entropy tabanlı belirsizlik ayrımı yapar."""
        # Basitleştirilmiş Shannon Entropy
        # probabilities: [p_home, p_draw, p_away]
        total_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        
        # Epistemic: Modelin varyansı (Örnek: Bayesian posterior varyansı)
        # Aleatoric: Toplam entropi - Epistemic
        epistemic = 0.15 # Mock
        aleatoric = max(0, total_entropy - epistemic)
        
        return {
            "total_entropy": float(total_entropy),
            "epistemic": float(epistemic),
            "aleatoric": float(aleatoric),
            "risk_multiplier": 1.0 - epistemic # Bilgi eksikliği varsa stake'i kıs
        }

    def wrap_signal(self, signal: dict) -> dict:
        """Sinyale belirsizlik metriklerini ekler."""
        probs = np.array([signal.get("p_home", 0.33), signal.get("p_draw", 0.34), signal.get("p_away", 0.33)])
        metrics = self.calculate_uncertainty(probs)
        
        signal["uncertainty_metrics"] = metrics
        if metrics["epistemic"] > 0.3:
            signal["confidence"] *= 0.8
            signal["tags"] = signal.get("tags", []) + ["high_epistemic_risk"]
            
        return signal
