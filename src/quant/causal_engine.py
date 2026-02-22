"""
causal_engine.py – Nedensel Çıkarım (Causal Inference) Motoru.

Pearl'ün "Do-calculus" ve "Counterfactuals" (Karşı olgular) teorisine dayanır. 
Sadece X ve Y korele mi diye bakmaz, "Eğer X'i değiştirirsek (do(X)) Y ne olur?" 
sorusunu simüle eder.

Örn: "Forvetin sakat olması gol sayısını korele mi ediyor yoksa neden mi oluyor?"
"""
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger

class CausalInferenceEngine:
    def __init__(self, db: Any = None):
        self.db = db
        # Causal Directed Acyclic Graph (DAG) - Nedensel Hiyerarşi
        # Hangi değişken hangisini doğrudan etkiler?
        self.dag = {
            "fitness": ["performance"],
            "weather": ["performance", "tactics"],
            "tactics": ["performance"],
            "performance": ["expected_goals"],
            "expected_goals": ["win_prob"],
            "luck": ["win_prob"],
            "referee": ["win_prob"]
        }
        # Nedensel Ağırlıklar (Katsayılar)
        self.causal_weights = {
            "fitness": 0.4,
            "tactics": 0.35,
            "weather": 0.25,
            "performance": 0.9,
            "expected_goals": 0.85
        }

    def simulate_do_intervention(self, data: Dict[str, Any], variable: str, value: float) -> float:
        """
        Interventional Distribution: P(Y | do(X=x))
        'Eğer X değişkenine dışarıdan müdahale edersek sonuç ne olur?'
        """
        # Baseline probability
        p_y = data.get("confidence", 0.5)
        
        # Değişkenin nedensel hiyerarşideki yeri
        # Eğer müdahale edilen değişken 'performance'ı etkiliyorsa:
        if variable in ["fitness", "tactics", "weather"]:
            impact = (value - 0.5) * self.causal_weights.get(variable, 0.1)
            # Etkiyi hiyerarşi boyunca yay (Performance -> xG -> WinProb)
            p_y += impact * self.causal_weights["performance"] * self.causal_weights["expected_goals"]
            
        return float(np.clip(p_y, 0.01, 0.99))

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Sinyallerin nedensel geçerliliğini kontrol eder (Do-Calculus)."""
        if not signals: return []
        
        enhanced_signals = []
        logger.info(f"[Causal] {len(signals)} sinyal üzerinde Nedensel Analiz (Do-Calculus) başlatıldı.")
        
        for sig in signals:
            try:
                # 1. 'Fitness' (Yorgunluk) müdahalesi simülasyonu
                # Eğer takım %100 fit olsaydı ne olurdu? vs Eğer %50 fit olsaydı?
                p_do_fit = self.simulate_do_intervention(sig, "fitness", 1.0)
                p_do_fatigue = self.simulate_do_intervention(sig, "fitness", 0.0)
                
                # Causal Lift (ATE - Average Treatment Effect tahmini)
                causal_lift = p_do_fit - p_do_fatigue
                sig["causal_lift"] = round(causal_lift, 3)
                
                # 2. Causal Filter: Eğer nedensel etki çok düşükse, korelasyon yanıltıcı olabilir.
                # 'Confidence' değerini nedensel kanıta göre ayarla.
                # Eğer causal_lift negatifse veya çok küçükse, sinyal 'zayıf nedensellik' taşır.
                causal_score = sig.get("confidence", 0.5) * (1 + (causal_lift * 0.2))
                sig["confidence"] = round(float(np.clip(causal_score, 0, 1)), 3)
                
                if causal_lift > 0.15:
                    sig["tags"] = sig.get("tags", []) + ["high_causality"]
                    logger.debug(f"[Causal] {sig.get('match_id')} için güçlü nedensel bağ: {causal_lift:.2f}")
                
                enhanced_signals.append(sig)
            except Exception as e:
                logger.debug(f"[Causal] Sinyal analiz hatası: {e}")
                enhanced_signals.append(sig)

        return enhanced_signals
