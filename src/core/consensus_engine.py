"""
consensus_engine.py – Modeller arası diplomasi ve oylama katmanı.

Birden fazla modelin (Ajanın) farklı tahminler yaptığı durumlarda, 
başarı geçmişine ve model güvenine dayalı bir konsensüs mekanizması kurar.
"""
from loguru import logger
from typing import List, Dict, Any
import numpy as np

class ConsensusEngine:
    def __init__(self, db: Any = None):
        self.db = db
        # Modellerin tarihsel başarı ağırlıkları
        self.model_weights = {
            "bayesian": 0.40,
            "spectral": 0.20,
            "lstm": 0.20,
            "dixon_coles": 0.20
        }

    async def resolve_signals(self, signals: List[dict]) -> Dict[str, Any]:
        """Gelen sinyaller arasından konsensüs ile en iyisini seçer."""
        if not signals:
            return {}

        # 1. Oylama (Voting)
        votes = {}
        for sig in signals:
            selection = sig.get("selection")
            weight = self.model_weights.get(sig.get("model"), 0.1)
            votes[selection] = votes.get(selection, 0) + (sig.get("confidence", 0.5) * weight)

        # 2. Kazananı Belirle
        winner = max(votes, key=votes.get) if votes else None
        winning_vote_strength = votes.get(winner, 0)

        # 3. Konsensüs Kontrolü
        # Eğer kazananın ağırlığı toplam ağırlığın %60'ından az ise, debate (tartışma) başlatılabilir.
        total_weight = sum(votes.values())
        consensus_ratio = winning_vote_strength / total_weight if total_weight > 0 else 0

        logger.info(f"[Consensus] Karar: {winner} | Güç: {consensus_ratio:.2%}")

        if consensus_ratio < 0.60:
            logger.warning("[Consensus] Düşük konsensüs! Ajanlar arası tartışma gerekebilir.")
            return {"status": "DEBATE_REQUIRED", "selection": winner, "strength": consensus_ratio}

        return {"status": "APPROVED", "selection": winner, "strength": consensus_ratio}

    def update_model_weight(self, model_name: str, pnl: float):
        """Modelin kârlılığına göre konsensüs ağırlığını günceller."""
        current_weight = self.model_weights.get(model_name, 0.1)
        # Kâr varsa ağırlığı artır, zarar varsa azalt (Adaptive Weighting)
        adjustment = 1.05 if pnl > 0 else 0.95
        self.model_weights[model_name] = round(current_weight * adjustment, 4)
        logger.info(f"[Consensus:Adapt] {model_name} yeni ağırlığı: {self.model_weights[model_name]}")
