"""
debate_engine.py – Çok Ajanlı Tartışma ve Konsensüs Mekanizması.

Sistemde birden fazla model (Poisson, Dixon-Coles, RL, ELO) farklı sonuçlar üretebilir. 
DebateEngine, bu modelleri birer 'ajan' gibi konuşturur. Eğer modeller arasında 
ciddi bir uyuşmazlık varsa (örn. bir model Ev Sahibi derken diğeri Beraberlik diyorsa),
sinyalin güven skoru düşürülür veya sinyal tamamen reddedilir.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from dataclasses import dataclass

@dataclass
class AgentOpinion:
    agent_name: str
    prediction: str # Örn: 'HOME_WIN', 'DRAW', 'AWAY_WIN'
    confidence: float
    probability: float

class MultiAgentDebateEngine:
    def __init__(self, threshold_agreement: float = 0.7):
        self.threshold = threshold_agreement

    async def debate(self, match_id: str, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """
        Modeller arasındaki uyuşmazlığı analiz eder ve nihai konsensüs kararı verir.
        """
        if not opinions:
            return {"status": "ABSTAIN", "consensus_prob": 0.0, "reason": "No opinions"}

        logger.info(f"[DebateEngine] {match_id} için {len(opinions)} ajan tartışıyor...")

        # 1. Tahmin Dağılımı
        votes = {}
        for op in opinions:
            votes[op.prediction] = votes.get(op.prediction, 0) + op.confidence
            
        # En çok oy alan tahmin
        winning_pred = max(votes, key=votes.get)
        total_confidence = sum(votes.values())
        agreement_ratio = votes[winning_pred] / total_confidence

        # 2. Ortalama Olasılık
        avg_prob = sum(op.probability for op in opinions) / len(opinions)

        # 3. Konsensüs Kararı
        status = "CONSENSUS"
        if agreement_ratio < self.threshold:
            status = "DISSENT" # Modeller anlaşamadı
            logger.warning(f"[DebateEngine] {match_id} için uyuşmazlık! Oran: {agreement_ratio:.2f}")

        # Eğer çok düşük bir anlaşma varsa (örn. 0.4 altı), reddet
        if agreement_ratio < 0.4:
            status = "REJECTED"

        return {
            "match_id": match_id,
            "prediction": winning_pred,
            "agreement_ratio": round(agreement_ratio, 2),
            "consensus_prob": round(avg_prob, 3),
            "status": status,
            "opinions": [vars(o) for o in opinions]
        }

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Orchestrator uyumlu batch metodu."""
        # Bu metod normalde modellerden gelen ham tahminleri alır ve debate eder.
        # Şimdilik örnek bir toplu işlem mantığı:
        results = []
        for sig in signals:
            # Sinyal içindeki model tahminlerini ajan görüşlerine çevir
            ops = []
            if "poisson_prob" in sig:
                ops.append(AgentOpinion("Poisson", "HOME_WIN" if sig["poisson_prob"] > 0.5 else "OTHER", 0.8, sig["poisson_prob"]))
            if "dc_prob" in sig:
                ops.append(AgentOpinion("DixonColes", "HOME_WIN" if sig["dc_prob"] > 0.5 else "OTHER", 0.9, sig["dc_prob"]))
            
            res = await self.debate(sig.get("match_id", "unknown"), ops)
            results.append(res)
        return results
