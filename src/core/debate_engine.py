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
    def __init__(self, threshold_agreement: float = 0.7, llm_backend: str = "ollama"):
        self.threshold = threshold_agreement
        self._llm = llm_backend

    async def debate(self, match_id: str, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """Modeller arasındaki uyuşmazlığı analiz eder."""
        if not opinions:
            return {"status": "ABSTAIN", "consensus_prob": 0.0, "reason": "No opinions"}

        logger.info(f"[DebateEngine] {match_id} için {len(opinions)} ajan tartışıyor...")
        votes = {}
        for op in opinions:
            votes[op.prediction] = votes.get(op.prediction, 0) + op.confidence
            
        winning_pred = max(votes, key=votes.get)
        total_confidence = sum(votes.values())
        agreement_ratio = votes[winning_pred] / total_confidence
        avg_prob = sum(op.probability for op in opinions) / len(opinions)

        status = "CONSENSUS"
        if agreement_ratio < self.threshold:
            status = "DISSENT"
        if agreement_ratio < 0.4:
            status = "REJECTED"

        return {
            "match_id": match_id,
            "prediction": winning_pred,
            "agreement_ratio": round(agreement_ratio, 2),
            "consensus_prob": round(avg_prob, 3),
            "status": status
        }

    async def socratic_debate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """LLM tabanlı Tez/Antitez diyaloğu."""
        home = signal.get("home_team", "Home")
        away = signal.get("away_team", "Away")
        selection = signal.get("selection", "?")
        
        # Yerel LLM simülasyonu (Gerçekte Ollama API çağrılır)
        # Tez: Neden kazanır?
        # Antitez: Neden kaybeder?
        tez = f"{home}, iç sahada çok güçlü ve {selection} için avantajlı."
        antitez = f"{away} savunması derin blokta beklerse sürpriz yapabilir."
        
        reduction = 0.05 if "sürpriz" in antitez.lower() else 0.0
        return {
            "thesis": tez,
            "antithesis": antitez,
            "adjustment": -reduction
        }

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Sinyalleri tartışma süzgecinden geçirir."""
        enhanced = []
        for sig in signals:
            try:
                # 1. Kantitatif Debate (Modeller arası)
                # (Ajan görüşleri burada oluşturulur)
                
                # 2. Kalitatif Debate (LLM)
                debate_res = await self.socratic_debate(sig)
                sig["confidence"] = round(sig.get("confidence", 0.5) + debate_res["adjustment"], 3)
                sig["debate_summary"] = f"Tez: {debate_res['thesis']} | Antitez: {debate_res['antithesis']}"
                enhanced.append(sig)
            except Exception:
                enhanced.append(sig)
        return enhanced
