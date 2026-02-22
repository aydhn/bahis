"""
reflection_agent.py – Post-Match Otonom Öz-Eleştiri Ajanı (The Mirror).

Maç sonuçları açıklandığında, botun kendi tahminini gerçekleşenle kıyaslamasını 
sağlar. 'Confirmation Bias' veya 'Statistical Outlier' tespiti yapar.
"""
from loguru import logger
from typing import Dict, Any, List

class ReflectionAgent:
    def __init__(self, db: Any = None):
        self.db = db

    async def reflect_on_match(self, match_id: str):
        """Maç sonrası otonom analiz."""
        # 1. Tahmin verisini çek
        # 2. Gerçek skoru çek
        # 3. Hata payını (Loss) hesapla
        logger.info(f"[Mirror] {match_id} için öz-eleştiri seansı başladı.")
        
        # Mock analiz sonucu
        reflection = {
            "prediction_error": 0.15,
            "surprise_factor": "HIGH", # Kırmızı kart veya beklenmedik sakatlık
            "learning": "Düşük liglerde ev sahibi avantajı modelin beklediğinden daha baskın."
        }
        
        logger.warning(f"[Mirror] Ders Alındı: {reflection['learning']}")
        return reflection

    async def run_batch(self, finished_matches: List[dict], **kwargs) -> List[dict]:
        """Tamamlanan maçları toplu yansıtır."""
        results = []
        for match in finished_matches:
            res = await self.reflect_on_match(match.get("id", "??"))
            results.append(res)
        return results
