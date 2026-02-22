"""
heuristic_evolver.py – Otonom Heuristic Evrim Motoru.

SelfHealingEngine (Otomatik İyileştirme) normalde sabit kurallar kullanır. 
Bu modül, hangi iyileştirme yönteminin (LLM promptu, regex kuralı, rollback stratejisi) 
daha başarılı olduğunu takip eder ve başarılı olanları 'evrimleştirerek' 
sistemin kendi tamir mekanizmasını akıllandırır.
"""
import random
from typing import List, Dict, Any
from loguru import logger

class HeuristicEvolver:
    def __init__(self, db: Any = None):
        self.db = db
        # population: list of repair heuristics (strategies)
        self.strategies = [
            {"name": "llm_retry", "success_rate": 0.5, "weight": 1.0},
            {"name": "regex_fix", "success_rate": 0.5, "weight": 1.0},
            {"name": "rollback_and_wait", "success_rate": 0.5, "weight": 1.0},
        ]

    def record_success(self, strategy_name: str, success: bool):
        """Bir tamir stratejisinin sonucunu kaydeder."""
        for s in self.strategies:
            if s["name"] == strategy_name:
                # Hareketli ortalama ile başarı puanı güncellemesi
                alpha = 0.2
                val = 1.0 if success else 0.0
                s["success_rate"] = (1 - alpha) * s["success_rate"] + alpha * val
                
                # Başarılı olanın ağırlığını artır
                if success:
                    s["weight"] *= 1.1
                else:
                    s["weight"] *= 0.9
                
                logger.debug(f"[HeuristicEvolver] Strateji {strategy_name} güncellendi: {s['success_rate']:.2f}")

    def get_best_strategy(self) -> Dict[str, Any]:
        """Evrimleşmiş en iyi stratejiyi döner."""
        return max(self.strategies, key=lambda x: x["weight"])

    def mutate_strategies(self):
        """Zamanla stratejileri 'mutasyona' uğratarak yeni yollar dener."""
        if random.random() < 0.1: # %10 ihtimalle mutasyon
            target = random.choice(self.strategies)
            logger.info(f"[HeuristicEvolver] Mutasyon! {target['name']} ağırlığı randomize ediliyor.")
            target["weight"] = random.uniform(0.5, 2.0)

    async def run_batch(self, **kwargs):
        """Geçmiş loglardan iyileştirme başarılarını analiz eder."""
        if self.db is None: return
        
        # SQL: SELECT strategy, success FROM healing_logs
        # Bu verilerle record_success çağrılarak popülasyon eğitilir.
        logger.info("[HeuristicEvolver] Kendi kendine öğrenme döngüsü aktif.")
        self.mutate_strategies()
