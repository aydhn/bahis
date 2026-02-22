"""
news_aggregator.py – Bilgi Grafiği ve Haber Agregatörü.

Bu modül, internet üzerindeki haber sitelerini tarar ve NLP (spacy/transformers) 
kullanarak önemli bilgileri (sakatlık, transfer, moral durumu) ayıklar.
KnowledgeGraph yapısı kurarak takımlar arası ilişkileri günceller.
"""
from typing import List, Dict, Any
import random
from loguru import logger

class NewsAggregator:
    def __init__(self, db: Any = None):
        self.db = db
        self.sources = ["bbc.com/sport", "skysports.com", "mackolik.com"]
        self.knowledge_graph = {} # {team: [events]}

    async def crawl_news(self) -> List[Dict]:
        """Haber kaynaklarını tarar (Simüle)."""
        logger.info("[NewsAggregator] Haber taraması başlatıldı...")
        
        # Gerçekte Scrapy veya Requests ile çekilir
        mock_news = [
            {"team": "Galatasaray", "content": "Icardi'nin durumu belirsiz.", "sentiment": -0.6},
            {"team": "Real Madrid", "content": "Antrenmanda moral yüksek.", "sentiment": 0.8},
        ]
        return mock_news

    def extract_entities(self, text: str) -> Dict:
        """NLP ile metinden oyuncu ve durum çıkarımı yapar."""
        # spacy.load("en_core_web_sm") benzeri bir işlem yapılır
        return {"player": "Unknown", "condition": "Neutral"}

    async def run_batch(self, **kwargs):
        """Knowledge Graph'ı güncelleyen asıl döngü."""
        news = await self.crawl_news()
        for n in news:
            team = n["team"]
            if team not in self.knowledge_graph:
                self.knowledge_graph[team] = []
            self.knowledge_graph[team].append(n)
            
        logger.success(f"[NewsAggregator] {len(news)} yeni haber işlendi ve grafiğe eklendi.")

    def get_team_sentiment(self, team: str) -> float:
        """Bir takım hakkındaki son haberlerin duyarlılık skorunu döner."""
        events = self.knowledge_graph.get(team, [])
        if not events: return 0.0
        return sum(e["sentiment"] for e in events) / len(events)
