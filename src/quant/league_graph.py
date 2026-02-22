"""
league_graph.py – Lig Çizge Modellemesi.

Futbol ligi kapalı bir sistemdir. Takımlar birbirleriyle tekrar eden maçlar yapar.
Bu yapı, bir 'Graph' (çizge) olarak modellenebilir:
  - Düğüm (Node): Takım
  - Kenar (Edge): Maç sonucu (Ağırlıklı directed edge)

Bu modül, NetworkX kullanarak merkeziyet puanları (Eigenvector, PageRank) 
ve 'Nemesis' (ters gelen rakip) ilişkilerini hesaplar.
"""
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Tuple
from loguru import logger

class LeagueGraphModel:
    def __init__(self, db: Any = None):
        self.db = db
        self.graph = nx.DiGraph()

    def build_graph(self, matches: List[Dict]):
        """Geçmiş maçlardan lig çizgesini oluşturur."""
        self.graph.clear()
        for m in matches:
            home = m["home_team"]
            away = m["away_team"]
            home_goals = m["home_score"]
            away_goals = m["away_score"]
            
            # Galibiyete göre edge ağırlığı (Home -> Away)
            if home_goals > away_goals:
                weight = home_goals - away_goals
                # Home, Away'i 'yendi'
                if self.graph.has_edge(home, away):
                    self.graph[home][away]["weight"] += weight
                else:
                    self.graph.add_edge(home, away, weight=weight)
            elif away_goals > home_goals:
                weight = away_goals - home_goals
                # Away, Home'u 'yendi'
                if self.graph.has_edge(away, home):
                    self.graph[away][home]["weight"] += weight
                else:
                    self.graph.add_edge(away, home, weight=weight)

    def get_centrality_ranks(self) -> Dict[str, float]:
        """PageRank algoritması ile takımların 'dominance' puanlarını hesaplar."""
        if not self.graph.nodes:
            return {}
        try:
            return nx.pagerank(self.graph, weight="weight")
        except Exception as e:
            logger.error(f"PageRank hatası: {e}")
            return {}

    def find_nemesis(self, team: str) -> List[Tuple[str, float]]:
        """Bir takımı en çok zorlayan rakipleri bulur."""
        if team not in self.graph:
            return []
        
        # Takıma karşı gelen (in_edges) en güçlü edge'ler
        nemesis_list = []
        for u, v, d in self.graph.in_edges(team, data=True):
            nemesis_list.append((u, d["weight"]))
        
        return sorted(nemesis_list, key=lambda x: x[1], reverse=True)

    async def run_batch(self, days: int = 365, **kwargs):
        """DB'den veriyi çekip analizi çalıştırır."""
        if self.db is None:
            return

        logger.info("[LeagueGraph] Çizge analizi başlatılıyor...")
        query = f"""
        SELECT home_team, away_team, home_score, away_score
        FROM matches
        WHERE status = 'finished' AND match_date >= CURRENT_DATE - INTERVAL '{days} DAY'
        """
        try:
            results = self.db.query(query).to_dicts()
            if not results:
                return
                
            self.build_graph(results)
            ranks = self.get_centrality_ranks()
            
            # En güçlü 5 takımı logla
            top_5 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"[LeagueGraph] Top 5 Dominant Takım: {top_5}")
            
        except Exception as e:
            logger.error(f"LeagueGraph batch hatası: {e}")
