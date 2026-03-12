"""
graph_rag.py – GraphRAG (Knowledge Graph + LLM).

Standart RAG sadece metinleri bulur. GraphRAG ise noktaları
birleştirir: "TD istifa etti" + "Kaptan mutsuz" → "Kaos Var".

Kavramlar:
  - Knowledge Graph: Haberleri düğüm (Node) ve ilişki (Edge) olarak
    Neo4j'de saklar
  - GraphRAG: LLM'in grafik üzerinde gezinerek gizli bağlantıları
    bulması (Reasoning over Graph)
  - Entity Extraction: Haberlerden varlık (takım, oyuncu, olay) çıkarma
  - Cypher QA Chain: LLM'in Neo4j'e Cypher sorgusu yazıp cevap alması
  - Community Detection: İlişkili haberlerin kümelenmesi (Louvain)

Akış:
  1. Haber RAG'den (news_rag.py) gelen haberleri al
  2. Varlık çıkarımı (NER): Takım, Oyuncu, Olay, Duygu
  3. Neo4j'e kaydet: (Haber)-[:HAKKINDA]->(Takım), (Oyuncu)-[:BAHSEDILDI]->(Haber)
  4. LLM'e sor: "Bu maçta gizli bir kriz var mı?" → Cypher ile grafik tara
  5. Sentiment + bağlam = "Kriz Skoru"

Teknoloji: LangChain + Neo4j (GraphCypherQAChain)
Fallback: NetworkX + basit NER + şablon çıkarım
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from loguru import logger

# Neo4j sürücü
try:
    from neo4j import GraphDatabase as Neo4jDriver
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False

# LangChain GraphQA
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain
    from langchain_community.llms import Ollama
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False
    try:
        from langchain.graphs import Neo4jGraph
        from langchain.chains import GraphCypherQAChain
        LANGCHAIN_OK = True
    except ImportError:
        pass

# NetworkX fallback
try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False

# LLM
try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False



OLLAMA_URL = "http://localhost:11434/api/generate"


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class Entity:
    """Haberden çıkarılan varlık."""
    name: str = ""
    entity_type: str = ""   # "team" | "player" | "coach" | "event" | "venue"
    sentiment: float = 0.0  # -1 ile 1 arası


@dataclass
class NewsNode:
    """Bilgi grafiğindeki haber düğümü."""
    news_id: str = ""
    title: str = ""
    source: str = ""
    timestamp: float = 0.0
    sentiment: float = 0.0
    entities: list[Entity] = field(default_factory=list)


@dataclass
class CrisisReport:
    """Kriz analizi raporu."""
    team: str = ""
    match_id: str = ""
    crisis_score: float = 0.0      # 0-1 (0=sakin, 1=kriz)
    crisis_level: str = "stable"   # "stable" | "tension" | "crisis" | "meltdown"
    # Kanıtlar
    negative_news_count: int = 0
    connected_events: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    hidden_connections: list[str] = field(default_factory=list)
    # LLM
    llm_analysis: str = ""
    # Meta
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  BASIT NER (Named Entity Recognition)
# ═══════════════════════════════════════════════
TEAM_KEYWORDS = [
    "galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor",
    "başakşehir", "adana demirspor", "alanyaspor", "antalyaspor",
    "gaziantep", "hatayspor", "kayserispor", "konyaspor",
    "sivasspor", "kasımpaşa", "pendikspor", "samsunspor",
    "rizespor", "ankaragücü", "bodrum", "eyüpspor",
]

EVENT_KEYWORDS = {
    "sakatlık": "injury",
    "sakatlandı": "injury",
    "ceza": "suspension",
    "kırmızı kart": "red_card",
    "istifa": "resignation",
    "transfer": "transfer",
    "kadro dışı": "exclusion",
    "kavga": "conflict",
    "moral": "morale",
    "şampiyonluk": "title_race",
    "küme düşme": "relegation",
    "gol": "goal",
}

NEGATIVE_WORDS = {
    "sakatlık", "sakatlandı", "istifa", "kavga", "kaos", "kriz",
    "mağlubiyet", "hezimet", "kırmızı", "ceza", "kadro dışı",
    "moral bozukluğu", "gerginlik", "tartışma", "hayal kırıklığı",
    "düşüş", "çöküş", "panik", "deprem", "fiyasko",
}

POSITIVE_WORDS = {
    "galibiyet", "zafer", "form", "motivasyon", "transfer",
    "şampiyonluk", "gol", "rekor", "başarı", "moral",
    "coşku", "umut", "iyileşme", "dönüş",
}


def simple_ner(text: str) -> list[Entity]:
    """Basit kural tabanlı varlık çıkarımı."""
    entities = []
    text_lower = text.lower()

    # Takım tespiti
    for team in TEAM_KEYWORDS:
        if team in text_lower:
            entities.append(Entity(
                name=team.title(),
                entity_type="team",
            ))

    # Olay tespiti
    for keyword, event_type in EVENT_KEYWORDS.items():
        if keyword in text_lower:
            entities.append(Entity(
                name=keyword,
                entity_type="event",
            ))

    # Duygu analizi (basit)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    total = neg_count + pos_count
    if total > 0:
        sentiment = (pos_count - neg_count) / total
        for e in entities:
            e.sentiment = round(sentiment, 2)

    return entities


def text_sentiment(text: str) -> float:
    """Basit duygu skoru."""
    text_lower = text.lower()
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    total = neg + pos
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


# ═══════════════════════════════════════════════
#  GRAPHRAG ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class GraphRAG:
    """Knowledge Graph + LLM ile haber analizi.

    Kullanım:
        grag = GraphRAG()

        # Haber ekle
        grag.ingest_news([
            {"title": "GS kaptanı sakatlandı", "source": "spor"},
            {"title": "GS teknik direktörü istifa sinyali verdi", "source": "spor"},
        ], team="Galatasaray")

        # Kriz analizi
        report = grag.analyze_crisis("Galatasaray", match_id="gs_fb_2026")

        # Soru sor
        answer = grag.ask("Bu maçta takımı etkileyen gizli kriz var mı?")
    """

    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: Optional[str] = None,
                 llm_backend: str = "auto"):
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_pass = neo4j_password
        self._llm_backend = llm_backend
        self._driver: Any = None
        self._qa_chain: Any = None

        # In-memory grafik (fallback)
        self._mem_graph: Any = nx.DiGraph() if NX_OK else None
        self._news_store: list[NewsNode] = []

        self._connect()
        logger.debug(
            f"[GraphRAG] Başlatıldı (backend={llm_backend}, "
            f"neo4j={'bağlı' if self._driver else 'fallback'})"
        )

    def _connect(self) -> None:
        """Neo4j bağlantısı."""
        if NEO4J_OK:
            try:
                self._driver = Neo4jDriver.driver(
                    self._neo4j_uri,
                    auth=(self._neo4j_user, self._neo4j_pass),
                )
                self._driver.verify_connectivity()

                if LANGCHAIN_OK:
                    try:
                        graph = Neo4jGraph(
                            url=self._neo4j_uri,
                            username=self._neo4j_user,
                            password=self._neo4j_pass,
                        )
                        llm = Ollama(model="llama3:8b")
                        self._qa_chain = GraphCypherQAChain.from_llm(
                            llm, graph=graph, verbose=False,
                        )
                    except Exception as e:
                        logger.debug(f"Exception caught: {e}")
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                self._driver = None

    def ingest_news(self, news_items: list[dict],
                      team: str = "") -> int:
        """Haberleri bilgi grafiğine yükle."""
        ingested = 0

        for item in news_items:
            title = item.get("title", "")
            source = item.get("source", "")
            if not title:
                continue

            news_id = hashlib.md5(title.encode()).hexdigest()[:12]
            entities = simple_ner(title)
            sentiment = text_sentiment(title)

            node = NewsNode(
                news_id=news_id,
                title=title,
                source=source,
                timestamp=time.time(),
                sentiment=sentiment,
                entities=entities,
            )
            self._news_store.append(node)

            # Neo4j'e yaz
            if self._driver:
                try:
                    with self._driver.session() as session:
                        session.run(
                            "MERGE (n:News {id: $id}) "
                            "SET n.title = $title, n.source = $source, "
                            "n.sentiment = $sentiment, n.timestamp = $ts",
                            id=news_id, title=title, source=source,
                            sentiment=sentiment, ts=time.time(),
                        )
                        for ent in entities:
                            label = ent.entity_type.title()
                            session.run(
                                f"MERGE (e:{label} {{name: $name}}) "
                                f"MERGE (n:News {{id: $nid}}) "
                                f"MERGE (n)-[:MENTIONS {{sentiment: $sent}}]->(e)",
                                name=ent.name, nid=news_id,
                                sent=ent.sentiment,
                            )
                        if team:
                            session.run(
                                "MERGE (t:Team {name: $team}) "
                                "MERGE (n:News {id: $nid}) "
                                "MERGE (n)-[:ABOUT]->(t)",
                                team=team, nid=news_id,
                            )
                except Exception as e:
                    logger.debug(f"Exception caught: {e}")

            # In-memory grafik
            if self._mem_graph is not None:
                self._mem_graph.add_node(
                    news_id, type="news", title=title,
                    sentiment=sentiment,
                )
                for ent in entities:
                    ent_id = f"{ent.entity_type}:{ent.name}"
                    self._mem_graph.add_node(
                        ent_id, type=ent.entity_type, name=ent.name,
                    )
                    self._mem_graph.add_edge(
                        news_id, ent_id, relation="mentions",
                        sentiment=ent.sentiment,
                    )
                if team:
                    team_id = f"team:{team}"
                    self._mem_graph.add_node(
                        team_id, type="team", name=team,
                    )
                    self._mem_graph.add_edge(
                        news_id, team_id, relation="about",
                    )

            ingested += 1

        logger.debug(f"[GraphRAG] {ingested} haber yüklendi.")
        return ingested

    def analyze_crisis(self, team: str,
                         match_id: str = "",
                         lookback_hours: float = 48.0) -> CrisisReport:
        """Takım için kriz analizi."""
        report = CrisisReport(team=team, match_id=match_id)
        cutoff = time.time() - lookback_hours * 3600

        # İlgili haberleri bul
        relevant = [
            n for n in self._news_store
            if n.timestamp >= cutoff
            and any(
                e.name.lower() == team.lower()
                or team.lower() in e.name.lower()
                for e in n.entities
            )
        ]

        if not relevant:
            report.crisis_level = "stable"
            report.crisis_score = 0.0
            report.recommendation = f"'{team}' için son {lookback_hours:.0f}h'de haber bulunamadı."
            report.method = "no_data"
            return report

        # Negatif haber sayısı
        neg_news = [n for n in relevant if n.sentiment < -0.1]
        report.negative_news_count = len(neg_news)

        # Bağlı olaylar
        all_events = set()
        all_entities = set()
        for n in relevant:
            for e in n.entities:
                all_entities.add(e.name)
                if e.entity_type == "event":
                    all_events.add(e.name)

        report.connected_events = list(all_events)[:10]
        report.key_entities = list(all_entities)[:10]

        # Grafik üzerinde gizli bağlantılar
        if self._mem_graph is not None and NX_OK:
            team_id = f"team:{team}"
            if self._mem_graph.has_node(team_id):
                neighbors = set()
                for news_id in self._mem_graph.predecessors(team_id):
                    for ent_id in self._mem_graph.successors(news_id):
                        if ent_id != team_id:
                            node_data = self._mem_graph.nodes.get(ent_id, {})
                            neighbors.add(node_data.get("name", ent_id))

                # İki-hop bağlantılar (gizli)
                for ent_id in list(neighbors):
                    for n2 in self._mem_graph.predecessors(f"event:{ent_id}"):
                        n2_data = self._mem_graph.nodes.get(n2, {})
                        if n2_data.get("type") == "news":
                            title = n2_data.get("title", "")
                            if title and title not in [n.title for n in relevant]:
                                report.hidden_connections.append(
                                    title[:80],
                                )

        report.hidden_connections = report.hidden_connections[:5]

        # Kriz skoru hesapla
        n_total = len(relevant)
        neg_ratio = len(neg_news) / max(n_total, 1)
        event_severity = len(all_events) * 0.1
        avg_sentiment = float(np.mean([n.sentiment for n in relevant]))

        crisis = (
            neg_ratio * 0.4
            + event_severity * 0.3
            + max(0, -avg_sentiment) * 0.3
        )
        report.crisis_score = round(float(min(1.0, crisis)), 3)

        if report.crisis_score < 0.2:
            report.crisis_level = "stable"
        elif report.crisis_score < 0.5:
            report.crisis_level = "tension"
        elif report.crisis_score < 0.8:
            report.crisis_level = "crisis"
        else:
            report.crisis_level = "meltdown"

        # LLM analizi
        llm_text = self._ask_llm(team, relevant)
        if llm_text:
            report.llm_analysis = llm_text
            report.method = "graphrag_llm"
        else:
            report.method = "graphrag_heuristic"

        report.recommendation = self._advice(report)
        return report

    def ask(self, question: str, team: str = "") -> str:
        """Bilgi grafiğine soru sor."""
        # LangChain QA Chain
        if self._qa_chain:
            try:
                result = self._qa_chain.invoke({"query": question})
                return result.get("result", "")
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        # LLM fallback
        context = self._build_context(team)
        return self._ask_llm_raw(
            f"Soru: {question}\n\nBağlam:\n{context}",
        )

    def _ask_llm(self, team: str, news: list[NewsNode]) -> str:
        """LLM ile kriz analizi."""
        headlines = "\n".join(
            f"- [{n.sentiment:+.1f}] {n.title}" for n in news[:15]
        )
        prompt = (
            f"Takım: {team}\n"
            f"Son haberler:\n{headlines}\n\n"
            f"Bu haberlere bakarak, maç sonucunu etkileyebilecek "
            f"gizli bir kriz veya motivasyon değişikliği var mı? "
            f"Kısa ve net cevap ver (3 cümle)."
        )
        return self._ask_llm_raw(prompt)

    def _ask_llm_raw(self, prompt: str) -> str:
        """Ham LLM sorgusu."""
        system = (
            "Sen bir futbol analisti ve kriz yönetim uzmanısın. "
            "Kısa, net ve Türkçe cevaplar ver."
        )

        # Ollama
        if self._llm_backend in ("ollama", "auto") and HTTPX_OK:
            try:
                resp = httpx.post(OLLAMA_URL, json={
                    "model": "llama3:8b",
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200},
                }, timeout=30.0)
                if resp.status_code == 200:
                    text = resp.json().get("response", "")
                    if text:
                        return text.strip()
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        return ""

    def _build_context(self, team: str) -> str:
        """Grafik bağlamını metin olarak hazırla."""
        relevant = [
            n for n in self._news_store[-50:]
            if not team or any(
                team.lower() in e.name.lower() for e in n.entities
            )
        ]
        return "\n".join(
            f"[{n.sentiment:+.1f}] {n.title}" for n in relevant[:20]
        )

    def get_stats(self) -> dict:
        """Grafik istatistikleri."""
        stats = {
            "total_news": len(self._news_store),
        }
        if self._mem_graph is not None:
            stats["nodes"] = self._mem_graph.number_of_nodes()
            stats["edges"] = self._mem_graph.number_of_edges()
            if NX_OK and self._mem_graph.number_of_nodes() > 0:
                type_counts: dict[str, int] = {}
                for _, data in self._mem_graph.nodes(data=True):
                    t = data.get("type", "unknown")
                    type_counts[t] = type_counts.get(t, 0) + 1
                stats["node_types"] = type_counts
        return stats

    def _advice(self, r: CrisisReport) -> str:
        if r.crisis_level == "meltdown":
            return (
                f"KRİTİK: {r.team} çöküş modunda! "
                f"Kriz skoru: {r.crisis_score:.0%}. "
                f"{r.negative_news_count} negatif haber. "
                f"Bahis YAPMA."
            )
        if r.crisis_level == "crisis":
            return (
                f"KRİZ: {r.team} stres altında. "
                f"Kriz: {r.crisis_score:.0%}. "
                f"Stake %50 düşür."
            )
        if r.crisis_level == "tension":
            return (
                f"GERGİNLİK: {r.team} gergin. "
                f"Kriz: {r.crisis_score:.0%}. Dikkatli ol."
            )
        return (
            f"STABİL: {r.team} sakin. "
            f"Kriz: {r.crisis_score:.0%}."
        )
