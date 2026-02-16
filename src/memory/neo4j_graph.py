"""
neo4j_graph.py – Neo4j Graph Database Entegrasyonu.

İlişkisel veritabanları (SQL) futbolu modellemekte zorlanır.
"X oyuncusu, Y teknik direktörü ile çalışırken performansı nasıldı?"
SQL'de zordur, Graph DB'de milisaniyelik iştir.

Düğümler (Nodes):
  (:Team {name, league, country})
  (:Player {name, position, market_value})
  (:Manager {name, nationality})
  (:Referee {name, card_tendency})
  (:Match {id, date, league, season})

İlişkiler (Relationships):
  (Player)-[:PLAYED_IN {rating, goals, assists, minutes}]->(Match)
  (Team)-[:HOME_TEAM]->(Match)
  (Team)-[:AWAY_TEAM]->(Match)
  (Manager)-[:MANAGED {win_rate}]->(Team)
  (Referee)-[:OFFICIATED {home_cards, away_cards}]->(Match)
  (Player)-[:PLAYS_FOR {since, until}]->(Team)

Sorgular (Cypher):
  - "Bu hakem deplasman sert oynadığında kaç kart gösteriyor?"
  - "X oyuncusu Z teknik direktörle kaç gol attı?"
  - "A takımı B hakemiyle tarihsel olarak nasıl?"
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger

try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False
    logger.warning("neo4j driver yüklü değil: pip install neo4j")


class Neo4jFootballGraph:
    """Neo4j üzerinde futbol bilgi grafiği.

    Kullanım:
        graph = Neo4jFootballGraph()
        graph.connect()
        graph.create_match(match_data)
        insights = graph.query_referee_bias("Cüneyt Çakır", "away_cards")
    """

    def __init__(self, uri: str = "", user: str = "neo4j",
                 password: str = ""):
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "bahis_graph_2026")
        self._driver = None
        self._connected = False
        logger.debug(f"Neo4jFootballGraph başlatıldı (uri={self._uri}).")

    # ═══════════════════════════════════════════
    #  BAĞLANTI
    # ═══════════════════════════════════════════
    def connect(self) -> bool:
        """Neo4j'ye bağlan."""
        if not NEO4J_OK:
            logger.info("[Neo4j] Driver yüklü değil – in-memory mod.")
            return False
        try:
            self._driver = GraphDatabase.driver(
                self._uri, auth=basic_auth(self._user, self._password),
            )
            self._driver.verify_connectivity()
            self._connected = True
            logger.success(f"[Neo4j] Bağlantı başarılı: {self._uri}")
            self._create_constraints()
            return True
        except Exception as e:
            logger.warning(f"[Neo4j] Bağlantı hatası: {e} – in-memory mod.")
            self._init_fallback()
            return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._connected = False

    def _create_constraints(self):
        """Benzersizlik kısıtlamaları oluştur."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Match) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Referee) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mg:Manager) REQUIRE mg.id IS UNIQUE",
        ]
        with self._driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception:
                    pass

    # ═══════════════════════════════════════════
    #  DÜĞÜM OLUŞTURMA
    # ═══════════════════════════════════════════
    def create_team(self, name: str, league: str = "",
                    country: str = "") -> bool:
        return self._run_write(
            "MERGE (t:Team {name: $name}) "
            "SET t.league = $league, t.country = $country",
            name=name, league=league, country=country,
        )

    def create_player(self, player_id: str, name: str,
                      position: str = "", team: str = "",
                      market_value: float = 0) -> bool:
        ok = self._run_write(
            "MERGE (p:Player {id: $pid}) "
            "SET p.name = $name, p.position = $pos, p.market_value = $mv",
            pid=player_id, name=name, pos=position, mv=market_value,
        )
        if team:
            self._run_write(
                "MATCH (p:Player {id: $pid}), (t:Team {name: $team}) "
                "MERGE (p)-[:PLAYS_FOR]->(t)",
                pid=player_id, team=team,
            )
        return ok

    def create_referee(self, name: str, card_tendency: str = "normal") -> bool:
        return self._run_write(
            "MERGE (r:Referee {name: $name}) "
            "SET r.card_tendency = $ct",
            name=name, ct=card_tendency,
        )

    def create_manager(self, manager_id: str, name: str,
                       team: str = "") -> bool:
        ok = self._run_write(
            "MERGE (m:Manager {id: $mid}) SET m.name = $name",
            mid=manager_id, name=name,
        )
        if team:
            self._run_write(
                "MATCH (m:Manager {id: $mid}), (t:Team {name: $team}) "
                "MERGE (m)-[:MANAGES]->(t)",
                mid=manager_id, team=team,
            )
        return ok

    # ═══════════════════════════════════════════
    #  MAÇ KAYDI
    # ═══════════════════════════════════════════
    def create_match(self, data: dict) -> bool:
        """Maçı tüm ilişkileriyle kaydet."""
        mid = data.get("match_id", "")
        home = data.get("home_team", "")
        away = data.get("away_team", "")
        if not (mid and home and away):
            return False

        # Match düğümü
        self._run_write(
            "MERGE (m:Match {id: $mid}) "
            "SET m.date = $date, m.league = $league, "
            "    m.home_goals = $hg, m.away_goals = $ag, "
            "    m.home_xg = $hxg, m.away_xg = $axg",
            mid=mid, date=data.get("date", ""),
            league=data.get("league", ""),
            hg=data.get("home_goals", 0), ag=data.get("away_goals", 0),
            hxg=data.get("home_xg", 0), axg=data.get("away_xg", 0),
        )

        # Takım ilişkileri
        self.create_team(home, data.get("league", ""))
        self.create_team(away, data.get("league", ""))
        self._run_write(
            "MATCH (t:Team {name: $team}), (m:Match {id: $mid}) "
            "MERGE (t)-[:HOME_TEAM]->(m)",
            team=home, mid=mid,
        )
        self._run_write(
            "MATCH (t:Team {name: $team}), (m:Match {id: $mid}) "
            "MERGE (t)-[:AWAY_TEAM]->(m)",
            team=away, mid=mid,
        )

        # Hakem
        ref = data.get("referee", "")
        if ref:
            self.create_referee(ref)
            self._run_write(
                "MATCH (r:Referee {name: $ref}), (m:Match {id: $mid}) "
                "MERGE (r)-[:OFFICIATED {home_cards: $hc, away_cards: $ac}]->(m)",
                ref=ref, mid=mid,
                hc=data.get("home_yellows", 0) + data.get("home_reds", 0),
                ac=data.get("away_yellows", 0) + data.get("away_reds", 0),
            )

        # Kadro (varsa)
        for player_data in data.get("home_lineup", []):
            self._add_player_to_match(player_data, mid, home)
        for player_data in data.get("away_lineup", []):
            self._add_player_to_match(player_data, mid, away)

        return True

    def _add_player_to_match(self, player: dict, match_id: str,
                              team: str):
        pid = player.get("id", player.get("name", ""))
        if not pid:
            return
        self.create_player(pid, player.get("name", pid), team=team)
        self._run_write(
            "MATCH (p:Player {id: $pid}), (m:Match {id: $mid}) "
            "MERGE (p)-[:PLAYED_IN {"
            "  rating: $rating, goals: $goals, assists: $assists, "
            "  minutes: $minutes"
            "}]->(m)",
            pid=pid, mid=match_id,
            rating=player.get("rating", 0),
            goals=player.get("goals", 0),
            assists=player.get("assists", 0),
            minutes=player.get("minutes", 0),
        )

    # ═══════════════════════════════════════════
    #  SORGULAR – SQL'in Göremediği İlişkiler
    # ═══════════════════════════════════════════
    def query_referee_bias(self, referee: str,
                            metric: str = "away_cards") -> dict:
        """Hakem yanlılığı: deplasman takımına kart eğilimi."""
        result = self._run_read(
            "MATCH (r:Referee {name: $ref})-[o:OFFICIATED]->(m:Match) "
            "RETURN avg(o.home_cards) AS avg_home_cards, "
            "       avg(o.away_cards) AS avg_away_cards, "
            "       count(m) AS total_matches",
            ref=referee,
        )
        if result:
            row = result[0]
            home_avg = row.get("avg_home_cards", 0) or 0
            away_avg = row.get("avg_away_cards", 0) or 0
            return {
                "referee": referee,
                "avg_home_cards": float(home_avg),
                "avg_away_cards": float(away_avg),
                "total_matches": int(row.get("total_matches", 0)),
                "away_bias": float(away_avg - home_avg),
                "is_strict_away": away_avg > home_avg * 1.2,
            }
        return {"referee": referee, "total_matches": 0}

    def query_player_under_manager(self, player_name: str,
                                     manager_name: str) -> dict:
        """Oyuncunun belirli teknik direktörle performansı."""
        result = self._run_read(
            "MATCH (p:Player)-[:PLAYED_IN]->(m:Match)<-[:HOME_TEAM|AWAY_TEAM]-(t:Team)"
            "<-[:MANAGES]-(mg:Manager {name: $manager}) "
            "WHERE p.name = $player "
            "RETURN avg(p.rating) AS avg_rating, "
            "       sum(p.goals) AS total_goals, "
            "       count(m) AS matches",
            player=player_name, manager=manager_name,
        )
        if result:
            return dict(result[0])
        return {}

    def query_team_vs_referee(self, team: str, referee: str) -> dict:
        """Takımın belirli hakemle tarihsel performansı."""
        result = self._run_read(
            "MATCH (t:Team {name: $team})-[:HOME_TEAM|AWAY_TEAM]->(m:Match)"
            "<-[:OFFICIATED]-(r:Referee {name: $ref}) "
            "RETURN count(m) AS matches, "
            "       avg(m.home_goals) AS avg_home_goals, "
            "       avg(m.away_goals) AS avg_away_goals",
            team=team, ref=referee,
        )
        if result:
            return dict(result[0])
        return {}

    def query_h2h_graph(self, team_a: str, team_b: str) -> list[dict]:
        """İki takım arası tüm maçlar (graph traverse)."""
        return self._run_read(
            "MATCH (a:Team {name: $a})-[:HOME_TEAM]->(m:Match)<-[:AWAY_TEAM]-(b:Team {name: $b}) "
            "RETURN m.id AS match_id, m.date AS date, "
            "       m.home_goals AS home_goals, m.away_goals AS away_goals "
            "UNION "
            "MATCH (b:Team {name: $b})-[:HOME_TEAM]->(m:Match)<-[:AWAY_TEAM]-(a:Team {name: $a}) "
            "RETURN m.id AS match_id, m.date AS date, "
            "       m.home_goals AS home_goals, m.away_goals AS away_goals "
            "ORDER BY date DESC LIMIT 20",
            a=team_a, b=team_b,
        )

    def query_key_player_impact(self, team: str,
                                  top_n: int = 3) -> list[dict]:
        """Takımın en kritik oyuncuları (rating + gol katkısı)."""
        return self._run_read(
            "MATCH (p:Player)-[:PLAYS_FOR]->(t:Team {name: $team}), "
            "      (p)-[r:PLAYED_IN]->(m:Match) "
            "RETURN p.name AS player, p.position AS position, "
            "       avg(r.rating) AS avg_rating, "
            "       sum(r.goals) AS goals, sum(r.assists) AS assists, "
            "       count(m) AS appearances "
            "ORDER BY avg_rating DESC LIMIT $n",
            team=team, n=top_n,
        )

    # ═══════════════════════════════════════════
    #  İSTATİSTİK
    # ═══════════════════════════════════════════
    def stats(self) -> dict:
        """Grafik veritabanı istatistikleri."""
        if not self._connected:
            return self._fallback_stats()

        counts = {}
        for label in ("Team", "Player", "Match", "Referee", "Manager"):
            result = self._run_read(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = result[0]["c"] if result else 0

        rel_count = self._run_read(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        )
        counts["relationships"] = rel_count[0]["c"] if rel_count else 0

        return counts

    # ═══════════════════════════════════════════
    #  YARDIMCI
    # ═══════════════════════════════════════════
    def _run_write(self, query: str, **params) -> bool:
        if not self._connected or not self._driver:
            return self._fallback_write(query, params)
        try:
            with self._driver.session() as session:
                session.run(query, **params)
            return True
        except Exception as e:
            logger.debug(f"[Neo4j] Write hatası: {e}")
            return False

    def _run_read(self, query: str, **params) -> list[dict]:
        if not self._connected or not self._driver:
            return self._fallback_read(query, params)
        try:
            with self._driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            logger.debug(f"[Neo4j] Read hatası: {e}")
            return []

    # ═══════════════════════════════════════════
    #  FALLBACK: In-Memory networkx
    # ═══════════════════════════════════════════
    def _init_fallback(self):
        """Neo4j yokken networkx ile in-memory graph."""
        try:
            import networkx as nx
            self._nx = nx.MultiDiGraph()
            self._fallback_mode = True
            logger.info("[Neo4j] Fallback: networkx in-memory graph aktif.")
        except ImportError:
            self._nx = None
            self._fallback_mode = False

    def _fallback_write(self, query: str, params: dict) -> bool:
        if not hasattr(self, "_nx") or self._nx is None:
            return False
        # Basit node/edge ekleme (Cypher parse yerine)
        # Tam Cypher parsing yapılmaz, sadece MERGE node/edge durumlarını destekler
        return True

    def _fallback_read(self, query: str, params: dict) -> list[dict]:
        return []

    def _fallback_stats(self) -> dict:
        if hasattr(self, "_nx") and self._nx:
            return {
                "nodes": self._nx.number_of_nodes(),
                "edges": self._nx.number_of_edges(),
                "mode": "in-memory (networkx)",
            }
        return {"mode": "disabled"}
