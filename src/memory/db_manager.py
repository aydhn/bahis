"""
db_manager.py – Polars + DuckDB tabanlı veri çerçevesi yönetimi.
Tüm maç, oran ve sinyal verisini yönetir.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import duckdb
import polars as pl
from loguru import logger

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "bahis.duckdb"


class DBManager:
    """Merkezi veri yöneticisi – DuckDB + Polars."""

    # Security: Valid columns for matches table to prevent SQL Injection
    ALLOWED_MATCH_COLUMNS = {
        "match_id", "league", "home_team", "away_team", "kickoff", "status",
        "home_odds", "draw_odds", "away_odds", "over25_odds", "under25_odds",
        "btts_yes", "btts_no", "home_score", "away_score", "created_at"
    }

    def __init__(self, db_path: Path | str = DB_PATH):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._path))
        self._init_schema()
        logger.debug(f"DBManager başlatıldı → {self._path}")

    # ── Şema ──
    def _init_schema(self):
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id     VARCHAR PRIMARY KEY,
                league       VARCHAR,
                home_team    VARCHAR,
                away_team    VARCHAR,
                kickoff      TIMESTAMP,
                status       VARCHAR DEFAULT 'upcoming',
                home_odds    DOUBLE,
                draw_odds    DOUBLE,
                away_odds    DOUBLE,
                over25_odds  DOUBLE,
                under25_odds DOUBLE,
                btts_yes     DOUBLE,
                btts_no      DOUBLE,
                home_score   INTEGER,
                away_score   INTEGER,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id    VARCHAR PRIMARY KEY,
                match_id     VARCHAR,
                market       VARCHAR,
                selection    VARCHAR,
                odds         DOUBLE,
                stake_pct    DOUBLE,
                confidence   DOUBLE,
                ev           DOUBLE,
                cycle        INTEGER,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS historical_stats (
                stat_id      VARCHAR PRIMARY KEY,
                team         VARCHAR,
                league       VARCHAR,
                season       VARCHAR,
                matches_played INTEGER,
                wins         INTEGER,
                draws        INTEGER,
                losses       INTEGER,
                goals_for    INTEGER,
                goals_against INTEGER,
                xg_for       DOUBLE,
                xg_against   DOUBLE,
                corners_avg  DOUBLE,
                cards_avg    DOUBLE,
                possession_avg DOUBLE,
                form_last5   VARCHAR,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS odds_history (
                id           INTEGER PRIMARY KEY,
                match_id     VARCHAR,
                bookmaker    VARCHAR,
                market       VARCHAR,
                selection    VARCHAR,
                odds         DOUBLE,
                timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._con.execute("""
            CREATE SEQUENCE IF NOT EXISTS odds_seq START 1
        """)

        self._con.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                bet_id       VARCHAR PRIMARY KEY,
                match_id     VARCHAR,
                selection    VARCHAR,
                odds         DOUBLE,
                stake        DOUBLE,
                is_paper     BOOLEAN,
                status       VARCHAR DEFAULT 'pending', -- pending, won, lost, void
                pnl          DOUBLE DEFAULT 0.0,
                placed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settled_at   TIMESTAMP
            )
        """)

    # ── Veri ekleme ──
    def upsert_match(self, data: dict):
        # Security: Whitelist columns to prevent SQL Injection
        valid_data = {
            k: v for k, v in data.items()
            if k in self.ALLOWED_MATCH_COLUMNS
        }

        if not valid_data:
            return

        if len(valid_data) < len(data):
            dropped = set(data.keys()) - set(valid_data.keys())
            logger.debug(f"Dropped invalid columns in upsert_match: {dropped}")

        cols = ", ".join(valid_data.keys())
        placeholders = ", ".join(["?"] * len(valid_data))
        updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in valid_data if k != "match_id")

        conflict_clause = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"

        sql = f"""
            INSERT INTO matches ({cols}) VALUES ({placeholders})
            ON CONFLICT (match_id) {conflict_clause}
        """
        self._con.execute(sql, list(valid_data.values()))

    def upsert_matches_bulk(self, df: pl.DataFrame):
        for row in df.iter_rows(named=True):
            self.upsert_match(row)

    def insert_odds_tick(self, match_id: str, bookmaker: str, market: str, selection: str, odds: float):
        self._con.execute(
            "INSERT INTO odds_history VALUES (nextval('odds_seq'), ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [match_id, bookmaker, market, selection, odds],
        )

    def save_signals(self, signals: list[dict] | pl.DataFrame, cycle: int = 0):
        if isinstance(signals, pl.DataFrame):
            rows = signals.iter_rows(named=True)
        elif isinstance(signals, list):
            rows = signals
        else:
            return
        import uuid
        for s in rows:
            sid = s.get("signal_id", str(uuid.uuid4())[:12])
            self._con.execute(
                """INSERT OR REPLACE INTO signals
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                [
                    sid,
                    s.get("match_id", ""),
                    s.get("market", ""),
                    s.get("selection", ""),
                    s.get("odds", 0.0),
                    s.get("stake_pct", 0.0),
                    s.get("confidence", 0.0),
                    s.get("ev", 0.0),
                    cycle,
                ],
            )
        logger.info(f"{cycle}. döngü – sinyaller kaydedildi.")

    def insert_bet(self, bet: dict):
        """Bahsi veritabanına kaydeder."""
        import uuid
        bet_id = bet.get("bet_id", str(uuid.uuid4())[:12])
        self._con.execute(
            """INSERT INTO bets (bet_id, match_id, selection, odds, stake, is_paper, status, placed_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP)
               ON CONFLICT (bet_id) DO NOTHING""",
            [
                bet_id,
                bet.get("match_id", ""),
                bet.get("selection", ""),
                bet.get("odds", 1.0),
                bet.get("stake", 0.0),
                bet.get("is_paper", True),
            ]
        )

    def update_bet_result(self, bet_id: str, pnl: float, status: str):
        """Bahis sonucunu günceller."""
        self._con.execute(
            """UPDATE bets
               SET pnl = ?, status = ?, settled_at = CURRENT_TIMESTAMP
               WHERE bet_id = ?""",
            [pnl, status, bet_id]
        )

    # ── Sorgulamalar ──
    def get_upcoming_matches(self, hours_ahead: int = 48) -> pl.DataFrame:
        cutoff = (datetime.utcnow() + timedelta(hours=hours_ahead)).isoformat()
        result = self._con.execute(
            "SELECT * FROM matches WHERE status = 'upcoming' AND kickoff <= ? ORDER BY kickoff",
            [cutoff],
        ).pl()
        return result

    def get_match(self, match_id: str) -> pl.DataFrame:
        return self._con.execute("SELECT * FROM matches WHERE match_id = ?", [match_id]).pl()

    def get_team_stats(self, team: str) -> pl.DataFrame:
        return self._con.execute("SELECT * FROM historical_stats WHERE team = ?", [team]).pl()

    def get_odds_history(self, match_id: str) -> pl.DataFrame:
        return self._con.execute(
            "SELECT * FROM odds_history WHERE match_id = ? ORDER BY timestamp", [match_id]
        ).pl()

    def get_finished_matches(self, limit: int = 1000) -> pl.DataFrame:
        """Sonuçlanmış maçları getirir (Elo eğitimi için)."""
        return self._con.execute(
            """SELECT * FROM matches
               WHERE status IN ('finished', 'FT', 'AET', 'PEN')
                 AND home_score IS NOT NULL
                 AND away_score IS NOT NULL
               ORDER BY kickoff DESC LIMIT ?""",
            [limit]
        ).pl()

    def get_pending_bets(self) -> pl.DataFrame:
        """Henüz sonuçlanmamış bahisleri getirir."""
        return self._con.execute(
            "SELECT * FROM bets WHERE status = 'pending'"
        ).pl()

    def get_settled_bets(self, limit: int = 100) -> pl.DataFrame:
        """Sonuçlanmış bahisleri getirir (PnL analizi için)."""
        return self._con.execute(
            "SELECT * FROM bets WHERE status != 'pending' ORDER BY settled_at DESC LIMIT ?",
            [limit]
        ).pl()

    def get_signals(self, cycle: int | None = None) -> pl.DataFrame:
        if cycle is not None:
            return self._con.execute("SELECT * FROM signals WHERE cycle = ?", [cycle]).pl()
        return self._con.execute("SELECT * FROM signals ORDER BY created_at DESC LIMIT 100").pl()

    def build_feature_matrix(self, matches: pl.DataFrame) -> pl.DataFrame:
        """Maç + tarihsel istatistik + oran geçmişini birleştirerek feature matrisi oluşturur."""
        if matches.is_empty():
            return matches

        features_rows = []
        for row in matches.iter_rows(named=True):
            mid = row["match_id"]
            home = row.get("home_team", "")
            away = row.get("away_team", "")

            home_stats = self.get_team_stats(home)
            away_stats = self.get_team_stats(away)
            odds_hist = self.get_odds_history(mid)

            feat = {
                "match_id": mid,
                "home_team": home,
                "away_team": away,
                "home_odds": row.get("home_odds", 0.0),
                "draw_odds": row.get("draw_odds", 0.0),
                "away_odds": row.get("away_odds", 0.0),
                "over25_odds": row.get("over25_odds", 0.0),
                "under25_odds": row.get("under25_odds", 0.0),
            }

            if not home_stats.is_empty():
                hs = home_stats.row(0, named=True)
                feat["home_xg"] = hs.get("xg_for", 0.0)
                feat["home_xga"] = hs.get("xg_against", 0.0)
                feat["home_win_rate"] = hs.get("wins", 0) / max(hs.get("matches_played", 1), 1)
                feat["home_possession"] = hs.get("possession_avg", 50.0)
            else:
                feat.update({"home_xg": 0.0, "home_xga": 0.0, "home_win_rate": 0.0, "home_possession": 50.0})

            if not away_stats.is_empty():
                aws = away_stats.row(0, named=True)
                feat["away_xg"] = aws.get("xg_for", 0.0)
                feat["away_xga"] = aws.get("xg_against", 0.0)
                feat["away_win_rate"] = aws.get("wins", 0) / max(aws.get("matches_played", 1), 1)
                feat["away_possession"] = aws.get("possession_avg", 50.0)
            else:
                feat.update({"away_xg": 0.0, "away_xga": 0.0, "away_win_rate": 0.0, "away_possession": 50.0})

            # Oran volatilitesi
            if not odds_hist.is_empty() and odds_hist.height > 1:
                feat["odds_volatility"] = odds_hist["odds"].std()
            else:
                feat["odds_volatility"] = 0.0

            features_rows.append(feat)

        return pl.DataFrame(features_rows)

    # ── SQL sorgulama ──
    def query(self, sql: str, params: list | None = None) -> pl.DataFrame:
        if params:
            return self._con.execute(sql, params).pl()
        return self._con.execute(sql).pl()

    def close(self):
        self._con.close()
