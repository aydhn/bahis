"""
db_manager.py – Polars + DuckDB tabanlı veri çerçevesi yönetimi.
Tüm maç, oran ve sinyal verisini yönetir.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta

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
        if df.is_empty():
            return

        # Select only allowed columns to prevent errors
        valid_cols = [c for c in df.columns if c in self.ALLOWED_MATCH_COLUMNS]
        df_valid = df.select(valid_cols)

        updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in valid_cols if k != "match_id")
        conflict_clause = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"

        # Register Polars DataFrame in DuckDB and execute bulk insert
        try:
            self._con.register('df_view', df_valid)
            cols = ", ".join(valid_cols)
            sql = f"""
                INSERT INTO matches ({cols})
                SELECT {cols} FROM df_view
                ON CONFLICT (match_id) {conflict_clause}
            """
            self._con.execute(sql)
        finally:
            self._con.unregister('df_view')

    def insert_odds_tick(self, match_id: str, bookmaker: str, market: str, selection: str, odds: float):
        self._con.execute(
            "INSERT INTO odds_history VALUES (nextval('odds_seq'), ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [match_id, bookmaker, market, selection, odds],
        )

    def save_signals(self, signals: list[dict] | pl.DataFrame, cycle: int = 0):
        if isinstance(signals, pl.DataFrame):
            if signals.is_empty():
                return
            rows = signals.iter_rows(named=True)
        elif isinstance(signals, list):
            if not signals:
                return
            rows = signals
        else:
            return

        import uuid

        params = []
        for s in rows:
            sid = s.get("signal_id")
            if not sid:
                sid = str(uuid.uuid4())[:12]

            params.append((
                sid,
                s.get("match_id", ""),
                s.get("market", ""),
                s.get("selection", ""),
                s.get("odds", 0.0) or 0.0,
                s.get("stake_pct", 0.0) or 0.0,
                s.get("confidence", 0.0) or 0.0,
                s.get("ev", 0.0) or 0.0,
                cycle,
            ))

        if params:
            self._con.executemany(
                """INSERT OR REPLACE INTO signals
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                params
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
        cutoff = (datetime.now(timezone.utc) + timedelta(hours=hours_ahead)).isoformat()
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

        teams = set(matches["home_team"].to_list()) | set(matches["away_team"].to_list())
        teams = [t for t in teams if t]
        match_ids = [m for m in matches["match_id"].to_list() if m]

        # Batch load team stats
        stats_df = pl.DataFrame()
        if teams:
            placeholders = ", ".join(["?"] * len(teams))
            stats_df = self._con.execute(f"SELECT * FROM historical_stats WHERE team IN ({placeholders})", teams).pl()

        # Batch load odds history and calculate volatility
        odds_df = pl.DataFrame()
        if match_ids:
            placeholders = ", ".join(["?"] * len(match_ids))
            raw_odds = self._con.execute(f"SELECT * FROM odds_history WHERE match_id IN ({placeholders})", match_ids).pl()
            if not raw_odds.is_empty():
                odds_df = raw_odds.group_by("match_id").agg([
                    pl.col("odds").std().fill_null(0.0).alias("odds_volatility")
                ])

        # Fill missing columns in stats_df if empty
        if stats_df.is_empty():
            stats_df = pl.DataFrame({
                "team": pl.Series(dtype=pl.Utf8),
                "xg_for": pl.Series(dtype=pl.Float64),
                "xg_against": pl.Series(dtype=pl.Float64),
                "wins": pl.Series(dtype=pl.Int64),
                "matches_played": pl.Series(dtype=pl.Int64),
                "possession_avg": pl.Series(dtype=pl.Float64),
            })

        # Calculate additional columns for stats_df
        stats_df = stats_df.with_columns([
            (pl.col("wins") / pl.max_horizontal(pl.col("matches_played"), 1)).fill_null(0.0).alias("win_rate")
        ])

        # Prepare base features from matches
        features = matches.select([
            "match_id", "home_team", "away_team", "home_odds", "draw_odds",
            "away_odds", "over25_odds", "under25_odds"
        ]).with_columns([
            pl.col("home_odds").fill_null(0.0),
            pl.col("draw_odds").fill_null(0.0),
            pl.col("away_odds").fill_null(0.0),
            pl.col("over25_odds").fill_null(0.0),
            pl.col("under25_odds").fill_null(0.0)
        ])

        # Join home stats
        home_stats = stats_df.rename({
            "xg_for": "home_xg",
            "xg_against": "home_xga",
            "win_rate": "home_win_rate",
            "possession_avg": "home_possession"
        }).select(["team", "home_xg", "home_xga", "home_win_rate", "home_possession"])

        features = features.join(
            home_stats, left_on="home_team", right_on="team", how="left"
        ).with_columns([
            pl.col("home_xg").fill_null(0.0),
            pl.col("home_xga").fill_null(0.0),
            pl.col("home_win_rate").fill_null(0.0),
            pl.col("home_possession").fill_null(50.0)
        ])

        # Join away stats
        away_stats = stats_df.rename({
            "xg_for": "away_xg",
            "xg_against": "away_xga",
            "win_rate": "away_win_rate",
            "possession_avg": "away_possession"
        }).select(["team", "away_xg", "away_xga", "away_win_rate", "away_possession"])

        features = features.join(
            away_stats, left_on="away_team", right_on="team", how="left"
        ).with_columns([
            pl.col("away_xg").fill_null(0.0),
            pl.col("away_xga").fill_null(0.0),
            pl.col("away_win_rate").fill_null(0.0),
            pl.col("away_possession").fill_null(50.0)
        ])

        # Join odds volatility
        if not odds_df.is_empty():
            features = features.join(
                odds_df.select(["match_id", "odds_volatility"]),
                on="match_id", how="left"
            ).with_columns(pl.col("odds_volatility").fill_null(0.0))
        else:
            features = features.with_columns(pl.lit(0.0).alias("odds_volatility"))

        return features


    def query(self, sql: str, params: list | None = None) -> pl.DataFrame:
        if params:
            return self._con.execute(sql, params).pl()
        return self._con.execute(sql).pl()

    def close(self):
        self._con.close()
