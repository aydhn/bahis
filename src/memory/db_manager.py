"""
db_manager.py – Polars + DuckDB tabanlı veri çerçevesi yönetimi.
Tüm maç, oran ve sinyal verisini yönetir.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta

try:
    import duckdb
    DUCKDB_OK = True
except Exception:
    duckdb = None
    DUCKDB_OK = False
import polars as pl
from loguru import logger

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "bahis.duckdb"


def _kill_stale_holders(db_path: Path) -> bool:
    """DuckDB dosyasını tutan eski Python süreçlerini tespit edip sonlandırır."""
    if not PSUTIL_OK:
        return False

    current_pid = os.getpid()
    killed = False
    db_path_str = str(db_path).lower()

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == current_pid:
                continue
            pname = (proc.info.get("name") or "").lower()
            if "python" not in pname:
                continue
            # Sürecin açtığı dosyalara bakarak DuckDB kilidini tespit et
            try:
                open_files = proc.open_files()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            for f in open_files:
                if db_path_str in f.path.lower():
                    logger.warning(
                        f"[DBManager] Eski süreç tespit edildi: PID={proc.pid} "
                        f"({pname}) – DuckDB dosyasını kilitliyor. Sonlandırılıyor…"
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        logger.warning(f"[DBManager] PID={proc.pid} zorla sonlandırıldı (KILL)")
                    killed = True
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed


def _connect_with_retry(db_path: Path, max_retries: int = 5,
                        base_delay: float = 1.0):
    """DuckDB varsa DuckDB, yoksa SQLite bağlantısını açar."""
    if not DUCKDB_OK:
        con = sqlite3.connect(
            str(db_path.with_suffix(".sqlite3")),
            timeout=30,
            isolation_level=None,
        )
        logger.warning("[DBManager] duckdb yok – SQLite fallback modu aktif.")
        return con

    last_exc = None
    for attempt in range(max_retries):
        try:
            con = duckdb.connect(str(db_path))
            if attempt > 0:
                logger.info(f"[DBManager] DuckDB bağlantısı {attempt + 1}. denemede başarılı")
            return con
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning(
                    f"[DBManager] DuckDB dosyası kilitli – eski süreçler taranıyor…"
                )
                if _kill_stale_holders(db_path):
                    time.sleep(1.0)
                    continue

            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"[DBManager] DuckDB bağlantı denemesi {attempt + 1}/{max_retries} "
                f"başarısız – {delay:.1f}s sonra tekrar denenecek. Hata: {exc}"
            )
            time.sleep(delay)

    # .wal ve .tmp dosyalarını temizleyerek son bir deneme
    for suffix in (".wal", ".tmp"):
        wal = db_path.with_suffix(db_path.suffix + suffix)
        if wal.exists():
            try:
                wal.unlink()
                logger.info(f"[DBManager] Temizlendi: {wal.name}")
            except OSError:
                pass

    try:
        return duckdb.connect(str(db_path))
    except Exception:
        raise RuntimeError(
            f"DuckDB dosyası {max_retries} deneme sonrasında hâlâ açılamıyor: "
            f"{db_path}. Lütfen kilitleyen süreci (python.exe) manuel kapatın. "
            f"Son hata: {last_exc}"
        ) from last_exc


class DBManager:
    """Merkezi veri yöneticisi – DuckDB + Polars."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = _connect_with_retry(self._path)
        self._is_duckdb = DUCKDB_OK and self._con.__class__.__module__.startswith("duckdb")
        # WAL mode + performans ayarları
        try:
            if self._is_duckdb:
                self._con.execute("PRAGMA enable_progress_bar=false")
                self._con.execute("PRAGMA threads=4")
        except Exception:
            pass
        self._reconnect_if_needed()
        self._backup_db()
        self._init_schema()
        logger.info(f"[DBManager] Başlatıldı → {self._path}")

    def _backup_db(self):
        """DuckDB dosyasını yedekler."""
        import shutil
        backup_path = self._path.with_suffix(".duckdb.backup")
        try:
            if self._path.exists():
                shutil.copy2(self._path, backup_path)
                logger.debug(f"[DBManager] Yedek oluşturuldu: {backup_path.name}")
        except Exception as e:
            logger.warning(f"[DBManager] Yedekleme başarısız: {e}")

    def _query_pl(self, sql: str, params: list | None = None) -> pl.DataFrame:
        params = params or []
        try:
            cur = self._con.execute(sql, params)
            if self._is_duckdb and hasattr(cur, "pl"):
                return cur.pl()
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            if not cols:
                return pl.DataFrame()
            if rows:
                return pl.DataFrame(rows, schema=cols, orient="row")
            return pl.DataFrame({c: [] for c in cols})
        except Exception:
            return pl.DataFrame()

    # ── Şema ──
    def _init_schema(self):
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id     VARCHAR PRIMARY KEY,
                sport        VARCHAR DEFAULT 'football',
                league       VARCHAR,
                country      VARCHAR DEFAULT '',
                competition_type VARCHAR DEFAULT 'league',
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
        # Mevcut tabloya yeni sütunları ekle (migration)
        for col, dtype, default in [
            ("sport", "VARCHAR", "'football'"),
            ("country", "VARCHAR", "''"),
            ("competition_type", "VARCHAR", "'league'"),
        ]:
            try:
                self._con.execute(f"ALTER TABLE matches ADD COLUMN {col} {dtype} DEFAULT {default}")
            except Exception:
                pass  # Zaten var
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
            CREATE TABLE IF NOT EXISTS bets (
                match_id     VARCHAR PRIMARY KEY,
                selection     VARCHAR,
                odds          DOUBLE,
                stake         DOUBLE,
                payout        DOUBLE DEFAULT 0.0,
                pnl           DOUBLE DEFAULT 0.0,
                result        VARCHAR DEFAULT 'PENDING',
                model_name    VARCHAR DEFAULT 'MANUAL',
                timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy      VARCHAR,
                confidence    DOUBLE,
                ev            DOUBLE
            )
        """)
        if self._is_duckdb:
            self._con.execute("""
                CREATE SEQUENCE IF NOT EXISTS odds_seq START 1
            """)

    # ── Veri ekleme ──
    def upsert_match(self, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in data if k != "match_id")
        sql = f"""
            INSERT INTO matches ({cols}) VALUES ({placeholders})
            ON CONFLICT (match_id) DO UPDATE SET {updates}
        """
        self._con.execute(sql, list(data.values()))

    def upsert_matches_bulk(self, df: pl.DataFrame):
        """Toplu upsert – DuckDB native registering ile hızlandırılmış."""
        if df.is_empty():
            return
        if not self._is_duckdb:
            for row in df.iter_rows(named=True):
                self.upsert_match(row)
            return
        try:
            # DuckDB'nin Polars'ı doğrudan okumasını kullan (sıfır kopya)
            self._con.register("_tmp_bulk", df)
            cols = [c for c in df.columns if c in (
                "match_id", "sport", "league", "country", "competition_type",
                "home_team", "away_team", "kickoff",
                "status", "home_odds", "draw_odds", "away_odds",
                "over25_odds", "under25_odds", "btts_yes", "btts_no",
                "home_score", "away_score",
            )]
            if "match_id" not in cols:
                # Fallback: satır satır
                for row in df.iter_rows(named=True):
                    self.upsert_match(row)
                return
            col_list = ", ".join(cols)
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "match_id")
            self._con.execute(f"""
                INSERT INTO matches ({col_list})
                SELECT {col_list} FROM _tmp_bulk
                ON CONFLICT (match_id) DO UPDATE SET {updates}
            """)
            self._con.unregister("_tmp_bulk")
            logger.debug(f"[DBManager] Toplu upsert: {df.height} satır")
        except Exception as e:
            logger.debug(f"[DBManager] Toplu upsert fallback (row-by-row): {e}")
            try:
                self._con.unregister("_tmp_bulk")
            except Exception:
                pass
            for row in df.iter_rows(named=True):
                self.upsert_match(row)

    def insert_odds_tick(self, match_id: str, bookmaker: str, market: str, selection: str, odds: float):
        if self._is_duckdb:
            self._con.execute(
                "INSERT INTO odds_history VALUES (nextval('odds_seq'), ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [match_id, bookmaker, market, selection, odds],
            )
        else:
            self._con.execute(
                """INSERT INTO odds_history
                   (match_id, bookmaker, market, selection, odds, timestamp)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
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

    # ── Sorgulamalar ──
    def get_upcoming_matches(self, hours_ahead: int = 72) -> pl.DataFrame:
        cutoff = (datetime.utcnow() + timedelta(hours=hours_ahead)).isoformat()
        try:
            result = self._query_pl(
                "SELECT * FROM matches WHERE status = 'upcoming' AND kickoff <= ? ORDER BY kickoff",
                [cutoff],
            )
            if not result.is_empty():
                return result
        except Exception as e:
            logger.debug(f"[DBManager] get_upcoming_matches sorgu hatası: {e}")

        # Fallback: tüm upcoming maçları al (kickoff filtresi olmadan)
        try:
            result = self._query_pl(
                "SELECT * FROM matches WHERE status = 'upcoming' ORDER BY kickoff LIMIT 100",
            )
            if not result.is_empty():
                return result
        except Exception:
            pass

        # Son fallback: herhangi bir maç
        try:
            return self._query_pl(
                "SELECT * FROM matches ORDER BY created_at DESC LIMIT 50",
            )
        except Exception:
            return pl.DataFrame()

    def get_match(self, match_id: str) -> pl.DataFrame:
        return self._query_pl("SELECT * FROM matches WHERE match_id = ?", [match_id])

    def get_team_stats(self, team: str) -> pl.DataFrame:
        return self._query_pl("SELECT * FROM historical_stats WHERE team = ?", [team])

    def get_odds_history_df(self, match_id: str, market: str | None = None) -> pl.DataFrame:
        """Ham DataFrame versiyonu – iç kullanım için."""
        if market:
            return self._query_pl(
                "SELECT * FROM odds_history WHERE match_id = ? AND market = ? ORDER BY timestamp",
                [match_id, market],
            )
        return self._query_pl(
            "SELECT * FROM odds_history WHERE match_id = ? ORDER BY timestamp", [match_id]
        )

    def get_odds_history(self, match_id: str, market: str | None = None) -> list[float]:
        """Oran geçmişini float listesi olarak döndür.

        !! BREAKING CHANGE !!
        Artık Polars DataFrame değil, list[float] döndürüyor.
        `if odds_hist:` ve `len(odds_hist)` güvenli çalışır.
        """
        df = self.get_odds_history_df(match_id, market)
        if df.is_empty():
            return []
        try:
            return [float(x) for x in df["odds"].to_list() if x is not None]
        except Exception:
            return []

    def get_odds_history_list(self, match_id: str, market: str | None = None) -> list[float]:
        """Alias – get_odds_history ile aynı."""
        return self.get_odds_history(match_id, market)

    def get_match_events_list(self, match_id: str, event_type: str = "goal") -> list[dict]:
        """Maç olaylarını list of dict olarak döndür."""
        try:
            df = self._query_pl(
                "SELECT * FROM match_events WHERE match_id = ? AND event_type = ? ORDER BY minute",
                [match_id, event_type],
            )
            if df.is_empty():
                return []
            return df.to_dicts()
        except Exception:
            return []

    def get_team_history_list(self, team: str, last_n: int = 20) -> list[dict]:
        """Takım maç geçmişini list of dict olarak döndür."""
        try:
            df = self._query_pl(
                """SELECT * FROM matches
                   WHERE (home_team = ? OR away_team = ?)
                   AND status = 'finished'
                   ORDER BY kickoff DESC LIMIT ?""",
                [team, team, last_n],
            )
            if df.is_empty():
                return []
            return df.to_dicts()
        except Exception:
            return []

    def get_concede_times_list(self, team: str, last_n: int = 50) -> list[dict]:
        """Gol yeme sürelerini döndür – survival analizi için."""
        try:
            df = self._query_pl(
                """SELECT * FROM goal_events
                   WHERE concede_team = ? ORDER BY match_date DESC LIMIT ?""",
                [team, last_n],
            )
            if df.is_empty():
                return []
            return df.to_dicts()
        except Exception:
            return []

    def get_signals(self, cycle: int | None = None) -> pl.DataFrame:
        if cycle is not None:
            return self._query_pl("SELECT * FROM signals WHERE cycle = ?", [cycle])
        return self._query_pl("SELECT * FROM signals ORDER BY created_at DESC LIMIT 100")

    def build_feature_matrix(self, matches: pl.DataFrame) -> pl.DataFrame:
        """Maç + tarihsel istatistik + oran geçmişini birleştirerek feature matrisi oluşturur.

        Batch SQL sorguları ile N+1 problem çözüldü. Tüm None/null güvenli varsayılanlara dönüştürülür.
        """
        import math as _math

        if matches.is_empty():
            return matches

        def _sf(val, default=0.0):
            if val is None:
                return default
            try:
                f = float(val)
                return f if _math.isfinite(f) else default
            except (TypeError, ValueError):
                return default

        # Tüm takımları ve match_id'leri topla – tek sorguda çek
        all_teams = set()
        all_mids = set()
        for row in matches.iter_rows(named=True):
            all_teams.add(row.get("home_team", ""))
            all_teams.add(row.get("away_team", ""))
            all_mids.add(row.get("match_id", ""))
        all_teams.discard("")
        all_mids.discard("")

        # Batch: Tüm takım istatistiklerini tek sorguda çek
        stats_map: dict[str, dict] = {}
        if all_teams:
            try:
                placeholders = ", ".join(["?"] * len(all_teams))
                all_stats = self._query_pl(
                    f"SELECT * FROM historical_stats WHERE team IN ({placeholders})",
                    list(all_teams),
                )
                for sr in all_stats.iter_rows(named=True):
                    stats_map[sr.get("team", "")] = sr
            except Exception as e:
                logger.debug(f"[DBManager] Batch stats hatası: {e}")

        # Batch: Tüm oran volatilitelerini tek sorguda hesapla
        vol_map: dict[str, float] = {}
        if all_mids:
            try:
                placeholders = ", ".join(["?"] * len(all_mids))
                vol_df = self._query_pl(
                    f"""SELECT match_id, STDDEV(odds) as vol
                        FROM odds_history
                        WHERE match_id IN ({placeholders})
                        GROUP BY match_id
                        HAVING COUNT(*) > 1""",
                    list(all_mids),
                )
                for vr in vol_df.iter_rows(named=True):
                    vol_map[vr.get("match_id", "")] = _sf(vr.get("vol"), 0.0)
            except Exception as e:
                logger.debug(f"[DBManager] Batch volatility hatası: {e}")

        # Feature matrisini oluştur (artık N+1 sorgu yok)
        features_rows = []
        for row in matches.iter_rows(named=True):
            mid = row.get("match_id", "")
            home = row.get("home_team", "")
            away = row.get("away_team", "")

            feat = {
                "match_id": mid,
                "home_team": home,
                "away_team": away,
                "home_odds": _sf(row.get("home_odds"), 0.0),
                "draw_odds": _sf(row.get("draw_odds"), 0.0),
                "away_odds": _sf(row.get("away_odds"), 0.0),
                "over25_odds": _sf(row.get("over25_odds"), 0.0),
                "under25_odds": _sf(row.get("under25_odds"), 0.0),
            }

            hs = stats_map.get(home)
            if hs:
                mp = max(_sf(hs.get("matches_played"), 1), 1)
                feat["home_xg"] = _sf(hs.get("xg_for"), 0.0)
                feat["home_xga"] = _sf(hs.get("xg_against"), 0.0)
                feat["home_win_rate"] = _sf(hs.get("wins"), 0) / mp
                feat["home_possession"] = _sf(hs.get("possession_avg"), 50.0)
            else:
                feat.update({"home_xg": 0.0, "home_xga": 0.0,
                             "home_win_rate": 0.0, "home_possession": 50.0})

            aws = stats_map.get(away)
            if aws:
                mp = max(_sf(aws.get("matches_played"), 1), 1)
                feat["away_xg"] = _sf(aws.get("xg_for"), 0.0)
                feat["away_xga"] = _sf(aws.get("xg_against"), 0.0)
                feat["away_win_rate"] = _sf(aws.get("wins"), 0) / mp
                feat["away_possession"] = _sf(aws.get("possession_avg"), 50.0)
            else:
                feat.update({"away_xg": 0.0, "away_xga": 0.0,
                             "away_win_rate": 0.0, "away_possession": 50.0})

            feat["odds_volatility"] = vol_map.get(mid, 0.0)
            features_rows.append(feat)

        return pl.DataFrame(features_rows)

    # ── SQL sorgulama ──
    def query(self, sql: str, params: list | None = None) -> pl.DataFrame:
        return self._query_pl(sql, params)

    def close(self):
        try:
            self._con.close()
        except Exception as e:
            logger.debug(f"[DBManager] Bağlantı kapatma hatası: {e}")

    def _check_connection(self) -> bool:
        """Bağlantının aktif olup olmadığını kontrol et."""
        try:
            self._con.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def _reconnect_if_needed(self):
        """Bağlantı kopuksa yeniden bağlan."""
        if not self._check_connection():
            logger.warning("[DBManager] Bağlantı kopuk – yeniden bağlanıyor…")
            try:
                self._con.close()
            except Exception:
                pass
            self._con = _connect_with_retry(self._path)
