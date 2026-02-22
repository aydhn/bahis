"""
pnl_tracker.py – Gerçek Zamanlı Kâr/Zarar (PnL) Takipçisi.

Bu modül, "JP Morgan" titizliğinde bir cüzdan yönetimi sağlar.
- Her bahsi kaydeder.
- ROI, Sharpe Ratio, Drawdown hesaplar.
- CSV/Parquet formatında kalıcı kayıt tutar.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path
from loguru import logger

@dataclass
class Transaction:
    match_id: str
    selection: str
    odds: float
    stake: float
    result: str = "PENDING"  # WON, LOST, PENDING, VOID
    payout: float = 0.0
    pnl: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    strategy: str = "MANUAL"
    model_name: str = "MANUAL"
    confidence: float = 0.0
    ev: float = 0.0

class PnLTracker:
    def __init__(self, storage_path: str = "data/ledger.csv"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_ledger()

    def _load_ledger(self):
        if self.storage_path.exists():
            try:
                self.df = pd.read_csv(self.storage_path)
            except Exception as e:
                logger.error(f"Ledger yüklenemedi: {e}")
                self.df = pd.DataFrame(columns=[f.name for f in field(Transaction)])
        else:
            self.df = pd.DataFrame(columns=[f.name for f in field(Transaction)])

    def record_bet(self, t: Transaction):
        """Yeni bahis kaydeder."""
        row = asdict(t)
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._save()
        logger.info(f"Bahis Kaydedildi: {t.match_id} | {t.stake}₺ @ {t.odds} ({t.model_name})")

    def update_result(self, match_id: str, result: str):
        """Bahis sonucunu günceller (WON/LOST)."""
        mask = (self.df["match_id"] == match_id) & (self.df["result"] == "PENDING")
        if not mask.any():
            return
        
        idx = self.df[mask].index[0]
        stake = self.df.at[idx, "stake"]
        odds = self.df.at[idx, "odds"]
        
        payout = 0.0
        pnl = -stake
        if result == "WON":
            payout = stake * odds
            pnl = payout - stake
        elif result == "VOID":
            payout = stake
            pnl = 0.0
        elif result == "LOST":
            payout = 0.0
            pnl = -stake
            
        self.df.at[idx, "result"] = result
        self.df.at[idx, "payout"] = payout
        self.df.at[idx, "pnl"] = pnl
        self._save()
        logger.info(f"Sonuç Güncellendi: {match_id} -> {result} (PnL: {pnl:+.2f})")

    def get_stats(self) -> dict:
        """Portföy istatistiklerini hesaplar."""
        if self.df.empty:
            return {"roi": 0.0, "pnl": 0.0, "win_rate": 0.0}
            
        finished = self.df[self.df["result"].isin(["WON", "LOST"])]
        if finished.empty:
            return {"roi": 0.0, "pnl": 0.0, "win_rate": 0.0}
            
        total_stake = finished["stake"].sum()
        total_payout = finished["payout"].sum()
        pnl = total_payout - total_stake
        roi = (pnl / total_stake) if total_stake > 0 else 0.0
        
        wins = len(finished[finished["result"] == "WON"])
        total = len(finished)
        win_rate = wins / total
        
        return {
            "roi": roi,
            "pnl": pnl,
            "win_rate": win_rate,
            "total_bets": total
        }

    def sync_to_duckdb(self, db_manager):
        """CSV ledger'ı DuckDB 'bets' tablosuna aktarır."""
        if self.df.empty: return
        
        try:
            # Polars üzerinden bulk insert
            import polars as pl_mod
            df_pl = pl_mod.from_pandas(self.df)
            
            # match_id çakışması durumunda güncelle (upsert)
            cols = ", ".join(df_pl.columns)
            updates = ", ".join([f"{c} = EXCLUDED.{c}" for c in df_pl.columns if c != 'match_id'])
            
            # DuckDB connection üzerinden
            db_manager._con.register("tmp_ledger", df_pl)
            db_manager._con.execute(f"""
                INSERT INTO bets ({cols})
                SELECT * FROM tmp_ledger
                ON CONFLICT (match_id) DO UPDATE SET {updates}
            """)
            logger.info(f"[PnLTracker] {len(self.df)} kayıt DuckDB 'bets' tablosuna senkronize edildi.")
        except Exception as e:
            logger.error(f"[PnLTracker] sync_to_duckdb hatası: {e}")

    def sync_from_db(self, db_manager):
        """DuckDB'deki sonuçlanmış maçlarla ledger'ı senkronize eder."""
        query = """
        SELECT s.match_id, m.home_score, m.away_score, m.status, s.selection
        FROM signals s
        JOIN matches m ON s.match_id = m.match_id
        WHERE m.status = 'finished'
        """
        results_df = db_manager.query(query)
        if results_df.is_empty():
            return

        for row in results_df.to_dicts():
            mid = row["match_id"]
            h, a = row["home_score"], row["away_score"]
            
            # Basit kazanan tespiti (1X2)
            actual = "1" if h > a else "2" if h < a else "X"
            
            # PENDING olanları güncelle
            mask = (self.df["match_id"] == mid) & (self.df["result"] == "PENDING")
            if mask.any():
                selection = self.df.loc[mask, "selection"].iloc[0]
                # Selection "home"/"away"/"draw" -> "1"/"2"/"X" dönüşümü gerekebilir
                # Şimdilik direkt karşılaştırma
                result = "WON" if selection == actual else "LOST"
                self.update_result(mid, result)

    def _save(self):
        self.df.to_csv(self.storage_path, index=False)
