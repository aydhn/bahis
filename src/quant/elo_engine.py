"""
elo_engine.py – Dinamik Elo Reyting hesaplayıcı.

Lig tablosuna (puan durumu) bakmak yerine, takımların gerçek gücünü
Elo (Satrançtaki gibi) sistemiyle hesaplar.

Özellikler:
- Ev sahibi avantajı ekler.
- Gol farkına göre puanı ayarlar (Farklı galibiyet daha çok puan).
- Ligler arası güç farkını ileride destekleyebilir.
"""
import math
from dataclasses import dataclass
import polars as pl
from typing import Any
from loguru import logger

@dataclass
class EloRating:
    team: str
    rating: float = 1500.0  # Başlangıç
    matches_played: int = 0
    last_update: str = ""

class EloEngine:
    def __init__(self, db: Any = None, k_factor: float = 30.0, home_advantage: float = 50.0):
        self.db = db
        self.k = k_factor
        self.home_adv = home_advantage
        self.ratings: dict[str, EloRating] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, EloRating(team)).rating

    def get_win_prob(self, home_team: str, away_team: str) -> float:
        """Ev sahibinin kazanma olasılığı (Beraberlik hariç, teorik)."""
        r_home = self.get_rating(home_team) + self.home_adv
        r_away = self.get_rating(away_team)
        
        # Logistic curve
        # P(A) = 1 / (1 + 10^((Rb - Ra) / 400))
        diff = r_away - r_home
        prob = 1 / (1 + math.pow(10, diff / 400))
        return prob

    def run_batch(self, **kwargs):
        """Batch modu: DB'den bitmiş maçları çekip reytingleri günceller."""
        if self.db is None:
            logger.warning("EloEngine: DB bağlantısı yok.")
            return

        # Bitmiş ve henüz işlenmemiş maçları çek (mock: son 24 saat bitenler)
        # matches_df = self.db.get_finished_matches(hours_back=24)
        # Şimdilik mock veri olmadığı için boş geçiyoruz veya log basıyoruz
        logger.info("EloEngine: Batch güncellemesi çalıştırıldı (Veri akışı bekleniyor).")
        
        # Örnek:
        # for match in matches:
        #     self.update_ratings(...)

    def update_ratings(self, home: str, away: str, score_home: int, score_away: int):
        """Maç sonucuna göre reytingleri güncelle."""
        if home not in self.ratings: self.ratings[home] = EloRating(home)
        if away not in self.ratings: self.ratings[away] = EloRating(away)
        
        rh = self.ratings[home].rating + self.home_adv
        ra = self.ratings[away].rating
        
        # Beklenen skor (Expected Score)
        e_home = 1 / (1 + math.pow(10, (ra - rh) / 400))
        e_away = 1 - e_home
        
        # Gerçek skor (Actual Score): Win=1, Draw=0.5, Loss=0
        if score_home > score_away:
            s_home, s_away = 1.0, 0.0
        elif score_home == score_away:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0
            
        # Margin of Victory Multiplier (Gol farkı çarpanı)
        goal_diff = abs(score_home - score_away)
        # Formül: ln(|diff| + 1) * 2.2 / ((WinningElo - LosingElo)*0.001 + 2.2)
        # Basitleştirilmiş: (diff)^0.5
        margin_mult = math.log(goal_diff + 1) if goal_diff > 0 else 1.0
        
        # Delta hesapla
        delta_home = self.k * margin_mult * (s_home - e_home)
        
        # Güncelle
        self.ratings[home].rating += delta_home
        self.ratings[away].rating -= delta_home # Zero-sum update (biri artar, biri azalır)
        
        self.ratings[home].matches_played += 1
        self.ratings[away].matches_played += 1
        
        logger.debug(f"[ELO] {home} ({self.ratings[home].rating:.1f}) vs {away} ({self.ratings[away].rating:.1f}) -> Delta: {delta_home:+.1f}")

    def bulk_process_history(self, history: list[dict]):
        """Geçmiş maçları sırayla işleyip bugünkü reytingleri oluşturur."""
        # history: [{"home": "A", "away": "B", "sh": 1, "sa": 0, "date": ...}]
        # Tarihe göre sıralı olmalı
        for m in history:
            self.update_ratings(m["home"], m["away"], m["sh"], m["sa"])
