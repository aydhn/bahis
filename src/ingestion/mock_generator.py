"""
mock_generator.py – Otonom Sistem için Sentetik Veri Üretici.

API kesintilerinde veya test aşamasında sistemin durmaması için
gerçekçi futbol verisi (maçlar, oranlar, istatistikler) üretir.
"""
from typing import List, Dict, Any
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from loguru import logger

class MockGenerator:
    """Sentetik veri üretim motoru."""

    TEAMS = [
        "Galatasaray", "Fenerbahce", "Besiktas", "Trabzonspor",
        "Basaksehir", "Adana Demirspor", "Kayserispor", "Konyaspor",
        "Antalyaspor", "Kasimpasa", "Alanyaspor", "Sivasspor",
        "Ankaragucu", "Hatayspor", "Gaziantep FK", "Istanbulspor",
        "Samsunspor", "Rizespor", "Pendikspor", "Fatih Karagumruk"
    ]

    def __init__(self):
        logger.info("MockGenerator başlatıldı: Sentetik veri modu aktif.")

    def generate_matches(self, n: int = 10) -> pl.DataFrame:
        """Rastgele maç fikstürü ve oranları üret."""
        matches = []

        # Rastgele eşleşmeler
        teams = np.random.choice(self.TEAMS, size=n*2, replace=True)

        for i in range(n):
            home = teams[2*i]
            away = teams[2*i+1]
            if home == away:
                away = "Bursaspor" # Çakışma önlemi

            # Rastgele oranlar (Margin dahil)
            prob_h = np.random.uniform(0.3, 0.7)
            prob_d = np.random.uniform(0.2, 0.35)
            prob_a = 1.0 - prob_h - prob_d

            # Margin ekle (%5)
            margin = 1.05
            odds_h = round(1 / prob_h * margin, 2)
            odds_d = round(1 / prob_d * margin, 2)
            odds_a = round(1 / prob_a * margin, 2)

            matches.append({
                "match_id": f"{home}_{away}",
                "date": (datetime.now() + timedelta(days=np.random.randint(0, 3))).strftime("%Y-%m-%d"),
                "league": "Super Lig",
                "home_team": home,
                "away_team": away,
                "home_odds": odds_h,
                "draw_odds": odds_d,
                "away_odds": odds_a,
                "status": "Not Started"
            })

        return pl.DataFrame(matches)

    def generate_features(self, match_ids: List[str]) -> pl.DataFrame:
        """Maçlar için rastgele feature (xg, form, eksik) üret."""
        features = []

        for mid in match_ids:
            # Home/Away xG beklentisi
            h_xg = round(np.random.normal(1.4, 0.4), 2)
            a_xg = round(np.random.normal(1.1, 0.4), 2)

            features.append({
                "match_id": mid,
                "home_xg": max(0.1, h_xg),
                "away_xg": max(0.1, a_xg),
                "home_last_5_points": np.random.randint(0, 15),
                "away_last_5_points": np.random.randint(0, 15),
                "home_missing_players": np.random.randint(0, 4),
                "away_missing_players": np.random.randint(0, 4),
                "rain_forecast": bool(np.random.choice([True, False], p=[0.2, 0.8])),
                "derby_match": bool(np.random.choice([True, False], p=[0.1, 0.9]))
            })

        return pl.DataFrame(features)

    def generate_news(self, match_ids: List[str]) -> List[Dict[str, Any]]:
        """Rastgele haber özetleri üret (RAG simülasyonu)."""
        news = []
        templates = [
            "Teknik direktör istifa etti.",
            "Yıldız oyuncu sakatlandı.",
            "Takım formunun zirvesinde.",
            "Taraftar desteği muazzam.",
            "Mali kriz söylentileri var."
        ]

        for mid in match_ids:
            news.append({
                "match_id": mid,
                "summary": np.random.choice(templates),
                "sentiment_score": np.random.uniform(-0.8, 0.8)
            })
        return news
