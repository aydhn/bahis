"""
mock_generator.py – API kesintilerinde devreye giren sahte veri üreteci.
Sistemin mantıksal akışını test etmek ve "HİÇ İŞLEM OLMUYOR" durumunu engellemek için kullanılır.
"""
import random
import time
from datetime import datetime, timedelta
from loguru import logger

class MockGenerator:
    """Rastgele ama tutarlı maç verisi üretir."""

    TEAMS = [
        "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor",
        "Real Madrid", "Barcelona", "Bayern Munich", "Dortmund",
        "Manchester City", "Arsenal", "Liverpool", "Chelsea",
        "Juventus", "Inter", "Milan", "Napoli",
        "PSG", "Marsilya", "Ajax", "PSV"
    ]

    def __init__(self):
        self._match_cache = {}

    def generate_live_matches(self, n: int = 10) -> list[dict]:
        """Canlı maç listesi üret."""
        matches = []
        used_teams = set()

        for _ in range(n):
            # Takım seçimi
            available = [t for t in self.TEAMS if t not in used_teams]
            if len(available) < 2:
                break

            home = random.choice(available)
            available.remove(home)
            away = random.choice(available)
            available.remove(away)

            used_teams.add(home)
            used_teams.add(away)

            # Oran üretimi (Margin dahil)
            prob_home = random.uniform(0.3, 0.6)
            prob_draw = random.uniform(0.2, 0.3)
            prob_away = 1.0 - prob_home - prob_draw

            # Margin ekle (Bookmaker payı)
            margin = 1.05
            odds_home = round(1 / (prob_home * margin), 2)
            odds_draw = round(1 / (prob_draw * margin), 2)
            odds_away = round(1 / (prob_away * margin), 2)

            match_id = f"mock_{home[:3].lower()}{away[:3].lower()}_{int(time.time())}"

            # Rastgele xG ve skor
            home_xg = round(random.uniform(0.5, 2.5), 2)
            away_xg = round(random.uniform(0.2, 1.8), 2)

            matches.append({
                "source": "mock_generator",
                "match_id": match_id,
                "sport": "football",
                "league": "Mock Super League",
                "country": "Simulasyon",
                "home_team": home,
                "away_team": away,
                "kickoff": datetime.utcnow().isoformat(),
                "status": "live",
                "minute": random.randint(10, 80),
                "home_score": random.randint(0, 3),
                "away_score": random.randint(0, 2),
                "home_odds": odds_home,
                "draw_odds": odds_draw,
                "away_odds": odds_away,
                "home_xg": home_xg,
                "away_xg": away_xg,
            })

        logger.warning(f"[MockGenerator] {len(matches)} adet sahte canlı veri üretildi (API YEDEK).")
        return matches

    def generate_upcoming_matches(self, n: int = 5) -> list[dict]:
        """Gelecek maç listesi üret."""
        matches = self.generate_live_matches(n)
        for m in matches:
            m["status"] = "upcoming"
            m["kickoff"] = (datetime.utcnow() + timedelta(hours=random.randint(1, 24))).isoformat()
            del m["minute"]
            del m["home_score"]
            del m["away_score"]
        return matches
