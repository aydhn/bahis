"""
elo_glicko_rating.py – Dinamik takım güç derecelendirmesi.
Lig tablosuna güvenmek yerine, takımların anlık güç dengesini hesaplar.
Elo + Glicko-2: güçlü takımı yenen zayıf takım daha çok puan kazanır.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class EloTeam:
    name: str
    rating: float = 1500.0
    home_rating: float = 1500.0
    away_rating: float = 1500.0
    matches_played: int = 0
    form: list[str] = field(default_factory=list)  # Son 5 maç


@dataclass
class GlickoTeam:
    name: str
    rating: float = 1500.0
    rd: float = 350.0        # Rating Deviation (belirsizlik)
    volatility: float = 0.06  # Volatilite
    matches_played: int = 0


class EloRating:
    """Klasik Elo rating sistemi – futbola uyarlanmış."""

    HOME_ADVANTAGE = 65  # Ev sahibi avantajı (Elo puanı)

    def __init__(self, k_factor: float = 32.0, home_advantage: float = 65.0):
        self._k = k_factor
        self._home_adv = home_advantage
        self._teams: dict[str, EloTeam] = {}
        logger.debug("EloRating başlatıldı.")

    def get_or_create(self, name: str) -> EloTeam:
        if name not in self._teams:
            self._teams[name] = EloTeam(name=name)
        return self._teams[name]

    def to_dict(self) -> Dict[str, Any]:
        """Durumu sözlük olarak döndür."""
        return {
            "k_factor": self._k,
            "home_advantage": self._home_adv,
            "teams": {name: asdict(team) for name, team in self._teams.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EloRating":
        """Sözlükten durumu yükle."""
        instance = cls(
            k_factor=data.get("k_factor", 32.0),
            home_advantage=data.get("home_advantage", 65.0)
        )

        teams_data = data.get("teams", {})
        for name, team_data in teams_data.items():
            instance._teams[name] = EloTeam(**team_data)

        return instance

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Beklenen skor (0-1 arası)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def predict(self, home: str, away: str) -> dict:
        """Maç olasılıklarını tahmin eder."""
        h = self.get_or_create(home)
        a = self.get_or_create(away)

        # Ev sahibi avantajı ekle
        home_eff = h.rating + self._home_adv
        away_eff = a.rating

        exp_home = self.expected_score(home_eff, away_eff)
        exp_away = 1 - exp_home

        # Beraberlik olasılığı: iki takım yakınsa artar
        rating_diff = abs(home_eff - away_eff)
        draw_factor = max(0.05, 0.28 - rating_diff / 2000)

        p_home = exp_home * (1 - draw_factor)
        p_away = exp_away * (1 - draw_factor)
        p_draw = draw_factor

        # Normalize
        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total

        return {
            "home_team": home,
            "away_team": away,
            "elo_home": h.rating,
            "elo_away": a.rating,
            "prob_home": float(p_home),
            "prob_draw": float(p_draw),
            "prob_away": float(p_away),
            "elo_diff": float(home_eff - away_eff),
        }

    def update(self, home: str, away: str, home_goals: int, away_goals: int):
        """Maç sonucuna göre ratinglari günceller."""
        h = self.get_or_create(home)
        a = self.get_or_create(away)

        # Gerçek skor
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
            form_h, form_a = "W", "L"
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
            form_h, form_a = "D", "D"
        else:
            actual_home, actual_away = 0.0, 1.0
            form_h, form_a = "L", "W"

        # Gol farkı çarpanı (büyük fark = büyük güncelleme)
        goal_diff = abs(home_goals - away_goals)
        gd_multiplier = math.log(max(goal_diff, 1) + 1) + 1

        # Beklenen vs gerçek
        exp_h = self.expected_score(h.rating + self._home_adv, a.rating)
        exp_a = 1 - exp_h

        # K faktörü ayarı (deneyimli takımlarda düşük)
        k_h = self._k * gd_multiplier / (1 + h.matches_played * 0.002)
        k_a = self._k * gd_multiplier / (1 + a.matches_played * 0.002)

        h.rating += k_h * (actual_home - exp_h)
        a.rating += k_a * (actual_away - exp_a)

        # İç saha / deplasman ayrı ratinglar
        h.home_rating += k_h * 0.5 * (actual_home - exp_h)
        a.away_rating += k_a * 0.5 * (actual_away - exp_a)

        h.matches_played += 1
        a.matches_played += 1

        # Form (son 5 maç)
        h.form.append(form_h)
        a.form.append(form_a)
        h.form = h.form[-5:]
        a.form = a.form[-5:]

    def get_rankings(self, top_n: int = 30) -> list[dict]:
        """Rating sıralamasını döndürür."""
        sorted_teams = sorted(self._teams.values(), key=lambda t: t.rating, reverse=True)
        return [
            {
                "rank": i + 1,
                "team": t.name,
                "rating": round(t.rating, 1),
                "home_rating": round(t.home_rating, 1),
                "away_rating": round(t.away_rating, 1),
                "matches": t.matches_played,
                "form": "".join(t.form[-5:]),
            }
            for i, t in enumerate(sorted_teams[:top_n])
        ]


class Glicko2Rating:
    """Glicko-2 rating sistemi – belirsizlik ve volatilite dahil."""

    TAU = 0.5  # Sistem sabiti

    def __init__(self):
        self._teams: dict[str, GlickoTeam] = {}
        logger.debug("Glicko2Rating başlatıldı.")

    def get_or_create(self, name: str) -> GlickoTeam:
        if name not in self._teams:
            self._teams[name] = GlickoTeam(name=name)
        return self._teams[name]

    def to_dict(self) -> Dict[str, Any]:
        """Durumu sözlük olarak döndür."""
        return {
            "teams": {name: asdict(team) for name, team in self._teams.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Glicko2Rating":
        """Sözlükten durumu yükle."""
        instance = cls()
        teams_data = data.get("teams", {})
        for name, team_data in teams_data.items():
            instance._teams[name] = GlickoTeam(**team_data)
        return instance

    def _g(self, rd: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * (rd / 400.0) ** 2 / math.pi ** 2)

    def _E(self, mu: float, mu_j: float, rd_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-self._g(rd_j) * (mu - mu_j)))

    def predict(self, home: str, away: str) -> dict:
        h = self.get_or_create(home)
        a = self.get_or_create(away)

        # Glicko ölçeğine dönüştür
        mu_h = (h.rating - 1500) / 173.7178
        mu_a = (a.rating - 1500) / 173.7178

        exp_h = self._E(mu_h, mu_a, a.rd)
        exp_a = 1 - exp_h

        # Belirsizlik: yüksek RD = düşük güven
        confidence = 1 - (h.rd + a.rd) / 700

        return {
            "home_team": home,
            "away_team": away,
            "glicko_home": h.rating,
            "glicko_away": a.rating,
            "rd_home": h.rd,
            "rd_away": a.rd,
            "prob_home": float(np.clip(exp_h, 0.05, 0.95)),
            "prob_away": float(np.clip(exp_a, 0.05, 0.95)),
            "confidence": float(np.clip(confidence, 0.1, 0.95)),
        }

    def update(self, home: str, away: str, result: float):
        """Glicko-2 güncellemesi. result: 1.0=home win, 0.5=draw, 0.0=away win"""
        h = self.get_or_create(home)
        a = self.get_or_create(away)

        # Step 2: Ölçek dönüşümü
        mu_h = (h.rating - 1500) / 173.7178
        phi_h = h.rd / 173.7178
        mu_a = (a.rating - 1500) / 173.7178
        phi_a = a.rd / 173.7178

        # Step 3-4: v ve delta hesabı (home perspektifi)
        g_a = self._g(a.rd)
        E_a = self._E(mu_h, mu_a, a.rd)
        v = 1.0 / (g_a ** 2 * E_a * (1 - E_a) + 1e-10)
        delta = v * g_a * (result - E_a)

        # Step 5: Yeni volatilite (basitleştirilmiş)
        new_vol_h = h.volatility  # Tam iterasyon yerine sabit tutuyoruz

        # Step 6-7: Yeni phi ve mu
        phi_star = math.sqrt(phi_h ** 2 + new_vol_h ** 2)
        new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
        new_mu = mu_h + new_phi ** 2 * g_a * (result - E_a)

        # Geri dönüştür
        h.rating = new_mu * 173.7178 + 1500
        h.rd = max(new_phi * 173.7178, 30)  # Min RD
        h.matches_played += 1

        # Away için simetrik güncelleme
        g_h = self._g(h.rd)
        E_h = self._E(mu_a, mu_h, h.rd)
        v_a = 1.0 / (g_h ** 2 * E_h * (1 - E_h) + 1e-10)
        phi_star_a = math.sqrt(phi_a ** 2 + a.volatility ** 2)
        new_phi_a = 1.0 / math.sqrt(1.0 / phi_star_a ** 2 + 1.0 / v_a)
        new_mu_a = mu_a + new_phi_a ** 2 * g_h * ((1 - result) - E_h)

        a.rating = new_mu_a * 173.7178 + 1500
        a.rd = max(new_phi_a * 173.7178, 30)
        a.matches_played += 1


class EloGlickoSystem:
    """Elo ve Glicko-2'yi birleştiren hibrit sistem."""

    def __init__(self):
        self.elo = EloRating()
        self.glicko = Glicko2Rating()
        self.processed_matches: set[str] = set()
        logger.debug("EloGlickoSystem başlatıldı.")

    def predict(self, home: str, away: str) -> dict:
        elo_pred = self.elo.predict(home, away)
        glicko_pred = self.glicko.predict(home, away)

        # Ağırlıklı ortalama (Glicko daha sofistike)
        w_elo, w_glicko = 0.4, 0.6
        return {
            "match_id": f"{home}_vs_{away}",
            "home_team": home,
            "away_team": away,
            "prob_home": w_elo * elo_pred["prob_home"] + w_glicko * glicko_pred["prob_home"],
            "prob_draw": elo_pred["prob_draw"] * 0.8 + 0.2 * (1 - glicko_pred["prob_home"] - glicko_pred["prob_away"]),
            "prob_away": w_elo * elo_pred["prob_away"] + w_glicko * glicko_pred["prob_away"],
            "elo_home": elo_pred["elo_home"],
            "elo_away": elo_pred["elo_away"],
            "glicko_home": glicko_pred["glicko_home"],
            "glicko_away": glicko_pred["glicko_away"],
            "confidence": glicko_pred["confidence"],
            "elo_diff": elo_pred["elo_diff"],
        }

    def update(self, home: str, away: str, home_goals: int, away_goals: int):
        self.elo.update(home, away, home_goals, away_goals)
        if home_goals > away_goals:
            result = 1.0
        elif home_goals == away_goals:
            result = 0.5
        else:
            result = 0.0
        self.glicko.update(home, away, result)

    def process_batch(self, matches_df: pl.DataFrame):
        """Toplu veri ile sistemi eğitir."""
        if matches_df.is_empty():
            return

        # Tarihe göre sırala
        if "kickoff" in matches_df.columns:
            sorted_df = matches_df.sort("kickoff")
        else:
            sorted_df = matches_df

        count = 0
        for row in sorted_df.iter_rows(named=True):
            mid = row.get("match_id")
            if mid and mid in self.processed_matches:
                continue

            home = row.get("home_team")
            away = row.get("away_team")
            h_score = row.get("home_score")
            a_score = row.get("away_score")

            if home and away and h_score is not None and a_score is not None:
                self.update(home, away, int(h_score), int(a_score))
                if mid:
                    self.processed_matches.add(mid)
                count += 1

        if count > 0:
            logger.info(f"EloGlickoSystem: {count} yeni maç ile güncellendi.")

    def predict_for_dataframe(self, features: pl.DataFrame) -> pl.DataFrame:
        results = []
        for row in features.iter_rows(named=True):
            pred = self.predict(row.get("home_team", ""), row.get("away_team", ""))
            pred["match_id"] = row.get("match_id", "")
            results.append(pred)
        return pl.DataFrame(results) if results else pl.DataFrame()

    def save_state(self, filepath: str | Path):
        """Durumu diske JSON formatında kaydeder."""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Use .json extension if not present, but respect original path for now
            # We will use text write mode

            state = {
                "elo": self.elo.to_dict(),
                "glicko": self.glicko.to_dict(),
                "processed": list(self.processed_matches)
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Elo state saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save Elo state: {e}")

    def load_state(self, filepath: str | Path) -> bool:
        """Durumu diskten (JSON) yükler."""
        path = Path(filepath)
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

                if "elo" in state:
                    self.elo = EloRating.from_dict(state["elo"])
                if "glicko" in state:
                    self.glicko = Glicko2Rating.from_dict(state["glicko"])
                if "processed" in state:
                    self.processed_matches = set(state["processed"])

            logger.info(f"Elo state loaded from {path} (Matches: {len(self.processed_matches)})")
            return True
        except Exception as e:
            logger.error(f"Failed to load Elo state: {e}")
            return False
