"""
digital_twin_sim.py – Agent-Based Modeling (ABM) ile Maçın Sayısal İkizi.

İstatistik geçmişe bakar. ABM ise geleceği simüle eder.
22 tane "Sanal Futbolcu" yaratıp maçı oynatacağız.

Kavramlar:
  - Agent: Her futbolcu bir ajan (hız, şut, pas, yorgunluk)
  - Environment: 105x68 metre saha
  - Interaction: Ajanlar topla temas, pas, şut, müdahale yapabilir
  - Simulation: Maç 90 dakika + uzatma olarak simüle edilir
  - Fatigue: Dakika ilerledikçe yorgunluk artar, performans düşer
  - Tactics: Formasyon ve taktik ayarları simülasyonu etkiler

Monte Carlo'dan farkı: Oyuncu etkileşimlerine dayalı sonuç!

Teknoloji: mesa (Python Agent-Based Modeling Framework)
Fallback: Pure Python simulation engine
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import ContinuousSpace
    MESA_OK = True
except ImportError:
    MESA_OK = False
    logger.info("mesa yüklü değil – pure Python simulation fallback.")


@dataclass
class PlayerAttributes:
    """Futbolcu özellikleri."""
    name: str = ""
    position: str = "MID"        # GK | DEF | MID | FWD
    team: str = ""
    speed: float = 70.0          # 0-100
    shooting: float = 60.0
    passing: float = 65.0
    defending: float = 55.0
    stamina: float = 80.0        # Başlangıç dayanıklılık
    fatigue: float = 0.0         # Anlık yorgunluk (0-1)
    x: float = 0.0               # Saha pozisyonu
    y: float = 0.0


@dataclass
class MatchEvent:
    """Maç olayı."""
    minute: int = 0
    event_type: str = ""         # goal | shot | foul | substitution | card
    team: str = ""
    player: str = ""
    description: str = ""


@dataclass
class SimulationResult:
    """Tek simülasyon sonucu."""
    home_goals: int = 0
    away_goals: int = 0
    events: list[MatchEvent] = field(default_factory=list)
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_possession: float = 0.5
    decisive_moment: str = ""


@dataclass
class TwinReport:
    """Sayısal İkiz toplam raporu."""
    match_id: str = ""
    home_team: str = ""
    away_team: str = ""
    n_simulations: int = 0
    # Olasılıklar
    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0
    # Gol dağılımı
    avg_home_goals: float = 0.0
    avg_away_goals: float = 0.0
    avg_total_goals: float = 0.0
    prob_over25: float = 0.0
    prob_btts: float = 0.0
    # En sık skor
    most_common_score: str = ""
    score_distribution: dict = field(default_factory=dict)
    # Simülasyon detayları
    avg_home_xg: float = 0.0
    avg_away_xg: float = 0.0
    fatigue_impact: str = ""
    key_player_impact: str = ""
    method: str = ""


# ═══════════════════════════════════════════════
#  FUTBOLCU AJANI
# ═══════════════════════════════════════════════
class FootballPlayerAgent:
    """Basit futbolcu ajanı (mesa Agent veya standalone)."""

    def __init__(self, attrs: PlayerAttributes):
        self.attrs = attrs
        self._base_speed = attrs.speed
        self._base_shooting = attrs.shooting
        self._base_passing = attrs.passing
        self._base_defending = attrs.defending

    def update_fatigue(self, minute: int) -> None:
        """Yorgunluk güncelle (dakikaya göre)."""
        # 0-45: yavaş artış, 45-70: orta, 70-90: hızlı
        if minute < 45:
            rate = 0.005
        elif minute < 70:
            rate = 0.010
        else:
            rate = 0.020

        base_fatigue = minute * rate
        stamina_factor = (100 - self.attrs.stamina) / 100
        self.attrs.fatigue = min(1.0, base_fatigue * (1 + stamina_factor))

        # Performans düşüşü
        decay = 1 - self.attrs.fatigue * 0.4
        self.attrs.speed = self._base_speed * decay
        self.attrs.shooting = self._base_shooting * decay
        self.attrs.passing = self._base_passing * decay
        self.attrs.defending = self._base_defending * decay

    def attempt_shot(self) -> tuple[bool, float]:
        """Şut girişimi. Returns: (isabet, xG katkısı)."""
        accuracy = self.attrs.shooting / 100
        distance_factor = random.uniform(0.3, 1.0)
        xg_contribution = accuracy * distance_factor * 0.3

        is_goal = random.random() < xg_contribution
        return is_goal, xg_contribution

    def attempt_pass(self) -> bool:
        """Pas girişimi."""
        success_rate = self.attrs.passing / 100
        return random.random() < success_rate * 0.85

    def attempt_tackle(self) -> tuple[bool, bool]:
        """Müdahale. Returns: (başarılı, faul)."""
        success = random.random() < self.attrs.defending / 100
        is_foul = random.random() < 0.15
        return success, is_foul


# ═══════════════════════════════════════════════
#  MAÇ SİMÜLASYON MOTORU
# ═══════════════════════════════════════════════
class MatchSimulator:
    """Tek maç simülasyonu."""

    PITCH_LENGTH = 105
    PITCH_WIDTH = 68
    MATCH_DURATION = 90

    def __init__(self, home_players: list[PlayerAttributes],
                 away_players: list[PlayerAttributes]):
        self._home = [FootballPlayerAgent(p) for p in home_players]
        self._away = [FootballPlayerAgent(p) for p in away_players]

    def simulate(self) -> SimulationResult:
        """90 dakikalık maç simüle et."""
        result = SimulationResult()
        ball_team = "home" if random.random() < 0.5 else "away"

        for minute in range(1, self.MATCH_DURATION + 1):
            # Yorgunluk güncelle
            for agent in self._home + self._away:
                agent.update_fatigue(minute)

            # Her dakika için olay simülasyonu
            events = self._simulate_minute(minute, ball_team, result)
            result.events.extend(events)

            # Top el değişimi
            if random.random() < 0.45:
                ball_team = "away" if ball_team == "home" else "home"

        # İstatistikler
        result.home_possession = self._calc_possession()

        return result

    def _simulate_minute(self, minute: int, ball_team: str,
                          result: SimulationResult) -> list[MatchEvent]:
        """Tek dakika simülasyonu."""
        events = []
        attackers = self._home if ball_team == "home" else self._away
        defenders = self._away if ball_team == "home" else self._home

        # Şut şansı (dakikaya ve takım gücüne bağlı)
        attack_power = np.mean([a.attrs.shooting for a in attackers])
        defense_power = np.mean([d.attrs.defending for d in defenders])

        shot_chance = (attack_power - defense_power * 0.6) / 300
        shot_chance = max(0.01, min(0.15, shot_chance))

        if random.random() < shot_chance:
            # Şut atan oyuncu (forvet öncelikli)
            forwards = [a for a in attackers if a.attrs.position == "FWD"]
            mids = [a for a in attackers if a.attrs.position == "MID"]
            shooter = random.choice(forwards or mids or attackers)

            is_goal, xg = shooter.attempt_shot()

            if ball_team == "home":
                result.home_xg += xg
            else:
                result.away_xg += xg

            if is_goal:
                if ball_team == "home":
                    result.home_goals += 1
                else:
                    result.away_goals += 1

                events.append(MatchEvent(
                    minute=minute,
                    event_type="goal",
                    team=ball_team,
                    player=shooter.attrs.name,
                    description=f"GOL! {shooter.attrs.name} ({minute}')",
                ))
            else:
                events.append(MatchEvent(
                    minute=minute,
                    event_type="shot",
                    team=ball_team,
                    player=shooter.attrs.name,
                ))

        # Faul şansı (yorgunluk arttıkça faul artar)
        avg_fatigue = np.mean([d.attrs.fatigue for d in defenders])
        foul_chance = 0.03 + avg_fatigue * 0.10

        if random.random() < foul_chance:
            fouler = random.choice(defenders)
            team = "away" if ball_team == "home" else "home"

            card = ""
            if random.random() < 0.15:
                card = "yellow"
            elif random.random() < 0.02:
                card = "red"

            events.append(MatchEvent(
                minute=minute,
                event_type="foul" if not card else "card",
                team=team,
                player=fouler.attrs.name,
                description=f"{'Sarı kart' if card == 'yellow' else 'Kırmızı kart' if card == 'red' else 'Faul'}: {fouler.attrs.name} ({minute}')",
            ))

        return events

    def _calc_possession(self) -> float:
        """Top oynama oranı."""
        home_pass = np.mean([a.attrs.passing for a in self._home])
        away_pass = np.mean([a.attrs.passing for a in self._away])
        total = home_pass + away_pass
        return home_pass / total if total > 0 else 0.5


# ═══════════════════════════════════════════════
#  DIGITAL TWIN (Ana Sınıf)
# ═══════════════════════════════════════════════
class DigitalTwinSimulator:
    """Agent-Based Modeling ile maçın sayısal ikizi.

    Kullanım:
        twin = DigitalTwinSimulator()
        # Oyuncu özelliklerini yükle
        home = twin.generate_team("Galatasaray", quality=78)
        away = twin.generate_team("Fenerbahçe", quality=76)
        # 1000 kez simüle et
        report = twin.simulate_match("GS_FB", home, away, n_sims=1000)
    """

    FORMATIONS = {
        "442": {"DEF": 4, "MID": 4, "FWD": 2},
        "433": {"DEF": 4, "MID": 3, "FWD": 3},
        "352": {"DEF": 3, "MID": 5, "FWD": 2},
        "4231": {"DEF": 4, "MID": 5, "FWD": 1},
        "343": {"DEF": 3, "MID": 4, "FWD": 3},
    }

    def __init__(self):
        logger.debug("[Twin] Digital Twin Simulator başlatıldı.")

    def generate_team(self, team_name: str, quality: float = 70,
                       formation: str = "442",
                       player_attrs: list[dict] | None = None
                       ) -> list[PlayerAttributes]:
        """Takım kadrosu üret.

        Eğer player_attrs verilmişse (Neo4j'den), onları kullan.
        Yoksa quality bazlı rastgele oluştur.
        """
        if player_attrs:
            return [
                PlayerAttributes(
                    name=p.get("name", f"Player_{i}"),
                    position=p.get("position", "MID"),
                    team=team_name,
                    speed=p.get("speed", quality),
                    shooting=p.get("shooting", quality),
                    passing=p.get("passing", quality),
                    defending=p.get("defending", quality),
                    stamina=p.get("stamina", 80),
                )
                for i, p in enumerate(player_attrs[:11])
            ]

        form = self.FORMATIONS.get(formation, self.FORMATIONS["442"])
        players = []

        # Kaleci
        players.append(PlayerAttributes(
            name=f"{team_name}_GK",
            position="GK",
            team=team_name,
            speed=quality * 0.7 + random.uniform(-5, 5),
            shooting=quality * 0.3,
            passing=quality * 0.6 + random.uniform(-5, 5),
            defending=quality * 0.9 + random.uniform(-5, 5),
            stamina=85 + random.uniform(-5, 5),
        ))

        # Alan oyuncuları
        idx = 1
        for pos, count in form.items():
            for _ in range(count):
                noise = random.uniform(-8, 8)
                if pos == "DEF":
                    attr = PlayerAttributes(
                        name=f"{team_name}_DEF{idx}",
                        position="DEF", team=team_name,
                        speed=quality * 0.85 + noise,
                        shooting=quality * 0.4 + noise,
                        passing=quality * 0.7 + noise,
                        defending=quality * 1.1 + noise,
                        stamina=78 + random.uniform(-5, 5),
                    )
                elif pos == "MID":
                    attr = PlayerAttributes(
                        name=f"{team_name}_MID{idx}",
                        position="MID", team=team_name,
                        speed=quality * 0.9 + noise,
                        shooting=quality * 0.7 + noise,
                        passing=quality * 1.1 + noise,
                        defending=quality * 0.65 + noise,
                        stamina=82 + random.uniform(-5, 5),
                    )
                else:
                    attr = PlayerAttributes(
                        name=f"{team_name}_FWD{idx}",
                        position="FWD", team=team_name,
                        speed=quality * 1.05 + noise,
                        shooting=quality * 1.15 + noise,
                        passing=quality * 0.75 + noise,
                        defending=quality * 0.35 + noise,
                        stamina=76 + random.uniform(-5, 5),
                    )
                players.append(attr)
                idx += 1

        return players[:11]

    def simulate_match(self, match_id: str,
                        home_players: list[PlayerAttributes],
                        away_players: list[PlayerAttributes],
                        n_sims: int = 1000) -> TwinReport:
        """Maçı n_sims kez simüle et ve rapor çıkar."""
        report = TwinReport(
            match_id=match_id,
            home_team=home_players[0].team if home_players else "",
            away_team=away_players[0].team if away_players else "",
            n_simulations=n_sims,
            method="agent_based_mesa" if MESA_OK else "agent_based_pure",
        )

        results: list[SimulationResult] = []
        scores: list[tuple[int, int]] = []

        for _ in range(n_sims):
            # Her simülasyon için oyuncuları sıfırla
            home_copy = [
                PlayerAttributes(**{
                    k: getattr(p, k) for k in PlayerAttributes.__dataclass_fields__
                })
                for p in home_players
            ]
            away_copy = [
                PlayerAttributes(**{
                    k: getattr(p, k) for k in PlayerAttributes.__dataclass_fields__
                })
                for p in away_players
            ]

            sim = MatchSimulator(home_copy, away_copy)
            result = sim.simulate()
            results.append(result)
            scores.append((result.home_goals, result.away_goals))

        # ── İstatistikler ──
        home_wins = sum(1 for h, a in scores if h > a)
        draws = sum(1 for h, a in scores if h == a)
        away_wins = sum(1 for h, a in scores if h < a)

        report.prob_home = round(home_wins / n_sims, 4)
        report.prob_draw = round(draws / n_sims, 4)
        report.prob_away = round(away_wins / n_sims, 4)

        report.avg_home_goals = round(np.mean([s[0] for s in scores]), 2)
        report.avg_away_goals = round(np.mean([s[1] for s in scores]), 2)
        report.avg_total_goals = round(
            report.avg_home_goals + report.avg_away_goals, 2,
        )

        report.prob_over25 = round(
            sum(1 for h, a in scores if h + a > 2) / n_sims, 4,
        )
        report.prob_btts = round(
            sum(1 for h, a in scores if h > 0 and a > 0) / n_sims, 4,
        )

        # En sık skor
        score_counter = Counter(scores)
        most_common = score_counter.most_common(5)
        if most_common:
            mc = most_common[0][0]
            report.most_common_score = f"{mc[0]}-{mc[1]}"
            report.score_distribution = {
                f"{s[0]}-{s[1]}": round(c / n_sims, 4)
                for s, c in most_common
            }

        # xG ortalamaları
        report.avg_home_xg = round(np.mean([r.home_xg for r in results]), 2)
        report.avg_away_xg = round(np.mean([r.away_xg for r in results]), 2)

        # Yorgunluk etkisi
        late_goals = sum(
            1 for r in results
            for e in r.events
            if e.event_type == "goal" and e.minute > 70
        )
        total_goals = sum(r.home_goals + r.away_goals for r in results)
        if total_goals > 0:
            late_ratio = late_goals / total_goals
            report.fatigue_impact = (
                f"Geç gollerin oranı: {late_ratio:.0%} "
                f"(70+ dk sonrası)"
            )

        return report

    def what_if_scenario(self, match_id: str,
                          home_players: list[PlayerAttributes],
                          away_players: list[PlayerAttributes],
                          scenario: str = "key_player_injured",
                          target_player: str = "",
                          n_sims: int = 500) -> dict:
        """'Eğer X olursa ne olur?' senaryosu.

        Senaryolar:
          - key_player_injured: Yıldız oyuncu çıkarsa
          - fatigue_boost: Takımın kondisyonu iyi
          - formation_change: Formasyon değişikliği
          - red_card_60: 60. dakikada kırmızı kart
        """
        # Normal simülasyon
        normal = self.simulate_match(match_id, home_players, away_players, n_sims)

        # Senaryo uygula
        modified_home = [
            PlayerAttributes(**{
                k: getattr(p, k) for k in PlayerAttributes.__dataclass_fields__
            })
            for p in home_players
        ]

        if scenario == "key_player_injured":
            # En iyi forvet/orta saha çıkar
            if not target_player:
                key = max(
                    [p for p in modified_home if p.position in ("FWD", "MID")],
                    key=lambda p: p.shooting + p.passing,
                    default=None,
                )
                if key:
                    target_player = key.name

            for p in modified_home:
                if p.name == target_player:
                    p.shooting *= 0.3
                    p.passing *= 0.3
                    p.speed *= 0.3
                    break

        elif scenario == "fatigue_boost":
            for p in modified_home:
                p.stamina = min(99, p.stamina + 10)

        elif scenario == "red_card_60":
            if modified_home:
                # Rastgele bir oyuncuyu çıkar
                modified_home.pop(random.randint(1, len(modified_home) - 1))
                # 10 kişi: kalan oyuncuların yükü artar
                for p in modified_home:
                    p.stamina *= 0.85

        scenario_result = self.simulate_match(
            f"{match_id}_scenario", modified_home, away_players, n_sims,
        )

        return {
            "scenario": scenario,
            "target_player": target_player,
            "normal": {
                "prob_home": normal.prob_home,
                "prob_draw": normal.prob_draw,
                "prob_away": normal.prob_away,
                "avg_goals": normal.avg_total_goals,
            },
            "modified": {
                "prob_home": scenario_result.prob_home,
                "prob_draw": scenario_result.prob_draw,
                "prob_away": scenario_result.prob_away,
                "avg_goals": scenario_result.avg_total_goals,
            },
            "impact": {
                "prob_home_change": round(
                    scenario_result.prob_home - normal.prob_home, 4,
                ),
                "prob_away_change": round(
                    scenario_result.prob_away - normal.prob_away, 4,
                ),
                "goals_change": round(
                    scenario_result.avg_total_goals - normal.avg_total_goals, 2,
                ),
            },
        }
