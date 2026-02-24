"""
kalman_tracker.py – Dinamik Güç Takibi (Kalman Filtresi).

Takımların gücü sabit değildir. Bir füzenin rotası gibi sürekli
hareket halindedir ve gürültülüdür.

Kalman Filtresi:
  - "Gerçek Güç" (Hidden State) = gizli değişken
  - "Maç Sonucu" (Observation) = gürültülü gözlem
  - Her maçtan sonra güncelle: predict → update → smooth
  - Elo'dan çok daha hassas ve hızlı tepki verir

State Vector: [strength, momentum]
  strength: Takımın anlık gücü (0-100 arası normalize)
  momentum: Gücün değişim hızı (ivme)

Observation:
  Maç sonucundan türetilen gürültülü güç gözlemi
  (kazanma/kaybetme, skor farkı, xG farkı, rakip gücü)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_OK = True
except ImportError:
    FILTERPY_OK = False
    logger.info("filterpy yüklü değil – basit exponential smooth aktif.")


@dataclass
class TeamState:
    """Bir takımın Kalman state vektörü."""
    team: str
    strength: float = 50.0       # Anlık güç (0-100)
    momentum: float = 0.0        # İvme (pozitif = yükselişte)
    uncertainty: float = 10.0    # State kovaryansı (düşük = emin)
    last_updated: str = ""
    n_observations: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class MatchObservation:
    """Maç sonucundan türetilen gözlem."""
    team: str
    opponent: str
    date: str = ""
    is_home: bool = True

    # Ham sonuç
    goals_for: int = 0
    goals_against: int = 0
    result: str = ""       # W / D / L
    points: int = 0

    # Gelişmiş metrikler (varsa)
    xg_for: float = 0.0
    xg_against: float = 0.0
    possession: float = 50.0
    shots_on_target: int = 0
    opponent_strength: float = 50.0

    # Türetilmiş gözlem değeri
    observed_strength: float = 50.0


class KalmanTeamTracker:
    """Her takım için ayrı Kalman filtresi ile dinamik güç takibi.

    Kullanım:
        tracker = KalmanTeamTracker()
        # Maç sonucu güncelle
        tracker.update("Galatasaray", MatchObservation(...))
        # Güncel güç al
        state = tracker.get_state("Galatasaray")
        # Maç tahmini
        pred = tracker.predict_match("Galatasaray", "Fenerbahce")
    """

    # Gözlem ağırlıkları (hangi metrik ne kadar etkili)
    RESULT_WEIGHT = 0.30         # Kazanma/kaybetme
    SCORE_DIFF_WEIGHT = 0.20     # Skor farkı
    XG_WEIGHT = 0.25             # xG performansı
    OPPONENT_WEIGHT = 0.15       # Rakip güçe göre normalizasyon
    POSSESSION_WEIGHT = 0.10     # Top kontrolü

    # Kalman parametreleri
    PROCESS_NOISE = 2.0          # Güç ne kadar hızlı değişebilir
    MEASUREMENT_NOISE = 8.0      # Gözlem ne kadar gürültülü
    INITIAL_STRENGTH = 50.0
    INITIAL_UNCERTAINTY = 15.0

    # Ev sahibi avantajı
    HOME_ADVANTAGE = 5.0

    def __init__(self, process_noise: float = 2.0,
                 measurement_noise: float = 8.0):
        self.PROCESS_NOISE = process_noise
        self.MEASUREMENT_NOISE = measurement_noise
        self._filters: dict[str, Any] = {}     # team → KalmanFilter
        self._states: dict[str, TeamState] = {}
        logger.debug(
            f"KalmanTeamTracker başlatıldı "
            f"(Q={process_noise}, R={measurement_noise})"
        )

    # ═══════════════════════════════════════════
    #  FİLTRE OLUŞTURMA
    # ═══════════════════════════════════════════
    def _create_filter(self, team: str) -> Any:
        """Takım için Kalman filtresi oluştur."""
        if FILTERPY_OK:
            return self._create_filterpy(team)
        return self._create_simple(team)

    def _create_filterpy(self, team: str):
        """filterpy.KalmanFilter ile 2D state [strength, momentum]."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition: x(k+1) = F * x(k) + noise
        # [strength]   [1  1] [strength]
        # [momentum] = [0  1] [momentum]
        dt = 1.0  # Birim zaman adımı (maç)
        kf.F = np.array([
            [1, dt],
            [0, 1],
        ])

        # Measurement: z = H * x + noise
        kf.H = np.array([[1, 0]])

        # Process noise
        kf.Q = np.array([
            [self.PROCESS_NOISE, 0],
            [0, self.PROCESS_NOISE * 0.5],
        ])

        # Measurement noise
        kf.R = np.array([[self.MEASUREMENT_NOISE]])

        # Başlangıç state
        kf.x = np.array([[self.INITIAL_STRENGTH], [0.0]])

        # Başlangıç uncertainty
        kf.P = np.eye(2) * self.INITIAL_UNCERTAINTY ** 2

        self._filters[team] = kf
        return kf

    def _create_simple(self, team: str):
        """filterpy yoksa basit exponential smoothing."""
        state = {
            "strength": self.INITIAL_STRENGTH,
            "momentum": 0.0,
            "alpha": 0.15,  # Smoothing factor
        }
        self._filters[team] = state
        return state

    # ═══════════════════════════════════════════
    #  GÖZLEM HESAPLAMA
    # ═══════════════════════════════════════════
    def _compute_observation(self, obs: MatchObservation) -> float:
        """Maç sonucundan gürültülü güç gözlemi üret."""
        # 1) Sonuç bazlı (0-100)
        result_score = {"W": 80, "D": 50, "L": 20}.get(obs.result, 50)
        if obs.result == "":
            if obs.goals_for > obs.goals_against:
                result_score = 80
            elif obs.goals_for == obs.goals_against:
                result_score = 50
            else:
                result_score = 20

        # 2) Skor farkı
        goal_diff = obs.goals_for - obs.goals_against
        score_diff_score = 50 + goal_diff * 10  # Her gol farkı ±10
        score_diff_score = max(10, min(90, score_diff_score))

        # 3) xG performansı
        if obs.xg_for > 0 or obs.xg_against > 0:
            xg_diff = obs.xg_for - obs.xg_against
            xg_score = 50 + xg_diff * 15
            xg_score = max(10, min(90, xg_score))
        else:
            xg_score = score_diff_score  # xG yoksa skor farkını kullan

        # 4) Rakip gücüne göre düzeltme
        opponent_factor = obs.opponent_strength / 50.0
        # Güçlü rakibe karşı iyi oynamak daha değerli
        opponent_bonus = (opponent_factor - 1.0) * 10 if result_score > 50 else 0

        # 5) Top kontrolü
        poss_score = obs.possession  # 0-100 zaten

        # Ağırlıklı toplam
        observed = (
            result_score * self.RESULT_WEIGHT +
            score_diff_score * self.SCORE_DIFF_WEIGHT +
            xg_score * self.XG_WEIGHT +
            (50 + opponent_bonus) * self.OPPONENT_WEIGHT +
            poss_score * self.POSSESSION_WEIGHT
        )

        # Ev/deplasman düzeltmesi
        if not obs.is_home:
            observed += self.HOME_ADVANTAGE * 0.5  # Deplasmanda iyi oynamak bonus

        return max(5, min(95, observed))

    # ═══════════════════════════════════════════
    #  GÜNCELLEME
    # ═══════════════════════════════════════════
    def update(self, team: str, obs: MatchObservation) -> TeamState:
        """Maç sonucu ile takımın state'ini güncelle."""
        if team not in self._filters:
            self._create_filter(team)
            self._states[team] = TeamState(team=team)

        observed_strength = self._compute_observation(obs)
        obs.observed_strength = observed_strength

        if FILTERPY_OK and isinstance(self._filters[team], KalmanFilter):
            state = self._update_filterpy(team, observed_strength)
        else:
            state = self._update_simple(team, observed_strength)

        # State güncelle
        state.last_updated = obs.date or datetime.now().isoformat()
        state.n_observations += 1
        state.history.append({
            "date": obs.date,
            "opponent": obs.opponent,
            "result": obs.result,
            "observed": round(observed_strength, 2),
            "filtered_strength": round(state.strength, 2),
            "momentum": round(state.momentum, 3),
            "uncertainty": round(state.uncertainty, 2),
        })

        # Son 50 kayıt tut
        if len(state.history) > 50:
            state.history = state.history[-50:]

        self._states[team] = state

        logger.debug(
            f"[Kalman] {team}: gözlem={observed_strength:.1f} → "
            f"strength={state.strength:.1f} (±{state.uncertainty:.1f}), "
            f"momentum={state.momentum:+.2f}"
        )

        return state

    def _update_filterpy(self, team: str, z: float) -> TeamState:
        """filterpy Kalman predict + update."""
        kf = self._filters[team]
        state = self._states[team]

        # Predict
        kf.predict()

        # Update
        kf.update(np.array([[z]]))

        state.strength = float(kf.x[0, 0])
        state.momentum = float(kf.x[1, 0])
        state.uncertainty = float(np.sqrt(kf.P[0, 0]))

        # Clamp
        state.strength = max(5, min(95, state.strength))

        return state

    def _update_simple(self, team: str, z: float) -> TeamState:
        """filterpy yoksa exponential smoothing."""
        filt = self._filters[team]
        state = self._states[team]
        alpha = filt["alpha"]

        old_strength = filt["strength"]
        new_strength = alpha * z + (1 - alpha) * old_strength
        momentum = new_strength - old_strength

        filt["strength"] = new_strength
        filt["momentum"] = momentum

        state.strength = max(5, min(95, new_strength))
        state.momentum = momentum
        state.uncertainty = abs(z - new_strength) * 0.5

        return state

    # ═══════════════════════════════════════════
    #  DURUM SORGULAMA
    # ═══════════════════════════════════════════
    def get_state(self, team: str) -> TeamState:
        """Takımın güncel Kalman state'i."""
        if team in self._states:
            return self._states[team]
        return TeamState(team=team)

    def get_strength(self, team: str) -> float:
        """Sadece güç değerini döndür."""
        return self.get_state(team).strength

    def get_momentum(self, team: str) -> float:
        """Sadece ivme değerini döndür."""
        return self.get_state(team).momentum

    # ═══════════════════════════════════════════
    #  MAÇ TAHMİNİ
    # ═══════════════════════════════════════════
    def predict_match(self, home: str, away: str) -> dict:
        """İki takımın Kalman state'lerine göre maç tahmini."""
        home_state = self.get_state(home)
        away_state = self.get_state(away)

        # Güç farkı + ev avantajı
        strength_diff = (
            home_state.strength + self.HOME_ADVANTAGE - away_state.strength
        )

        # Momentum katkısı
        momentum_bonus = (home_state.momentum - away_state.momentum) * 2

        total_diff = strength_diff + momentum_bonus

        # Sigmoid → olasılık
        home_prob = 1.0 / (1.0 + np.exp(-total_diff / 15.0))
        draw_prob = 0.25 * np.exp(-(total_diff ** 2) / 400.0)
        away_prob = 1.0 - home_prob - draw_prob

        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        # Güvenilirlik: iki takımın da uncertainty'si düşükse güvenilir
        combined_unc = (home_state.uncertainty + away_state.uncertainty) / 2
        reliability = max(0, 1.0 - combined_unc / 30.0)

        return {
            "prob_home": round(float(home_prob), 4),
            "prob_draw": round(float(draw_prob), 4),
            "prob_away": round(float(away_prob), 4),
            "home_strength": round(home_state.strength, 2),
            "away_strength": round(away_state.strength, 2),
            "strength_diff": round(strength_diff, 2),
            "home_momentum": round(home_state.momentum, 3),
            "away_momentum": round(away_state.momentum, 3),
            "reliability": round(reliability, 3),
            "method": "kalman" if FILTERPY_OK else "exponential_smooth",
        }

    # ═══════════════════════════════════════════
    #  TOPLU GÜNCELLEMEdef
    # ═══════════════════════════════════════════
    def bulk_update(self, results: list[dict]):
        """Geçmiş maç sonuçlarını toplu yükle.

        results: [
            {"home": "GS", "away": "FB", "hg": 2, "ag": 1, "date": "...",
             "home_xg": 1.8, "away_xg": 1.2},
            ...
        ]
        """
        sorted_results = sorted(results, key=lambda r: r.get("date", ""))

        for r in sorted_results:
            home = r.get("home", r.get("home_team", ""))
            away = r.get("away", r.get("away_team", ""))
            hg = r.get("hg", r.get("home_goals", 0))
            ag = r.get("ag", r.get("away_goals", 0))
            date = r.get("date", "")

            if not (home and away):
                continue

            # Ev sahibi gözlemi
            home_obs = MatchObservation(
                team=home, opponent=away, date=date, is_home=True,
                goals_for=hg, goals_against=ag,
                xg_for=r.get("home_xg", 0), xg_against=r.get("away_xg", 0),
                possession=r.get("home_possession", 50),
                opponent_strength=self.get_strength(away),
            )
            self.update(home, home_obs)

            # Deplasman gözlemi
            away_obs = MatchObservation(
                team=away, opponent=home, date=date, is_home=False,
                goals_for=ag, goals_against=hg,
                xg_for=r.get("away_xg", 0), xg_against=r.get("home_xg", 0),
                possession=r.get("away_possession", 50),
                opponent_strength=self.get_strength(home),
            )
            self.update(away, away_obs)

        logger.info(
            f"[Kalman] {len(results)} maç yüklendi, "
            f"{len(self._states)} takım takipte."
        )

    # ═══════════════════════════════════════════
    #  SKOR TAHTASI
    # ═══════════════════════════════════════════
    def power_rankings(self, top_n: int = 20) -> list[dict]:
        """Güncel güç sıralaması."""
        ranked = sorted(
            self._states.values(),
            key=lambda s: s.strength,
            reverse=True,
        )
        return [
            {
                "rank": i + 1,
                "team": s.team,
                "strength": round(s.strength, 2),
                "momentum": round(s.momentum, 3),
                "uncertainty": round(s.uncertainty, 2),
                "trend": (
                    "rising" if s.momentum > 1 else
                    "falling" if s.momentum < -1 else
                    "stable"
                ),
                "n_obs": s.n_observations,
            }
            for i, s in enumerate(ranked[:top_n])
        ]

    @property
    def tracked_teams(self) -> list[str]:
        return list(self._states.keys())
