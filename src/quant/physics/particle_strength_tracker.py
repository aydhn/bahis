"""
particle_strength_tracker.py – Particle Filter ile Dinamik Güç Takibi.

Kalman filtresinin üst seviyesi. Maç içinde takımların anlık gücü
(Hidden State) bir parçacık bulutu gibi simüle edilir. Non-linear
ve non-Gaussian dağılımları destekler.

Kavramlar:
  - Particle Filter (Sequential Monte Carlo – SMC): Bayesian filtering
    yöntemi. Gizli durumu binlerce "parçacık" (olasılık) ile temsil eder
  - State Space: [home_power, away_power, momentum, fatigue_h, fatigue_a]
  - Observation: Şutlar, pas yüzdeleri, top kontrolü, tehlikeli ataklar
  - Transition Model: Güç zamanla değişir (momentum + yorgunluk)
  - Observation Model: Gözlemler gürültülü — gerçek gücü tam yansıtmaz
  - Resampling: Düşük ağırlıklı parçacıklar elenir (Systematic Resampling)
  - Effective Sample Size (ESS): Parçacıkların çeşitliliği ölçüsü

Akış:
  1. Başlangıçta N parçacık (ör: 1000) rastgele dağıtılır
  2. Her dakikada gözlem gelir → parçacıklar güncellenir (update)
  3. Parçacıklar propagate edilir (transition) → güç değişimi simüle
  4. Ağırlıklar hesaplanır → gözleme uyum (likelihood)
  5. Resampling → düşük ağırlıklı parçacıklar elenir
  6. Weighted mean → anlık güç tahmini

Teknoloji: filterpy + numpy
Fallback: Pure numpy implementasyonu
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from filterpy.monte_carlo import systematic_resample
    FILTERPY_OK = True
except ImportError:
    FILTERPY_OK = False
    logger.debug("filterpy.monte_carlo yüklü değil – numpy resampling fallback.")


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class MatchObservation:
    """Maç gözlemi (bir dakikalık veri)."""
    minute: int = 0
    home_shots: int = 0
    away_shots: int = 0
    home_possession: float = 50.0
    away_possession: float = 50.0
    home_dangerous_attacks: int = 0
    away_dangerous_attacks: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_fouls: int = 0
    away_fouls: int = 0
    score_home: int = 0
    score_away: int = 0


@dataclass
class PowerState:
    """Anlık güç durumu."""
    home_power: float = 0.5       # [0, 1] ev sahibi gücü
    away_power: float = 0.5       # [0, 1] deplasman gücü
    momentum: float = 0.0         # [-1, 1] pozitif=ev sahibi
    fatigue_home: float = 0.0     # [0, 1] 0=taze, 1=bitkin
    fatigue_away: float = 0.0
    power_diff: float = 0.0       # home - away
    confidence: float = 0.0       # ESS / N_particles


@dataclass
class MomentumShift:
    """Momentum kayması tespiti."""
    detected: bool = False
    minute: int = 0
    direction: str = ""           # "home_surge" | "away_surge" | "equalized"
    magnitude: float = 0.0        # Kayma büyüklüğü
    previous_power_diff: float = 0.0
    current_power_diff: float = 0.0


@dataclass
class ParticleReport:
    """Parçacık filtresi analiz raporu."""
    match_id: str = ""
    minute: int = 0
    state: PowerState = field(default_factory=PowerState)
    # Parçacık istatistikleri
    n_particles: int = 0
    ess: float = 0.0              # Effective Sample Size
    ess_ratio: float = 0.0        # ESS / N
    resampled: bool = False
    # Momentum
    momentum_shift: MomentumShift = field(default_factory=MomentumShift)
    # Tahmin
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    # Tarihçe
    power_history: list[tuple[int, float, float]] = field(default_factory=list)
    method: str = ""


# ═══════════════════════════════════════════════
#  SYSTEMATIC RESAMPLING (Fallback)
# ═══════════════════════════════════════════════
def _systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Sistematik yeniden örnekleme (filterpy yoksa)."""
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform()) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return indices.astype(int)


# ═══════════════════════════════════════════════
#  PARTICLE STRENGTH TRACKER (Ana Sınıf)
# ═══════════════════════════════════════════════
class ParticleStrengthTracker:
    """Parçacık Filtresi ile Dinamik Güç Takibi.

    Kullanım:
        pst = ParticleStrengthTracker(n_particles=1000)

        # Maç başlangıcı
        pst.initialize(
            home_prior=0.55,  # Ev sahibi başlangıç gücü (modellerden)
            away_prior=0.45,
        )

        # Her dakikada gözlem gelir
        obs = MatchObservation(
            minute=15,
            home_shots=3, away_shots=1,
            home_possession=62.0, away_possession=38.0,
        )
        report = pst.update(obs, match_id="gs_fb")

        # Momentum kayması?
        if report.momentum_shift.detected:
            print(f"Momentum: {report.momentum_shift.direction}")

        # Güç durumu
        print(f"Home: {report.state.home_power:.2f}")
        print(f"Away: {report.state.away_power:.2f}")
    """

    # Durum vektörü: [home_power, away_power, momentum, fatigue_h, fatigue_a]
    STATE_DIM = 5
    # Gözlem: [shots_ratio, poss_ratio, attack_ratio, corner_ratio]
    OBS_DIM = 4

    def __init__(self, n_particles: int = 1000,
                 process_noise: float = 0.02,
                 observation_noise: float = 0.15,
                 fatigue_rate: float = 0.005,
                 momentum_decay: float = 0.95,
                 ess_threshold: float = 0.5,
                 momentum_shift_threshold: float = 0.15):
        self._n = n_particles
        self._proc_noise = process_noise
        self._obs_noise = observation_noise
        self._fatigue_rate = fatigue_rate
        self._mom_decay = momentum_decay
        self._ess_thresh = ess_threshold
        self._shift_thresh = momentum_shift_threshold

        # Parçacıklar: (N, STATE_DIM)
        self._particles: np.ndarray | None = None
        self._weights: np.ndarray | None = None

        # Tarihçe
        self._history: list[tuple[int, float, float]] = []
        self._prev_power_diff: float = 0.0
        self._initialized = False

        logger.debug(
            f"[Particle] Tracker başlatıldı: "
            f"N={n_particles}, σ_proc={process_noise}, σ_obs={observation_noise}"
        )

    def initialize(self, home_prior: float = 0.5,
                     away_prior: float = 0.5,
                     home_std: float = 0.1,
                     away_std: float = 0.1) -> None:
        """Parçacıkları başlangıç dağılımıyla başlat."""
        self._particles = np.zeros((self._n, self.STATE_DIM))

        # home_power ~ N(home_prior, home_std)
        self._particles[:, 0] = np.clip(
            np.random.normal(home_prior, home_std, self._n), 0.05, 0.95,
        )
        # away_power ~ N(away_prior, away_std)
        self._particles[:, 1] = np.clip(
            np.random.normal(away_prior, away_std, self._n), 0.05, 0.95,
        )
        # momentum ~ N(0, 0.05)
        self._particles[:, 2] = np.random.normal(0, 0.05, self._n)
        # fatigue_h = 0, fatigue_a = 0
        self._particles[:, 3] = 0.0
        self._particles[:, 4] = 0.0

        self._weights = np.ones(self._n) / self._n
        self._history.clear()
        self._prev_power_diff = home_prior - away_prior
        self._initialized = True

        logger.debug(
            f"[Particle] Başlatıldı: home={home_prior:.2f}±{home_std:.2f}, "
            f"away={away_prior:.2f}±{away_std:.2f}"
        )

    def update(self, obs: MatchObservation,
                 match_id: str = "") -> ParticleReport:
        """Yeni gözlem ile parçacıkları güncelle."""
        report = ParticleReport(
            match_id=match_id,
            minute=obs.minute,
            n_particles=self._n,
        )

        if not self._initialized or self._particles is None or self._weights is None:
            self.initialize()
            assert self._particles is not None and self._weights is not None

        # 1) Transition (propagate)
        self._propagate(obs.minute)

        # 2) Observation → ağırlık güncelleme
        z = self._observation_vector(obs)
        self._update_weights(z)

        # 3) ESS hesapla
        ess = self._compute_ess()
        report.ess = round(float(ess), 1)
        report.ess_ratio = round(float(ess / self._n), 3)

        # 4) Resampling (gerekirse)
        if ess < self._ess_thresh * self._n:
            self._resample()
            report.resampled = True

        # 5) Durum tahmini (weighted mean)
        state = self._estimate_state()
        report.state = state
        report.method = "particle_filter"

        # 6) Momentum kayması tespiti
        shift = self._detect_momentum_shift(state, obs.minute)
        report.momentum_shift = shift

        # 7) Kazanma olasılıkları
        probs = self._win_probabilities()
        report.home_win_prob = round(probs[0], 4)
        report.draw_prob = round(probs[1], 4)
        report.away_win_prob = round(probs[2], 4)

        # Tarihçe
        self._history.append((obs.minute, state.home_power, state.away_power))
        report.power_history = list(self._history)
        self._prev_power_diff = state.power_diff

        return report

    # ─────────────────────────────────────────────
    #  TRANSITION MODEL
    # ─────────────────────────────────────────────
    def _propagate(self, minute: int) -> None:
        """Parçacıkları ileri taşı (transition)."""
        assert self._particles is not None
        n = self._n

        # Yorgunluk artışı (dakika ilerledikçe)
        fatigue_inc = self._fatigue_rate * (1 + minute / 90.0)
        self._particles[:, 3] = np.clip(
            self._particles[:, 3] + fatigue_inc
            + np.random.normal(0, 0.002, n),
            0, 1,
        )
        self._particles[:, 4] = np.clip(
            self._particles[:, 4] + fatigue_inc
            + np.random.normal(0, 0.002, n),
            0, 1,
        )

        # Momentum decay
        self._particles[:, 2] *= self._mom_decay

        # Güç = güç - yorgunluk etkisi + gürültü
        noise = np.random.normal(0, self._proc_noise, (n, 2))
        self._particles[:, 0] = np.clip(
            self._particles[:, 0]
            - 0.1 * self._particles[:, 3]
            + 0.05 * self._particles[:, 2]  # Momentum etkisi
            + noise[:, 0],
            0.05, 0.95,
        )
        self._particles[:, 1] = np.clip(
            self._particles[:, 1]
            - 0.1 * self._particles[:, 4]
            - 0.05 * self._particles[:, 2]  # Ters momentum
            + noise[:, 1],
            0.05, 0.95,
        )

    # ─────────────────────────────────────────────
    #  OBSERVATION MODEL
    # ─────────────────────────────────────────────
    def _observation_vector(self, obs: MatchObservation) -> np.ndarray:
        """Gözlemi normalleştirilmiş vektöre çevir."""
        total_shots = max(obs.home_shots + obs.away_shots, 1)
        total_attacks = max(
            obs.home_dangerous_attacks + obs.away_dangerous_attacks, 1,
        )
        total_corners = max(obs.home_corners + obs.away_corners, 1)

        z = np.array([
            obs.home_shots / total_shots,           # Şut oranı
            obs.home_possession / 100.0,            # Top kontrolü
            obs.home_dangerous_attacks / total_attacks,  # Atak oranı
            obs.home_corners / total_corners,        # Korner oranı
        ])
        return z

    def _update_weights(self, z: np.ndarray) -> None:
        """Gözlem likelihood ile ağırlıkları güncelle."""
        assert self._particles is not None and self._weights is not None

        # Her parçacığın "beklenen gözlemi"
        home_power = self._particles[:, 0]
        away_power = self._particles[:, 1]

        # Beklenen gözlem: güç oranlarından türetilir
        power_ratio = home_power / np.clip(home_power + away_power, 0.01, 2.0)

        # Optimize edilmiş likelihood hesaplaması (vektörize edilmiş)
        # Orijinal: expected = [p, 0.3+0.4p, p, 0.4+0.2p]
        # A = [1.0, 0.4, 1.0, 0.2]
        # B = [0.0, 0.3, 0.0, 0.4]
        # diff^2 = (z - (p*A + B))^2 = ((z - B) - p*A)^2
        #        = (z-B)^2 - 2*p*A*(z-B) + p^2*A^2

        A = np.array([1.0, 0.4, 1.0, 0.2])
        B = np.array([0.0, 0.3, 0.0, 0.4])

        z_prime = z - B

        C0 = np.sum(z_prime**2)
        C1 = 2 * np.sum(z_prime * A)
        C2 = np.sum(A**2)  # Scalar: 1 + 0.16 + 1 + 0.04 = 2.2

        # sum_sq_diff = C0 - C1 * p + C2 * p^2
        sum_sq_diff = C0 - C1 * power_ratio + C2 * (power_ratio**2)

        log_likelihood = -0.5 * sum_sq_diff / (self._obs_noise ** 2)

        # Log-weight update (sayısal kararlılık)
        log_weights = np.log(self._weights + 1e-300) + log_likelihood
        log_weights -= np.max(log_weights)  # Overflow önleme
        self._weights = np.exp(log_weights)
        self._weights /= np.sum(self._weights) + 1e-300

    # ─────────────────────────────────────────────
    #  RESAMPLING
    # ─────────────────────────────────────────────
    def _compute_ess(self) -> float:
        """Effective Sample Size."""
        assert self._weights is not None
        return 1.0 / np.sum(self._weights ** 2)

    def _resample(self) -> None:
        """Sistematik yeniden örnekleme."""
        assert self._particles is not None and self._weights is not None

        if FILTERPY_OK:
            indices = systematic_resample(self._weights)
        else:
            indices = _systematic_resample(self._weights)

        self._particles = self._particles[indices]
        self._weights = np.ones(self._n) / self._n

    # ─────────────────────────────────────────────
    #  DURUM TAHMİNİ
    # ─────────────────────────────────────────────
    def _estimate_state(self) -> PowerState:
        """Ağırlıklı ortalama ile durum tahmini."""
        assert self._particles is not None and self._weights is not None
        w = self._weights

        hp = float(np.average(self._particles[:, 0], weights=w))
        ap = float(np.average(self._particles[:, 1], weights=w))
        mom = float(np.average(self._particles[:, 2], weights=w))
        fh = float(np.average(self._particles[:, 3], weights=w))
        fa = float(np.average(self._particles[:, 4], weights=w))

        ess = self._compute_ess()

        return PowerState(
            home_power=round(hp, 4),
            away_power=round(ap, 4),
            momentum=round(mom, 4),
            fatigue_home=round(fh, 4),
            fatigue_away=round(fa, 4),
            power_diff=round(hp - ap, 4),
            confidence=round(float(ess / self._n), 4),
        )

    def _win_probabilities(self) -> tuple[float, float, float]:
        """Parçacıklardan kazanma olasılıkları."""
        assert self._particles is not None and self._weights is not None
        w = self._weights
        hp = self._particles[:, 0]
        ap = self._particles[:, 1]

        diff = hp - ap
        home_wins = float(np.sum(w[diff > 0.05]))
        away_wins = float(np.sum(w[diff < -0.05]))
        draws = 1.0 - home_wins - away_wins

        return (
            max(home_wins, 0.01),
            max(draws, 0.01),
            max(away_wins, 0.01),
        )

    # ─────────────────────────────────────────────
    #  MOMENTUM KAYMASI TESPİTİ
    # ─────────────────────────────────────────────
    def _detect_momentum_shift(self, state: PowerState,
                                  minute: int) -> MomentumShift:
        """Momentum kayması tespiti."""
        shift = MomentumShift(minute=minute)
        delta = state.power_diff - self._prev_power_diff

        if abs(delta) >= self._shift_thresh:
            shift.detected = True
            shift.magnitude = round(abs(delta), 4)
            shift.previous_power_diff = round(self._prev_power_diff, 4)
            shift.current_power_diff = round(state.power_diff, 4)

            if delta > 0:
                shift.direction = "home_surge"
            else:
                shift.direction = "away_surge"

            logger.info(
                f"[Particle] MOMENTUM SHIFT dk.{minute}: "
                f"{shift.direction} (Δ={delta:+.3f}, "
                f"güç: {state.home_power:.2f} vs {state.away_power:.2f})"
            )

        return shift

    # ─────────────────────────────────────────────
    #  TOPLU ANALİZ
    # ─────────────────────────────────────────────
    def simulate_match(self, observations: list[MatchObservation],
                         match_id: str = "",
                         home_prior: float = 0.5,
                         away_prior: float = 0.5) -> list[ParticleReport]:
        """Tam maç simülasyonu (gözlem dizisi)."""
        self.initialize(home_prior, away_prior)
        reports = []
        for obs in observations:
            report = self.update(obs, match_id)
            reports.append(report)
        return reports

    def get_history(self) -> list[tuple[int, float, float]]:
        """Güç tarihçesi: [(minute, home_power, away_power), ...]"""
        return list(self._history)

    def reset(self) -> None:
        """Durumu sıfırla."""
        self._particles = None
        self._weights = None
        self._history.clear()
        self._prev_power_diff = 0.0
        self._initialized = False
