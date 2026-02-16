"""
fluid_pitch.py – Pitch Control via Fluid Dynamics (Akışkanlar Dinamiği).

Voronoi diyagramları (Kim kime yakın?) statiktir. Futbol ise
dinamiktir. Oyuncuları sahaya "Baskı Yayan Isı Kaynakları" gibi
modelleyip, fiziksel difüzyon denklemleriyle "Gol İhtimali Akışını"
hesaplıyoruz.

Kavramlar:
  - Potential Field: Her oyuncu bir potansiyel alan yayar
  - Diffusion Equation: ∂u/∂t = D∇²u + S (ısı denklemi)
  - Source: Topa sahip olan oyuncu (pozitif kaynak)
  - Sink: Rakip defans (negatif kaynak / engel)
  - Pressure Map: Sahanın her noktasındaki "Kontrol" değeri
  - Least Resistance Path: Topun en az dirençle gidebileceği yol
  - Expected Threat (xT): Fizik tabanlı gol tehdidi

Saha: 105m × 68m → 210×136 grid (0.5m çözünürlük)

Teknoloji: scipy.ndimage (Gaussian filter = difüzyon yaklaşımı)
Fallback: numpy konvolüsyon
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from scipy.ndimage import gaussian_filter, label
    from scipy.interpolate import RectBivariateSpline
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    logger.debug("scipy.ndimage yüklü değil – numpy fallback.")


# ═══════════════════════════════════════════════
#  SABITLER
# ═══════════════════════════════════════════════
PITCH_LENGTH = 105.0   # metre
PITCH_WIDTH = 68.0
GRID_RES = 0.5         # metre / hücre
GRID_X = int(PITCH_LENGTH / GRID_RES)   # 210
GRID_Y = int(PITCH_WIDTH / GRID_RES)    # 136

# Kale koordinatları (grid indeksi)
GOAL_HOME_X = 0
GOAL_AWAY_X = GRID_X - 1
GOAL_Y_MIN = int((PITCH_WIDTH / 2 - 3.66) / GRID_RES)   # Kale genişliği ~7.32m
GOAL_Y_MAX = int((PITCH_WIDTH / 2 + 3.66) / GRID_RES)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class PlayerState:
    """Oyuncu durumu (konum + hız)."""
    player_id: str = ""
    team: str = "home"       # "home" | "away"
    x: float = 52.5          # metre (0–105)
    y: float = 34.0          # metre (0–68)
    vx: float = 0.0          # hız x (m/s)
    vy: float = 0.0          # hız y (m/s)
    speed: float = 0.0       # toplam hız
    has_ball: bool = False
    rating: float = 70.0     # oyuncu kalitesi


@dataclass
class FluidReport:
    """Akışkanlar dinamiği raporu."""
    # Kontrol haritası
    home_control_pct: float = 0.0    # Ev sahibi saha kontrolü (%)
    away_control_pct: float = 0.0
    # Tehdit metrikleri
    home_xt: float = 0.0             # Expected Threat (home hücum)
    away_xt: float = 0.0
    # Basınç
    home_pressure_index: float = 0.0  # Baskı endeksi
    away_pressure_index: float = 0.0
    # Akış kanalları
    open_channels: int = 0            # Açık pas kanalı sayısı
    dominant_channel: str = ""        # "left" | "center" | "right"
    # Momentum
    flow_direction: str = ""          # "home_attacking" | "away_attacking" | "neutral"
    momentum_score: float = 0.0       # -1 (away) to +1 (home)
    # Savunma
    defensive_compactness: float = 0.0  # 0–1 (1=çok kompakt)
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  FİZİK HESAPLAMALARI
# ═══════════════════════════════════════════════
def player_influence_field(grid_shape: tuple[int, int],
                            player: PlayerState,
                            sigma_base: float = 5.0,
                            intensity: float = 1.0) -> np.ndarray:
    """Tek bir oyuncunun potansiyel alanını hesapla.

    Gaussian kaynak: I(x,y) = A · exp(-((x-px)² + (y-py)²) / (2σ²))
    σ hız vektörü yönünde genişler (anizotropik).
    """
    gx, gy = grid_shape
    px = int(player.x / GRID_RES)
    py = int(player.y / GRID_RES)

    # Hıza göre sigma ayarla (koşan oyuncunun etkisi ileride genişler)
    speed = math.sqrt(player.vx ** 2 + player.vy ** 2)
    sigma = sigma_base + speed * 1.5

    # Rating ile yoğunluk
    amplitude = intensity * (player.rating / 80.0)

    # Grid koordinatları
    yy, xx = np.mgrid[0:gx, 0:gy]

    # Anizotropik Gaussian
    if speed > 0.5:
        # Hız yönünde genişle
        angle = math.atan2(player.vy, player.vx)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = xx - py
        dy = yy - px
        # Döndürülmüş koordinatlar
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        sigma_u = sigma * 1.5  # Hız yönü
        sigma_v = sigma * 0.7  # Çapraz yön
        dist_sq = (u ** 2) / (2 * sigma_u ** 2) + (v ** 2) / (2 * sigma_v ** 2)
    else:
        dist_sq = ((xx - py) ** 2 + (yy - px) ** 2) / (2 * sigma ** 2)

    influence = amplitude * np.exp(-dist_sq)
    return influence


def compute_pressure_map(players: list[PlayerState],
                          diffusion_sigma: float = 3.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Saha basınç haritası hesapla.

    Returns: (home_pressure, away_pressure) – her ikisi de (GRID_X, GRID_Y)
    """
    home_field = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    away_field = np.zeros((GRID_X, GRID_Y), dtype=np.float64)

    for p in players:
        influence = player_influence_field(
            (GRID_X, GRID_Y), p,
            sigma_base=5.0,
            intensity=1.0 if not p.has_ball else 2.0,
        )
        if p.team == "home":
            home_field += influence
        else:
            away_field += influence

    # Difüzyon (Gaussian blur = ısı denkleminin çözümü)
    if SCIPY_OK:
        home_field = gaussian_filter(home_field, sigma=diffusion_sigma)
        away_field = gaussian_filter(away_field, sigma=diffusion_sigma)
    else:
        # Basit box blur (numpy fallback)
        kernel_size = int(diffusion_sigma * 2) + 1
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        from numpy import convolve
        # 2D convolve basit yaklaşım
        for _ in range(2):
            home_field = _simple_blur(home_field, kernel_size)
            away_field = _simple_blur(away_field, kernel_size)

    return home_field, away_field


def _simple_blur(field: np.ndarray, k: int = 5) -> np.ndarray:
    """Basit 2D ortalama blur (numpy fallback)."""
    result = np.zeros_like(field)
    pad = k // 2
    padded = np.pad(field, pad, mode="edge")
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            result[i, j] = padded[i:i + k, j:j + k].mean()
    return result


def compute_control_map(home_pressure: np.ndarray,
                          away_pressure: np.ndarray) -> np.ndarray:
    """Kontrol haritası: home_control ∈ [0, 1].

    Softmax tabanlı: C = P_home / (P_home + P_away + ε)
    """
    eps = 1e-8
    total = home_pressure + away_pressure + eps
    return home_pressure / total


def find_flow_channels(control_map: np.ndarray,
                         threshold: float = 0.7) -> list[dict]:
    """Açık pas kanallarını bul (yüksek kontrol bölgeleri)."""
    channels = []

    # Sahayı 3 dikey bölgeye ayır (sol, orta, sağ)
    third = GRID_Y // 3

    for zone_name, y_start, y_end in [
        ("left", 0, third),
        ("center", third, 2 * third),
        ("right", 2 * third, GRID_Y),
    ]:
        zone = control_map[:, y_start:y_end]
        high_control = (zone > threshold).sum()
        total = zone.size
        openness = high_control / max(total, 1)

        if openness > 0.3:
            channels.append({
                "zone": zone_name,
                "openness": round(float(openness), 4),
                "avg_control": round(float(zone.mean()), 4),
            })

    return sorted(channels, key=lambda c: -c["openness"])


def expected_threat(control_map: np.ndarray,
                      attacking_right: bool = True) -> float:
    """Expected Threat (xT) – fizik tabanlı gol tehdidi.

    Kaleye yakın bölgelerdeki kontrol ağırlıklı toplamı.
    """
    if attacking_right:
        # Son 1/3 saha (rakip yarı alan)
        attack_zone = control_map[2 * GRID_X // 3:, :]
        # Penalty alanı (son 16.5m)
        penalty_zone = control_map[-int(16.5 / GRID_RES):,
                                    GOAL_Y_MIN:GOAL_Y_MAX]
    else:
        attack_zone = control_map[:GRID_X // 3, :]
        penalty_zone = control_map[:int(16.5 / GRID_RES),
                                    GOAL_Y_MIN:GOAL_Y_MAX]

    # xT = ortalama kontrol × alan ağırlığı
    attack_score = float(attack_zone.mean()) * 0.6
    penalty_score = float(penalty_zone.mean()) * 0.4 if penalty_zone.size > 0 else 0

    return round(attack_score + penalty_score, 4)


# ═══════════════════════════════════════════════
#  FLUID PITCH ANALYZER (Ana Sınıf)
# ═══════════════════════════════════════════════
class FluidPitchAnalyzer:
    """Akışkanlar dinamiği ile saha kontrolü analizi.

    Kullanım:
        fpa = FluidPitchAnalyzer()

        players = [
            PlayerState("gk1", "home", 5, 34, 0, 0, 0, False, 80),
            PlayerState("cb1", "home", 25, 20, 1, 0, 1, False, 78),
            # ... 22 oyuncu
        ]

        report = fpa.analyze(players)
        print(report.home_control_pct, report.home_xt)
    """

    def __init__(self, diffusion_sigma: float = 3.0,
                 control_threshold: float = 0.6):
        self._sigma = diffusion_sigma
        self._threshold = control_threshold
        logger.debug("[Fluid] PitchAnalyzer başlatıldı.")

    def analyze(self, players: list[PlayerState]) -> FluidReport:
        """22 oyuncunun konumundan saha analizi yap."""
        report = FluidReport()

        if not players:
            report.recommendation = "Oyuncu verisi yok."
            return report

        # Basınç haritaları
        home_p, away_p = compute_pressure_map(players, self._sigma)

        # Kontrol haritası
        control = compute_control_map(home_p, away_p)

        # Kontrol yüzdeleri
        report.home_control_pct = round(
            float((control > 0.5).sum()) / control.size * 100, 1,
        )
        report.away_control_pct = round(100.0 - report.home_control_pct, 1)

        # Expected Threat
        report.home_xt = expected_threat(control, attacking_right=True)
        report.away_xt = expected_threat(1.0 - control, attacking_right=False)

        # Baskı endeksi (rakip yarı alandaki toplam basınç)
        half = GRID_X // 2
        report.home_pressure_index = round(
            float(home_p[half:, :].sum()) / max(float(home_p.sum()), 1e-8), 4,
        )
        report.away_pressure_index = round(
            float(away_p[:half, :].sum()) / max(float(away_p.sum()), 1e-8), 4,
        )

        # Akış kanalları
        channels = find_flow_channels(control, self._threshold)
        report.open_channels = len(channels)
        if channels:
            report.dominant_channel = channels[0]["zone"]

        # Momentum
        home_attack = float(home_p[half:, :].mean())
        away_attack = float(away_p[:half, :].mean())
        total_attack = home_attack + away_attack + 1e-8
        report.momentum_score = round(
            (home_attack - away_attack) / total_attack, 4,
        )
        if report.momentum_score > 0.2:
            report.flow_direction = "home_attacking"
        elif report.momentum_score < -0.2:
            report.flow_direction = "away_attacking"
        else:
            report.flow_direction = "neutral"

        # Savunma kompaktlığı
        home_defenders = [
            p for p in players
            if p.team == "home" and p.x < PITCH_LENGTH * 0.4
        ]
        if len(home_defenders) >= 3:
            positions = np.array([[p.x, p.y] for p in home_defenders])
            spread = float(np.std(positions, axis=0).mean())
            report.defensive_compactness = round(
                1.0 / (1.0 + spread / 10.0), 4,
            )

        report.recommendation = self._advice(report)
        return report

    def analyze_from_coordinates(self, home_coords: list[tuple[float, float]],
                                   away_coords: list[tuple[float, float]],
                                   ball_owner: str = "home",
                                   ball_idx: int = 0) -> FluidReport:
        """Sadece koordinatlardan analiz (basit API)."""
        players = []
        for i, (x, y) in enumerate(home_coords):
            players.append(PlayerState(
                player_id=f"h{i}", team="home",
                x=x, y=y,
                has_ball=(ball_owner == "home" and i == ball_idx),
                rating=70,
            ))
        for i, (x, y) in enumerate(away_coords):
            players.append(PlayerState(
                player_id=f"a{i}", team="away",
                x=x, y=y,
                has_ball=(ball_owner == "away" and i == ball_idx),
                rating=70,
            ))
        return self.analyze(players)

    def _advice(self, r: FluidReport) -> str:
        if r.home_control_pct > 60:
            return (
                f"Ev sahibi dominant: %{r.home_control_pct:.0f} saha kontrolü, "
                f"xT={r.home_xt:.3f}. "
                f"Kanal: {r.dominant_channel or '?'}, momentum={r.momentum_score:+.2f}."
            )
        if r.away_control_pct > 60:
            return (
                f"Deplasman dominant: %{r.away_control_pct:.0f} kontrolde, "
                f"xT={r.away_xt:.3f}."
            )
        return (
            f"Dengeli maç: ev={r.home_control_pct:.0f}%, "
            f"dep={r.away_control_pct:.0f}%. "
            f"Momentum={r.momentum_score:+.2f}."
        )
