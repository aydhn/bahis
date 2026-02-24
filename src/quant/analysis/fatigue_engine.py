"""
fatigue_engine.py – Biomechanics & Fatigue Modeling (Yorgunluk Fiziği).

Oyuncular robot değildir. Maçın 70. dakikasından sonra karar
mekanizmaları (Cognitive Decline) bozulur. Bunu fiziksel iş
yükü ile modelliyoruz.

Kavramlar:
  - Metabolic Power: Hızlanma ve yavaşlama verilerinden enerji tüketimi
  - Stamina (Enerji Tankı): 0-100 arası, 0 = tamamen bitkin
  - Lactic Acid: Dakika bazlı laktik asit birikimi
  - Cognitive Decline: Enerji < %30 → hata yapma olasılığı %50 artar
  - Critical Fatigue Zone: Enerji < %20 → savunma çöküşü riski

Formüller:
  Metabolik Güç:
    P = m × v × (dv/dt + g × sin(θ) + f × g)
    Basitleştirilmiş: P ≈ m × (v² + a² × c) / t

  Enerji Tüketimi:
    E(t) = E₀ − ∫₀ᵗ P(τ) dτ + R(t)
    R(t) = recovery_rate × rest_time

  Laktik Asit:
    La(t) = La₀ + k₁ × high_intensity_time − k₂ × low_intensity_time

  Hata Olasılığı:
    P(error) = base_error × (1 + decay_factor × (1 − stamina/100))

Teknoloji: scikit-learn (regresyon), fizik formülleri
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ═══════════════════════════════════════════════
#  FİZİKSEL SABITLER
# ═══════════════════════════════════════════════
PLAYER_MASS_KG = 78.0          # Ortalama futbolcu kitlesi
GRAVITY = 9.81                 # m/s²
GROUND_FRICTION = 0.05         # Sürtünme katsayısı
METABOLIC_EFFICIENCY = 0.25    # Mekanik verimlilik
MAX_STAMINA = 100.0            # Tam enerji
RECOVERY_RATE = 0.3            # dk başına dinlenme geri kazanımı
LACTIC_PRODUCTION_K1 = 0.08    # Yüksek yoğunluk → laktik üretimi
LACTIC_CLEARANCE_K2 = 0.03     # Düşük yoğunluk → laktik temizleme
COGNITIVE_DECLINE_THRESHOLD = 30.0  # Enerji % → bilişsel düşüş başlangıcı
CRITICAL_FATIGUE_THRESHOLD = 20.0   # Enerji % → kritik yorgunluk


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class PlayerFatigue:
    """Bir oyuncunun yorgunluk durumu."""
    player_id: str = ""
    player_name: str = ""
    position: str = ""            # "GK" | "DEF" | "MID" | "FWD"
    # Fiziksel metrikler
    stamina: float = 100.0       # Enerji (0-100)
    lactic_acid: float = 0.0     # mmol/L tahmini
    distance_covered: float = 0.0  # km
    sprint_count: int = 0
    high_intensity_mins: float = 0.0
    # Metabolik
    metabolic_power: float = 0.0   # W/kg
    total_energy_kj: float = 0.0   # kJ toplam harcama
    # Bilişsel
    cognitive_factor: float = 1.0  # 1.0 = normal, 0.5 = ciddi düşüş
    error_probability: float = 0.05  # Hata yapma olasılığı
    # Durum
    is_critical: bool = False      # Kritik yorgunluk
    is_sub_candidate: bool = False  # Oyuncu değişikliği adayı


@dataclass
class FatigueReport:
    """Takım yorgunluk raporu."""
    team: str = ""
    match_id: str = ""
    current_minute: float = 0.0
    # Takım özeti
    avg_stamina: float = 100.0
    min_stamina: float = 100.0
    critical_count: int = 0       # Kritik yorgunluktaki oyuncu sayısı
    weakest_player: str = ""
    weakest_stamina: float = 100.0
    # Savunma analizi
    defense_avg_stamina: float = 100.0
    defense_vulnerability: float = 0.0  # 0-1 arası
    # Hücum analizi
    attack_avg_stamina: float = 100.0
    attack_effectiveness: float = 1.0   # 1.0 = tam performans
    # Takım farkı
    team_fatigue_advantage: float = 0.0  # +: avantaj, -: dezavantaj
    # Sinyaller
    defense_collapse_risk: bool = False
    counter_attack_risk: bool = False    # Hızlı forvet avantajı
    substitution_impact: float = 0.0     # Oyuncu değişikliği etkisi (%)
    # Bahis sinyalleri
    over_signal: bool = False
    late_goal_signal: bool = False
    recommendation: str = ""
    method: str = ""
    # Oyuncu detayları
    players: list[PlayerFatigue] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  METABOLİK GÜÇ HESAPLAMALARI
# ═══════════════════════════════════════════════
def metabolic_power(velocity: float, acceleration: float,
                      mass: float = PLAYER_MASS_KG) -> float:
    """Metabolik güç hesaplama (W/kg).

    P = (v² + (a² × c_drag)) / efficiency
    """
    c_drag = 0.5
    power = (velocity**2 + (acceleration**2 * c_drag)) / METABOLIC_EFFICIENCY
    return max(0, power)


def stamina_decay(initial: float, minute: float,
                    intensity: float = 0.5,
                    has_extra_time: bool = False) -> float:
    """Dakikaya göre enerji düşüşü.

    Yoğunluğa göre farklı azalma oranları:
    - Düşük yoğunluk (< 0.3): 0.4%/dk
    - Orta yoğunluk (0.3-0.7): 0.8%/dk
    - Yüksek yoğunluk (> 0.7): 1.5%/dk

    70. dakikadan sonra +%50 ek düşüş (fizyolojik gerçeklik)
    """
    if intensity < 0.3:
        rate = 0.4
    elif intensity < 0.7:
        rate = 0.8
    else:
        rate = 1.5

    # 70. dakikadan sonra yorgunluk hızlanır
    if minute > 70:
        rate *= 1.5
    elif minute > 80:
        rate *= 2.0

    if has_extra_time and minute > 90:
        rate *= 1.8

    remaining = initial - rate * minute
    return max(0, min(100, remaining))


def lactic_acid_model(high_intensity_mins: float,
                        low_intensity_mins: float,
                        base_lactate: float = 1.0) -> float:
    """Laktik asit birikimi tahmini (mmol/L).

    Normal istirahat: ~1 mmol/L
    Eşik: ~4 mmol/L (anaerobik eşik)
    Kritik: > 8 mmol/L (ciddi performans düşüşü)
    """
    production = LACTIC_PRODUCTION_K1 * high_intensity_mins
    clearance = LACTIC_CLEARANCE_K2 * low_intensity_mins
    lactate = base_lactate + production - clearance
    return max(0.5, lactate)


def cognitive_decline(stamina: float, lactic: float) -> float:
    """Bilişsel performans faktörü.

    1.0 = tam performans
    0.5 = ciddi düşüş (karar hataları)
    """
    if stamina > COGNITIVE_DECLINE_THRESHOLD:
        stamina_factor = 1.0
    else:
        stamina_factor = 0.5 + 0.5 * (stamina / COGNITIVE_DECLINE_THRESHOLD)

    # Laktik asit > 6 mmol/L → bilişsel düşüş
    if lactic > 6:
        lactic_factor = max(0.5, 1 - (lactic - 6) * 0.1)
    else:
        lactic_factor = 1.0

    return round(min(stamina_factor, lactic_factor), 3)


def error_probability(stamina: float, cognitive: float,
                        base_error: float = 0.05) -> float:
    """Hata yapma olasılığı.

    P(error) = base × (1 + decay × (1 - stamina/100))
    """
    decay = 2.0 if stamina < CRITICAL_FATIGUE_THRESHOLD else 1.0
    prob = base_error * (1 + decay * (1 - stamina / 100)) / cognitive
    return round(float(np.clip(prob, 0, 0.5)), 4)


# ═══════════════════════════════════════════════
#  FATIGUE ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class FatigueEngine:
    """Biyomekanik yorgunluk modeli.

    Kullanım:
        fe = FatigueEngine()

        # Oyuncu verileri (vision_tracker veya API'den)
        players = [
            {"id": "p1", "name": "Muslera", "position": "GK",
             "distance_km": 4.5, "sprints": 2, "hi_mins": 5},
            {"id": "p2", "name": "Torreira", "position": "MID",
             "distance_km": 11.2, "sprints": 18, "hi_mins": 22},
            ...
        ]

        report = fe.analyze_team(
            players, current_minute=75,
            team="Galatasaray", match_id="gs_fb",
        )

        if report.defense_collapse_risk:
            increase_over_bet()
    """

    DEFENSE_COLLAPSE_THRESHOLD = 30.0   # Savunma ort. enerji
    COUNTER_SPRINT_THRESHOLD = 25.0     # Hızlı forvet avantajı
    SUB_CANDIDATE_THRESHOLD = 25.0      # Oyuncu değişikliği eşiği

    def __init__(self, base_intensity: float = 0.5):
        self._base_intensity = base_intensity
        logger.debug("[Fatigue] Engine başlatıldı.")

    def compute_player_fatigue(self, player_data: dict,
                                 current_minute: float) -> PlayerFatigue:
        """Tek bir oyuncunun yorgunluk durumunu hesapla."""
        pf = PlayerFatigue(
            player_id=player_data.get("id", ""),
            player_name=player_data.get("name", ""),
            position=player_data.get("position", "MID"),
        )

        distance = player_data.get("distance_km", 0.0)
        sprints = player_data.get("sprints", 0)
        hi_mins = player_data.get("hi_mins", 0.0)

        pf.distance_covered = distance
        pf.sprint_count = sprints
        pf.high_intensity_mins = hi_mins

        # Yoğunluk tahmini: mesafe + sprint bazlı
        if current_minute > 0:
            dist_rate = distance / (current_minute / 90)
            sprint_rate = sprints / max(current_minute, 1)
            intensity = min(1.0, (dist_rate / 12.0 + sprint_rate * 3) / 2)
        else:
            intensity = self._base_intensity

        # Enerji düşüşü
        pf.stamina = stamina_decay(
            MAX_STAMINA, current_minute, intensity,
        )

        # Pozisyona göre enerji düzeltmesi
        position_factor = {
            "GK": 1.3,     # Kaleci daha az yorulur
            "DEF": 1.0,
            "MID": 0.85,   # Orta saha en çok koşar
            "FWD": 0.95,
        }.get(pf.position, 1.0)
        pf.stamina = min(100, pf.stamina * position_factor)

        # Laktik asit
        low_mins = max(0, current_minute - hi_mins)
        pf.lactic_acid = lactic_acid_model(hi_mins, low_mins)

        # Metabolik güç (ortalama)
        avg_velocity = (distance * 1000) / max(current_minute * 60, 1)
        avg_accel = sprints * 0.5 / max(current_minute, 1)
        pf.metabolic_power = round(
            metabolic_power(avg_velocity, avg_accel), 2,
        )
        pf.total_energy_kj = round(
            pf.metabolic_power * PLAYER_MASS_KG * current_minute * 60 / 1000,
            1,
        )

        # Bilişsel
        pf.cognitive_factor = cognitive_decline(pf.stamina, pf.lactic_acid)
        pf.error_probability = error_probability(
            pf.stamina, pf.cognitive_factor,
        )

        # Durum bayrakları
        pf.is_critical = pf.stamina < CRITICAL_FATIGUE_THRESHOLD
        pf.is_sub_candidate = pf.stamina < self.SUB_CANDIDATE_THRESHOLD

        return pf

    def analyze_team(self, players: list[dict],
                       current_minute: float = 0.0,
                       team: str = "",
                       match_id: str = "",
                       opponent_fatigue: list[dict] | None = None) -> FatigueReport:
        """Takım yorgunluk analizi."""
        report = FatigueReport(
            team=team,
            match_id=match_id,
            current_minute=current_minute,
            method="biomechanics_v1",
        )

        if not players:
            report.recommendation = "Oyuncu verisi yok."
            return report

        # Her oyuncu için hesapla
        for p_data in players:
            pf = self.compute_player_fatigue(p_data, current_minute)
            report.players.append(pf)

        # Takım ortalamaları
        all_stamina = [p.stamina for p in report.players]
        report.avg_stamina = round(float(np.mean(all_stamina)), 1)
        report.min_stamina = round(float(np.min(all_stamina)), 1)
        report.critical_count = sum(1 for p in report.players if p.is_critical)

        weakest = min(report.players, key=lambda p: p.stamina)
        report.weakest_player = weakest.player_name
        report.weakest_stamina = weakest.stamina

        # Savunma analizi
        defenders = [p for p in report.players if p.position == "DEF"]
        if defenders:
            report.defense_avg_stamina = round(
                float(np.mean([d.stamina for d in defenders])), 1,
            )
            # Savunma kırılganlığı: 0-1
            vuln = 1 - report.defense_avg_stamina / 100
            # Hata olasılığı ağırlıklı
            avg_error = np.mean([d.error_probability for d in defenders])
            report.defense_vulnerability = round(
                float(np.clip(vuln * 0.5 + avg_error * 5, 0, 1)), 3,
            )

        # Hücum analizi
        attackers = [p for p in report.players if p.position in ("FWD", "MID")]
        if attackers:
            report.attack_avg_stamina = round(
                float(np.mean([a.stamina for a in attackers])), 1,
            )
            report.attack_effectiveness = round(
                float(np.mean([a.cognitive_factor for a in attackers])), 3,
            )

        # Rakip takım farkı
        if opponent_fatigue:
            opp_stamina = []
            for op in opponent_fatigue:
                opp_pf = self.compute_player_fatigue(op, current_minute)
                opp_stamina.append(opp_pf.stamina)
            if opp_stamina:
                report.team_fatigue_advantage = round(
                    report.avg_stamina - float(np.mean(opp_stamina)), 1,
                )

        # Sinyaller
        if report.defense_avg_stamina < self.DEFENSE_COLLAPSE_THRESHOLD:
            report.defense_collapse_risk = True

        # Hızlı forvet avantajı: rakip savunma yorgun, forvet taze
        if (report.defense_vulnerability > 0.5 and
                current_minute > 65):
            report.counter_attack_risk = True

        # Oyuncu değişikliği etkisi
        sub_candidates = [
            p for p in report.players if p.is_sub_candidate
        ]
        if sub_candidates:
            avg_sub_stamina = np.mean([s.stamina for s in sub_candidates])
            report.substitution_impact = round(
                (MAX_STAMINA - avg_sub_stamina) / MAX_STAMINA * 100, 1,
            )

        # Bahis sinyalleri
        if report.defense_collapse_risk:
            report.over_signal = True
        if current_minute > 70 and report.avg_stamina < 40:
            report.late_goal_signal = True

        report.recommendation = self._advice(report)
        return report

    def predict_stamina_at(self, current_stamina: float,
                              current_minute: float,
                              target_minute: float,
                              intensity: float = 0.5) -> float:
        """Gelecekteki enerji tahmini."""
        remaining_mins = target_minute - current_minute
        if remaining_mins <= 0:
            return current_stamina
        decay = stamina_decay(current_stamina, remaining_mins, intensity)
        return round(decay, 1)

    def _advice(self, r: FatigueReport) -> str:
        if r.defense_collapse_risk:
            return (
                f"SAVUNMA ÇÖKÜYOR: Defans ort. enerji={r.defense_avg_stamina:.0f}%, "
                f"kırılganlık={r.defense_vulnerability:.0%}. "
                f"Kritik oyuncu: {r.weakest_player} ({r.weakest_stamina:.0f}%). "
                f"ÜST BAHİS SİNYALİ!"
            )
        if r.late_goal_signal:
            return (
                f"GEÇ GOL RİSKİ: Takım ort. enerji={r.avg_stamina:.0f}% "
                f"(dk {r.current_minute:.0f}). "
                f"{r.critical_count} oyuncu kritik seviyede."
            )
        if r.counter_attack_risk:
            return (
                f"KONTRA ATAK RİSKİ: Savunma yorgun, "
                f"hızlı forvet avantajı var."
            )
        return (
            f"Normal: Takım enerji={r.avg_stamina:.0f}%, "
            f"en düşük={r.weakest_player} ({r.weakest_stamina:.0f}%)."
        )
