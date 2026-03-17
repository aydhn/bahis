"""
fuzzy_reasoning.py – Fuzzy Logic (Bulanık Mantık).

Bilgisayarlar 0-1 ile çalışır. Futbol ise "Belki", "Biraz",
"Galiba" oyunudur. Bulanık mantık, botun insani kavramları
matematiğe dökmesini sağlar.

Kavramlar:
  - Fuzzy Set: Bir elemanın bir kümeye ait olma derecesi (0-1)
  - Membership Function: Üyelik fonksiyonu (Triangular, Trapezoidal)
  - Fuzzy Rule: IF hava=yağmurlu AND yorgunluk=yüksek THEN risk=çok_yüksek
  - Defuzzification: Bulanık sonucu keskin sayıya çevir (Centroid)
  - Linguistic Variable: "düşük", "orta", "yüksek" gibi sözel değerler

Değişkenler:
  - Hava Durumu: güneşli / bulutlu / yağmurlu / karlı
  - Yorgunluk: düşük / orta / yüksek / kritik
  - Deplasman Mesafesi: yakın / orta / uzak
  - Sakat Oyuncu Sayısı: az / orta / çok
  - Motivasyon: düşük / normal / yüksek

Çıktı:
  - Risk Skoru: 0-100 (düşük / orta / yüksek / çok yüksek)
  - Confidence Modifier: 0.5 - 1.5 (model güvenini çarpan)

Teknoloji: scikit-fuzzy
Fallback: Manuel üçgen üyelik fonksiyonları + centroid
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from dataclasses import field

import numpy as np
from loguru import logger

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_OK = True
except ImportError:
    FUZZY_OK = False
    logger.debug("scikit-fuzzy yüklü değil – manuel fuzzy fallback.")


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class FuzzyInput:
    """Bulanık mantık girdileri."""
    weather: float = 0.5         # 0=güneşli, 0.5=bulutlu, 1.0=yağmurlu
    fatigue: float = 0.3         # 0=dinlenmiş, 1=bitkin
    travel_distance: float = 0.2  # 0=ev, 0.5=yakın, 1=uzak deplasman
    injury_count: float = 0.0    # 0=sağlam, 1=5+ sakat
    motivation: float = 0.7      # 0=düşük, 1=çok yüksek
    form: float = 0.6            # 0=düşük, 1=mükemmel form
    crowd_factor: float = 0.5    # 0=deplasman, 1=tam ev avantajı


@dataclass
class FuzzyOutput:
    """Bulanık mantık çıktıları."""
    risk_score: float = 50.0          # 0-100
    risk_level: str = "orta"          # "düşük" | "orta" | "yüksek" | "çok_yüksek"
    confidence_modifier: float = 1.0  # 0.5-1.5
    goal_expectation_mod: float = 1.0  # Gol beklentisi çarpanı
    # Açıklama
    active_rules: list[str] = field(default_factory=list)
    recommendation: str = ""
    method: str = ""


# ═══════════════════════════════════════════════
#  MANUEL ÜÇGEN ÜYELİK FONKSİYONLARI
# ═══════════════════════════════════════════════
def trimf(x: float, a: float, b: float, c: float) -> float:
    """Üçgen üyelik fonksiyonu.

    a=sol taban, b=tepe, c=sağ taban.
    """
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / max(b - a, 1e-10)
    return (c - x) / max(c - b, 1e-10)


def trapmf(x: float, a: float, b: float,
             c: float, d: float) -> float:
    """Trapez üyelik fonksiyonu."""
    if x <= a or x >= d:
        return 0.0
    if a < x <= b:
        return (x - a) / max(b - a, 1e-10)
    if b < x <= c:
        return 1.0
    return (d - x) / max(d - c, 1e-10)


# ═══════════════════════════════════════════════
#  BULANIK KURAL TABANI
# ═══════════════════════════════════════════════
FUZZY_RULES = [
    # (koşullar, sonuç_risk, açıklama)
    ({"weather": "yağmurlu", "fatigue": "yüksek"},
     "çok_yüksek", "Yağmur + yorgunluk = savunma hataları"),
    ({"weather": "yağmurlu", "travel_distance": "uzak"},
     "yüksek", "Yağmurlu uzak deplasman = düşük performans"),
    ({"fatigue": "kritik", "injury_count": "çok"},
     "çok_yüksek", "Bitkin + sakat kadro = çöküş riski"),
    ({"motivation": "yüksek", "form": "yüksek"},
     "düşük", "Motive + formda = güvenilir"),
    ({"motivation": "düşük", "fatigue": "yüksek"},
     "yüksek", "Motivasyonsuz + yorgun = sürpriz"),
    ({"crowd_factor": "yüksek", "form": "yüksek"},
     "düşük", "Ev avantajı + form = güçlü"),
    ({"travel_distance": "yakın", "weather": "güneşli"},
     "düşük", "İdeal koşullar"),
    ({"injury_count": "çok", "motivation": "düşük"},
     "çok_yüksek", "Sakat kadro + düşük moral"),
]


def evaluate_membership(value: float, var_name: str) -> dict[str, float]:
    """Bir değişkenin bulanık üyelik derecelerini hesapla."""
    memberships: dict[str, float] = {}

    if var_name == "weather":
        memberships["güneşli"] = trimf(value, -0.1, 0.0, 0.3)
        memberships["bulutlu"] = trimf(value, 0.2, 0.5, 0.7)
        memberships["yağmurlu"] = trimf(value, 0.6, 0.85, 1.0)
        memberships["karlı"] = trapmf(value, 0.85, 0.95, 1.0, 1.1)
    elif var_name in ("fatigue", "injury_count"):
        memberships["düşük"] = trimf(value, -0.1, 0.0, 0.3)
        memberships["orta"] = trimf(value, 0.2, 0.4, 0.6)
        memberships["yüksek"] = trimf(value, 0.5, 0.7, 0.85)
        memberships["kritik"] = trapmf(value, 0.8, 0.9, 1.0, 1.1)
    elif var_name == "travel_distance":
        memberships["yakın"] = trimf(value, -0.1, 0.0, 0.35)
        memberships["orta"] = trimf(value, 0.25, 0.5, 0.75)
        memberships["uzak"] = trapmf(value, 0.65, 0.8, 1.0, 1.1)
    elif var_name in ("motivation", "form", "crowd_factor"):
        memberships["düşük"] = trimf(value, -0.1, 0.0, 0.35)
        memberships["normal"] = trimf(value, 0.25, 0.5, 0.75)
        memberships["yüksek"] = trapmf(value, 0.65, 0.85, 1.0, 1.1)
    else:
        memberships["düşük"] = trimf(value, -0.1, 0.0, 0.4)
        memberships["orta"] = trimf(value, 0.3, 0.5, 0.7)
        memberships["yüksek"] = trapmf(value, 0.6, 0.8, 1.0, 1.1)

    return memberships


# ═══════════════════════════════════════════════
#  FUZZY REASONING ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class FuzzyReasoningEngine:
    """Bulanık mantık tabanlı risk değerlendirme.

    Kullanım:
        fre = FuzzyReasoningEngine()

        inputs = FuzzyInput(
            weather=0.8,           # Yağmurlu
            fatigue=0.7,           # Yorgun
            travel_distance=0.6,   # Orta mesafe
            injury_count=0.4,      # 2 sakat
            motivation=0.5,        # Normal
            form=0.6,              # İyi form
            crowd_factor=0.8,      # Ev avantajı
        )

        output = fre.evaluate(inputs, team="Galatasaray")
    """

    RISK_MAP = {
        "düşük": 20.0,
        "orta": 50.0,
        "yüksek": 75.0,
        "çok_yüksek": 95.0,
    }

    CONFIDENCE_MAP = {
        "düşük": 1.2,      # Düşük risk → güveni artır
        "orta": 1.0,
        "yüksek": 0.7,     # Yüksek risk → güveni düşür
        "çok_yüksek": 0.5,
    }

    def __init__(self):
        self._sim: Any = None
        if FUZZY_OK:
            self._build_fuzzy_system()
        logger.debug("[Fuzzy] Reasoning Engine başlatıldı.")

    def _build_fuzzy_system(self) -> None:
        """scikit-fuzzy kontrol sistemi kur."""
        try:
            weather = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "weather")
            fatigue_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "fatigue")
            risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

            weather.automf(3, names=["güneşli", "bulutlu", "yağmurlu"])
            fatigue_var.automf(3, names=["düşük", "orta", "yüksek"])
            risk.automf(4, names=["düşük", "orta", "yüksek", "çok_yüksek"])

            rules = [
                ctrl.Rule(weather["yağmurlu"] & fatigue_var["yüksek"], risk["çok_yüksek"]),
                ctrl.Rule(weather["güneşli"] & fatigue_var["düşük"], risk["düşük"]),
                ctrl.Rule(weather["bulutlu"] & fatigue_var["orta"], risk["orta"]),
                ctrl.Rule(weather["yağmurlu"] & fatigue_var["düşük"], risk["orta"]),
                ctrl.Rule(weather["güneşli"] & fatigue_var["yüksek"], risk["yüksek"]),
            ]

            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
        except Exception as e:
            logger.debug(f"[Fuzzy] Sistem kurulum hatası: {e}")
            self._sim = None

    def evaluate(self, inputs: FuzzyInput,
                   team: str = "",
                   match_id: str = "") -> FuzzyOutput:
        """Bulanık mantık değerlendirmesi."""
        output = FuzzyOutput()

        # scikit-fuzzy denemesi
        if FUZZY_OK and self._sim:
            try:
                output = self._evaluate_skfuzzy(inputs)
                output.method = "scikit-fuzzy"
                return output
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        # Manuel fallback
        output = self._evaluate_manual(inputs)
        output.method = "manual_fuzzy"
        return output

    def _evaluate_skfuzzy(self, inputs: FuzzyInput) -> FuzzyOutput:
        """scikit-fuzzy ile değerlendirme."""
        output = FuzzyOutput()

        self._sim.input["weather"] = inputs.weather
        self._sim.input["fatigue"] = inputs.fatigue
        self._sim.compute()

        output.risk_score = round(float(self._sim.output["risk"]), 1)

        if output.risk_score < 30:
            output.risk_level = "düşük"
        elif output.risk_score < 55:
            output.risk_level = "orta"
        elif output.risk_score < 80:
            output.risk_level = "yüksek"
        else:
            output.risk_level = "çok_yüksek"

        output.confidence_modifier = self.CONFIDENCE_MAP.get(
            output.risk_level, 1.0,
        )
        output.goal_expectation_mod = self._goal_mod(inputs)
        output.recommendation = self._advice(output, inputs)
        return output

    def _evaluate_manual(self, inputs: FuzzyInput) -> FuzzyOutput:
        """Manuel bulanık mantık değerlendirmesi."""
        output = FuzzyOutput()

        # Her değişkenin üyelik derecelerini hesapla
        memberships = {
            "weather": evaluate_membership(inputs.weather, "weather"),
            "fatigue": evaluate_membership(inputs.fatigue, "fatigue"),
            "travel_distance": evaluate_membership(inputs.travel_distance, "travel_distance"),
            "injury_count": evaluate_membership(inputs.injury_count, "injury_count"),
            "motivation": evaluate_membership(inputs.motivation, "motivation"),
            "form": evaluate_membership(inputs.form, "form"),
            "crowd_factor": evaluate_membership(inputs.crowd_factor, "crowd_factor"),
        }

        # Kuralları değerlendir (Mamdani inference)
        risk_activations: dict[str, float] = {
            "düşük": 0.0, "orta": 0.0,
            "yüksek": 0.0, "çok_yüksek": 0.0,
        }

        for conditions, risk_level, description in FUZZY_RULES:
            # AND (minimum) operatörü
            activation = 1.0
            for var_name, fuzzy_term in conditions.items():
                var_memberships = memberships.get(var_name, {})
                degree = var_memberships.get(fuzzy_term, 0.0)
                activation = min(activation, degree)

            if activation > 0.01:
                current = risk_activations.get(risk_level, 0.0)
                risk_activations[risk_level] = max(current, activation)
                output.active_rules.append(
                    f"{description} (ateşlenme={activation:.2f})"
                )

        # Defuzzification (ağırlıklı ortalama / centroid yaklaşımı)
        total_weight = 0.0
        total_risk = 0.0
        for level, activation in risk_activations.items():
            if activation > 0:
                center = self.RISK_MAP[level]
                total_risk += center * activation
                total_weight += activation

        if total_weight > 0:
            output.risk_score = round(total_risk / total_weight, 1)
        else:
            # Default: girdilerin basit ortalaması
            avg_input = np.mean([
                inputs.weather, inputs.fatigue,
                inputs.travel_distance, inputs.injury_count,
                1 - inputs.motivation, 1 - inputs.form,
                1 - inputs.crowd_factor,
            ])
            output.risk_score = round(float(avg_input * 100), 1)

        # Risk seviyesi
        if output.risk_score < 30:
            output.risk_level = "düşük"
        elif output.risk_score < 55:
            output.risk_level = "orta"
        elif output.risk_score < 80:
            output.risk_level = "yüksek"
        else:
            output.risk_level = "çok_yüksek"

        output.confidence_modifier = self.CONFIDENCE_MAP.get(
            output.risk_level, 1.0,
        )
        output.goal_expectation_mod = self._goal_mod(inputs)
        output.recommendation = self._advice(output, inputs)
        return output

    def _goal_mod(self, inputs: FuzzyInput) -> float:
        """Gol beklentisi çarpanı."""
        # Yağmur → düşük gol, yorgunluk → hata = fazla gol
        weather_effect = 1.0 - inputs.weather * 0.15
        fatigue_effect = 1.0 + inputs.fatigue * 0.2
        return round(weather_effect * fatigue_effect, 3)

    def _advice(self, output: FuzzyOutput,
                  inputs: FuzzyInput) -> str:
        if output.risk_level == "çok_yüksek":
            return (
                f"ÇOK YÜKSEK RİSK: skor={output.risk_score:.0f}/100. "
                f"Güven çarpanı x{output.confidence_modifier:.1f}. "
                f"{'Yağmurlu, ' if inputs.weather > 0.7 else ''}"
                f"{'Yorgun, ' if inputs.fatigue > 0.6 else ''}"
                f"{'Çok sakat. ' if inputs.injury_count > 0.5 else ''}"
                f"Bahisten uzak dur veya stake'i düşür."
            )
        if output.risk_level == "yüksek":
            return (
                f"Yüksek risk: skor={output.risk_score:.0f}/100. "
                f"Stake %30 düşürülmeli."
            )
        if output.risk_level == "düşük":
            return (
                f"Düşük risk: skor={output.risk_score:.0f}/100. "
                f"Güven çarpanı x{output.confidence_modifier:.1f}. "
                f"Model tahminlerine güvenilebilir."
            )
        return (
            f"Orta risk: skor={output.risk_score:.0f}/100. "
            f"Normal işlem."
        )
