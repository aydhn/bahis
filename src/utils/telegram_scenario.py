"""
telegram_scenario.py – İnteraktif Senaryo Simülatörü.

Bot analiz attığında altında butonlar olur:
  [Senaryo: Gol Erken Gelirse?]
  [Senaryo: Kırmızı Kart Çıkarsa?]
  [Senaryo: Penaltı Verilirse?]
  [Senaryo: Yağmur Başlarsa?]

Kullanıcı butona bastığında, analiz parametreleri değiştirilip
model tahminleri anlık olarak yeniden hesaplanır ve sonuç güncellenir.

Telegram InlineKeyboard + Callback Query
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class Scenario:
    """Tek bir senaryo tanımı."""
    id: str
    label: str                   # Buton metni
    emoji: str = ""
    description: str = ""
    # Parametre değişiklikleri
    adjustments: dict = field(default_factory=dict)
    # Örn: {"home_xg_mult": 1.3, "minute": 10, "score_home": 1}


@dataclass
class ScenarioResult:
    """Senaryo simülasyon sonucu."""
    scenario_id: str
    scenario_label: str
    original: dict = field(default_factory=dict)    # Orijinal tahmin
    adjusted: dict = field(default_factory=dict)    # Senaryo sonrası tahmin
    impact: dict = field(default_factory=dict)       # Fark
    explanation: str = ""
    telegram_text: str = ""


# ═══════════════════════════════════════════════
#  HAZIR SENARYOLAR
# ═══════════════════════════════════════════════
DEFAULT_SCENARIOS = [
    Scenario(
        id="early_goal_home",
        label="Ev Sahibi Erken Gol",
        emoji="⚽",
        description="10. dakikada ev sahibi 1-0 öne geçerse",
        adjustments={
            "minute": 10, "score_home": 1, "score_away": 0,
            "home_xg_mult": 1.15, "away_xg_mult": 1.10,
            "home_morale": 0.85, "away_morale": 0.40,
        },
    ),
    Scenario(
        id="early_goal_away",
        label="Deplasman Erken Gol",
        emoji="⚽",
        description="10. dakikada deplasman 0-1 öne geçerse",
        adjustments={
            "minute": 10, "score_home": 0, "score_away": 1,
            "home_xg_mult": 1.20, "away_xg_mult": 0.90,
            "home_morale": 0.45, "away_morale": 0.80,
        },
    ),
    Scenario(
        id="red_card_home",
        label="Ev Sahibi Kırmızı Kart",
        emoji="🟥",
        description="Ev sahibi takımdan kırmızı kart çıkarsa",
        adjustments={
            "home_xg_mult": 0.70, "away_xg_mult": 1.25,
            "home_players": 10, "home_morale": 0.30,
        },
    ),
    Scenario(
        id="red_card_away",
        label="Deplasman Kırmızı Kart",
        emoji="🟥",
        description="Deplasman takımından kırmızı kart çıkarsa",
        adjustments={
            "home_xg_mult": 1.25, "away_xg_mult": 0.70,
            "away_players": 10, "away_morale": 0.30,
        },
    ),
    Scenario(
        id="penalty_home",
        label="Ev Sahibi Penaltı",
        emoji="🎯",
        description="Ev sahibi lehine penaltı verilirse",
        adjustments={
            "home_xg_mult": 1.35, "home_morale": 0.80,
            "penalty_xg": 0.76,
        },
    ),
    Scenario(
        id="rain_start",
        label="Yağmur Başlarsa",
        emoji="🌧️",
        description="Maç sırasında şiddetli yağmur başlarsa",
        adjustments={
            "home_xg_mult": 0.85, "away_xg_mult": 0.85,
            "weather": "rain", "pitch_quality": 0.60,
        },
    ),
    Scenario(
        id="star_injury",
        label="Yıldız Oyuncu Sakatlanırsa",
        emoji="🏥",
        description="Ev sahibi yıldız oyuncu sakatlanırsa",
        adjustments={
            "home_xg_mult": 0.80, "home_morale": 0.50,
            "key_player_out": True,
        },
    ),
    Scenario(
        id="halftime_draw",
        label="İlk Yarı 0-0",
        emoji="⏸️",
        description="İlk yarı 0-0 biterse ikinci yarı ne olur?",
        adjustments={
            "minute": 45, "score_home": 0, "score_away": 0,
            "home_xg_mult": 1.10, "away_xg_mult": 1.10,
        },
    ),
]


class ScenarioSimulator:
    """İnteraktif senaryo simülatörü.

    Kullanım:
        sim = ScenarioSimulator(predict_fn=my_model.predict)
        # Telegram inline keyboard oluştur
        keyboard = sim.build_keyboard("match_123")
        # Senaryo çalıştır
        result = sim.simulate("match_123", "early_goal_home", base_features)
    """

    def __init__(self, predict_fn: Callable | None = None,
                 scenarios: list[Scenario] | None = None):
        """
        Args:
            predict_fn: Tahmin fonksiyonu. features → {"prob_home": ..., "prob_draw": ...}
            scenarios: Özel senaryo listesi (veya DEFAULT_SCENARIOS kullanılır)
        """
        self._predict_fn = predict_fn
        self._scenarios = {s.id: s for s in (scenarios or DEFAULT_SCENARIOS)}
        self._cache: dict[str, dict] = {}  # match_id → base prediction
        logger.debug(f"ScenarioSimulator: {len(self._scenarios)} senaryo.")

    # ═══════════════════════════════════════════
    #  TELEGRAM KLAVYE
    # ═══════════════════════════════════════════
    def build_keyboard(self, match_id: str) -> list[list[dict]]:
        """Telegram InlineKeyboard satırları oluştur.

        Returns: [
            [{"text": "⚽ Erken Gol (Ev)", "callback_data": "scenario_match123_early_goal_home"}],
            ...
        ]
        """
        rows = []
        scenarios = list(self._scenarios.values())

        for i in range(0, len(scenarios), 2):
            row = []
            for s in scenarios[i:i + 2]:
                row.append({
                    "text": f"{s.emoji} {s.label}",
                    "callback_data": f"scenario_{match_id}_{s.id}",
                })
            rows.append(row)

        return rows

    def build_inline_markup(self, match_id: str) -> Any:
        """python-telegram-bot InlineKeyboardMarkup oluştur."""
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            rows = []
            scenarios = list(self._scenarios.values())

            for i in range(0, len(scenarios), 2):
                row = []
                for s in scenarios[i:i + 2]:
                    row.append(InlineKeyboardButton(
                        text=f"{s.emoji} {s.label}",
                        callback_data=f"scenario_{match_id}_{s.id}",
                    ))
                rows.append(row)

            return InlineKeyboardMarkup(rows)
        except ImportError:
            return None

    # ═══════════════════════════════════════════
    #  SENARYO SİMÜLASYONU
    # ═══════════════════════════════════════════
    def simulate(self, match_id: str, scenario_id: str,
                  base_features: dict) -> ScenarioResult:
        """Senaryoyu çalıştır ve sonucu karşılaştır."""
        scenario = self._scenarios.get(scenario_id)
        if not scenario:
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_label="Bilinmeyen senaryo",
                explanation="Bu senaryo tanımlı değil.",
            )

        # Orijinal tahmin
        if match_id in self._cache:
            original = self._cache[match_id]
        else:
            original = self._run_prediction(base_features)
            self._cache[match_id] = original

        # Parametreleri ayarla
        adjusted_features = self._apply_adjustments(
            base_features.copy(), scenario.adjustments,
        )

        # Ayarlanmış tahmin
        adjusted = self._run_prediction(adjusted_features)

        # Fark hesapla
        impact = {}
        for key in ("prob_home", "prob_draw", "prob_away",
                     "over_25", "btts", "xg_home", "xg_away"):
            orig_val = original.get(key, 0)
            adj_val = adjusted.get(key, 0)
            diff = adj_val - orig_val
            if abs(diff) > 0.001:
                impact[key] = {
                    "original": round(orig_val, 3),
                    "adjusted": round(adj_val, 3),
                    "change": round(diff, 3),
                    "change_pct": f"{diff:+.1%}",
                }

        # Açıklama oluştur
        explanation = self._generate_explanation(scenario, impact)

        # Telegram formatı
        telegram_text = self._format_telegram(
            match_id, scenario, original, adjusted, impact,
        )

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_label=f"{scenario.emoji} {scenario.label}",
            original=original,
            adjusted=adjusted,
            impact=impact,
            explanation=explanation,
            telegram_text=telegram_text,
        )

    def _apply_adjustments(self, features: dict,
                            adjustments: dict) -> dict:
        """Senaryo parametrelerini uygula."""
        for key, value in adjustments.items():
            if key.endswith("_mult"):
                base_key = key.replace("_mult", "")
                if base_key in features:
                    features[base_key] = features[base_key] * value
            else:
                features[key] = value
        return features

    def _run_prediction(self, features: dict) -> dict:
        """Model tahmini çalıştır."""
        if self._predict_fn:
            try:
                return self._predict_fn(features)
            except Exception as e:
                logger.debug(f"[Scenario] Tahmin hatası: {e}")

        # Heuristic fallback
        home_xg = features.get("home_xg", 1.4) * features.get("home_xg_mult", 1.0)
        away_xg = features.get("away_xg", 1.1) * features.get("away_xg_mult", 1.0)
        morale_h = features.get("home_morale", 0.60)
        morale_a = features.get("away_morale", 0.50)

        total_xg = home_xg + away_xg
        home_ratio = (home_xg * morale_h) / max(home_xg * morale_h + away_xg * morale_a, 0.01)

        prob_home = min(0.80, max(0.10, home_ratio * 0.7 + 0.15))
        prob_away = min(0.80, max(0.10, (1 - home_ratio) * 0.7 + 0.10))
        prob_draw = max(0.05, 1.0 - prob_home - prob_away)

        # Normalize
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        import math
        over_25 = 1.0 - sum(
            math.exp(-total_xg) * total_xg ** k / math.factorial(k)
            for k in range(3)
        )

        return {
            "prob_home": round(prob_home, 3),
            "prob_draw": round(prob_draw, 3),
            "prob_away": round(prob_away, 3),
            "xg_home": round(home_xg, 2),
            "xg_away": round(away_xg, 2),
            "over_25": round(max(0, min(1, over_25)), 3),
            "btts": round(min(0.9, (1 - math.exp(-home_xg)) * (1 - math.exp(-away_xg))), 3),
        }

    def _generate_explanation(self, scenario: Scenario,
                                impact: dict) -> str:
        """Türkçe açıklama üret."""
        lines = [f"{scenario.emoji} <b>{scenario.label}</b>", ""]
        lines.append(f"<i>{scenario.description}</i>\n")

        if not impact:
            lines.append("Bu senaryo modelin tahminini önemli ölçüde değiştirmiyor.")
            return "\n".join(lines)

        lines.append("<b>Etkilenen Metrikler:</b>")
        metric_names = {
            "prob_home": "Ev Kazanır",
            "prob_draw": "Beraberlik",
            "prob_away": "Dep. Kazanır",
            "over_25": "Üst 2.5",
            "btts": "KG Var",
            "xg_home": "Ev xG",
            "xg_away": "Dep xG",
        }

        for key, data in impact.items():
            name = metric_names.get(key, key)
            change = data["change"]
            arrow = "📈" if change > 0 else "📉"
            lines.append(
                f"  {arrow} {name}: {data['original']:.1%} → "
                f"{data['adjusted']:.1%} ({data['change_pct']})"
            )

        return "\n".join(lines)

    def _format_telegram(self, match_id: str, scenario: Scenario,
                          original: dict, adjusted: dict,
                          impact: dict) -> str:
        """Telegram HTML mesaj formatı."""
        text = (
            f"{scenario.emoji} <b>SENARYO: {scenario.label}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<i>{scenario.description}</i>\n\n"
        )

        # Orijinal vs Düzeltilmiş
        text += "<b>Orijinal → Senaryo Sonrası:</b>\n"
        text += (
            f"  Ev:  {original.get('prob_home', 0):.0%} → "
            f"{adjusted.get('prob_home', 0):.0%}\n"
            f"  Ber: {original.get('prob_draw', 0):.0%} → "
            f"{adjusted.get('prob_draw', 0):.0%}\n"
            f"  Dep: {original.get('prob_away', 0):.0%} → "
            f"{adjusted.get('prob_away', 0):.0%}\n"
        )

        if "over_25" in impact:
            text += (
                f"\n  Ü2.5: {original.get('over_25', 0):.0%} → "
                f"{adjusted.get('over_25', 0):.0%}\n"
            )

        # En büyük etki
        if impact:
            biggest = max(impact.items(), key=lambda x: abs(x[1]["change"]))
            metric_names = {
                "prob_home": "Ev Sahibi", "prob_draw": "Beraberlik",
                "prob_away": "Deplasman", "over_25": "Üst 2.5",
            }
            name = metric_names.get(biggest[0], biggest[0])
            text += (
                f"\n💡 <b>En büyük etki:</b> {name} "
                f"({biggest[1]['change_pct']})"
            )

        return text

    # ═══════════════════════════════════════════
    #  CALLBACK İŞLEYİCİ
    # ═══════════════════════════════════════════
    async def handle_callback(self, callback_data: str,
                                base_features: dict,
                                notifier: Any = None) -> ScenarioResult | None:
        """Telegram callback query işle.

        callback_data format: "scenario_{match_id}_{scenario_id}"
        """
        parts = callback_data.split("_", 2)
        if len(parts) < 3 or parts[0] != "scenario":
            return None

        match_id = parts[1]
        scenario_id = parts[2]

        result = self.simulate(match_id, scenario_id, base_features)

        # Telegram'a gönder
        if notifier and result.telegram_text:
            try:
                await notifier.send(result.telegram_text)
            except Exception as e:
                logger.debug(f"[Scenario] Telegram gönderim hatası: {e}")

        return result

    def parse_callback(self, callback_data: str) -> tuple[str, str] | None:
        """callback_data'dan match_id ve scenario_id çıkar."""
        if not callback_data.startswith("scenario_"):
            return None
        parts = callback_data.split("_", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
        return None

    @property
    def available_scenarios(self) -> list[str]:
        return list(self._scenarios.keys())
