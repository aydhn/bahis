"""
devils_advocate.py – Karşıt analiz ajanı.
"Bu bahis neden yatabilir?" raporu üreterek overconfidence bias'ı kırar.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class DevilsAdvocate:
    """Her bahse karşı argüman üreten karşıt analiz modülü."""

    RISK_FACTORS = {
        "low_confidence": {
            "threshold": lambda sig: sig.get("confidence", 0) < 0.45,
            "message": "Güven seviyesi düşük – model kararsız.",
        },
        "negative_ev": {
            "threshold": lambda sig: sig.get("ev", 0) < 0.01,
            "message": "Beklenen değer (EV) yetersiz.",
        },
        "high_stake": {
            "threshold": lambda sig: sig.get("stake_pct", 0) > 0.04,
            "message": "Stake oranı agresif – tek maça çok yükleniyor.",
        },
        "draw_trap": {
            "threshold": lambda sig: sig.get("selection") == "draw" and sig.get("confidence", 0) < 0.55,
            "message": "Beraberlik tuzağı – draw tahminleri genellikle düşük isabetle.",
        },
        "odds_too_low": {
            "threshold": lambda sig: sig.get("odds", 0) < 1.25,
            "message": "Oran çok düşük – olası getiri riski karşılamıyor.",
        },
        "odds_too_high": {
            "threshold": lambda sig: sig.get("odds", 0) > 6.0,
            "message": "Oran çok yüksek – muhtemel value trap.",
        },
        "overbet": {
            "threshold": lambda sig: sig.get("ev", 0) > 0 and sig.get("stake_pct", 0) > sig.get("ev", 0) * 0.5,
            "message": "Stake, EV'ye göre orantısız yüksek.",
        },
    }

    def __init__(self):
        self._challenges: list[dict] = []
        logger.debug("DevilsAdvocate başlatıldı.")

    def challenge(self, bets: list[dict]) -> list[dict]:
        """Her bahise karşı argüman üretir."""
        self._challenges = []

        for bet in bets:
            if bet.get("selection") == "skip":
                continue

            warnings = []
            for factor_name, factor in self.RISK_FACTORS.items():
                try:
                    if factor["threshold"](bet):
                        warnings.append({
                            "factor": factor_name,
                            "message": factor["message"],
                        })
                except Exception as e:
                    logger.debug(f"Exception caught: {e}")

            # Sürpriz riski
            surprise_score = self._surprise_risk(bet)
            if surprise_score > 0.6:
                warnings.append({
                    "factor": "surprise_risk",
                    "message": f"Sürpriz riski yüksek ({surprise_score:.0%}) – beklenmeyeni düşün.",
                })

            challenge = {
                "match_id": bet.get("match_id", ""),
                "selection": bet.get("selection", ""),
                "warnings": warnings,
                "risk_score": len(warnings) / max(len(self.RISK_FACTORS), 1),
                "verdict": self._verdict(warnings),
            }
            self._challenges.append(challenge)

            if warnings:
                logger.warning(
                    f"Şeytanın Avukatı [{bet.get('match_id', '')}]: "
                    f"{len(warnings)} uyarı – {challenge['verdict']}"
                )

        return self._challenges

    def _surprise_risk(self, bet: dict) -> float:
        """Sürpriz (underdog kazanma) riskini hesaplar."""
        prob = bet.get("confidence", 0.5)
        odds = bet.get("odds", 2.0)

        if odds <= 0:
            return 0.5

        implied_loss = 1 - 1.0 / odds
        entropy = -prob * np.log(prob + 1e-10) - (1-prob) * np.log(1-prob + 1e-10)
        normalized_entropy = entropy / np.log(2)

        return float(np.clip(implied_loss * normalized_entropy, 0, 1))

    def _verdict(self, warnings: list[dict]) -> str:
        n = len(warnings)
        if n == 0:
            return "ONAY – Ciddi risk faktörü tespit edilmedi."
        elif n <= 2:
            return "DİKKAT – Bazı risk faktörleri mevcut."
        elif n <= 4:
            return "UYARI – Yüksek risk, stake azaltılmalı."
        else:
            return "RED – Bu bahisten uzak dur!"

    def print_report(self):
        """Karşıt analiz raporunu konsola yazdırır."""
        console = Console()
        for ch in self._challenges:
            if not ch["warnings"]:
                continue

            text = Text()
            text.append(f"\n  Maç: {ch['match_id']}\n", style="bold cyan")
            text.append(f"  Seçim: {ch['selection']}\n", style="white")
            text.append(f"  Karar: {ch['verdict']}\n", style="bold yellow")

            for w in ch["warnings"]:
                text.append(f"    ⚠ {w['message']}\n", style="red")

            console.print(Panel(text, title="Şeytanın Avukatı", border_style="red"))
