"""
hedge_calculator.py – Arbitraj ve Hedge (Kar Al) Hesaplayıcı.

Bazen kazanmak değil, "kaybetmemek" önemlidir.
Canlı bahiste "Kar Al" veya "Ters Bahis" fırsatı sunulmalıdır.

Senaryo:
  - Maç öncesi "Ev Sahibi" oynadık @ 2.10
  - Maçta Ev Sahibi 1-0 öne geçti
  - Canlı'da "Beraberlik/Deplasman" oranları yükseldi
  - Ters bahise para yatırarak GARANTİ KAR elde edebiliriz

Arbitraj (Surebet):
  Tüm olasılıkları kapsayarak, sonuç ne olursa olsun kâr

Hedge:
  Mevcut bahsi korumak için ters pozisyon alarak riski sınırla

Telegram: "🚨 HEDGE FIRSATI: X TL karşı bahse bas → maç ne biterse
          bitsin Y TL kâr!"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class HedgeOpportunity:
    """Tespit edilen hedge/arbitraj fırsatı."""
    match_id: str
    opportunity_type: str      # "surebet" / "hedge" / "cash_out"
    original_bet: dict = field(default_factory=dict)
    hedge_bet: dict = field(default_factory=dict)
    guaranteed_profit: float = 0.0
    guaranteed_profit_pct: float = 0.0
    risk_free: bool = False
    action_text: str = ""
    urgency: str = "medium"    # low / medium / high / critical


class HedgeCalculator:
    """Arbitraj ve Hedge hesaplayıcı.

    Kullanım:
        calc = HedgeCalculator()

        # Surebet kontrolü (maç öncesi)
        surebet = calc.check_surebet(home_odds=2.10, draw_odds=3.40, away_odds=3.80)

        # Hedge hesaplama (canlı maç)
        hedge = calc.calculate_hedge(
            original_stake=100, original_odds=2.10, original_selection="home",
            current_live_odds={"home": 1.30, "draw": 5.00, "away": 12.00},
            current_score={"home": 1, "away": 0},
        )

        # Cash-out değeri
        cashout = calc.calculate_cashout(
            original_stake=100, original_odds=2.10,
            current_odds=1.50,
        )
    """

    MIN_SUREBET_MARGIN = -0.03  # -%3'ten düşükse surebet var

    def __init__(self, min_profit_pct: float = 0.01):
        self._min_profit = min_profit_pct
        self._opportunities: list[HedgeOpportunity] = []
        logger.debug("HedgeCalculator başlatıldı.")

    # ═══════════════════════════════════════════
    #  SUREBET (ARBİTRAJ) KONTROLÜ
    # ═══════════════════════════════════════════
    def check_surebet(self, home_odds: float, draw_odds: float,
                      away_odds: float,
                      match_id: str = "") -> HedgeOpportunity | None:
        """3 yollu surebet kontrolü.

        Surebet koşulu: 1/home + 1/draw + 1/away < 1.0

        Returns:
            HedgeOpportunity if surebet exists, else None
        """
        if any(o <= 1.0 for o in (home_odds, draw_odds, away_odds)):
            return None

        margin = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)

        if margin >= 1.0:
            return None

        # Surebet var! Optimal stake dağılımını hesapla
        total_stake = 100.0
        stake_home = total_stake / (margin * home_odds)
        stake_draw = total_stake / (margin * draw_odds)
        stake_away = total_stake / (margin * away_odds)

        profit = total_stake * (1 / margin - 1)
        profit_pct = profit / total_stake

        opp = HedgeOpportunity(
            match_id=match_id,
            opportunity_type="surebet",
            original_bet={
                "home_odds": home_odds, "draw_odds": draw_odds,
                "away_odds": away_odds, "margin": margin,
            },
            hedge_bet={
                "stake_home": round(stake_home, 2),
                "stake_draw": round(stake_draw, 2),
                "stake_away": round(stake_away, 2),
                "total_stake": round(total_stake, 2),
            },
            guaranteed_profit=round(profit, 2),
            guaranteed_profit_pct=round(profit_pct, 4),
            risk_free=True,
            action_text=(
                f"🚨 SUREBET: Ev={home_odds:.2f} (₺{stake_home:.0f}), "
                f"Ber={draw_odds:.2f} (₺{stake_draw:.0f}), "
                f"Dep={away_odds:.2f} (₺{stake_away:.0f}) → "
                f"Garanti kâr: ₺{profit:.2f} ({profit_pct:.1%})"
            ),
            urgency="critical",
        )

        self._opportunities.append(opp)
        logger.warning(f"[Hedge] 🚨 SUREBET tespit: margin={margin:.4f}, profit={profit_pct:.1%}")
        return opp

    def check_surebet_2way(self, odds_a: float, odds_b: float,
                            match_id: str = "",
                            labels: tuple[str, str] = ("over", "under")
                            ) -> HedgeOpportunity | None:
        """2 yollu surebet (Alt/Üst, Var/Yok, vb.)."""
        if odds_a <= 1.0 or odds_b <= 1.0:
            return None

        margin = (1 / odds_a) + (1 / odds_b)
        if margin >= 1.0:
            return None

        total = 100.0
        stake_a = total / (margin * odds_a)
        stake_b = total / (margin * odds_b)
        profit = total * (1 / margin - 1)

        opp = HedgeOpportunity(
            match_id=match_id,
            opportunity_type="surebet_2way",
            hedge_bet={
                f"stake_{labels[0]}": round(stake_a, 2),
                f"stake_{labels[1]}": round(stake_b, 2),
            },
            guaranteed_profit=round(profit, 2),
            guaranteed_profit_pct=round(profit / total, 4),
            risk_free=True,
            urgency="critical",
        )
        self._opportunities.append(opp)
        return opp

    # ═══════════════════════════════════════════
    #  HEDGE HESAPLAMA (Canlı maç)
    # ═══════════════════════════════════════════
    def calculate_hedge(self, original_stake: float,
                        original_odds: float,
                        original_selection: str,
                        current_live_odds: dict[str, float],
                        current_score: dict[str, int] | None = None,
                        match_id: str = "") -> HedgeOpportunity | None:
        """Canlı maçta hedge fırsatı hesapla.

        Senaryo: "Ev Sahibi" oynadık @ 2.10, maçta 1-0 oldu,
        canlı'da "Beraberlik/Dep" oranları yükseldi.

        Args:
            original_stake: İlk bahis miktarı (₺)
            original_odds: İlk bahis oranı
            original_selection: "home" / "draw" / "away"
            current_live_odds: Anlık canlı oranlar
            current_score: Anlık skor {"home": 1, "away": 0}
        """
        potential_return = original_stake * original_odds

        # Ters bahis seçenekleri
        counter_selections = [s for s in current_live_odds if s != original_selection]

        best_hedge = None
        best_min_profit = -float("inf")

        for counter in counter_selections:
            counter_odds = current_live_odds.get(counter, 0)
            if counter_odds <= 1.0:
                continue

            # Hedge stake hesapla: tüm senaryolarda kâr/zarar dengeleme
            hedge_stake = potential_return / counter_odds

            # Senaryo analizi
            profit_if_original_wins = potential_return - original_stake - hedge_stake
            profit_if_hedge_wins = (hedge_stake * counter_odds) - original_stake - hedge_stake

            min_profit = min(profit_if_original_wins, profit_if_hedge_wins)

            if min_profit > best_min_profit:
                best_min_profit = min_profit
                best_hedge = {
                    "selection": counter,
                    "odds": counter_odds,
                    "stake": round(hedge_stake, 2),
                    "profit_if_original": round(profit_if_original_wins, 2),
                    "profit_if_hedge": round(profit_if_hedge_wins, 2),
                    "min_profit": round(min_profit, 2),
                }

        if not best_hedge or best_min_profit < 0:
            # Kâr garantisi yok ama zarar sınırlaması var mı?
            if best_hedge:
                return self._partial_hedge(
                    original_stake, original_odds, original_selection,
                    current_live_odds, match_id,
                )
            return None

        total_investment = original_stake + best_hedge["stake"]
        profit_pct = best_min_profit / total_investment

        opp = HedgeOpportunity(
            match_id=match_id,
            opportunity_type="hedge",
            original_bet={
                "selection": original_selection,
                "odds": original_odds,
                "stake": original_stake,
            },
            hedge_bet=best_hedge,
            guaranteed_profit=best_min_profit,
            guaranteed_profit_pct=profit_pct,
            risk_free=best_min_profit > 0,
            action_text=(
                f"🚨 HEDGE FIRSATI: {best_hedge['selection'].upper()} "
                f"@ {best_hedge['odds']:.2f} → ₺{best_hedge['stake']:.0f} yatır.\n"
                f"Maç ne biterse bitsin en az ₺{best_min_profit:.0f} kâr!"
            ),
            urgency="critical" if profit_pct > 0.05 else "high",
        )

        self._opportunities.append(opp)
        logger.info(
            f"[Hedge] 💰 Fırsat: {best_hedge['selection']} "
            f"@ {best_hedge['odds']:.2f}, garanti: ₺{best_min_profit:.0f}"
        )
        return opp

    def _partial_hedge(self, original_stake: float,
                        original_odds: float,
                        original_selection: str,
                        live_odds: dict[str, float],
                        match_id: str) -> HedgeOpportunity | None:
        """Tam garanti kâr yoksa, kısmi hedge (zarar sınırlama)."""
        counter_selections = [s for s in live_odds if s != original_selection]
        if not counter_selections:
            return None

        # En düşük kayıp senaryosunu bul
        best = None
        for counter in counter_selections:
            c_odds = live_odds.get(counter, 0)
            if c_odds <= 1.0:
                continue

            # Hedge stake'i = orijinalin yarısı (zarar sınırlama)
            hedge_stake = original_stake * 0.5
            hedge_return = hedge_stake * c_odds

            worst_case = -(original_stake + hedge_stake)
            best_case = (original_stake * original_odds) - original_stake - hedge_stake

            if best is None or worst_case > best.get("worst", -float("inf")):
                best = {
                    "selection": counter,
                    "odds": c_odds,
                    "stake": round(hedge_stake, 2),
                    "worst_case": round(worst_case + hedge_return, 2),
                    "best_case": round(best_case, 2),
                }

        if not best:
            return None

        return HedgeOpportunity(
            match_id=match_id,
            opportunity_type="partial_hedge",
            original_bet={"selection": original_selection, "odds": original_odds, "stake": original_stake},
            hedge_bet=best,
            guaranteed_profit=best["worst_case"],
            risk_free=False,
            action_text=(
                f"⚠️ KISMİ HEDGE: {best['selection'].upper()} "
                f"@ {best['odds']:.2f} → ₺{best['stake']:.0f}\n"
                f"En kötü: ₺{best['worst_case']:.0f} | En iyi: ₺{best['best_case']:.0f}"
            ),
            urgency="medium",
        )

    # ═══════════════════════════════════════════
    #  CASH-OUT DEĞERİ
    # ═══════════════════════════════════════════
    def calculate_cashout(self, original_stake: float,
                          original_odds: float,
                          current_odds: float) -> dict:
        """Cash-out (erken çekim) değerini hesapla.

        Bahis şirketinin sunacağı cash-out tahmini:
        CashOut = Stake * (Original_Odds / Current_Odds)
        """
        if current_odds <= 0:
            return {"cashout_value": 0, "profit": -original_stake}

        cashout = original_stake * (original_odds / current_odds)
        profit = cashout - original_stake
        roi = profit / original_stake if original_stake > 0 else 0

        return {
            "cashout_value": round(cashout, 2),
            "profit": round(profit, 2),
            "roi": round(roi, 4),
            "original_stake": original_stake,
            "original_odds": original_odds,
            "current_odds": current_odds,
            "recommendation": (
                "CASH OUT" if roi > 0.3 else
                "CONSIDER" if roi > 0.1 else
                "HOLD" if roi > -0.1 else
                "HEDGE"
            ),
        }

    # ═══════════════════════════════════════════
    #  TOPLU KONTROL
    # ═══════════════════════════════════════════
    def scan_active_bets(self, active_bets: list[dict],
                          live_odds: dict[str, dict]) -> list[HedgeOpportunity]:
        """Tüm aktif bahisleri tarayarak hedge fırsatlarını bul."""
        opportunities = []

        for bet in active_bets:
            match_id = bet.get("match_id", "")
            odds_data = live_odds.get(match_id, {})
            if not odds_data:
                continue

            opp = self.calculate_hedge(
                original_stake=bet.get("stake", 0),
                original_odds=bet.get("odds", 0),
                original_selection=bet.get("selection", ""),
                current_live_odds=odds_data,
                match_id=match_id,
            )
            if opp:
                opportunities.append(opp)

        return opportunities

    async def notify_opportunities(self, opportunities: list[HedgeOpportunity],
                                    notifier=None):
        """Fırsatları Telegram'a bildir."""
        if not notifier:
            return

        for opp in opportunities:
            emoji = "🚨" if opp.risk_free else "⚠️"
            text = (
                f"{emoji} <b>{'SUREBET' if opp.opportunity_type == 'surebet' else 'HEDGE'} FIRSATI</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{opp.action_text}\n\n"
                f"💰 <b>Garanti Kâr:</b> ₺{opp.guaranteed_profit:.0f} "
                f"({opp.guaranteed_profit_pct:.1%})\n"
                f"🔒 <b>Risksiz:</b> {'Evet ✅' if opp.risk_free else 'Hayır ❌'}"
            )
            await notifier.send(text)

    @property
    def recent_opportunities(self) -> list[HedgeOpportunity]:
        return self._opportunities[-20:]
