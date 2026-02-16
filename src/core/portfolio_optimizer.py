"""
portfolio_optimizer.py – Markowitz Modern Portföy Teorisi (Uyarlanmış) +
                        Drawdown Control + Paper Trading.

Tek tek Kelly ile bakmak yerine, o günün tüm önerilerini
bir havuz (pool) olarak değerlendiren portföy optimizasyonu.

Eğer A ve B maçı arasında yüksek pozitif korelasyon varsa
(biri yatarsa diğeri de yatıyorsa), risk marjı otomatik düşer.

Drawdown Control:
  Kasa -%10 eridiğinde → stake'ler yarıya iner
  Kasa -%15 eridiğinde → Paper Trading moduna geçer (sanal bahis)
  Kasa toparlandığında → Normal moda geri döner
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from loguru import logger


class TradingMode(Enum):
    LIVE = "LIVE"            # Gerçek bahis
    REDUCED = "REDUCED"      # Yarı stake
    PAPER = "PAPER"          # Sanal bahis (cool-down)
    FROZEN = "FROZEN"        # Tamamen dondurulmuş


@dataclass
class DrawdownConfig:
    """Drawdown kontrol parametreleri."""
    reduce_threshold: float = 0.10   # -%10'da stake yarıya
    paper_threshold: float = 0.15    # -%15'te paper trading
    freeze_threshold: float = 0.25   # -%25'te tamamen dondur
    recovery_threshold: float = 0.05 # -%5'e toparlanınca normal moda dön
    cooldown_hours: float = 6.0      # Paper mode minimum süresi


@dataclass
class PortfolioBet:
    """Portföydeki tek bir bahis."""
    match_id: str
    selection: str       # home / draw / away / over / under
    odds: float
    prob_model: float    # Model tahmin olasılığı
    ev: float            # Expected Value
    stake_pct: float     # Kelly önerisi (kasa yüzdesi)
    league: str = ""
    correlation_group: str = ""  # Benzer bahisleri grupla


class PortfolioOptimizer:
    """Markowitz-uyarlanmış portföy optimizasyonu.

    Kelly her maça bağımsız bakar. Markowitz tüm portföyü birlikte
    değerlendirir:
    - Yüksek korelasyonlu bahislerde toplam stake düşer
    - Bağımsız bahislerde stake korunur (diversification)
    - Drawdown kontrolü ile otomatik risk yönetimi
    """

    # Aynı lig/gün korelasyon tahmini
    LEAGUE_CORRELATION = 0.15  # Aynı lig maçları arası baz korelasyon
    TIME_CORRELATION = 0.10    # Aynı gün maçları arası baz korelasyon

    def __init__(self, initial_bankroll: float = 10000.0,
                 max_portfolio_risk: float = 0.15,
                 drawdown_config: DrawdownConfig | None = None):
        self._initial_bankroll = initial_bankroll
        self._current_bankroll = initial_bankroll
        self._peak_bankroll = initial_bankroll
        self._max_risk = max_portfolio_risk
        self._dd_config = drawdown_config or DrawdownConfig()
        self._mode = TradingMode.LIVE
        self._mode_entered_at = time.time()
        self._pnl_history: list[float] = []
        self._paper_pnl: list[float] = []
        logger.debug(
            f"PortfolioOptimizer başlatıldı: bankroll=₺{initial_bankroll:,.0f}, "
            f"max_risk={max_portfolio_risk:.0%}"
        )

    # ═══════════════════════════════════════════
    #  MARKOWITZ OPTİMİZASYONU
    # ═══════════════════════════════════════════
    def optimize(self, candidates: list[PortfolioBet]) -> list[dict]:
        """Portföy optimizasyonu uygula.

        1. Korelasyon matrisini oluştur
        2. Portföy varyansını hesapla
        3. Yüksek korelasyonlu bahislerde stake'i düşür
        4. Toplam riski sınırla
        5. Drawdown kontrolü uygula
        """
        if not candidates:
            return []

        # Drawdown kontrolünü güncelle
        self._update_trading_mode()

        if self._mode == TradingMode.FROZEN:
            logger.warning("[Portfolio] FROZEN – tüm bahisler donduruldu.")
            return []

        n = len(candidates)

        # 1. Korelasyon matrisi
        corr = self._build_correlation_matrix(candidates)

        # 2. Her bahisin ham stake'i (Kelly)
        raw_stakes = np.array([c.stake_pct for c in candidates])
        evs = np.array([c.ev for c in candidates])

        # Sadece pozitif EV olan bahisleri al
        positive_mask = evs > 0
        if not positive_mask.any():
            logger.info("[Portfolio] Pozitif EV yok – bahis yok.")
            return []

        # 3. Korelasyon düzeltmesi
        adjusted_stakes = self._adjust_for_correlation(
            raw_stakes, corr, positive_mask
        )

        # 4. Toplam risk sınırlaması
        total_risk = adjusted_stakes.sum()
        if total_risk > self._max_risk:
            scale = self._max_risk / total_risk
            adjusted_stakes *= scale

        # 5. Drawdown moduna göre ek düzeltme
        mode_multiplier = self._mode_multiplier()
        adjusted_stakes *= mode_multiplier

        # Sonuçları oluştur
        results = []
        for i, bet in enumerate(candidates):
            if not positive_mask[i] or adjusted_stakes[i] < 0.001:
                continue

            is_paper = self._mode == TradingMode.PAPER

            results.append({
                "match_id": bet.match_id,
                "selection": bet.selection,
                "odds": bet.odds,
                "prob_model": bet.prob_model,
                "ev": bet.ev,
                "raw_stake_pct": float(raw_stakes[i]),
                "adjusted_stake_pct": float(adjusted_stakes[i]),
                "stake_amount": float(adjusted_stakes[i] * self._current_bankroll),
                "correlation_penalty": float(1 - adjusted_stakes[i] / max(raw_stakes[i], 0.001)),
                "trading_mode": self._mode.value,
                "is_paper": is_paper,
            })

        # Portföy özet metrikleri
        if results:
            total_stake = sum(r["adjusted_stake_pct"] for r in results)
            portfolio_ev = sum(r["ev"] * r["adjusted_stake_pct"] for r in results)
            logger.info(
                f"[Portfolio] {len(results)} bahis, toplam stake: {total_stake:.2%}, "
                f"portföy EV: {portfolio_ev:.4f}, mod: {self._mode.value}"
            )

        return results

    def _build_correlation_matrix(self, bets: list[PortfolioBet]) -> np.ndarray:
        """Bahisler arası korelasyon matrisini oluştur."""
        n = len(bets)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                c = self._estimate_correlation(bets[i], bets[j])
                corr[i][j] = c
                corr[j][i] = c

        return corr

    def _estimate_correlation(self, a: PortfolioBet, b: PortfolioBet) -> float:
        """İki bahis arasındaki tahmini korelasyon."""
        corr = 0.0

        # Aynı maç → yüksek korelasyon
        if a.match_id == b.match_id:
            return 0.7

        # Aynı lig
        if a.league and a.league == b.league:
            corr += self.LEAGUE_CORRELATION

        # Aynı seçim türü
        if a.selection == b.selection:
            corr += 0.1

        # Aynı korelasyon grubu
        if a.correlation_group and a.correlation_group == b.correlation_group:
            corr += 0.2

        return min(corr, 0.8)

    def _adjust_for_correlation(self, stakes: np.ndarray,
                                 corr: np.ndarray,
                                 mask: np.ndarray) -> np.ndarray:
        """Korelasyon bazlı stake düzeltmesi.

        Yüksek korelasyonlu bahislerde portföy varyansı artar.
        Bunu telafi etmek için stake'leri düşür.
        """
        adjusted = stakes.copy()
        n = len(stakes)

        for i in range(n):
            if not mask[i]:
                adjusted[i] = 0
                continue

            # Bu bahisin diğerleriyle ortalama korelasyonu
            avg_corr = 0.0
            count = 0
            for j in range(n):
                if i != j and mask[j]:
                    avg_corr += abs(corr[i][j])
                    count += 1

            if count > 0:
                avg_corr /= count
                # Korelasyon cezası: yüksek korelasyon → düşük stake
                penalty = 1.0 - (avg_corr * 0.5)
                adjusted[i] *= max(penalty, 0.3)

        return adjusted

    # ═══════════════════════════════════════════
    #  DRAWDOWN CONTROL
    # ═══════════════════════════════════════════
    def update_bankroll(self, pnl: float, is_paper: bool = False):
        """Kasa güncellemesi ve drawdown kontrolü."""
        if is_paper:
            self._paper_pnl.append(pnl)
            logger.debug(f"[Paper] PnL: {pnl:+.2f}")
            return

        self._current_bankroll += pnl
        self._pnl_history.append(pnl)

        if self._current_bankroll > self._peak_bankroll:
            self._peak_bankroll = self._current_bankroll

        self._update_trading_mode()

    def _update_trading_mode(self):
        """Drawdown seviyesine göre trading modunu güncelle."""
        if self._peak_bankroll <= 0:
            return

        drawdown = (self._peak_bankroll - self._current_bankroll) / self._peak_bankroll

        old_mode = self._mode

        if drawdown >= self._dd_config.freeze_threshold:
            self._mode = TradingMode.FROZEN
        elif drawdown >= self._dd_config.paper_threshold:
            # Paper mode minimum süre kontrolü
            if self._mode == TradingMode.PAPER:
                elapsed_hours = (time.time() - self._mode_entered_at) / 3600
                if elapsed_hours < self._dd_config.cooldown_hours:
                    return  # Cool-down süresi dolmadı
            self._mode = TradingMode.PAPER
        elif drawdown >= self._dd_config.reduce_threshold:
            self._mode = TradingMode.REDUCED
        elif drawdown <= self._dd_config.recovery_threshold:
            if self._mode in (TradingMode.PAPER, TradingMode.REDUCED):
                self._mode = TradingMode.LIVE

        if old_mode != self._mode:
            self._mode_entered_at = time.time()
            logger.warning(
                f"[Drawdown] {old_mode.value} → {self._mode.value} "
                f"(drawdown={drawdown:.1%}, "
                f"kasa=₺{self._current_bankroll:,.0f})"
            )

    def _mode_multiplier(self) -> float:
        """Trading moduna göre stake çarpanı."""
        return {
            TradingMode.LIVE: 1.0,
            TradingMode.REDUCED: 0.5,
            TradingMode.PAPER: 1.0,
            TradingMode.FROZEN: 0.0,
        }[self._mode]

    @property
    def drawdown(self) -> float:
        if self._peak_bankroll <= 0:
            return 0
        return (self._peak_bankroll - self._current_bankroll) / self._peak_bankroll

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def bankroll(self) -> float:
        return self._current_bankroll

    def status(self) -> dict:
        return {
            "bankroll": self._current_bankroll,
            "peak": self._peak_bankroll,
            "drawdown": self.drawdown,
            "drawdown_pct": f"{self.drawdown:.1%}",
            "mode": self._mode.value,
            "total_bets": len(self._pnl_history),
            "paper_bets": len(self._paper_pnl),
            "total_pnl": sum(self._pnl_history),
            "paper_pnl": sum(self._paper_pnl),
            "win_rate": (
                sum(1 for p in self._pnl_history if p > 0) / max(len(self._pnl_history), 1)
            ),
        }
