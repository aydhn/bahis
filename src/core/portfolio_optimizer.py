"""
portfolio_optimizer.py – Markowitz Modern Portföy Teorisi (Uyarlanmış) +
                        Drawdown Control + Paper Trading + Regime Awareness.

Tek tek Kelly ile bakmak yerine, o günün tüm önerilerini
bir havuz (pool) olarak değerlendiren portföy optimizasyonu.

Eğer A ve B maçı arasında yüksek pozitif korelasyon varsa
(biri yatarsa diğeri de yatıyorsa), risk marjı otomatik düşer.

Drawdown Control:
  Kasa -%10 eridiğinde → stake'ler yarıya iner
  Kasa -%15 eridiğinde → Paper Trading moduna geçer (sanal bahis)
  Kasa toparlandığında → Normal moda geri döner

Regime Awareness:
  Piyasa rejimine (STABLE, VOLATILE, CHAOTIC, CRASH) göre
  risk iştahını ve maksimum pozisyon büyüklüklerini dinamik ayarlar.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger

from src.quant.finance.liquidity_engine import LiquidityEngine


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
    - Likidite kontrolü ile slippage önlemi
    - Piyasa rejimi ile dinamik risk yönetimi
    """

    # Aynı lig/gün korelasyon tahmini
    LEAGUE_CORRELATION = 0.15  # Aynı lig maçları arası baz korelasyon
    TIME_CORRELATION = 0.10    # Aynı gün maçları arası baz korelasyon

    def __init__(self, initial_bankroll: float = 10000.0,
                 max_portfolio_risk: float = 0.15,
                 drawdown_config: DrawdownConfig | None = None,
                 liquidity_engine: LiquidityEngine | None = None):
        self._initial_bankroll = initial_bankroll
        self._current_bankroll = initial_bankroll
        self._peak_bankroll = initial_bankroll
        self._max_risk = max_portfolio_risk
        self._dd_config = drawdown_config or DrawdownConfig()
        self._mode = TradingMode.LIVE
        self._mode_entered_at = time.time()
        self._pnl_history: list[float] = []
        self._paper_pnl: list[float] = []

        self.liquidity_engine = liquidity_engine or LiquidityEngine()

        logger.debug(
            f"PortfolioOptimizer başlatıldı: bankroll=₺{initial_bankroll:,.0f}, "
            f"max_risk={max_portfolio_risk:.0%}"
        )

    # ═══════════════════════════════════════════
    #  MARKOWITZ OPTİMİZASYONU (REGIME-AWARE)
    # ═══════════════════════════════════════════
    def optimize(self, candidates: list[PortfolioBet], regime: str = "STABLE") -> list[dict]:
        """Portföy optimizasyonu uygula.

        1. Korelasyon matrisini oluştur
        2. Portföy varyansını hesapla
        3. Yüksek korelasyonlu bahislerde stake'i düşür
        4. Rejime göre risk limitlerini uygula
        5. Likidite kontrolü yap
        6. Drawdown kontrolü uygula
        """
        if not candidates or not self._check_trading_state(regime):
            return []

        # 1. Korelasyon matrisi
        corr = self._build_correlation_matrix(candidates)

        # 2. Her bahisin ham stake'i (Kelly) ve EV'si
        raw_stakes = np.array([c.stake_pct for c in candidates])
        evs = np.array([c.ev for c in candidates])

        # Sadece pozitif EV olan bahisleri al
        positive_mask = evs > 0
        if not positive_mask.any():
            logger.info("[Portfolio] Pozitif EV yok – bahis yok.")
            return []

        # 3. Korelasyon düzeltmesi (Rejime duyarlı)
        risk_aversion = self._get_risk_aversion(regime)
        try:
            adjusted_stakes = self._adjust_for_correlation(
                raw_stakes, corr, positive_mask, risk_aversion=risk_aversion
            )
        except Exception as e:
            logger.error(f"[Portfolio] Optimization crashed ({e}). Fallback to Heuristic.")
            adjusted_stakes = self._heuristic_adjust(raw_stakes, corr, positive_mask)

        # 4. Toplam risk sınırlaması (Rejime göre scale et)
        adjusted_stakes = self._apply_risk_limits(adjusted_stakes, regime)

        # 5. Drawdown moduna göre ek düzeltme
        mode_multiplier = self._mode_multiplier()
        adjusted_stakes *= mode_multiplier

        # 6. Likidite kontrolü ve sonuçları oluştur
        return self._build_results(candidates, raw_stakes, adjusted_stakes, positive_mask, regime)

    def _check_trading_state(self, regime: str) -> bool:
        """Trading durumunu (Drawdown ve Regime) kontrol eder."""
        self._update_trading_mode()

        if self._mode == TradingMode.FROZEN:
            logger.warning("[Portfolio] FROZEN – tüm bahisler donduruldu.")
            return False

        if regime == "CRASH":
            logger.warning("[Portfolio] CRASH rejimi tespit edildi. Güvenli mod - sadece minimum risk.")
            if self._mode == TradingMode.LIVE:
                 # CRASH modunda otomatik olarak REDUCED veya PAPER moda geçici geçiş mantığı eklenebilir
                 pass

        return True

    def _apply_risk_limits(self, adjusted_stakes: np.ndarray, regime: str) -> np.ndarray:
        """Piyasa rejimine göre portföyün toplam risk sınırlarını uygular."""
        regime_scale = self._get_regime_risk_factor(regime)
        current_max_risk = self._max_risk * regime_scale

        total_risk = adjusted_stakes.sum()
        if total_risk > current_max_risk:
            scale = current_max_risk / total_risk
            adjusted_stakes *= scale

        return adjusted_stakes

    def _build_results(self, candidates: list[PortfolioBet], raw_stakes: np.ndarray,
                       adjusted_stakes: np.ndarray, positive_mask: np.ndarray, regime: str) -> list[dict]:
        """Adayları sonuç sözlüklerine dönüştürür ve likidite sınırlarını uygular."""
        results = []
        is_paper = self._mode == TradingMode.PAPER

        for i, bet in enumerate(candidates):
            if not positive_mask[i] or adjusted_stakes[i] < 0.001:
                continue

            # Likidite Kontrolü: Max Safe Stake
            max_safe_amount = self.liquidity_engine.calculate_max_safe_stake(
                odds=bet.odds,
                edge=bet.ev,
                league=bet.league
            )
            # Portföy yüzdesine çevir
            max_safe_pct = max_safe_amount / self._current_bankroll

            # Eğer optimize edilmiş stake, likidite sınırını aşıyorsa kes
            final_stake_pct = min(adjusted_stakes[i], max_safe_pct)

            # Stake miktarı
            stake_amount = final_stake_pct * self._current_bankroll

            results.append({
                "match_id": bet.match_id,
                "selection": bet.selection,
                "odds": bet.odds,
                "prob_model": bet.prob_model,
                "ev": bet.ev,
                "raw_stake_pct": float(raw_stakes[i]),
                "adjusted_stake_pct": float(final_stake_pct),
                "stake_amount": float(stake_amount),
                "correlation_penalty": float(1 - final_stake_pct / max(raw_stakes[i], 0.001)),
                "liquidity_cap_hit": (final_stake_pct < adjusted_stakes[i]),
                "trading_mode": self._mode.value,
                "regime": regime,
                "is_paper": is_paper,
            })

        # Portföy özet metrikleri
        if results:
            total_stake = sum(r["adjusted_stake_pct"] for r in results)
            portfolio_ev = sum(r["ev"] * r["adjusted_stake_pct"] for r in results)
            logger.info(
                f"[Portfolio] {len(results)} bahis, toplam stake: {total_stake:.2%}, "
                f"portföy EV: {portfolio_ev:.4f}, mod: {self._mode.value}, rejim: {regime}"
            )

        return results

    def _get_regime_risk_factor(self, regime: str) -> float:
        """Rejime göre maksimum risk çarpanı."""
        return {
            "STABLE": 1.0,
            "VOLATILE": 0.6,  # Riski %40 düşür
            "CHAOTIC": 0.3,   # Riski %70 düşür
            "CRASH": 0.1,     # Riski %90 düşür
        }.get(regime, 1.0)

    def _get_risk_aversion(self, regime: str) -> float:
        """Rejime göre risk aversion parametresi."""
        # Yüksek değer = daha fazla varyans cezası (daha güvenli)
        return {
            "STABLE": 1.5,
            "VOLATILE": 3.0,
            "CHAOTIC": 5.0,
            "CRASH": 10.0,
        }.get(regime, 2.0)

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
                                 mask: np.ndarray,
                                 risk_aversion: float = 2.0) -> np.ndarray:
        """Korelasyon bazlı stake düzeltmesi (Markowitz Mean-Variance).

        Yüksek korelasyonlu bahislerde portföy varyansı artar.
        Scipy minimize ile optimal ağırlıkları bulur.
        Objective: Maximize (Expected Return - Risk Penalty * Variance)
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("Scipy yüklü değil, Eigen Risk Parity metoduna dönülüyor.")
            return self._eigen_risk_parity(stakes, corr, mask)

        n = len(stakes)
        indices = np.where(mask)[0]
        k = len(indices)

        if k == 0:
            return np.zeros(n)

        # Sadece aktif bahisler için alt matrisler
        sub_stakes = stakes[indices] # Bunlar Kelly üst sınırları olacak
        sub_corr = corr[np.ix_(indices, indices)]

        # Expected Return vector (Basitçe Kelly stake ile orantılı varsayıyoruz)
        # Çünkü Kelly ~ Edge / Odds. Edge arttıkça Kelly artar.
        expected_returns = sub_stakes

        # Objective Function: Minimize (-Utility)
        # Utility = w.T * mu - (lambda/2) * w.T * Sigma * w
        def objective(weights):
            port_return = np.dot(weights, expected_returns)
            port_var = np.dot(weights.T, np.dot(sub_corr, weights))
            utility = port_return - (risk_aversion / 2) * port_var
            return -utility # Minimize negative utility

        # Constraints & Bounds
        # 1. Weights <= Kelly suggestion (Kelly üst sınırdır)
        # 2. Weights >= 0
        bounds = [(0, s) for s in sub_stakes]

        # Initial guess: Kelly'nin yarısı (Fractional Kelly)
        x0 = sub_stakes * 0.5

        try:
            # SLSQP allows bounds and constraints
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, tol=1e-4)
            if res.success:
                optimized_sub = res.x
            else:
                logger.warning(f"Optimizasyon yakınsamadı: {res.message}")
                optimized_sub = sub_stakes * 0.5 # Fallback
        except Exception as e:
            logger.error(f"Optimizasyon hatası: {e}")
            # Fallback to simple mean if scipy crashes inside
            optimized_sub = sub_stakes * 0.5

        # Sonuçları ana vektöre yerleştir
        adjusted = np.zeros(n)
        adjusted[indices] = optimized_sub

        return adjusted


    def _eigen_risk_parity(self, stakes: np.ndarray, corr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Advanced Eigen Risk Parity for portfolio orthogonalization.
        Extracts eigenvectors to find uncorrelated risk factors and scales stakes down
        if they bunch up on a single dominant eigenvector (e.g., massive weekend league overlap).
        """
        indices = np.where(mask)[0]
        if len(indices) < 2:
            return stakes

        sub_corr = corr[np.ix_(indices, indices)]
        sub_stakes = stakes[indices]

        try:
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(sub_corr)

            # Risk contribution of each bet to the principal components
            # We want to penalize bets that load heavily on the largest eigenvalues (systemic risk)
            # Find the largest eigenvalue
            max_eig_idx = np.argmax(eigenvals)
            max_eig_val = eigenvals[max_eig_idx]

            if max_eig_val > 1.5:  # High systemic correlation detected
                dominant_vec = np.abs(eigenvecs[:, max_eig_idx])
                # Penalize proportional to their loading on the dominant risk factor
                penalty_factors = 1.0 - (dominant_vec * 0.5)
                # Bound penalties to reasonable levels
                penalty_factors = np.clip(penalty_factors, 0.3, 1.0)

                sub_stakes = sub_stakes * penalty_factors

            adjusted = stakes.copy()
            adjusted[indices] = sub_stakes
            return adjusted

        except Exception as e:
            logger.error(f"[Portfolio] Eigen Risk Parity failed: {e}")
            return stakes

    def _heuristic_adjust(self, stakes: np.ndarray, corr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        adjusted = stakes.copy()
        n = len(stakes)
        for i in range(n):
            if not mask[i]:
                adjusted[i] = 0
                continue
            avg_corr = 0.0
            count = 0
            for j in range(n):
                if i != j and mask[j]:
                    avg_corr += abs(corr[i][j])
                    count += 1
            if count > 0:
                avg_corr /= count
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
