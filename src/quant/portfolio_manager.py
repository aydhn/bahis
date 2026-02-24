"""
portfolio_manager.py - Modern Portfolio Theory & Kelly Optimization.

Bu modül, tekil bahisleri bir "Portföy" olarak ele alır ve
toplam riski minimize ederken Sharpe oranını maksimize eder.
"""
from typing import Dict, Any, List, Optional
from src.core.event_bus import EventBus, Event
from loguru import logger
import numpy as np
from scipy.optimize import minimize

class PortfolioManager:
    """
    Bahis portföyünü optimize eden yönetici.
    Predict -> Optimize -> Bet akışını sağlar.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.current_opportunities: List[Dict[str, Any]] = []

        # Olayları dinle
        if self.bus:
            self.bus.subscribe("prediction_ready", self.on_prediction)
            self.bus.subscribe("pipeline_cycle_end", self.on_cycle_end)

    def on_prediction(self, event: Event):
        """Yeni bir tahmin geldiğinde havuza ekle."""
        prediction = event.data
        # Tahmin güvenilirliği düşükse hiç ekleme
        if prediction.get("confidence", 0) < 0.5:
            return

        self.current_opportunities.append(prediction)
        logger.debug(f"PortfolioManager: Fırsat eklendi -> {prediction.get('match_id')}")

    async def on_cycle_end(self, event: Event):
        """Cycle bittiğinde optimizasyonu çalıştır ve emirleri gönder."""
        if not self.current_opportunities:
            return

        logger.info(f"PortfolioManager: {len(self.current_opportunities)} fırsat optimize ediliyor...")

        # Blocking optimization run in executor if heavy? For N<100 it's fast.
        allocations = self.optimize_portfolio(self.current_opportunities)

        # Emirleri oluştur
        bet_count = 0
        for opp in self.current_opportunities:
            opp_id = opp.get("match_id")
            stake_pct = allocations.get(opp_id, 0.0)

            if stake_pct > 0.001: # Minimum bahis %0.1
                # Emir olayını yayınla
                bet_order = {
                    "match_id": opp_id,
                    "selection": opp.get("selection"),
                    "odds": opp.get("odds"),
                    "stake_pct": round(stake_pct, 4),
                    "reason": "Portfolio Optimization (Sharpe Max)"
                }
                if self.bus:
                    await self.bus.emit(Event(
                        event_type="bet_placed",
                        source="PortfolioManager",
                        match_id=opp_id,
                        data=bet_order
                    ))
                bet_count += 1

        if bet_count > 0:
            logger.success(f"PortfolioManager: {bet_count} adet emir oluşturuldu.")

        # Havuzu temizle
        self.current_opportunities = []

    def optimize_portfolio(self, opportunities: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Markowitz Mean-Variance Optimization.
        Amaç: Sharpe Oranını Maksimize Etmek.
        """
        n = len(opportunities)
        if n == 0:
            return {}

        # 1. Veri Hazırlığı
        returns = [] # Beklenen Getiri
        stds = []    # Risk (Standart Sapma)

        for opp in opportunities:
            p = opp.get("prob_win", 0.5)
            o = opp.get("odds", 2.0)

            # Beklenen Getiri: E[R] = (p * (o-1)) - ((1-p) * 1) = p*o - 1
            exp_ret = (p * o) - 1

            # Varyans (Bernoulli): Var = p*(1-p) * (o)^2 roughly?
            # Basit varyans modeli: Kayıp (-1) veya Kazanç (o-1)
            # Var = E[X^2] - (E[X])^2
            # E[X^2] = p*(o-1)^2 + (1-p)*(-1)^2
            var = (p * (o - 1)**2 + (1 - p) * 1) - (exp_ret**2)
            std = np.sqrt(var) if var > 0 else 1.0

            returns.append(exp_ret)
            stds.append(std)

        returns = np.array(returns)
        stds = np.array(stds)

        # 2. Kovaryans Matrisi (Basitleştirilmiş: Korelasyon yok varsayımı)
        # Gerçekte aynı ligdeki takımlar koreledir, şimdilik diagonal.
        cov_matrix = np.diag(stds ** 2)

        # 3. Optimizasyon Fonksiyonu (Negative Sharpe)
        def neg_sharpe(weights):
            p_ret = np.sum(returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol <= 1e-6:
                return 0.0
            return -(p_ret / p_vol)

        # 4. Kısıtlar
        # Toplam ağırlık <= 1 (Kaldıraçsız)
        # Her bahis <= %5 (Risk Yönetimi)
        constraints = ({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x)})
        bounds = tuple((0.0, 0.05) for _ in range(n)) # 0 ile %5 arası

        # Başlangıç tahmin (Eşit dağılım)
        init_guess = np.array([1.0 / n] * n) * 0.1

        try:
            result = minimize(
                neg_sharpe,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            optimal_weights = result.x
        except Exception as e:
            logger.error(f"Optimizasyon hatası: {e}. Fallback to flat.")
            optimal_weights = np.zeros(n)

        # 5. Sonuçları Eşle
        allocations = {}
        for i, opp in enumerate(opportunities):
            allocations[opp.get("match_id")] = float(optimal_weights[i])

        return allocations
