"""
evt_risk_manager.py – Extreme Value Theory (EVT) ile Kuyruk Riski Yönetimi.

Normal dağılım (Çan eğrisi), futboldaki sürprizleri (7-0 biten
maçları veya 90+8'de gelen golleri) açıklayamaz. Bunlar "Kalın
Kuyruk" (Fat Tail) olaylarıdır.

Value at Risk (VaR): "Bu bahiste %99 ihtimalle en fazla ne kadar
kaybederim?" sorusunun cevabı.

EVT: Nadir görülen olayların (Sürprizlerin) olasılığını modeller.
Bahis bürolarının en çok yanıldığı yer burasıdır
(Underpricing the Tail Risk).

Kavramlar:
  - Generalized Pareto Distribution (GPD): Eşik aşımlarını modeller
  - Peaks Over Threshold (POT): Belirli eşiğin üzerindeki değerler
  - Return Level: "X yılda bir olacak olay" tahmini
  - Tail Index (ξ): Kuyruğun kalınlığını ölçer
  - VaR (Value at Risk): Maksimum kayıp tahmini
  - CVaR (Expected Shortfall): VaR'ı aştığında ortalama kayıp

Sinyal:
  Siyah Kuğu riski > eşik → Kelly stake %50 düşür!
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from scipy.stats import genpareto, genextreme, norm, kstest
    from scipy.optimize import minimize_scalar
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    logger.info("scipy yüklü değil – EVT basit fallback.")

try:
    import pyextremes
    PYEXT_OK = True
except ImportError:
    PYEXT_OK = False


@dataclass
class EVTReport:
    """EVT analiz raporu."""
    match_id: str = ""
    # GPD parametreleri
    shape_xi: float = 0.0             # Kuyruk indeksi (ξ): >0 kalın kuyruk
    scale_sigma: float = 0.0          # Ölçek parametresi
    threshold: float = 0.0            # Eşik değeri (POT)
    n_exceedances: int = 0            # Eşik aşan gözlem sayısı
    # Risk metrikleri
    var_95: float = 0.0               # %95 VaR (birim kayıp)
    var_99: float = 0.0               # %99 VaR
    cvar_95: float = 0.0              # %95 CVaR (Expected Shortfall)
    cvar_99: float = 0.0              # %99 CVaR
    # Siyah Kuğu
    black_swan_prob: float = 0.0      # Siyah Kuğu olasılığı
    return_level_10: float = 0.0      # 10 yılda bir olacak olay seviyesi
    return_level_50: float = 0.0      # 50 yılda bir
    # Karar
    is_fat_tail: bool = False         # Kalın kuyruk var mı?
    tail_severity: str = "normal"     # normal | moderate | heavy | extreme
    kelly_adjustment: float = 1.0     # Kelly çarpanı (1.0 = normal, 0.5 = %50 düşür)
    recommendation: str = ""
    method: str = ""


@dataclass
class PortfolioVaR:
    """Portföy bazlı VaR raporu."""
    total_var_95: float = 0.0
    total_var_99: float = 0.0
    total_cvar_95: float = 0.0
    component_vars: list[dict] = field(default_factory=list)
    diversification_benefit: float = 0.0
    max_loss_scenario: float = 0.0
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  GPD FIT (Peaks Over Threshold)
# ═══════════════════════════════════════════════
def fit_gpd(data: np.ndarray, threshold_quantile: float = 0.90
            ) -> dict:
    """Generalized Pareto Distribution fit.

    Peaks Over Threshold (POT) yöntemi ile kalın kuyruk
    modellemesi.
    """
    if not SCIPY_OK or len(data) < 20:
        return {"shape": 0.0, "scale": 1.0, "threshold": 0.0,
                "exceedances": 0, "method": "insufficient_data"}

    threshold = float(np.quantile(data, threshold_quantile))
    exceedances = data[data > threshold] - threshold

    if len(exceedances) < 5:
        return {"shape": 0.0, "scale": 1.0, "threshold": threshold,
                "exceedances": len(exceedances), "method": "too_few_exceedances"}

    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
        return {
            "shape": float(shape),
            "scale": float(scale),
            "threshold": threshold,
            "exceedances": len(exceedances),
            "method": "gpd_mle",
        }
    except Exception:
        # Moment yöntemi fallback
        mean_exc = float(np.mean(exceedances))
        var_exc = float(np.var(exceedances))
        if mean_exc > 0:
            shape = 0.5 * (mean_exc ** 2 / var_exc - 1) if var_exc > 0 else 0
            scale = mean_exc * (1 + shape) / 2 if shape > -0.5 else mean_exc
        else:
            shape, scale = 0.0, 1.0
        return {
            "shape": shape, "scale": scale,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "method": "moment",
        }


def compute_var(shape: float, scale: float, threshold: float,
                n_total: int, n_exceedances: int,
                confidence: float = 0.99) -> float:
    """GPD tabanlı VaR hesapla."""
    if n_total == 0 or n_exceedances == 0:
        return 0.0

    exceed_rate = n_exceedances / n_total
    p = 1 - confidence

    if SCIPY_OK and abs(shape) > 1e-8:
        try:
            var_val = threshold + (scale / shape) * (
                (p / exceed_rate) ** (-shape) - 1
            )
            return max(0.0, float(var_val))
        except Exception as e:
            logger.debug(f"Exception caught: {e}")

    return threshold + scale * np.log(exceed_rate / max(p, 1e-10))


def compute_cvar(var_val: float, shape: float, scale: float,
                  threshold: float) -> float:
    """CVaR (Expected Shortfall) hesapla.

    VaR'ı aştığında beklenen ortalama kayıp.
    """
    if shape >= 1:
        return float("inf")

    cvar = var_val / (1 - shape) + (scale - shape * threshold) / (1 - shape)
    return max(var_val, float(cvar))


# ═══════════════════════════════════════════════
#  EVT RISK MANAGER
# ═══════════════════════════════════════════════
class EVTRiskManager:
    """Extreme Value Theory ile kuyruk riski yöneticisi.

    Kullanım:
        evt = EVTRiskManager()
        # Geçmiş kayıpları/sürprizleri besle
        evt.add_observations(historical_losses)
        # Maç bazlı risk
        report = evt.analyze_match_risk(
            match_id="GS_FB",
            model_probs={"prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20},
            odds={"home": 1.80, "draw": 3.50, "away": 4.50},
            stake=100,
        )
        if report.kelly_adjustment < 1.0:
            kelly_stake *= report.kelly_adjustment
    """

    # Kuyruk indeksi eşikleri
    NORMAL_XI = 0.0
    MODERATE_XI = 0.25
    HEAVY_XI = 0.50

    # Siyah Kuğu olasılık eşiği
    BLACK_SWAN_THRESHOLD = 0.05

    def __init__(self, threshold_quantile: float = 0.90,
                 kelly_reduction: float = 0.50):
        self._threshold_q = threshold_quantile
        self._kelly_reduction = kelly_reduction
        self._observations: list[float] = []
        self._gpd_params: dict = {}
        self._fitted = False
        logger.debug("[EVT] Risk Manager başlatıldı.")

    def add_observations(self, values: list[float] | np.ndarray) -> None:
        """Geçmiş gözlemleri ekle (kayıplar, sürpriz skorları vb.)."""
        if isinstance(values, np.ndarray):
            values = values.tolist()
        self._observations.extend(values)
        self._fitted = False

    def fit(self) -> dict:
        """GPD modelini geçmiş verilere uydur."""
        if len(self._observations) < 20:
            logger.debug("[EVT] Yetersiz veri – fit atlandı.")
            return {}

        data = np.array(self._observations)
        self._gpd_params = fit_gpd(data, self._threshold_q)
        self._fitted = True
        logger.info(
            f"[EVT] GPD fit: ξ={self._gpd_params['shape']:.3f}, "
            f"σ={self._gpd_params['scale']:.3f}, "
            f"eşik={self._gpd_params['threshold']:.2f}, "
            f"aşım={self._gpd_params['exceedances']}"
        )
        return self._gpd_params

    # ═══════════════════════════════════════════
    #  MAÇ BAZLI RİSK ANALİZİ
    # ═══════════════════════════════════════════
    def analyze_match_risk(self, match_id: str = "",
                            model_probs: dict | None = None,
                            odds: dict | None = None,
                            stake: float = 100.0) -> EVTReport:
        """Tek maç için EVT risk analizi."""
        report = EVTReport(match_id=match_id)

        if not self._fitted:
            self.fit()

        if not self._gpd_params:
            report.recommendation = "Yetersiz veri – EVT analiz yapılamadı."
            report.method = "no_data"
            return report

        xi = self._gpd_params["shape"]
        sigma = self._gpd_params["scale"]
        thresh = self._gpd_params["threshold"]
        n_total = len(self._observations)
        n_exc = self._gpd_params["exceedances"]

        report.shape_xi = round(xi, 4)
        report.scale_sigma = round(sigma, 4)
        report.threshold = round(thresh, 4)
        report.n_exceedances = n_exc
        report.method = self._gpd_params.get("method", "")

        # ── VaR & CVaR ──
        report.var_95 = round(
            compute_var(xi, sigma, thresh, n_total, n_exc, 0.95) * stake / 100, 2,
        )
        report.var_99 = round(
            compute_var(xi, sigma, thresh, n_total, n_exc, 0.99) * stake / 100, 2,
        )
        report.cvar_95 = round(
            compute_cvar(report.var_95, xi, sigma, thresh), 2,
        )
        report.cvar_99 = round(
            compute_cvar(report.var_99, xi, sigma, thresh), 2,
        )

        # ── Return Levels ──
        if n_exc > 0:
            exceed_rate = n_exc / n_total
            for period, attr in [(10, "return_level_10"), (50, "return_level_50")]:
                try:
                    if abs(xi) > 1e-8:
                        rl = thresh + (sigma / xi) * (
                            (period / exceed_rate) ** xi - 1
                        )
                    else:
                        rl = thresh + sigma * np.log(period / exceed_rate)
                    setattr(report, attr, round(float(rl), 2))
                except Exception as e:
                    logger.debug(f"Exception caught: {e}")

        # ── Siyah Kuğu Olasılığı ──
        report.black_swan_prob = self._estimate_black_swan_prob(
            model_probs, odds,
        )

        # ── Kuyruk Kalınlığı ──
        if xi < self.NORMAL_XI:
            report.tail_severity = "normal"
            report.is_fat_tail = False
        elif xi < self.MODERATE_XI:
            report.tail_severity = "moderate"
            report.is_fat_tail = True
        elif xi < self.HEAVY_XI:
            report.tail_severity = "heavy"
            report.is_fat_tail = True
        else:
            report.tail_severity = "extreme"
            report.is_fat_tail = True

        # ── Kelly Adjustment ──
        if report.is_fat_tail and report.black_swan_prob > self.BLACK_SWAN_THRESHOLD:
            report.kelly_adjustment = self._kelly_reduction
        elif report.is_fat_tail:
            report.kelly_adjustment = 0.75
        else:
            report.kelly_adjustment = 1.0

        # ── Tavsiye ──
        report.recommendation = self._generate_advice(report)

        return report

    def _estimate_black_swan_prob(self, model_probs: dict | None,
                                    odds: dict | None) -> float:
        """Siyah Kuğu olasılığını tahmin et."""
        if not model_probs or not odds:
            return 0.0

        ph = model_probs.get("prob_home", 0.33)
        pd = model_probs.get("prob_draw", 0.33)
        pa = model_probs.get("prob_away", 0.34)

        oh = odds.get("home", 2.0)
        od = odds.get("draw", 3.5)
        oa = odds.get("away", 4.0)

        # Piyasa olasılıkları
        if oh > 1 and od > 1 and oa > 1:
            market_ph = 1 / oh
            market_pd = 1 / od
            market_pa = 1 / oa
            total = market_ph + market_pd + market_pa
            market_ph /= total
            market_pd /= total
            market_pa /= total
        else:
            return 0.0

        # En düşük olasılıklı sonuç → sürpriz
        min_prob = min(ph, pd, pa)

        # Model ve piyasa arasındaki maksimum fark
        max_divergence = max(
            abs(ph - market_ph),
            abs(pd - market_pd),
            abs(pa - market_pa),
        )

        # Siyah Kuğu skoru: düşük olasılık + yüksek divergans
        black_swan = min_prob * (1 + 3 * max_divergence)

        return round(min(black_swan, 1.0), 4)

    def _generate_advice(self, report: EVTReport) -> str:
        """EVT tavsiyesi."""
        parts = []

        if report.tail_severity == "extreme":
            parts.append(
                f"UYARI: Aşırı kalın kuyruk (ξ={report.shape_xi:.2f}). "
                f"Sürpriz riski çok yüksek!"
            )
        elif report.tail_severity == "heavy":
            parts.append(
                f"Kalın kuyruk tespit edildi (ξ={report.shape_xi:.2f}). "
                f"Dikkatli olun."
            )

        if report.kelly_adjustment < 1.0:
            parts.append(
                f"Kelly stake %{(1 - report.kelly_adjustment) * 100:.0f} düşürüldü."
            )

        parts.append(
            f"VaR(%99): {report.var_99:.1f}, "
            f"CVaR(%99): {report.cvar_99:.1f}"
        )

        return " | ".join(parts) if parts else "Normal kuyruk – standart devam."

    # ═══════════════════════════════════════════
    #  PORTFÖY VaR
    # ═══════════════════════════════════════════
    def portfolio_var(self, bets: list[dict]) -> PortfolioVaR:
        """Portföy bazlı VaR hesapla."""
        result = PortfolioVaR()

        if not bets:
            return result

        component_losses = []
        for bet in bets:
            stake = bet.get("stake", 100)
            odds = bet.get("odds", 2.0)
            prob = bet.get("confidence", 0.5)

            # En kötü durum: tam kayıp
            max_loss = stake
            expected_loss = stake * (1 - prob)
            component_losses.append(expected_loss)

            result.component_vars.append({
                "match_id": bet.get("match_id", ""),
                "var_95": round(expected_loss * 1.65, 2),
                "var_99": round(expected_loss * 2.33, 2),
                "max_loss": stake,
            })

        # Toplam (korelasyonsuz varsayım)
        losses = np.array(component_losses)
        total_std = float(np.sqrt(np.sum(losses ** 2)))
        result.total_var_95 = round(total_std * 1.65, 2)
        result.total_var_99 = round(total_std * 2.33, 2)
        result.total_cvar_95 = round(result.total_var_95 * 1.2, 2)

        # Basit toplam vs portföy VaR
        naive_var = float(np.sum(losses)) * 2.33
        if naive_var > 0:
            result.diversification_benefit = round(
                1 - result.total_var_99 / naive_var, 4,
            )

        result.max_loss_scenario = round(float(np.sum(
            [b.get("stake", 100) for b in bets]
        )), 2)

        result.recommendation = (
            f"Portföy VaR(%99): {result.total_var_99:.0f} TL. "
            f"Max kayıp: {result.max_loss_scenario:.0f} TL. "
            f"Diversifikasyon faydası: {result.diversification_benefit:.0%}"
        )

        return result

    # ═══════════════════════════════════════════
    #  FİLTRE
    # ═══════════════════════════════════════════
    def adjust_kelly_stakes(self, bets: list[dict]) -> list[dict]:
        """Kalın kuyruk riskine göre Kelly stake'leri ayarla."""
        for bet in bets:
            if not isinstance(bet, dict):
                continue

            report = self.analyze_match_risk(
                match_id=bet.get("match_id", ""),
                model_probs={
                    "prob_home": bet.get("prob_home", 0.33),
                    "prob_draw": bet.get("prob_draw", 0.33),
                    "prob_away": bet.get("prob_away", 0.34),
                },
                odds={
                    "home": bet.get("odds", 2.0),
                    "draw": bet.get("draw_odds", 3.5),
                    "away": bet.get("away_odds", 4.0),
                },
                stake=bet.get("stake", 100),
            )

            if report.kelly_adjustment < 1.0:
                original = bet.get("kelly_stake", bet.get("stake", 100))
                bet["evt_kelly_adj"] = report.kelly_adjustment
                bet["evt_original_stake"] = original
                bet["kelly_stake"] = round(original * report.kelly_adjustment, 2)
                bet["evt_tail"] = report.tail_severity
                bet["evt_var99"] = report.var_99
                bet["evt_black_swan"] = report.black_swan_prob
                logger.info(
                    f"[EVT] {bet.get('match_id','')}: "
                    f"Kelly {original:.0f} → {bet['kelly_stake']:.0f} "
                    f"(kuyruk={report.tail_severity}, "
                    f"ξ={report.shape_xi:.2f})"
                )

        return bets
