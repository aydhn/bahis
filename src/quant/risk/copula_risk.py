"""
copula_risk.py – Copula Fonksiyonları ile Bağımlılık Modellemesi.

Kombine (parlay) kupon yapıyorsanız, maçların birbirinden bağımsız
olduğunu varsaymak hatadır. Aynı ligdeki maçlar veya aynı saatteki
favoriler birbirine "görünmez bağlarla" bağlıdır.

Sinyal:
  Negatif kuyruk bağımlılığı varsa → aynı kupona koymayı YASAKLA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Dict

import numpy as np
from loguru import logger

try:
    from copulas.bivariate import Clayton, Frank, Gumbel
    from copulas.multivariate import GaussianMultivariate
    COPULAS_OK = True
except ImportError:
    COPULAS_OK = False
    logger.debug("copulas yüklü değil – empirical bağımlılık analizi aktif.")

try:
    from scipy import stats
    from scipy.stats import kendalltau, spearmanr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


@dataclass
class DependencyResult:
    """İki maç arasındaki bağımlılık analizi."""
    match_a: str = ""
    match_b: str = ""
    pearson: float = 0.0
    kendall_tau: float = 0.0
    spearman: float = 0.0
    lower_tail: float = 0.0     # Alt kuyruk bağımlılığı (birlikte batma)
    upper_tail: float = 0.0     # Üst kuyruk bağımlılığı (birlikte kazanma)
    copula_family: str = ""
    copula_param: float = 0.0
    is_dangerous: bool = False   # Aynı kupona konulmamalı mı?
    danger_reason: str = ""
    combined_prob_naive: float = 0.0   # P(A) * P(B) (naif)
    combined_prob_copula: float = 0.0  # Copula düzeltilmiş
    correction_factor: float = 1.0     # copula / naive


@dataclass
class CouponRiskReport:
    """Kupon bazlı bağımlılık raporu."""
    n_matches: int = 0
    pairs_analyzed: int = 0
    dangerous_pairs: List[DependencyResult] = field(default_factory=list)
    safe_combinations: List[List[str]] = field(default_factory=list)
    overall_tail_risk: float = 0.0
    naive_combined_prob: float = 0.0
    copula_combined_prob: float = 0.0
    risk_adjustment: float = 1.0
    recommendation: str = ""
    method: str = "empirical"


class CopulaRiskAnalyzer:
    """Copula ile kupon bağımlılık analizi.

    Kullanım:
        analyzer = CopulaRiskAnalyzer()
        # Kupon adaylarını kontrol et
        report = analyzer.analyze_coupon(coupon_matches)
    """

    # Tehlike eşikleri
    TAIL_DANGER_THRESHOLD = 0.30     # Kuyruk bağımlılığı > %30 → tehlikeli
    CORRELATION_DANGER = 0.50        # Korelasyon > %50 → aynı kupona koyma
    MAX_CORRECTION = 0.50            # Copula düzeltmesi max %50 düşürebilir

    def __init__(self, copula_family: str = "auto"):
        self._family = copula_family
        logger.debug(f"CopulaRiskAnalyzer başlatıldı (family={copula_family}).")

    def analyze_coupon(self, matches: List[Dict[str, Any]]) -> CouponRiskReport:
        """Kupon adaylarının bağımlılık riskini analiz et."""
        n = len(matches)
        report = CouponRiskReport(n_matches=n)

        if n < 2:
            report.recommendation = "Tek maç – bağımlılık analizi gereksiz."
            return report

        # Naif birleşik olasılık
        probs = [m.get("prob", 0.5) for m in matches]
        naive_combined = float(np.prod(probs))
        report.naive_combined_prob = round(naive_combined, 6)

        # Her çift için bağımlılık analizi
        dangerous_pairs = []
        correction_factors = []

        for i in range(n):
            for j in range(i + 1, n):
                ma = matches[i]
                mb = matches[j]

                # Heuristic Analiz (Futbolda en mantıklısı)
                dep = self._heuristic_dependency(ma, mb)

                report.pairs_analyzed += 1

                if dep.is_dangerous:
                    dangerous_pairs.append(dep)

                correction_factors.append(dep.correction_factor)

        report.dangerous_pairs = dangerous_pairs

        # Copula düzeltilmiş birleşik olasılık
        avg_correction = float(np.mean(correction_factors)) if correction_factors else 1.0
        # Çok agresif düşüşü engelle
        avg_correction = max(avg_correction, self.MAX_CORRECTION)

        copula_combined = naive_combined * avg_correction
        report.copula_combined_prob = round(copula_combined, 6)
        report.risk_adjustment = round(avg_correction, 3)

        # Kuyruk riski
        if dangerous_pairs:
            report.overall_tail_risk = max(d.lower_tail for d in dangerous_pairs)

        # Tavsiye
        if dangerous_pairs:
            pairs_str = ", ".join(f"{d.match_a}↔{d.match_b}" for d in dangerous_pairs[:3])
            report.recommendation = (
                f"UYARI: {len(dangerous_pairs)} tehlikeli çift tespit edildi ({pairs_str}). "
                f"Bu maçları aynı kupona koymayın! "
                f"Risk düzeltme: {avg_correction:.0%}"
            )
        else:
            report.recommendation = (
                f"Kupon güvenli görünüyor. "
                f"Düzeltme faktörü: {avg_correction:.0%}"
            )

        report.method = "heuristic" # Şimdilik sadece heuristic

        return report

    def _heuristic_dependency(self, ma: Dict[str, Any], mb: Dict[str, Any]) -> DependencyResult:
        """Veri yoksa heuristic bağımlılık tahmini."""
        dep = DependencyResult(
            match_a=ma.get("match_id", "A"),
            match_b=mb.get("match_id", "B"),
        )

        risk_score = 0.0
        reasons = []

        # 1. Aynı Lig
        if ma.get("league") == mb.get("league") and ma.get("league"):
            risk_score += 0.15
            reasons.append("Aynı Lig")

        # 2. Aynı Gün
        date_a = ma.get("date", "").split(" ")[0]
        date_b = mb.get("date", "").split(" ")[0]
        if date_a == date_b and date_a:
            risk_score += 0.05
            reasons.append("Aynı Gün")

        # 3. İkisi de Favori (Sürpriz Haftası Riski)
        # Eğer ikisi de yüksek olasılıklıysa (prob > 0.60)
        # Sürpriz haftalarında tüm favoriler aynı anda yatabilir.
        prob_a = ma.get("confidence", ma.get("prob", 0.5))
        prob_b = mb.get("confidence", mb.get("prob", 0.5))

        if prob_a > 0.65 and prob_b > 0.65:
            risk_score += 0.10
            reasons.append("Çifte Favori Riski")

        dep.lower_tail = round(min(risk_score, 0.50), 3)
        dep.pearson = round(risk_score * 0.5, 3) # Korelasyon tahmini

        dep.is_dangerous = risk_score > self.TAIL_DANGER_THRESHOLD

        if dep.is_dangerous:
            dep.danger_reason = " + ".join(reasons) + f" (Score: {risk_score:.2f})"
            # Tehlike varsa düzeltme faktörü uygula
            dep.correction_factor = max(1.0 - risk_score, 0.5)
        else:
            dep.correction_factor = 1.0

        return dep

    def filter_safe_coupon(self, matches: List[Dict[str, Any]], max_size: int = 5) -> List[Dict[str, Any]]:
        """Güvenli kupon kombinasyonunu seç (tehlikeli çiftleri çıkar)."""
        report = self.analyze_coupon(matches)

        if not report.dangerous_pairs:
            return matches[:max_size]

        # Tehlikeli maçları tespit et (basitçe ikincisini çıkar)
        dangerous_ids = set()
        for dp in report.dangerous_pairs:
            dangerous_ids.add(dp.match_b)

        safe = [m for m in matches if m.get("match_id") not in dangerous_ids]
        return safe[:max_size]
