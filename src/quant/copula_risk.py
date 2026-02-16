"""
copula_risk.py – Copula Fonksiyonları ile Bağımlılık Modellemesi.

Kombine (parlay) kupon yapıyorsanız, maçların birbirinden bağımsız
olduğunu varsaymak hatadır. Aynı ligdeki maçlar veya aynı saatteki
favoriler birbirine "görünmez bağlarla" bağlıdır.

Korelasyon katsayısı → sadece doğrusal ilişki
Copula → kuyruk bağımlılığı (Tail Dependence)
  "Sürpriz haftasında her şey beraber mi batıyor?"

Copula Aileleri:
  - Gaussian Copula: Normal bağımlılık (basit)
  - Clayton Copula: Alt kuyruk bağımlılığı (birlikte batma)
  - Gumbel Copula: Üst kuyruk bağımlılığı (birlikte çıkma)
  - Frank Copula: Simetrik bağımlılık
  - t-Copula: Ağır kuyruk (tail risk)

Sinyal:
  Negatif kuyruk bağımlılığı varsa → aynı kupona koymayı YASAKLA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from copulas.bivariate import Clayton, Frank, Gumbel
    from copulas.multivariate import GaussianMultivariate
    COPULAS_OK = True
except ImportError:
    COPULAS_OK = False
    logger.info("copulas yüklü değil – empirical bağımlılık analizi aktif.")

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
    dangerous_pairs: list[DependencyResult] = field(default_factory=list)
    safe_combinations: list[list[str]] = field(default_factory=list)
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
        # Geçmiş sonuçlarla eğit
        analyzer.fit(historical_results)
        # Kupon adaylarını kontrol et
        report = analyzer.analyze_coupon(coupon_matches)
    """

    # Tehlike eşikleri
    TAIL_DANGER_THRESHOLD = 0.30     # Kuyruk bağımlılığı > %30 → tehlikeli
    CORRELATION_DANGER = 0.50        # Korelasyon > %50 → aynı kupona koyma
    MAX_CORRECTION = 0.50            # Copula düzeltmesi max %50 düşürebilir

    def __init__(self, copula_family: str = "auto"):
        """
        Args:
            copula_family: 'gaussian', 'clayton', 'frank', 'gumbel', 'auto'
        """
        self._family = copula_family
        self._fitted = False
        self._historical: np.ndarray | None = None
        self._copula_model: Any = None
        self._n_matches = 0
        logger.debug(f"CopulaRiskAnalyzer başlatıldı (family={copula_family}).")

    # ═══════════════════════════════════════════
    #  EĞİTİM
    # ═══════════════════════════════════════════
    def fit(self, results: list[dict], outcome_key: str = "result_binary"):
        """Geçmiş sonuçlarla bağımlılık yapısını öğren.

        results: [
            {"match_id": "m1", "league": "super_lig", "week": 1,
             "result_binary": 1,  # 1=beklenen oldu, 0=sürpriz
             "odds": 1.80, "prob": 0.55},
            ...
        ]
        """
        if len(results) < 30:
            logger.warning("[Copula] Yetersiz veri (< 30 maç).")
            return

        # Haftalık sonuçları matris haline getir
        weeks = {}
        for r in results:
            week = r.get("week", r.get("round", 0))
            if week not in weeks:
                weeks[week] = []
            weeks[week].append(r.get(outcome_key, 0))

        # En az 2 maçlık haftaları al
        matrix_rows = [v for v in weeks.values() if len(v) >= 2]
        if not matrix_rows:
            logger.warning("[Copula] Yeterli çoklu maç haftası yok.")
            return

        # Sabit genişliğe pad
        max_len = max(len(row) for row in matrix_rows)
        padded = np.array([
            row + [0.5] * (max_len - len(row)) for row in matrix_rows
        ], dtype=np.float64)

        self._historical = padded
        self._n_matches = max_len
        self._fitted = True

        # Copula model (varsa)
        if COPULAS_OK and padded.shape[1] >= 2:
            try:
                self._copula_model = GaussianMultivariate()
                import pandas as pd
                df = pd.DataFrame(padded, columns=[f"m{i}" for i in range(padded.shape[1])])
                self._copula_model.fit(df)
                logger.info(f"[Copula] Gaussian Copula eğitildi: {padded.shape}")
            except Exception as e:
                logger.debug(f"[Copula] Model eğitim hatası: {e}")
                self._copula_model = None

        logger.info(f"[Copula] {len(matrix_rows)} hafta, {max_len} maç/hafta.")

    # ═══════════════════════════════════════════
    #  İKİLİ BAĞIMLILIK ANALİZİ
    # ═══════════════════════════════════════════
    def analyze_pair(self, results_a: list[float],
                      results_b: list[float],
                      match_a: str = "", match_b: str = "") -> DependencyResult:
        """İki maç serisi arasındaki bağımlılığı ölç."""
        result = DependencyResult(match_a=match_a, match_b=match_b)

        n = min(len(results_a), len(results_b))
        if n < 10:
            return result

        a = np.array(results_a[:n], dtype=np.float64)
        b = np.array(results_b[:n], dtype=np.float64)

        # Pearson korelasyon
        if np.std(a) > 0 and np.std(b) > 0:
            result.pearson = float(np.corrcoef(a, b)[0, 1])

        # Kendall tau (sıralama korelasyonu)
        if SCIPY_OK:
            try:
                tau, _ = kendalltau(a, b)
                result.kendall_tau = float(tau) if not np.isnan(tau) else 0.0
            except Exception:
                pass

            try:
                rho, _ = spearmanr(a, b)
                result.spearman = float(rho) if not np.isnan(rho) else 0.0
            except Exception:
                pass

        # Kuyruk bağımlılığı (empirical)
        lower_tail, upper_tail = self._empirical_tail_dependence(a, b)
        result.lower_tail = round(lower_tail, 3)
        result.upper_tail = round(upper_tail, 3)

        # Copula fit (ikili)
        if COPULAS_OK:
            result.copula_family, result.copula_param = self._fit_best_copula(a, b)

        # Tehlike değerlendirme
        is_dangerous = (
            result.lower_tail > self.TAIL_DANGER_THRESHOLD or
            abs(result.pearson) > self.CORRELATION_DANGER or
            abs(result.kendall_tau) > self.CORRELATION_DANGER
        )
        result.is_dangerous = is_dangerous

        if is_dangerous:
            reasons = []
            if result.lower_tail > self.TAIL_DANGER_THRESHOLD:
                reasons.append(
                    f"Alt kuyruk bağımlılığı yüksek ({result.lower_tail:.0%}): "
                    f"sürpriz haftasında ikisi de batar"
                )
            if abs(result.pearson) > self.CORRELATION_DANGER:
                reasons.append(f"Korelasyon yüksek ({result.pearson:.2f})")
            result.danger_reason = "; ".join(reasons)

        return result

    def _empirical_tail_dependence(self, a: np.ndarray, b: np.ndarray,
                                    quantile: float = 0.1) -> tuple[float, float]:
        """Empirical kuyruk bağımlılığı."""
        n = len(a)
        if n < 20:
            return 0.0, 0.0

        # Alt kuyruk: ikisi de kötü performans gösterdiğinde
        q_low = np.quantile(a, quantile)
        mask_a_low = a <= q_low
        if mask_a_low.sum() > 0:
            b_given_a_low = b[mask_a_low]
            q_b_low = np.quantile(b, quantile)
            lower_tail = float(np.mean(b_given_a_low <= q_b_low))
        else:
            lower_tail = 0.0

        # Üst kuyruk
        q_high = np.quantile(a, 1 - quantile)
        mask_a_high = a >= q_high
        if mask_a_high.sum() > 0:
            b_given_a_high = b[mask_a_high]
            q_b_high = np.quantile(b, 1 - quantile)
            upper_tail = float(np.mean(b_given_a_high >= q_b_high))
        else:
            upper_tail = 0.0

        return lower_tail, upper_tail

    def _fit_best_copula(self, a: np.ndarray, b: np.ndarray
                          ) -> tuple[str, float]:
        """En iyi copula ailesini seç."""
        if not COPULAS_OK:
            return ("none", 0.0)

        # Uniform marginals'a dönüştür (probability integral transform)
        try:
            if SCIPY_OK:
                u = stats.rankdata(a) / (len(a) + 1)
                v = stats.rankdata(b) / (len(b) + 1)
            else:
                u = np.argsort(np.argsort(a)).astype(float) / (len(a) + 1)
                v = np.argsort(np.argsort(b)).astype(float) / (len(b) + 1)

            import pandas as pd
            data = pd.DataFrame({"u": u, "v": v})

            best_family = "gaussian"
            best_param = 0.0

            for CopulaClass, name in [
                (Clayton, "clayton"),
                (Frank, "frank"),
                (Gumbel, "gumbel"),
            ]:
                try:
                    cop = CopulaClass()
                    cop.fit(data)
                    best_family = name
                    best_param = getattr(cop, "theta", 0)
                    break
                except Exception:
                    continue

            return best_family, float(best_param)

        except Exception:
            return ("none", 0.0)

    # ═══════════════════════════════════════════
    #  KUPON ANALİZİ
    # ═══════════════════════════════════════════
    def analyze_coupon(self, matches: list[dict]) -> CouponRiskReport:
        """Kupon adaylarının bağımlılık riskini analiz et.

        matches: [
            {"match_id": "m1", "prob": 0.60, "odds": 1.80,
             "league": "super_lig", "home_team": "GS", ...},
            ...
        ]
        """
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

                # Aynı lig / aynı gün → potansiyel bağımlılık
                same_league = ma.get("league", "") == mb.get("league", "")
                same_day = ma.get("date", "") == mb.get("date", "")

                # Simüle edilmiş bağımlılık (gerçek veriden)
                if self._fitted and self._historical is not None:
                    # Geçmiş veriden korelasyon tahmin et
                    dep = self._estimate_dependency(ma, mb)
                else:
                    dep = self._heuristic_dependency(ma, mb, same_league, same_day)

                report.pairs_analyzed += 1

                if dep.is_dangerous:
                    dangerous_pairs.append(dep)

                correction_factors.append(dep.correction_factor)

        report.dangerous_pairs = dangerous_pairs

        # Copula düzeltilmiş birleşik olasılık
        avg_correction = float(np.mean(correction_factors)) if correction_factors else 1.0
        avg_correction = max(avg_correction, self.MAX_CORRECTION)
        copula_combined = naive_combined * avg_correction
        report.copula_combined_prob = round(copula_combined, 6)
        report.risk_adjustment = round(avg_correction, 3)

        # Kuyruk riski
        if dangerous_pairs:
            report.overall_tail_risk = max(
                d.lower_tail for d in dangerous_pairs
            )

        # Tavsiye
        if dangerous_pairs:
            pairs_str = ", ".join(
                f"{d.match_a}↔{d.match_b}" for d in dangerous_pairs[:3]
            )
            report.recommendation = (
                f"UYARI: {len(dangerous_pairs)} tehlikeli çift tespit edildi ({pairs_str}). "
                f"Bu maçları aynı kupona koymayın! "
                f"Copula düzeltme: {avg_correction:.0%}"
            )
        else:
            report.recommendation = (
                f"Kupon güvenli görünüyor. "
                f"Düzeltme faktörü: {avg_correction:.0%}"
            )

        report.method = "copula" if COPULAS_OK else "empirical"

        return report

    def _estimate_dependency(self, ma: dict, mb: dict) -> DependencyResult:
        """Geçmiş veriden bağımlılık tahmin et."""
        dep = DependencyResult(
            match_a=ma.get("match_id", ""),
            match_b=mb.get("match_id", ""),
        )

        if self._historical is None or self._historical.shape[1] < 2:
            return dep

        # Geçmiş matrisin sütunlarından korelasyon al
        col_a = 0
        col_b = min(1, self._historical.shape[1] - 1)
        a_vals = self._historical[:, col_a]
        b_vals = self._historical[:, col_b]

        if np.std(a_vals) > 0 and np.std(b_vals) > 0:
            dep.pearson = float(np.corrcoef(a_vals, b_vals)[0, 1])

        lower, upper = self._empirical_tail_dependence(a_vals, b_vals)
        dep.lower_tail = lower
        dep.upper_tail = upper

        dep.is_dangerous = (
            lower > self.TAIL_DANGER_THRESHOLD or
            abs(dep.pearson) > self.CORRELATION_DANGER
        )

        # Correction factor
        if dep.is_dangerous:
            dep.correction_factor = 1.0 - lower * 0.5
            dep.danger_reason = f"Geçmiş veri: alt kuyruk={lower:.0%}"
        else:
            dep.correction_factor = 1.0

        return dep

    def _heuristic_dependency(self, ma: dict, mb: dict,
                                same_league: bool, same_day: bool
                                ) -> DependencyResult:
        """Veri yoksa heuristic bağımlılık tahmini."""
        dep = DependencyResult(
            match_a=ma.get("match_id", ""),
            match_b=mb.get("match_id", ""),
        )

        risk_score = 0.0

        # Aynı lig → bağımlılık riski
        if same_league:
            risk_score += 0.15

        # Aynı gün → psikolojik bağımlılık
        if same_day:
            risk_score += 0.05

        # İkisi de favori → sürpriz haftası riski
        prob_a = ma.get("prob", 0.5)
        prob_b = mb.get("prob", 0.5)
        if prob_a > 0.60 and prob_b > 0.60:
            risk_score += 0.10

        # Aynı ülke
        if ma.get("country", "") == mb.get("country", "") and ma.get("country", ""):
            risk_score += 0.05

        dep.lower_tail = round(min(risk_score, 0.50), 3)
        dep.pearson = round(risk_score * 0.5, 3)

        dep.is_dangerous = risk_score > self.TAIL_DANGER_THRESHOLD
        if dep.is_dangerous:
            dep.danger_reason = f"Heuristic risk: {risk_score:.0%}"

        dep.correction_factor = max(1.0 - risk_score * 0.3, self.MAX_CORRECTION)

        return dep

    def filter_safe_coupon(self, matches: list[dict],
                            max_size: int = 5) -> list[dict]:
        """Güvenli kupon kombinasyonunu seç (tehlikeli çiftleri çıkar)."""
        report = self.analyze_coupon(matches)

        if not report.dangerous_pairs:
            return matches[:max_size]

        # Tehlikeli maçları tespit et
        dangerous_ids = set()
        for dp in report.dangerous_pairs:
            dangerous_ids.add(dp.match_b)  # İkincisini çıkar

        safe = [m for m in matches if m.get("match_id", "") not in dangerous_ids]
        return safe[:max_size]
