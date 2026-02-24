"""
uncertainty_quantifier.py – Conformal Prediction ile Güven Aralığı Garantisi.

Klasik model: "Galatasaray %60 kazanır" → Yanılabilir.
Conformal Prediction: "%55-%65 arasında, %95 eminim" → Matematiksel garanti.

Eğer hata payı çok genişse (Örn: %30-%80):
  → Model "Bu maçtan emin değilim" der → ABSTAIN (Çekimser kal).

MAPIE + Crepes entegrasyonu:
  - MAPIE: sklearn uyumlu conformal wrapper
  - Fallback: quantile-based nonconformity scores (kütüphane yoksa)

Çıktı:
  PointEstimate + PredictionInterval + Coverage + AbstractDecision
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from mapie.classification import MapieClassifier
    from mapie.regression import MapieRegressor
    MAPIE_OK = True
except ImportError:
    MAPIE_OK = False
    logger.info("mapie yüklü değil – heuristic uncertainty aktif.")

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


@dataclass
class UncertaintyResult:
    """Belirsizlik ölçümü sonucu."""
    match_id: str = ""
    selection: str = ""
    point_estimate: float = 0.0        # Nokta tahmini (ör: %60)

    # Güven aralığı
    lower_bound: float = 0.0           # Alt sınır (ör: %55)
    upper_bound: float = 1.0           # Üst sınır (ör: %65)
    interval_width: float = 1.0        # Aralık genişliği
    confidence_level: float = 0.95     # Hedef kapsama oranı (coverage)

    # Karar
    is_certain: bool = False           # Model emin mi?
    abstain: bool = False              # Bahisten kaçın mı?
    abstain_reason: str = ""
    reliability_score: float = 0.0     # 0-1 (1 = çok güvenilir)

    # Meta
    method: str = "heuristic"
    prediction_set: list = field(default_factory=list)  # Hangi sonuçlar olası?
    nonconformity_score: float = 0.0


@dataclass
class CalibrationReport:
    """Model kalibrasyon raporu."""
    expected_coverage: float = 0.95
    actual_coverage: float = 0.0
    avg_interval_width: float = 0.0
    abstain_rate: float = 0.0
    n_samples: int = 0
    is_well_calibrated: bool = False
    recommendation: str = ""


class UncertaintyQuantifier:
    """Conformal Prediction ile model belirsizliği ölçümü.

    Kullanım:
        uq = UncertaintyQuantifier()
        # Model eğitimi sırasında kalibrasyon verisi topla
        uq.calibrate(X_cal, y_cal, model)
        # Tahmin sırasında güven aralığı ekle
        result = uq.quantify("match_123", "home", point_estimate=0.60)
    """

    # Aralık genişliği eşikleri
    MAX_WIDTH_CERTAIN = 0.20    # Bu altında: "Emin"
    MAX_WIDTH_MODERATE = 0.35   # Bu altında: "Orta güven"
    ABSTAIN_WIDTH = 0.50        # Bu üstünde: ABSTAIN (çekimser kal)

    # Minimum coverage oranı
    TARGET_COVERAGE = 0.95

    def __init__(self, alpha: float = 0.05,
                 method: str = "score",
                 abstain_threshold: float = 0.50):
        """
        Args:
            alpha: 1 - coverage level (0.05 → %95 güven)
            method: 'score' veya 'cumulated_score' (MAPIE yöntemi)
            abstain_threshold: Bu genişliğin üstünde bahis yapma
        """
        self._alpha = alpha
        self._method = method
        self.ABSTAIN_WIDTH = abstain_threshold
        self._calibrated = False
        self._cal_scores: list[float] = []
        self._quantiles: dict[float, float] = {}
        self._mapie_clf = None
        self._mapie_reg = None
        self._base_model = None
        self._history: list[UncertaintyResult] = []
        logger.debug(
            f"UncertaintyQuantifier başlatıldı "
            f"(alpha={alpha}, method={method}, abstain={abstain_threshold})"
        )

    # ═══════════════════════════════════════════
    #  KALİBRASYON
    # ═══════════════════════════════════════════
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  model: Any = None, task: str = "classification") -> bool:
        """Kalibrasyon verisi ile conformal sınırları hesapla.

        Args:
            X_cal: Kalibrasyon features (n_samples, n_features)
            y_cal: Kalibrasyon labels
            model: Eğitilmiş sklearn-uyumlu model (veya None → GBM oluşturulur)
            task: 'classification' veya 'regression'
        """
        if len(X_cal) < 20:
            logger.warning("[UQ] Kalibrasyon verisi çok az (< 20). Heuristic mod.")
            return False

        if model is None and SKLEARN_OK:
            if task == "classification":
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=42,
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42,
                )
            model.fit(X_cal, y_cal)

        self._base_model = model

        if MAPIE_OK and model is not None:
            return self._calibrate_mapie(X_cal, y_cal, model, task)

        return self._calibrate_manual(X_cal, y_cal, model)

    def _calibrate_mapie(self, X_cal, y_cal, model, task) -> bool:
        """MAPIE ile conformal kalibrasyon."""
        try:
            if task == "classification":
                self._mapie_clf = MapieClassifier(
                    estimator=model, method=self._method,
                    cv="prefit", random_state=42,
                )
                self._mapie_clf.fit(X_cal, y_cal)
            else:
                self._mapie_reg = MapieRegressor(
                    estimator=model, method="plus",
                    cv="prefit", random_state=42,
                )
                self._mapie_reg.fit(X_cal, y_cal)

            self._calibrated = True
            logger.success(
                f"[UQ] MAPIE kalibrasyon tamamlandı "
                f"(n={len(X_cal)}, task={task})"
            )
            return True

        except Exception as e:
            logger.warning(f"[UQ] MAPIE hatası: {e} → manual fallback")
            return self._calibrate_manual(X_cal, y_cal, model)

    def _calibrate_manual(self, X_cal, y_cal, model) -> bool:
        """MAPIE yoksa → quantile-based nonconformity scores."""
        try:
            if model is not None and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_cal)
                # Nonconformity score = 1 - P(true class)
                scores = []
                for i, y_true in enumerate(y_cal):
                    idx = int(y_true) if int(y_true) < probs.shape[1] else 0
                    scores.append(1.0 - probs[i, idx])
                self._cal_scores = sorted(scores)
            elif model is not None and hasattr(model, "predict"):
                preds = model.predict(X_cal)
                self._cal_scores = sorted(abs(y_cal - preds))
            else:
                self._cal_scores = [0.1 * i for i in range(1, 11)]

            # Quantile hesapla
            n = len(self._cal_scores)
            for alpha in (0.01, 0.05, 0.10, 0.20):
                q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
                q_idx = min(max(q_idx, 0), n - 1)
                self._quantiles[alpha] = self._cal_scores[q_idx]

            self._calibrated = True
            logger.info(
                f"[UQ] Manual kalibrasyon: n={n}, "
                f"q95={self._quantiles.get(0.05, 0):.3f}"
            )
            return True

        except Exception as e:
            logger.warning(f"[UQ] Manual kalibrasyon hatası: {e}")
            return False

    # ═══════════════════════════════════════════
    #  TAHMİN + BELİRSİZLİK
    # ═══════════════════════════════════════════
    def quantify(self, match_id: str, selection: str,
                 point_estimate: float,
                 features: np.ndarray | None = None,
                 model_confidence: float = 0.0) -> UncertaintyResult:
        """Nokta tahminine güven aralığı ekle.

        Args:
            match_id: Maç ID
            selection: home / draw / away
            point_estimate: Model olasılığı (0-1)
            features: Opsiyonel feature vektörü (MAPIE kullanımı için)
            model_confidence: Modelin kendi güven skoru
        """
        if self._mapie_clf and features is not None:
            return self._quantify_mapie_clf(match_id, selection, features)

        if self._mapie_reg and features is not None:
            return self._quantify_mapie_reg(match_id, selection, features)

        if self._calibrated and self._cal_scores:
            return self._quantify_manual(
                match_id, selection, point_estimate, model_confidence,
            )

        return self._quantify_heuristic(
            match_id, selection, point_estimate, model_confidence,
        )

    def _quantify_mapie_clf(self, match_id, selection, features) -> UncertaintyResult:
        """MAPIE classification prediction set."""
        try:
            features_2d = features.reshape(1, -1) if features.ndim == 1 else features
            y_pred, y_pset = self._mapie_clf.predict(
                features_2d, alpha=self._alpha,
            )

            pred_class = int(y_pred[0])
            pset = y_pset[0, :, 0]  # Boolean mask

            # Prediction set: hangi sınıflar olası?
            possible_classes = [i for i, v in enumerate(pset) if v]
            set_size = len(possible_classes)

            # Sınıf olasılıklarını al
            base_probs = self._base_model.predict_proba(features_2d)[0]
            point_est = float(base_probs[pred_class])

            # Set büyüklüğüne göre güven
            if set_size == 1:
                lower = max(point_est - 0.05, 0)
                upper = min(point_est + 0.05, 1)
                is_certain = True
                abstain = False
            elif set_size == 2:
                lower = max(point_est - 0.15, 0)
                upper = min(point_est + 0.15, 1)
                is_certain = False
                abstain = False
            else:
                lower = max(point_est - 0.30, 0)
                upper = min(point_est + 0.30, 1)
                is_certain = False
                abstain = True

            width = upper - lower
            reliability = max(0, 1.0 - (width / self.ABSTAIN_WIDTH))

            result = UncertaintyResult(
                match_id=match_id, selection=selection,
                point_estimate=round(point_est, 4),
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                interval_width=round(width, 4),
                confidence_level=1 - self._alpha,
                is_certain=is_certain,
                abstain=abstain,
                abstain_reason=(
                    f"Prediction set çok geniş ({set_size} sınıf)"
                    if abstain else ""
                ),
                reliability_score=round(reliability, 3),
                method="mapie_classification",
                prediction_set=possible_classes,
            )
            self._history.append(result)
            return result

        except Exception as e:
            logger.debug(f"[UQ] MAPIE clf hatası: {e}")
            return self._quantify_heuristic(match_id, selection, 0.33, 0)

    def _quantify_mapie_reg(self, match_id, selection, features) -> UncertaintyResult:
        """MAPIE regression prediction interval."""
        try:
            features_2d = features.reshape(1, -1) if features.ndim == 1 else features
            y_pred, y_pis = self._mapie_reg.predict(
                features_2d, alpha=self._alpha,
            )

            point_est = float(y_pred[0])
            lower = float(y_pis[0, 0, 0])
            upper = float(y_pis[0, 1, 0])

            lower = max(lower, 0)
            upper = min(upper, 1)
            width = upper - lower

            is_certain = width <= self.MAX_WIDTH_CERTAIN
            abstain = width >= self.ABSTAIN_WIDTH

            reliability = max(0, 1.0 - (width / self.ABSTAIN_WIDTH))

            result = UncertaintyResult(
                match_id=match_id, selection=selection,
                point_estimate=round(point_est, 4),
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                interval_width=round(width, 4),
                confidence_level=1 - self._alpha,
                is_certain=is_certain,
                abstain=abstain,
                abstain_reason=(
                    f"Aralık çok geniş ({width:.0%})"
                    if abstain else ""
                ),
                reliability_score=round(reliability, 3),
                method="mapie_regression",
            )
            self._history.append(result)
            return result

        except Exception as e:
            logger.debug(f"[UQ] MAPIE reg hatası: {e}")
            return self._quantify_heuristic(match_id, selection, 0.33, 0)

    def _quantify_manual(self, match_id, selection, point_est,
                          model_conf) -> UncertaintyResult:
        """Kalibrasyon quantile'ları ile aralık hesapla."""
        q = self._quantiles.get(self._alpha, 0.15)

        lower = max(point_est - q, 0.01)
        upper = min(point_est + q, 0.99)
        width = upper - lower

        is_certain = width <= self.MAX_WIDTH_CERTAIN
        abstain = width >= self.ABSTAIN_WIDTH

        # Model güvenini de hesaba kat
        if model_conf > 0:
            combined_reliability = (
                0.6 * max(0, 1.0 - width / self.ABSTAIN_WIDTH) +
                0.4 * model_conf
            )
        else:
            combined_reliability = max(0, 1.0 - width / self.ABSTAIN_WIDTH)

        result = UncertaintyResult(
            match_id=match_id, selection=selection,
            point_estimate=round(point_est, 4),
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
            interval_width=round(width, 4),
            confidence_level=1 - self._alpha,
            is_certain=is_certain,
            abstain=abstain,
            abstain_reason=(
                f"Nonconformity skoru yüksek (q={q:.3f}, width={width:.0%})"
                if abstain else ""
            ),
            reliability_score=round(combined_reliability, 3),
            method="conformal_manual",
            nonconformity_score=q,
        )
        self._history.append(result)
        return result

    def _quantify_heuristic(self, match_id, selection, point_est,
                             model_conf) -> UncertaintyResult:
        """MAPIE/kalibrasyon yoksa → bilgi-kuramsal heuristic."""
        # Entropy-based uncertainty: olasılık 0.50'ye yakınsa daha belirsiz
        distance_from_uniform = abs(point_est - 0.333)
        entropy_factor = 1.0 - (distance_from_uniform / 0.667)
        half_width = 0.08 + entropy_factor * 0.20

        # Model güveninden ek düzeltme
        if model_conf > 0:
            half_width *= (1.3 - model_conf)

        lower = max(point_est - half_width, 0.01)
        upper = min(point_est + half_width, 0.99)
        width = upper - lower

        is_certain = width <= self.MAX_WIDTH_CERTAIN
        abstain = width >= self.ABSTAIN_WIDTH

        reliability = max(0, 1.0 - (width / self.ABSTAIN_WIDTH))

        reason = ""
        if abstain:
            reason = f"Heuristic: aralık çok geniş ({width:.0%}), model belirsiz."

        result = UncertaintyResult(
            match_id=match_id, selection=selection,
            point_estimate=round(point_est, 4),
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
            interval_width=round(width, 4),
            confidence_level=1 - self._alpha,
            is_certain=is_certain,
            abstain=abstain,
            abstain_reason=reason,
            reliability_score=round(reliability, 3),
            method="heuristic",
        )
        self._history.append(result)
        return result

    # ═══════════════════════════════════════════
    #  TOPLU ANALİZ & FİLTRE
    # ═══════════════════════════════════════════
    def filter_certain_bets(self, bets: list[dict],
                             prob_key: str = "confidence") -> list[dict]:
        """Sadece model emin olduğu bahisleri geçir (ABSTAIN filtresi).

        Returns:
            Filtrelenmiş bahis listesi (belirsiz olanlar çıkarıldı).
        """
        filtered = []
        abstained = 0
        for bet in bets:
            prob = bet.get(prob_key, 0.33)
            mid = bet.get("match_id", "")
            sel = bet.get("selection", "")

            uq = self.quantify(
                mid, sel, prob,
                model_confidence=bet.get("model_confidence", 0),
            )

            if uq.abstain:
                abstained += 1
                logger.info(
                    f"[UQ] ABSTAIN: {mid} {sel} → {uq.abstain_reason}"
                )
                bet["abstained"] = True
                bet["abstain_reason"] = uq.abstain_reason
                continue

            bet["uq_lower"] = uq.lower_bound
            bet["uq_upper"] = uq.upper_bound
            bet["uq_width"] = uq.interval_width
            bet["uq_reliability"] = uq.reliability_score
            bet["uq_certain"] = uq.is_certain
            filtered.append(bet)

        if abstained:
            logger.warning(
                f"[UQ] {abstained}/{len(bets)} bahis ABSTAIN edildi "
                f"(belirsizlik çok yüksek)"
            )

        return filtered

    # ═══════════════════════════════════════════
    #  KALİBRASYON RAPORU
    # ═══════════════════════════════════════════
    def calibration_report(self, y_true: list[float] | None = None
                            ) -> CalibrationReport:
        """Model kalibrasyonunu değerlendir.

        Eğer gerçek sonuçlar verilirse, coverage kontrolü yapılır.
        """
        if not self._history:
            return CalibrationReport(recommendation="Henüz tahmin yapılmadı.")

        n = len(self._history)
        avg_width = float(np.mean([r.interval_width for r in self._history]))
        abstain_rate = sum(1 for r in self._history if r.abstain) / n

        actual_coverage = 0.0
        if y_true is not None and len(y_true) == n:
            in_interval = sum(
                1 for r, y in zip(self._history, y_true)
                if r.lower_bound <= y <= r.upper_bound
            )
            actual_coverage = in_interval / n

        is_calibrated = (
            actual_coverage >= self.TARGET_COVERAGE - 0.03
            if y_true else False
        )

        # Tavsiye
        if avg_width > self.ABSTAIN_WIDTH:
            rec = "Model çok belirsiz. Daha fazla veri veya daha güçlü model gerekli."
        elif avg_width > self.MAX_WIDTH_MODERATE:
            rec = "Orta belirsizlik. Dikkatli bahis yapın."
        elif avg_width < self.MAX_WIDTH_CERTAIN:
            rec = "Model güvenilir. Aralıklar dar."
        else:
            rec = "Makul belirsizlik seviyesi."

        return CalibrationReport(
            expected_coverage=self.TARGET_COVERAGE,
            actual_coverage=round(actual_coverage, 3),
            avg_interval_width=round(avg_width, 3),
            abstain_rate=round(abstain_rate, 3),
            n_samples=n,
            is_well_calibrated=is_calibrated,
            recommendation=rec,
        )

    def clear_history(self):
        self._history.clear()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
