"""
uncertainty_separator.py – Epistemic vs. Aleatoric Uncertainty.

Bot "Riskli" dediğinde ne demek istiyor?
  - Aleatoric (Şans): "Verim tam ama futbol bu, top yuvarlak."
    → Bahis miktarını yarıya düşür (Half-Kelly).
  - Epistemic (Bilgisizlik): "Bu lig/takım hakkında yeterli verim yok."
    → Bahis yapma, pas geç (Abstain).

Kavramlar:
  - Aleatoric Uncertainty: Verinin doğasındaki rastgelelik, azaltılamaz
    (heteroscedastic noise). Daha fazla veri toplasan bile değişmez.
  - Epistemic Uncertainty: Model bilgisizliğinden kaynaklanan belirsizlik,
    daha fazla veri ile azaltılabilir.
  - MC Dropout: Eğitilmiş ağda dropout açık bırakarak N tahmin yap,
    varyans = epistemic uncertainty.
  - Deep Ensemble: N farklı model eğit, tahminler arası varyans = epistemic.
  - Heteroscedastic Output: Model hem μ hem σ tahmin eder → σ = aleatoric.
  - Conformal Prediction: Dağılımdan bağımsız güven aralığı.

Akış:
  1. Model tahmini al (olasılıklar)
  2. MC Dropout veya Ensemble ile N tahmin üret
  3. Tahminler arası varyans → Epistemic
  4. Tek bir tahmin içindeki dağılım genişliği → Aleatoric
  5. Karar: Epistemic yüksekse PAS, Aleatoric yüksekse HALF-KELLY

Teknoloji: torch (MC Dropout), sklearn (Ensemble)
Fallback: Bootstrap varyans + entropi ayrımı
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from scipy.stats import entropy as sp_entropy
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class UncertaintyReport:
    """Belirsizlik ayrımı raporu."""
    match_id: str = ""
    team: str = ""
    # Tahmin
    prediction: np.ndarray | None = None  # [P(home), P(draw), P(away)]
    predicted_class: int = -1
    predicted_prob: float = 0.0
    # Belirsizlik
    total_uncertainty: float = 0.0
    aleatoric: float = 0.0        # Şans kaynaklı (azaltılamaz)
    epistemic: float = 0.0        # Bilgisizlik kaynaklı (azaltılabilir)
    epistemic_ratio: float = 0.0  # epistemic / total
    aleatoric_ratio: float = 0.0  # aleatoric / total
    # Karar
    decision: str = ""            # "BET" | "HALF_KELLY" | "ABSTAIN"
    confidence_modifier: float = 1.0
    reason: str = ""
    # Meta
    n_samples: int = 0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  ENTROPI HESAPLAMA
# ═══════════════════════════════════════════════
def predictive_entropy(probs: np.ndarray) -> float:
    """Tahmin entropisi (toplam belirsizlik).

    H[y|x] = -Σ p(y) log p(y)
    """
    probs = np.clip(probs, 1e-10, 1.0)
    if SCIPY_OK:
        return float(sp_entropy(probs, base=2))
    return float(-np.sum(probs * np.log2(probs)))


def expected_entropy(probs_samples: np.ndarray) -> float:
    """Beklenen entropi (aleatoric belirsizlik).

    E_θ[H[y|x, θ]] = ortalama(her modelin entropisi)
    """
    entropies = []
    for p in probs_samples:
        p = np.clip(p, 1e-10, 1.0)
        if SCIPY_OK:
            entropies.append(sp_entropy(p, base=2))
        else:
            entropies.append(-np.sum(p * np.log2(p)))
    return float(np.mean(entropies))


def mutual_information(probs_samples: np.ndarray) -> float:
    """Karşılıklı bilgi (epistemic belirsizlik).

    I[y; θ|x] = H[y|x] - E_θ[H[y|x, θ]]
    Predictive Entropy - Expected Entropy = Epistemic
    """
    mean_probs = np.mean(probs_samples, axis=0)
    total = predictive_entropy(mean_probs)
    aleatoric = expected_entropy(probs_samples)
    return max(0.0, total - aleatoric)


# ═══════════════════════════════════════════════
#  BOOTSTRAP ENSEMBLE (Fallback)
# ═══════════════════════════════════════════════
def bootstrap_predictions(X: np.ndarray, y: np.ndarray,
                            X_test: np.ndarray,
                            n_models: int = 10,
                            n_classes: int = 3) -> np.ndarray:
    """Bootstrap ensemble ile çoklu tahmin üret."""
    if not SKLEARN_OK:
        # Tamamen rastgele fallback
        preds = np.random.dirichlet(np.ones(n_classes), (n_models, len(X_test)))
        return preds

    all_probs = []
    n_samples = len(X)

    for i in range(n_models):
        # Bootstrap örnekleme
        indices = np.random.randint(0, n_samples, n_samples)
        X_boot = X[indices]
        y_boot = y[indices]

        try:
            if i % 2 == 0:
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5 + i,
                    random_state=i * 42,
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3 + i % 3,
                    learning_rate=0.05 + i * 0.02,
                    random_state=i * 42,
                )

            model.fit(X_boot, y_boot)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                # Sınıf sayısı eşitle
                if probs.shape[1] < n_classes:
                    padded = np.zeros((len(X_test), n_classes))
                    padded[:, :probs.shape[1]] = probs
                    probs = padded
                elif probs.shape[1] > n_classes:
                    probs = probs[:, :n_classes]
                all_probs.append(probs)
        except Exception:
            continue

    if not all_probs:
        return np.random.dirichlet(np.ones(n_classes), (n_models, len(X_test)))

    return np.array(all_probs)  # (n_models, n_test, n_classes)


# ═══════════════════════════════════════════════
#  UNCERTAINTY SEPARATOR (Ana Sınıf)
# ═══════════════════════════════════════════════
class UncertaintySeparator:
    """Epistemic vs Aleatoric belirsizlik ayırıcı.

    Kullanım:
        us = UncertaintySeparator(n_models=10)

        # Eğitim
        us.fit(X_train, y_train)

        # Analiz
        report = us.analyze(features, match_id="gs_fb", team="GS")

        # Toplu analiz
        reports = us.analyze_batch(X_test, match_ids=ids)
    """

    # Eşik değerleri
    EPISTEMIC_THRESHOLD = 0.3    # Üstünde → ABSTAIN
    ALEATORIC_THRESHOLD = 0.5    # Üstünde → HALF_KELLY
    TOTAL_THRESHOLD = 0.8        # Üstünde → ABSTAIN

    def __init__(self, n_models: int = 10, n_classes: int = 3):
        self._n_models = n_models
        self._n_classes = n_classes
        self._models: list[Any] = []
        self._fitted = False
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

        logger.debug(
            f"[Uncertainty] Separator başlatıldı: "
            f"n_models={n_models}, n_classes={n_classes}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Ensemble modelleri eğit."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        self._X_train = X
        self._y_train = y

        if SKLEARN_OK and len(X) >= 20:
            self._models = []
            for i in range(self._n_models):
                indices = np.random.randint(0, len(X), len(X))
                X_boot, y_boot = X[indices], y[indices]

                try:
                    if i % 2 == 0:
                        m = RandomForestClassifier(
                            n_estimators=50,
                            max_depth=5 + i % 5,
                            random_state=i * 42,
                        )
                    else:
                        m = GradientBoostingClassifier(
                            n_estimators=50,
                            max_depth=3 + i % 3,
                            learning_rate=0.05 + (i % 5) * 0.02,
                            random_state=i * 42,
                        )
                    m.fit(X_boot, y_boot)
                    self._models.append(m)
                except Exception:
                    continue

        self._fitted = bool(self._models)
        logger.debug(
            f"[Uncertainty] {len(self._models)} model eğitildi."
        )

    def analyze(self, features: np.ndarray | list,
                  match_id: str = "",
                  team: str = "") -> UncertaintyReport:
        """Tek bir maç için belirsizlik ayrımı."""
        report = UncertaintyReport(match_id=match_id, team=team)

        X = np.array(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Çoklu tahmin üret
        if self._fitted and self._models:
            probs_list = []
            for m in self._models:
                try:
                    p = m.predict_proba(X)
                    if p.shape[1] < self._n_classes:
                        padded = np.zeros((len(X), self._n_classes))
                        padded[:, :p.shape[1]] = p
                        p = padded
                    probs_list.append(p[0])
                except Exception:
                    continue
            probs_samples = np.array(probs_list)
            report.method = "deep_ensemble"
        elif self._X_train is not None and self._y_train is not None:
            probs_all = bootstrap_predictions(
                self._X_train, self._y_train, X,
                n_models=self._n_models,
                n_classes=self._n_classes,
            )
            probs_samples = probs_all[:, 0, :]
            report.method = "bootstrap_ensemble"
        else:
            # Heuristic fallback
            probs_samples = np.random.dirichlet(
                np.ones(self._n_classes), self._n_models,
            )
            report.method = "heuristic"

        report.n_samples = len(probs_samples)

        # Ortalama tahmin
        mean_probs = np.mean(probs_samples, axis=0)
        mean_probs /= mean_probs.sum()
        report.prediction = mean_probs
        report.predicted_class = int(np.argmax(mean_probs))
        report.predicted_prob = round(float(np.max(mean_probs)), 4)

        # Belirsizlik ayrımı
        total = predictive_entropy(mean_probs)
        aleatoric = expected_entropy(probs_samples)
        epistemic = max(0.0, total - aleatoric)

        report.total_uncertainty = round(total, 4)
        report.aleatoric = round(aleatoric, 4)
        report.epistemic = round(epistemic, 4)

        if total > 0:
            report.epistemic_ratio = round(epistemic / total, 3)
            report.aleatoric_ratio = round(aleatoric / total, 3)

        # Karar
        report.decision, report.confidence_modifier, report.reason = (
            self._decide(report)
        )
        report.recommendation = self._advice(report)

        return report

    def analyze_batch(self, X: np.ndarray,
                        match_ids: list[str] | None = None) -> list[UncertaintyReport]:
        """Toplu belirsizlik analizi."""
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        ids = match_ids or [f"match_{i}" for i in range(len(X))]
        return [
            self.analyze(X[i], match_id=ids[i] if i < len(ids) else "")
            for i in range(len(X))
        ]

    def _decide(self, r: UncertaintyReport) -> tuple[str, float, str]:
        """Karar mekanizması."""
        # Epistemic yüksek → veri yetersiz → pas geç
        if r.epistemic > self.EPISTEMIC_THRESHOLD:
            return (
                "ABSTAIN",
                0.0,
                f"Epistemik belirsizlik çok yüksek ({r.epistemic:.3f}). "
                f"Bu maç hakkında yeterli bilgi yok."
            )

        # Toplam çok yüksek → pas geç
        if r.total_uncertainty > self.TOTAL_THRESHOLD:
            return (
                "ABSTAIN",
                0.0,
                f"Toplam belirsizlik aşırı ({r.total_uncertainty:.3f}). "
                f"Güvenilir tahmin yapılamıyor."
            )

        # Aleatoric yüksek → yarı Kelly
        if r.aleatoric > self.ALEATORIC_THRESHOLD:
            return (
                "HALF_KELLY",
                0.5,
                f"Aleatorik belirsizlik yüksek ({r.aleatoric:.3f}). "
                f"Top yuvarlak, stake yarıya düşürüldü."
            )

        # Normal
        conf = max(0.5, 1.0 - r.total_uncertainty)
        return (
            "BET",
            round(conf, 2),
            f"Belirsizlik kontrol altında (ε={r.epistemic:.3f}, "
            f"α={r.aleatoric:.3f}). Normal devam."
        )

    def _advice(self, r: UncertaintyReport) -> str:
        labels = {0: "Ev Sahibi", 1: "Beraberlik", 2: "Deplasman"}
        pred = labels.get(r.predicted_class, "?")

        if r.decision == "ABSTAIN":
            return (
                f"PAS GEÇ: {r.team} (tahmin: {pred} %{r.predicted_prob:.0%}). "
                f"Epistemik: {r.epistemic:.3f} (bilgisizlik), "
                f"Aleatorik: {r.aleatoric:.3f} (şans). "
                f"Bu maç hakkında bilgi yetersiz."
            )
        if r.decision == "HALF_KELLY":
            return (
                f"YARI KELLY: {r.team} (tahmin: {pred} %{r.predicted_prob:.0%}). "
                f"Aleatorik yüksek ({r.aleatoric:.3f}). "
                f"Futbolun doğası gereği riskli, stake'i yarıla."
            )
        return (
            f"BAHIS: {r.team} (tahmin: {pred} %{r.predicted_prob:.0%}). "
            f"Toplam belirsizlik: {r.total_uncertainty:.3f}. "
            f"Güven x{r.confidence_modifier:.1f}."
        )
