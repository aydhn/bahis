"""
ensemble_stacking.py – Ensemble Stacking Meta-Model.

Tek modele güvenmek finansal intihardır. Farklı modellerin
güçlü yönlerini birleştiren bir "Üst Akıl" (Meta-Learner).

Katman 1 (Base Learners):
  - Poisson Model       → Alt/Üst'te güçlü
  - Dixon-Coles         → Beraberlik tespitinde güçlü
  - LightGBM            → Favorileri bilmede güçlü
  - Elo/Glicko           → Takım gücü trendlerinde güçlü
  - Bayesian Hierarchical → Sezon başı az veride güçlü
  - Sentiment            → Kalabalık duyarlılığında güçlü
  - Monte Carlo          → Varyans/belirsizlik ölçümünde güçlü

Katman 2 (Meta-Learner):
  - Logistic Regression: Base learner olasılıklarını girdi alır,
    hangi durumda kime güveneceğini öğrenir.

Walk-Forward Validation:
  - Klasik CV gelecekten bilgi sızdırır (data leakage).
  - Rolling window: sadece geçmişle eğit, geleceği test et.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn yüklü değil – stacking basit modda.")


# ═══════════════════════════════════════════════════════
#  BASE LEARNER SONUÇLARI
# ═══════════════════════════════════════════════════════
@dataclass
class BasePrediction:
    """Tek bir base learner'ın bir maç için tahmini."""
    model_name: str
    match_id: str
    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0
    prob_over25: float = 0.0
    prob_btts: float = 0.0
    confidence: float = 0.5


@dataclass
class StackingRecord:
    """Eğitim verisi: tüm base learner tahminleri + gerçek sonuç."""
    match_id: str
    timestamp: str = ""
    # Base learner olasılıkları (feature vector)
    features: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    # Gerçek sonuç (label): 0=home, 1=draw, 2=away
    actual_result: int = -1


class EnsembleStacking:
    """Meta-Model: Base learner çıktılarını birleştiren üst akıl.

    Örnek:
        stacker = EnsembleStacking()
        # Base learner'lardan olasılıkları topla
        stacker.add_base_prediction("poisson", match_id, 0.45, 0.28, 0.27)
        stacker.add_base_prediction("lightgbm", match_id, 0.50, 0.25, 0.25)
        stacker.add_base_prediction("elo", match_id, 0.42, 0.30, 0.28)
        # Meta tahmin
        result = stacker.predict(match_id)
    """

    # Her base learner'ın varsayılan ağırlığı (eğitim öncesi)
    DEFAULT_WEIGHTS = {
        "poisson": 0.15,
        "dixon_coles": 0.15,
        "lightgbm": 0.20,
        "elo": 0.12,
        "bayesian": 0.12,
        "monte_carlo": 0.10,
        "sentiment": 0.06,
        "gradient_boosting": 0.10,
    }

    BASE_MODELS = list(DEFAULT_WEIGHTS.keys())

    def __init__(self):
        self._current_predictions: dict[str, dict[str, BasePrediction]] = {}
        self._training_data: list[StackingRecord] = []
        self._meta_model = None
        self._scaler = StandardScaler() if SKLEARN_OK else None
        self._fitted = False
        self._walk_forward_results: list[dict] = []
        logger.debug("EnsembleStacking başlatıldı.")

    # ═══════════════════════════════════════════
    #  BASE LEARNER TAHMİNLERİ EKLE
    # ═══════════════════════════════════════════
    def add_base_prediction(self, model_name: str, match_id: str,
                            prob_home: float, prob_draw: float,
                            prob_away: float,
                            prob_over25: float = 0.5,
                            prob_btts: float = 0.5,
                            confidence: float = 0.5):
        """Bir base learner'ın tahminini ekler."""
        pred = BasePrediction(
            model_name=model_name, match_id=match_id,
            prob_home=prob_home, prob_draw=prob_draw, prob_away=prob_away,
            prob_over25=prob_over25, prob_btts=prob_btts,
            confidence=confidence,
        )
        if match_id not in self._current_predictions:
            self._current_predictions[match_id] = {}
        self._current_predictions[match_id][model_name] = pred

    def _build_feature_vector(self, match_id: str) -> tuple[list[float], list[str]]:
        """Bir maç için tüm base learner olasılıklarını feature vektörüne çevir."""
        preds = self._current_predictions.get(match_id, {})
        features = []
        names = []

        for model_name in self.BASE_MODELS:
            pred = preds.get(model_name)
            if pred:
                features.extend([pred.prob_home, pred.prob_draw, pred.prob_away,
                                 pred.prob_over25, pred.confidence])
                names.extend([
                    f"{model_name}_home", f"{model_name}_draw", f"{model_name}_away",
                    f"{model_name}_over25", f"{model_name}_conf",
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.5, 0.0])
                names.extend([
                    f"{model_name}_home", f"{model_name}_draw", f"{model_name}_away",
                    f"{model_name}_over25", f"{model_name}_conf",
                ])

        # Modeller arası fark feature'ları (interaction)
        home_probs = [preds[m].prob_home for m in self.BASE_MODELS if m in preds]
        if len(home_probs) >= 2:
            features.append(float(np.std(home_probs)))  # Consensus/disagreement
            names.append("home_prob_disagreement")
            features.append(max(home_probs) - min(home_probs))
            names.append("home_prob_range")
        else:
            features.extend([0.0, 0.0])
            names.extend(["home_prob_disagreement", "home_prob_range"])

        return features, names

    # ═══════════════════════════════════════════
    #  META-MODEL TAHMİN
    # ═══════════════════════════════════════════
    def predict(self, match_id: str) -> dict:
        """Meta-model ile nihai tahmin üretir."""
        features, names = self._build_feature_vector(match_id)

        if self._fitted and self._meta_model is not None and SKLEARN_OK:
            return self._predict_trained(match_id, features, names)
        else:
            return self._predict_weighted_avg(match_id)

    def _predict_trained(self, match_id: str,
                         features: list[float], names: list[str]) -> dict:
        """Eğitilmiş meta-model ile tahmin."""
        X = np.array([features])
        if self._scaler:
            X = self._scaler.transform(X)

        probs = self._meta_model.predict_proba(X)[0]

        # 3 sınıf: home, draw, away
        if len(probs) == 3:
            p_home, p_draw, p_away = probs
        elif len(probs) == 2:
            p_home, p_away = probs
            p_draw = 1 - p_home - p_away
        else:
            return self._predict_weighted_avg(match_id)

        prediction = ["home", "draw", "away"][np.argmax([p_home, p_draw, p_away])]
        confidence = float(max(p_home, p_draw, p_away))

        return {
            "match_id": match_id,
            "method": "stacking_meta_model",
            "prob_home": float(p_home),
            "prob_draw": float(p_draw),
            "prob_away": float(p_away),
            "prediction": prediction,
            "confidence": confidence,
            "n_base_models": len(self._current_predictions.get(match_id, {})),
        }

    def _predict_weighted_avg(self, match_id: str) -> dict:
        """Eğitim öncesi: ağırlıklı ortalama ile tahmin."""
        preds = self._current_predictions.get(match_id, {})
        if not preds:
            return {"match_id": match_id, "method": "no_data",
                    "prob_home": 0.33, "prob_draw": 0.33, "prob_away": 0.34}

        total_w = 0.0
        w_home = w_draw = w_away = 0.0

        for model_name, pred in preds.items():
            w = self.DEFAULT_WEIGHTS.get(model_name, 0.1) * pred.confidence
            w_home += pred.prob_home * w
            w_draw += pred.prob_draw * w
            w_away += pred.prob_away * w
            total_w += w

        if total_w > 0:
            w_home /= total_w
            w_draw /= total_w
            w_away /= total_w

        # Normalize
        total = w_home + w_draw + w_away
        if total > 0:
            w_home /= total
            w_draw /= total
            w_away /= total

        prediction = ["home", "draw", "away"][np.argmax([w_home, w_draw, w_away])]

        return {
            "match_id": match_id,
            "method": "weighted_average",
            "prob_home": float(w_home),
            "prob_draw": float(w_draw),
            "prob_away": float(w_away),
            "prediction": prediction,
            "confidence": float(max(w_home, w_draw, w_away)),
            "n_base_models": len(preds),
        }

    # ═══════════════════════════════════════════
    #  EĞİTİM VERİSİ KAYDI
    # ═══════════════════════════════════════════
    def record_result(self, match_id: str, actual_result: int):
        """Maç sonucunu kaydet (eğitim verisi).
        actual_result: 0=home_win, 1=draw, 2=away_win
        """
        features, names = self._build_feature_vector(match_id)
        record = StackingRecord(
            match_id=match_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            features=features,
            feature_names=names,
            actual_result=actual_result,
        )
        self._training_data.append(record)

        # Belirli sayıda veri birikince otomatik yeniden eğit
        if len(self._training_data) % 50 == 0 and len(self._training_data) >= 100:
            self.fit()

    # ═══════════════════════════════════════════
    #  META-MODEL EĞİTİMİ
    # ═══════════════════════════════════════════
    def fit(self):
        """Meta-model'i eğitim verisinden öğret."""
        if not SKLEARN_OK:
            logger.warning("sklearn yok – meta model eğitilemez.")
            return

        valid = [r for r in self._training_data if r.actual_result >= 0]
        if len(valid) < 50:
            logger.info(f"Stacking: {len(valid)} kayıt – minimum 50 gerekli.")
            return

        X = np.array([r.features for r in valid])
        y = np.array([r.actual_result for r in valid])

        # Walk-Forward ile eğit
        wf_result = self._walk_forward_fit(X, y)

        # Son modeli tüm veriyle eğit
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._meta_model = CalibratedClassifierCV(
            LogisticRegression(
                C=1.0, max_iter=1000, multi_class="multinomial",
                solver="lbfgs", class_weight="balanced",
            ),
            cv=3, method="isotonic",
        )
        self._meta_model.fit(X_scaled, y)
        self._fitted = True

        logger.success(
            f"Meta-model eğitildi: {len(valid)} kayıt, "
            f"WF accuracy={wf_result.get('avg_accuracy', 0):.3f}"
        )

    # ═══════════════════════════════════════════
    #  WALK-FORWARD VALIDATION
    # ═══════════════════════════════════════════
    def _walk_forward_fit(self, X: np.ndarray, y: np.ndarray,
                          n_splits: int = 5,
                          min_train_size: int = 30) -> dict:
        """Zaman serisi için Walk-Forward (Rolling Window) doğrulama.

        Klasik CV gelecekten bilgi sızdırır. Walk-Forward bunu önler:
        - Train: [0:t], Test: [t:t+step]
        - Her adımda train penceresi büyür/kayar
        """
        n = len(X)
        if n < min_train_size + 10:
            return {"status": "yetersiz_veri", "avg_accuracy": 0}

        step = max((n - min_train_size) // n_splits, 5)
        accuracies = []
        log_losses = []
        brier_scores = []

        for i in range(n_splits):
            train_end = min_train_size + i * step
            test_end = min(train_end + step, n)

            if train_end >= n or test_end <= train_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(np.unique(y_train)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            model = LogisticRegression(
                C=1.0, max_iter=1000, multi_class="multinomial",
                solver="lbfgs", class_weight="balanced",
            )
            try:
                model.fit(X_tr_s, y_train)
                y_pred = model.predict(X_te_s)
                y_prob = model.predict_proba(X_te_s)

                acc = accuracy_score(y_test, y_pred)
                ll = log_loss(y_test, y_prob, labels=[0, 1, 2])

                accuracies.append(acc)
                log_losses.append(ll)
            except Exception:
                continue

        result = {
            "n_splits": len(accuracies),
            "avg_accuracy": float(np.mean(accuracies)) if accuracies else 0,
            "std_accuracy": float(np.std(accuracies)) if accuracies else 0,
            "avg_log_loss": float(np.mean(log_losses)) if log_losses else 0,
        }

        self._walk_forward_results.append(result)
        logger.info(
            f"Walk-Forward: {result['n_splits']} fold, "
            f"acc={result['avg_accuracy']:.3f}±{result['std_accuracy']:.3f}"
        )
        return result

    # ═══════════════════════════════════════════
    #  TOPLU TAHMİN
    # ═══════════════════════════════════════════
    def predict_batch(self, match_ids: list[str]) -> list[dict]:
        """Birden fazla maç için meta-tahmin."""
        return [self.predict(mid) for mid in match_ids]

    def clear_predictions(self):
        """Günlük tahminleri temizle (yeni döngü için)."""
        self._current_predictions.clear()

    def model_agreement(self, match_id: str) -> dict:
        """Modellerin ne kadar hemfikir olduğunu ölç."""
        preds = self._current_predictions.get(match_id, {})
        if len(preds) < 2:
            return {"agreement": "unknown", "n_models": len(preds)}

        predictions = []
        for p in preds.values():
            winner = np.argmax([p.prob_home, p.prob_draw, p.prob_away])
            predictions.append(winner)

        from collections import Counter
        counter = Counter(predictions)
        majority = counter.most_common(1)[0]
        agreement = majority[1] / len(predictions)

        return {
            "agreement": float(agreement),
            "majority_prediction": ["home", "draw", "away"][majority[0]],
            "n_models": len(preds),
            "unanimous": agreement == 1.0,
            "split_vote": agreement < 0.5,
        }

    @property
    def stats(self) -> dict:
        return {
            "fitted": self._fitted,
            "training_records": len(self._training_data),
            "walk_forward_results": self._walk_forward_results[-3:],
            "active_matches": len(self._current_predictions),
        }
