"""
automl_engine.py – TPOT AutoML (Otonom Model Seçimi).

Botunuz binlerce farklı algoritmayı dener, kendi Python kodunu
kendi yazar ve en iyi modeli otomatik devreye alır.

Kavramlar:
  - AutoML: Otomatik Makine Öğrenmesi – hiperparametre arama
  - TPOT: Tree-based Pipeline Optimization Tool (Genetik Algoritma)
  - Pipeline: Veri ön-işleme + model → tek bir nesne
  - Walk-Forward Validation: Zaman serisi için uygun doğrulama
  - Model Registry: En iyi modellerin versiyonlanmış kaydı

Akış:
  1. Veri hazırla (Feature Engineering çıktıları)
  2. TPOT genetik algoritmayı başlat (n_generation * pop_size = binlerce deney)
  3. En iyi pipeline'ı bul → Python kodu olarak kaydet
  4. best_model.py olarak dışa aktar
  5. Ensemble'a yeni sinyal olarak ekle

Teknoloji: TPOT (TPOTClassifier / TPOTRegressor)
Fallback: RandomizedSearchCV (sklearn)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from tpot import TPOTClassifier, TPOTRegressor
    TPOT_OK = True
except ImportError:
    TPOT_OK = False
    logger.debug("tpot yüklü değil – RandomizedSearchCV fallback.")

try:
    from sklearn.model_selection import (
        RandomizedSearchCV,
        TimeSeriesSplit,
    )
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class AutoMLResult:
    """AutoML arama sonucu."""
    best_model_name: str = ""
    best_score: float = 0.0
    best_params: dict = field(default_factory=dict)
    n_models_tried: int = 0
    search_time_sec: float = 0.0
    pipeline_code: str = ""      # Dışa aktarılan Python kodu
    model_path: str = ""         # Kaydedilen model dosyası
    method: str = ""
    recommendation: str = ""


@dataclass
class ModelRegistry:
    """Model versiyonlama kaydı."""
    models: list[dict] = field(default_factory=list)
    active_model: str = ""
    active_score: float = 0.0


# ═══════════════════════════════════════════════
#  AUTOML ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class AutoMLEngine:
    """Otonom model seçimi ve eğitimi.

    Kullanım:
        aml = AutoMLEngine()

        # Veri hazırla
        X, y = load_features()

        # En iyi modeli bul
        result = aml.search(X, y, task="classify", time_budget_min=5)

        # Otomatik devreye al
        aml.deploy_best()

        # Tahmin
        preds = aml.predict(X_new)
    """

    def __init__(self, generations: int = 5, population_size: int = 50,
                 cv_folds: int = 5, random_state: int = 42):
        self._generations = generations
        self._pop_size = population_size
        self._cv = cv_folds
        self._seed = random_state
        self._best_pipeline: Any = None
        self._registry = ModelRegistry()
        self._search_history: list[AutoMLResult] = []

        logger.debug(
            f"[AutoML] Engine başlatıldı: gen={generations}, "
            f"pop={population_size}, cv={cv_folds}"
        )

    def search(self, X: np.ndarray, y: np.ndarray,
                 task: str = "classify",
                 time_budget_min: int = 5,
                 scoring: str = "accuracy") -> AutoMLResult:
        """En iyi modeli bul.

        Args:
            X: Özellik matrisi (n_samples, n_features)
            y: Hedef (sınıf etiketleri veya regresyon)
            task: "classify" veya "regress"
            time_budget_min: Arama süresi limiti (dakika)
            scoring: Metrik ("accuracy", "neg_log_loss", "r2")
        """
        result = AutoMLResult()
        t0 = time.perf_counter()

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        # NaN temizle
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        if len(X) < 20:
            result.recommendation = "Yetersiz veri (min 20 örnek)."
            result.method = "none"
            return result

        if TPOT_OK:
            result = self._search_tpot(X, y, task, time_budget_min, scoring)
        elif SKLEARN_OK:
            result = self._search_sklearn(X, y, task, scoring)
        else:
            result.recommendation = "tpot veya sklearn yüklü değil."
            result.method = "none"
            return result

        result.search_time_sec = round(time.perf_counter() - t0, 2)
        self._search_history.append(result)

        # Registry güncelle
        if result.best_score > self._registry.active_score:
            self._registry.active_model = result.best_model_name
            self._registry.active_score = result.best_score
            self._registry.models.append({
                "name": result.best_model_name,
                "score": result.best_score,
                "params": result.best_params,
                "timestamp": time.time(),
            })

        result.recommendation = self._advice(result)
        return result

    def _search_tpot(self, X: np.ndarray, y: np.ndarray,
                       task: str, time_min: int,
                       scoring: str) -> AutoMLResult:
        """TPOT ile genetik arama."""
        result = AutoMLResult(method="tpot_genetic")

        try:
            if task == "classify":
                tpot = TPOTClassifier(
                    generations=self._generations,
                    population_size=self._pop_size,
                    cv=self._cv,
                    scoring=scoring,
                    max_time_mins=time_min,
                    random_state=self._seed,
                    verbosity=0,
                    n_jobs=-1,
                )
            else:
                tpot = TPOTRegressor(
                    generations=self._generations,
                    population_size=self._pop_size,
                    cv=self._cv,
                    scoring=scoring,
                    max_time_mins=time_min,
                    random_state=self._seed,
                    verbosity=0,
                    n_jobs=-1,
                )

            tpot.fit(X, y)
            self._best_pipeline = tpot.fitted_pipeline_

            result.best_score = round(
                float(tpot.score(X, y)), 4,
            )
            result.best_model_name = type(
                tpot.fitted_pipeline_.steps[-1][1]
            ).__name__
            result.n_models_tried = self._generations * self._pop_size

            # Pipeline kodunu kaydet
            code_path = MODEL_DIR / "best_pipeline.py"
            tpot.export(str(code_path))
            result.pipeline_code = str(code_path)

            # Model kaydet
            model_path = MODEL_DIR / "best_model.pkl"
            joblib.dump(tpot.fitted_pipeline_, model_path)
            result.model_path = str(model_path)

            logger.info(
                f"[AutoML] TPOT: {result.best_model_name} "
                f"(score={result.best_score:.4f}, "
                f"{result.n_models_tried} model denendi)"
            )

        except Exception as e:
            logger.warning(f"[AutoML] TPOT hatası: {e}")
            if SKLEARN_OK:
                return self._search_sklearn(X, y, "classify", "accuracy")

        return result

    def _search_sklearn(self, X: np.ndarray, y: np.ndarray,
                          task: str,
                          scoring: str) -> AutoMLResult:
        """Sklearn RandomizedSearchCV ile arama."""
        result = AutoMLResult(method="sklearn_random_search")

        candidates = [
            ("RandomForest", RandomForestClassifier(random_state=self._seed), {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [3, 5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }),
            ("GradientBoosting", GradientBoostingClassifier(random_state=self._seed), {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
            }),
            ("ExtraTrees", ExtraTreesClassifier(random_state=self._seed), {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            }),
            ("LogisticReg", LogisticRegression(
                max_iter=1000, random_state=self._seed,
            ), {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
            }),
        ]

        best_score = -1
        total_tried = 0
        tscv = TimeSeriesSplit(n_splits=min(self._cv, len(X) // 10))

        for name, model, params in candidates:
            try:
                pipe = SkPipeline([
                    ("scaler", StandardScaler()),
                    ("model", model),
                ])
                param_grid = {f"model__{k}": v for k, v in params.items()}

                search = RandomizedSearchCV(
                    pipe, param_grid,
                    n_iter=30, cv=tscv,
                    scoring=scoring,
                    random_state=self._seed,
                    n_jobs=-1,
                    error_score="raise",
                )
                search.fit(X, y)
                total_tried += 30

                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    self._best_pipeline = search.best_estimator_
                    result.best_model_name = name
                    result.best_score = round(float(best_score), 4)
                    result.best_params = {
                        k.replace("model__", ""): v
                        for k, v in search.best_params_.items()
                    }
            except Exception as e:
                logger.debug(f"[AutoML] {name} hatası: {e}")
                continue

        result.n_models_tried = total_tried

        # Kaydet
        if self._best_pipeline:
            model_path = MODEL_DIR / "best_model.pkl"
            joblib.dump(self._best_pipeline, model_path)
            result.model_path = str(model_path)

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """En iyi model ile tahmin."""
        if self._best_pipeline is None:
            saved = MODEL_DIR / "best_model.pkl"
            if saved.exists() and SKLEARN_OK:
                self._best_pipeline = joblib.load(saved)
            else:
                return np.ones(len(X), dtype=int)

        return self._best_pipeline.predict(np.array(X, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Olasılık tahmini."""
        if self._best_pipeline is None:
            return np.full((len(X), 3), 1 / 3)

        X = np.array(X, dtype=np.float64)
        if hasattr(self._best_pipeline, "predict_proba"):
            return self._best_pipeline.predict_proba(X)
        return np.full((len(X), 3), 1 / 3)

    def deploy_best(self) -> str:
        """En iyi modeli devreye al."""
        path = MODEL_DIR / "best_model.pkl"
        if path.exists():
            logger.info(f"[AutoML] Model devrede: {path}")
            return str(path)
        return ""

    def get_registry(self) -> ModelRegistry:
        """Model kayıt defteri."""
        return self._registry

    def _advice(self, r: AutoMLResult) -> str:
        if r.best_score > 0.6:
            return (
                f"Mükemmel: {r.best_model_name} "
                f"(skor={r.best_score:.1%}, "
                f"{r.n_models_tried} model denendi, "
                f"{r.search_time_sec:.0f}s). Devreye alındı."
            )
        if r.best_score > 0.45:
            return (
                f"Orta: {r.best_model_name} "
                f"(skor={r.best_score:.1%}). "
                f"Daha fazla veri veya özellik gerekebilir."
            )
        return (
            f"Düşük: {r.best_model_name} "
            f"(skor={r.best_score:.1%}). "
            f"Veri kalitesini kontrol edin."
        )
