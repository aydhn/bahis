"""
gradient_boosting.py – LightGBM/XGBoost tabanlı ML tahmin modeli.

Kaggle yarışmalarının kralı Gradient Boosting, tabular veriler için
deep learning'den çok daha etkilidir. 3-sınıflı sınıflandırma:
  Ev Sahibi Kazanır (1), Beraberlik (0), Deplasman (2)

Feature Engineering: sadece gol sayısı değil, türetilmiş özellikler:
  - Son 5 maçtaki xG farkı
  - Son 3 H2H kart sayısı
  - Ligdeki sıralama farkı
  - Form (son 5 maç puan ortalaması)
  - Ev/deplasman ayrı performans
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm yüklü değil.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost yüklü değil.")

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, log_loss, classification_report
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class FeatureEngineer:
    """Özellik mühendisliği: ham veriden türetilmiş ML feature'ları üretir."""

    # Kullanılacak temel feature'lar
    BASE_FEATURES = [
        "home_odds", "draw_odds", "away_odds",
        "over25_odds", "under25_odds",
    ]

    # Türetilecek feature'lar
    DERIVED_FEATURES = [
        "odds_implied_home", "odds_implied_draw", "odds_implied_away",
        "odds_margin", "odds_entropy",
        "home_xg", "away_xg", "xg_diff", "xg_total",
        "home_win_rate", "away_win_rate", "win_rate_diff",
        "home_possession", "away_possession",
        "odds_volatility",
        "home_form_pts", "away_form_pts", "form_diff",
        "home_gd_per_match", "away_gd_per_match",
    ]

    def __init__(self):
        logger.debug("FeatureEngineer başlatıldı.")

    def build_features(self, row: dict) -> dict:
        """Tek bir maç satırından tüm feature'ları türetir."""
        ho = max(row.get("home_odds", 2.5) or 2.5, 1.01)
        do_ = max(row.get("draw_odds", 3.3) or 3.3, 1.01)
        ao = max(row.get("away_odds", 3.0) or 3.0, 1.01)
        o25 = max(row.get("over25_odds", 1.9) or 1.9, 1.01)
        u25 = max(row.get("under25_odds", 1.9) or 1.9, 1.01)

        # İmplied probabilities
        imp_h = 1.0 / ho
        imp_d = 1.0 / do_
        imp_a = 1.0 / ao
        total_imp = imp_h + imp_d + imp_a
        margin = total_imp - 1.0  # Bahisçi marjı

        # Fair probabilities
        fair_h = imp_h / total_imp
        fair_d = imp_d / total_imp
        fair_a = imp_a / total_imp

        # Entropi (yüksek = belirsiz maç)
        probs = np.array([fair_h, fair_d, fair_a])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # xG
        home_xg = row.get("home_xg", 0.0) or 0.0
        away_xg = row.get("away_xg", 0.0) or 0.0
        home_xga = row.get("home_xga", 0.0) or 0.0
        away_xga = row.get("away_xga", 0.0) or 0.0

        # Win rates
        home_wr = row.get("home_win_rate", 0.0) or 0.0
        away_wr = row.get("away_win_rate", 0.0) or 0.0

        # Possession
        home_poss = (row.get("home_possession", 50) or 50) / 100
        away_poss = (row.get("away_possession", 50) or 50) / 100

        # Form (son 5 maç puan ortalaması – W=3, D=1, L=0)
        home_form = self._parse_form(row.get("home_form", ""))
        away_form = self._parse_form(row.get("away_form", ""))

        return {
            # Oran feature'ları
            "home_odds": ho,
            "draw_odds": do_,
            "away_odds": ao,
            "over25_odds": o25,
            "under25_odds": u25,
            "odds_implied_home": fair_h,
            "odds_implied_draw": fair_d,
            "odds_implied_away": fair_a,
            "odds_margin": margin,
            "odds_entropy": entropy,
            "odds_volatility": row.get("odds_volatility", 0.0) or 0.0,

            # xG feature'ları
            "home_xg": home_xg,
            "away_xg": away_xg,
            "xg_diff": home_xg - away_xg,
            "xg_total": home_xg + away_xg,
            "home_xga": home_xga,
            "away_xga": away_xga,
            "xg_defensive_diff": away_xga - home_xga,

            # Performans
            "home_win_rate": home_wr,
            "away_win_rate": away_wr,
            "win_rate_diff": home_wr - away_wr,
            "home_possession": home_poss,
            "away_possession": away_poss,
            "possession_diff": home_poss - away_poss,

            # Form
            "home_form_pts": home_form,
            "away_form_pts": away_form,
            "form_diff": home_form - away_form,

            # Gol farkı
            "home_gd_per_match": home_xg - home_xga,
            "away_gd_per_match": away_xg - away_xga,
        }

    def _parse_form(self, form_str: str) -> float:
        """'WWDLW' → ortalama puan (W=3, D=1, L=0)."""
        if not form_str:
            return 1.5  # Varsayılan orta form
        points = {"W": 3, "D": 1, "L": 0}
        vals = [points.get(c.upper(), 1) for c in form_str[-5:]]
        return np.mean(vals) if vals else 1.5

    def build_matrix(self, features_df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
        """DataFrame'den feature matrisi oluşturur."""
        rows = []
        for row in features_df.iter_rows(named=True):
            feat = self.build_features(row)
            rows.append(feat)

        if not rows:
            return np.zeros((0, 0)), []

        feature_names = sorted(rows[0].keys())
        X = np.array([[r.get(f, 0.0) for f in feature_names] for r in rows], dtype=np.float32)
        return X, feature_names


class GradientBoostingModel:
    """LightGBM/XGBoost tabanlı 3-sınıflı maç sonucu tahmini."""

    def __init__(self, engine: str = "lightgbm", model_path: str | None = None):
        self._engine = engine
        self._model = None
        self._feature_engineer = FeatureEngineer()
        self._feature_names: list[str] = []
        self._fitted = False

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if model_path:
            self._load(model_path)

        logger.debug(f"GradientBoostingModel başlatıldı (engine={engine}).")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        """Modeli eğitir. y: 0=home, 1=draw, 2=away."""
        n_samples = len(X)
        if n_samples < 10:
            logger.warning("Yeterli eğitim verisi yok.")
            return

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        if self._engine == "lightgbm" and LGB_AVAILABLE:
            self._fit_lgb(X, y)
        elif self._engine == "xgboost" and XGB_AVAILABLE:
            self._fit_xgb(X, y)
        else:
            self._fit_sklearn(X, y)

    def _fit_lgb(self, X: np.ndarray, y: np.ndarray):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        }
        train_data = lgb.Dataset(X, label=y, feature_name=self._feature_names)

        # Cross-validation
        cv_results = lgb.cv(
            params, train_data, num_boost_round=500,
            nfold=5, stratified=True, seed=42,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        best_rounds = len(cv_results["valid multi_logloss-mean"])
        best_loss = cv_results["valid multi_logloss-mean"][-1]

        self._model = lgb.train(
            params, train_data, num_boost_round=best_rounds,
        )
        self._fitted = True
        logger.info(f"LightGBM eğitildi: {best_rounds} round, loss={best_loss:.4f}")

        # Feature importance
        importance = self._model.feature_importance(importance_type="gain")
        top_features = sorted(
            zip(self._feature_names, importance),
            key=lambda x: x[1], reverse=True
        )[:10]
        logger.info(f"Top 10 feature: {[(f, round(i, 1)) for f, i in top_features]}")

    def _fit_xgb(self, X: np.ndarray, y: np.ndarray):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        dtrain = xgb.DMatrix(X, label=y, feature_names=self._feature_names)
        cv_results = xgb.cv(
            params, dtrain, num_boost_round=500,
            nfold=5, stratified=True, seed=42,
            early_stopping_rounds=50, verbose_eval=False,
        )
        best_rounds = len(cv_results)
        self._model = xgb.train(params, dtrain, num_boost_round=best_rounds)
        self._fitted = True
        logger.info(f"XGBoost eğitildi: {best_rounds} round")

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray):
        """LightGBM/XGBoost yoksa sklearn GradientBoosting."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            self._model.fit(X, y)
            self._fitted = True
            logger.info("sklearn GradientBoosting eğitildi.")
        except Exception as e:
            logger.error(f"sklearn fit hatası: {e}")

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """Feature DataFrame'inden tahmin üretir."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")
            feat = self._feature_engineer.build_features(row)

            if self._fitted and self._model is not None:
                probs = self._predict_one(feat)
            else:
                probs = self._heuristic_predict(feat)

            results.append({
                "match_id": mid,
                "gb_prob_home": probs[0],
                "gb_prob_draw": probs[1],
                "gb_prob_away": probs[2],
                "gb_confidence": float(1 - np.std(probs) * 2),
                "gb_prediction": ["home", "draw", "away"][np.argmax(probs)],
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _predict_one(self, feat: dict) -> np.ndarray:
        x = np.array([[feat.get(f, 0.0) for f in self._feature_names]], dtype=np.float32)

        if self._engine == "lightgbm" and LGB_AVAILABLE:
            probs = self._model.predict(x)[0]
        elif self._engine == "xgboost" and XGB_AVAILABLE:
            dtest = xgb.DMatrix(x, feature_names=self._feature_names)
            probs = self._model.predict(dtest)[0]
        else:
            probs = self._model.predict_proba(x)[0]

        return np.array(probs)

    def _heuristic_predict(self, feat: dict) -> np.ndarray:
        """Model eğitilmemişken oran-tabanlı tahmin."""
        imp_h = feat.get("odds_implied_home", 0.4)
        imp_d = feat.get("odds_implied_draw", 0.3)
        imp_a = feat.get("odds_implied_away", 0.3)
        total = imp_h + imp_d + imp_a
        if total > 0:
            return np.array([imp_h/total, imp_d/total, imp_a/total])
        return np.array([0.4, 0.3, 0.3])

    def save(self, filename: str = "gradient_boosting.model"):
        if self._model is None:
            return
        path = MODEL_DIR / filename
        if self._engine == "lightgbm" and LGB_AVAILABLE:
            self._model.save_model(str(path))
        elif self._engine == "xgboost" and XGB_AVAILABLE:
            self._model.save_model(str(path))
        else:
            import joblib
            joblib.dump(self._model, str(path))
        logger.info(f"Model kaydedildi: {path}")

    def _load(self, path: str):
        try:
            p = Path(path)
            if not p.exists():
                return
            if self._engine == "lightgbm" and LGB_AVAILABLE:
                self._model = lgb.Booster(model_file=str(p))
            elif self._engine == "xgboost" and XGB_AVAILABLE:
                self._model = xgb.Booster()
                self._model.load_model(str(p))
            self._fitted = True
            logger.info(f"Model yüklendi: {p}")
        except Exception as e:
            logger.warning(f"Model yükleme hatası: {e}")
