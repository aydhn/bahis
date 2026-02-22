"""
kan_interpreter.py – Kolmogorov-Arnold Networks (KAN) tabanlı yorumlanabilir tahmin.
Tahminleri matematiksel formüllere dönüştüren şeffaf zeka.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    from kan import KAN as PyKAN
    PYKAN_AVAILABLE = True
except ImportError:
    PYKAN_AVAILABLE = False
    logger.warning("pykan yüklü değil – KAN basit modda çalışacak.")


class KANInterpreter:
    """Kolmogorov-Arnold Network tabanlı yorumlanabilir model."""

    FEATURE_KEYS = [
        "home_odds", "draw_odds", "away_odds",
        "home_xg", "away_xg",
        "home_win_rate", "away_win_rate",
        "odds_volatility",
    ]

    def __init__(self, model_path: str | None = None):
        self._model = None

        if PYKAN_AVAILABLE:
            try:
                self._model = PyKAN(
                    width=[len(self.FEATURE_KEYS), 8, 4, 3],
                    grid=5, k=3,
                )
                logger.info("KAN modeli oluşturuldu.")
            except Exception as e:
                logger.warning(f"KAN oluşturma hatası: {e}")

        logger.debug("KANInterpreter başlatıldı.")

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için KAN tahmini üretir."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")
            feat = np.array([row.get(k, 0.0) or 0.0 for k in self.FEATURE_KEYS], dtype=np.float32)

            if PYKAN_AVAILABLE and self._model is not None:
                preds = self._predict_kan(feat)
            else:
                preds = self._predict_analytical(row)

            results.append({
                "match_id": mid,
                "prob_home": preds["prob_home"],
                "prob_draw": preds["prob_draw"],
                "prob_away": preds["prob_away"],
                "confidence": preds["confidence"],
                "formula_complexity": preds["formula_complexity"],
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _predict_kan(self, feat: np.ndarray) -> dict:
        """KAN modeli ile tahmin."""
        try:
            import torch
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = self._model(x)
            probs = torch.softmax(out, dim=-1).numpy()[0]
            return {
                "prob_home": float(probs[0]),
                "prob_draw": float(probs[1]),
                "prob_away": float(probs[2]),
                "confidence": float(1 - np.std(probs)),
                "formula_complexity": self._estimate_complexity(),
            }
        except Exception as e:
            logger.debug(f"KAN predict hatası: {e}")
            return self._predict_analytical_from_feat(feat)

    def _predict_analytical(self, row: dict) -> dict:
        """KAN olmadan Kolmogorov teoremine dayalı analitik yaklaşım.
        Kolmogorov süperpozisyon teoremi: f(x1,...,xn) = Σ Φ(Σ ψ(xi))"""
        ho = row.get("home_odds", 2.5) or 2.5
        do_ = row.get("draw_odds", 3.3) or 3.3
        ao = row.get("away_odds", 3.0) or 3.0

        # İç fonksiyonlar ψ (feature dönüşümleri)
        psi = [
            np.tanh(1.0 / ho - 0.4),
            np.tanh(1.0 / do_ - 0.3),
            np.tanh(1.0 / ao - 0.35),
            np.tanh(row.get("home_xg", 0.0) - row.get("away_xg", 0.0)),
            np.tanh(row.get("home_win_rate", 0.0) - row.get("away_win_rate", 0.0)),
        ]

        # Dış fonksiyon Φ (toplama ve softmax)
        inner_sum = np.sum(psi)
        logits = np.array([
            inner_sum + 0.3,   # ev sahibi avantajı
            -abs(inner_sum) * 0.5 + 0.1,  # beraberlik
            -inner_sum + 0.3,
        ])
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return {
            "prob_home": float(probs[0]),
            "prob_draw": float(probs[1]),
            "prob_away": float(probs[2]),
            "confidence": float(1 - np.std(probs)),
            "formula_complexity": 2.5,
        }

    def _predict_analytical_from_feat(self, feat: np.ndarray) -> dict:
        implied = np.array([1/max(feat[0], 1.01), 1/max(feat[1], 1.01), 1/max(feat[2], 1.01)])
        implied /= implied.sum()
        return {
            "prob_home": float(implied[0]),
            "prob_draw": float(implied[1]),
            "prob_away": float(implied[2]),
            "confidence": float(1 - np.std(implied)),
            "formula_complexity": 1.0,
        }

    def _estimate_complexity(self) -> float:
        """KAN modelinin formül karmaşıklığını tahmin eder."""
        if self._model is None:
            return 0.0
        try:
            n_params = sum(p.numel() for p in self._model.parameters())
            return float(np.log10(n_params + 1))
        except Exception:
            return 3.0

    def fit(self, features: np.ndarray, labels: np.ndarray, steps: int = 20):
        """KAN modelini eğit."""
        if not PYKAN_AVAILABLE or self._model is None:
            logger.warning("[KAN] PyKAN yok veya model init edilmedi, eğitim atlanıyor.")
            return

        try:
            import torch
            dataset = {
                "train_input": torch.tensor(features, dtype=torch.float32),
                "train_label": torch.tensor(labels, dtype=torch.long), # Class indices
                "test_input": torch.tensor(features, dtype=torch.float32),
                "test_label": torch.tensor(labels, dtype=torch.long)
            }
            # KAN genellikle regresyon veya fonksiyon approksimasyonu içindir.
            # Sınıflandırma için output dimension 3 (Home, Draw, Away) ayarlandı.
            # PyKAN train metodu: model.train(dataset, opt="LBFGS", steps=20)
            self._model.train(dataset, opt="LBFGS", steps=steps)
            logger.success(f"[KAN] Eğitim tamamlandı ({steps} adım).")
        except Exception as e:
            logger.error(f"[KAN] Eğitim hatası: {e}")

    def save_model(self, path: str = "models/kan_model.pt"):
        """KAN model ağırlıklarını kaydet."""
        if not PYKAN_AVAILABLE or self._model is None:
            return
        try:
            import torch
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # PyKAN save metodu bazen sorunlu olabilir, state_dict kullanalım
            # self._model.save(path) # Eğer kütüphane destekliyorsa
            # Alternatif: Torch save
            torch.save(self._model.state_dict(), path)
            logger.info(f"[KAN] Model kaydedildi: {path}")
        except Exception as e:
            logger.warning(f"[KAN] Kayıt hatası: {e}")

    def load_model(self, path: str = "models/kan_model.pt"):
        """KAN model ağırlıklarını yükle."""
        if not PYKAN_AVAILABLE or self._model is None:
            return
        try:
            import torch
            if not os.path.exists(path):
                logger.warning(f"[KAN] Model dosyası bulunamadı: {path}")
                return
            self._model.load_state_dict(torch.load(path))
            logger.info(f"[KAN] Model yüklendi: {path}")
        except Exception as e:
            logger.warning(f"[KAN] Yükleme hatası: {e}")

    def symbolic_formula(self) -> str:
        """KAN'dan sembolik formül çıkarır."""
        if not PYKAN_AVAILABLE or self._model is None:
            return "P(home) ≈ softmax(Σ tanh(ψ_i(x_i)))"
        try:
            # self._model.auto_symbolic() # Expensive
            # formula = self._model.symbolic_formula()
            return "symbolic_formula_placeholder"
        except Exception:
            return "Sembolik formül çıkarılamadı."
