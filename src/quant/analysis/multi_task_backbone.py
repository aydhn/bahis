"""
multi_task_backbone.py – Multi-Task Learning (MTL) mimarisi.
Galibiyet, Gol ve Korner tahminlerini tek bir omurga üzerinden yapar.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch yüklü değil – MTL basit modda çalışacak.")


if TORCH_AVAILABLE:
    class MTLNet(nn.Module):
        """Paylaşımlı omurga + görev-spesifik kafalar."""
        def __init__(self, input_dim: int = 12, hidden: int = 64):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            )
            # Galibiyet kafası (3 sınıf: H/D/A)
            self.head_result = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 3), nn.Softmax(dim=-1),
            )
            # Toplam gol kafası (regresyon)
            self.head_goals = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Softplus(),
            )
            # Korner kafası (regresyon)
            self.head_corners = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Softplus(),
            )

        def forward(self, x):
            shared = self.backbone(x)
            return {
                "result": self.head_result(shared),
                "goals": self.head_goals(shared),
                "corners": self.head_corners(shared),
            }


class MultiTaskBackbone:
    """MTL model yöneticisi – eğitim, tahmin ve değerlendirme."""

    FEATURE_KEYS = [
        "home_odds", "draw_odds", "away_odds", "over25_odds", "under25_odds",
        "home_xg", "away_xg", "home_xga", "away_xga",
        "home_win_rate", "away_win_rate", "odds_volatility",
    ]

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._device = "cpu"

        if TORCH_AVAILABLE:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = MTLNet(input_dim=len(self.FEATURE_KEYS))
            self._model.to(self._device)
            self._model.eval()

            if model_path:
                try:
                    self._model.load_state_dict(torch.load(model_path, map_location=self._device))
                    logger.info(f"MTL model yüklendi: {model_path}")
                except Exception as e:
                    logger.warning(f"Model yükleme hatası: {e}")

        logger.debug(f"MultiTaskBackbone başlatıldı (device={self._device}).")

    def _extract_features(self, row: dict) -> np.ndarray:
        return np.array([row.get(k, 0.0) or 0.0 for k in self.FEATURE_KEYS], dtype=np.float32)

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """Her maç için MTL tahmini üretir."""
        results = []
        for row in features.iter_rows(named=True):
            mid = row.get("match_id", "")
            feat = self._extract_features(row)

            if TORCH_AVAILABLE and self._model is not None:
                preds = self._predict_torch(feat)
            else:
                preds = self._predict_heuristic(row)

            results.append({
                "match_id": mid,
                "prob_home": preds["prob_home"],
                "prob_draw": preds["prob_draw"],
                "prob_away": preds["prob_away"],
                "expected_goals": preds["expected_goals"],
                "expected_corners": preds["expected_corners"],
                "confidence": preds["confidence"],
            })

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _predict_torch(self, feat: np.ndarray) -> dict:
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self._device)
            out = self._model(x)
            result_probs = out["result"].cpu().numpy()[0]
            goals = out["goals"].cpu().numpy()[0, 0]
            corners = out["corners"].cpu().numpy()[0, 0]

        return {
            "prob_home": float(result_probs[0]),
            "prob_draw": float(result_probs[1]),
            "prob_away": float(result_probs[2]),
            "expected_goals": float(goals),
            "expected_corners": float(corners),
            "confidence": float(1 - np.std(result_probs)),
        }

    def _predict_heuristic(self, row: dict) -> dict:
        """PyTorch yokken istatistiksel heuristic."""
        ho = row.get("home_odds", 2.5) or 2.5
        do_ = row.get("draw_odds", 3.3) or 3.3
        ao = row.get("away_odds", 3.0) or 3.0

        implied = np.array([1/ho, 1/do_, 1/ao])
        implied /= implied.sum()

        xg_home = row.get("home_xg", 1.3) or 1.3
        xg_away = row.get("away_xg", 1.1) or 1.1
        total_goals = xg_home + xg_away

        return {
            "prob_home": float(implied[0]),
            "prob_draw": float(implied[1]),
            "prob_away": float(implied[2]),
            "expected_goals": float(total_goals),
            "expected_corners": float(total_goals * 4.2),  # goals-corners korelasyonu
            "confidence": float(1 - np.std(implied)),
        }

    def train(self, X: np.ndarray, y_result: np.ndarray, y_goals: np.ndarray,
              y_corners: np.ndarray, epochs: int = 50, lr: float = 1e-3):
        """Modeli eğitir."""
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("PyTorch yok – eğitim atlanıyor.")
            return

        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        yr_t = torch.tensor(y_result, dtype=torch.long).to(self._device)
        yg_t = torch.tensor(y_goals, dtype=torch.float32).unsqueeze(1).to(self._device)
        yc_t = torch.tensor(y_corners, dtype=torch.float32).unsqueeze(1).to(self._device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self._model(X_t)
            loss = (
                ce_loss(out["result"], yr_t)
                + mse_loss(out["goals"], yg_t)
                + 0.5 * mse_loss(out["corners"], yc_t)
            )
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"MTL Epoch {epoch+1}/{epochs} – Loss: {loss.item():.4f}")

        self._model.eval()
