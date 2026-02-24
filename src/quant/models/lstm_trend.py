"""
lstm_trend.py – LSTM (Long Short-Term Memory) ile Form/Momentum Analizi.

LightGBM tablosal verilerde iyidir ama "takımın son 5 haftadaki
psikolojik çöküşünü" veya "ivmesini" anlamakta zorlanır.
Zaman serileri için Derin Öğrenme gerekir.

Girdi:   Son 10 maçlık skor, xG, topla oynama serileri (sequence)
Model:   2-katman LSTM → Dense → Sigmoid
Çıktı:   Momentum skoru (0.0 = tam düşüş, 1.0 = tam yükseliş)

Bu skor Ensemble Stacking modeline yeni bir feature olarak eklenir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.warning("PyTorch yüklü değil – LSTM heuristik modda çalışacak.")


# ═══════════════════════════════════════════════
#  PYTORCH MODEL
# ═══════════════════════════════════════════════
if TORCH_OK:
    class LSTMNet(nn.Module):
        """Hafif 2-katman LSTM → Momentum tahmini."""

        def __init__(self, input_size: int = 6, hidden_size: int = 32,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
                nn.Sigmoid(),   # 0-1 arası momentum
            )

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Son zaman adımının çıktısını al
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden).squeeze(-1)

    class TeamSequenceDataset(Dataset):
        """Takım zaman serisi veri seti."""

        def __init__(self, sequences: list[np.ndarray],
                     labels: list[float]):
            self.sequences = [torch.FloatTensor(s) for s in sequences]
            self.labels = torch.FloatTensor(labels)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]


# ═══════════════════════════════════════════════
#  FEATURE BUILDER: Maç serisini tensöre çevir
# ═══════════════════════════════════════════════
@dataclass
class MatchSequenceFeatures:
    """Tek bir maçın seri feature'ları."""
    goals_scored: float = 0.0
    goals_conceded: float = 0.0
    xg: float = 0.0
    xga: float = 0.0           # xG against
    possession: float = 0.5
    points: float = 0.0        # 3=W, 1=D, 0=L


class SequenceBuilder:
    """Ham maç verilerinden LSTM girdisi oluşturur."""

    FEATURE_NAMES = [
        "goals_scored", "goals_conceded", "xg", "xga",
        "possession", "points",
    ]
    SEQUENCE_LEN = 10

    def build(self, match_history: list[dict],
              seq_len: int | None = None) -> np.ndarray:
        """Maç geçmişini (son N maç) numpy dizisine çevir.

        Returns: shape (seq_len, n_features)
        """
        sl = seq_len or self.SEQUENCE_LEN
        features = []

        for match in match_history[-sl:]:
            row = [
                float(match.get("goals_scored", 0)),
                float(match.get("goals_conceded", 0)),
                float(match.get("xg", match.get("goals_scored", 0))),
                float(match.get("xga", match.get("goals_conceded", 0))),
                float(match.get("possession", 0.5)),
                self._points(match),
            ]
            features.append(row)

        arr = np.array(features, dtype=np.float32)

        # Padding: yetersiz maç varsa sıfırla doldur
        if len(arr) < sl:
            pad = np.zeros((sl - len(arr), len(self.FEATURE_NAMES)), dtype=np.float32)
            arr = np.vstack([pad, arr])

        return arr

    @staticmethod
    def _points(match: dict) -> float:
        result = match.get("result", "")
        if result == "W":
            return 3.0
        elif result == "D":
            return 1.0
        return 0.0

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalizasyon (0-1 arası)."""
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (arr - mins) / denom


# ═══════════════════════════════════════════════
#  ANA SINIF
# ═══════════════════════════════════════════════
class LSTMTrendAnalyzer:
    """LSTM ile takım momentum/form analizi.

    Kullanım:
        analyzer = LSTMTrendAnalyzer()
        # Eğitim (geçmiş verilerle)
        analyzer.fit(training_data)
        # Tahmin
        momentum = analyzer.predict_momentum("Galatasaray", last_10_matches)
        # momentum: 0.0 (düşüş) ... 1.0 (yükseliş)
    """

    def __init__(self, hidden_size: int = 32, num_layers: int = 2,
                 lr: float = 0.001, epochs: int = 50,
                 device: str = "cpu"):
        self._hidden = hidden_size
        self._layers = num_layers
        self._lr = lr
        self._epochs = epochs
        self._device = device
        self._builder = SequenceBuilder()
        self._model = None
        self._fitted = False

        if TORCH_OK:
            self._model = LSTMNet(
                input_size=len(SequenceBuilder.FEATURE_NAMES),
                hidden_size=hidden_size,
                num_layers=num_layers,
            ).to(device)
        logger.debug(f"LSTMTrendAnalyzer başlatıldı (PyTorch={'✓' if TORCH_OK else '✗'}).")

    # ═══════════════════════════════════════════
    #  EĞİTİM
    # ═══════════════════════════════════════════
    def fit(self, training_data: list[dict], epochs: int | None = None):
        """Modeli eğit.

        training_data: [
            {"team": "GS", "matches": [...last_10...], "next_result": "W"},
            ...
        ]
        """
        if not TORCH_OK:
            logger.info("[LSTM] PyTorch yok – heuristik mod.")
            return

        ep = epochs or self._epochs
        sequences = []
        labels = []

        for sample in training_data:
            matches = sample.get("matches", [])
            if len(matches) < 5:
                continue

            seq = self._builder.build(matches)
            seq = self._builder.normalize(seq)
            sequences.append(seq)

            # Label: sonraki maçın sonucu → momentum
            result = sample.get("next_result", "D")
            label = {"W": 0.9, "D": 0.5, "L": 0.1}.get(result, 0.5)
            labels.append(label)

        if len(sequences) < 20:
            logger.warning(f"[LSTM] Yetersiz eğitim verisi: {len(sequences)}")
            return

        dataset = TeamSequenceDataset(sequences, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        self._model.train()
        best_loss = float("inf")

        for epoch in range(ep):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                logger.debug(f"[LSTM] Epoch {epoch+1}/{ep} – loss: {avg_loss:.4f}")

        self._fitted = True
        logger.success(f"[LSTM] Eğitim tamamlandı: {len(sequences)} örnek, best_loss={best_loss:.4f}")

    # ═══════════════════════════════════════════
    #  TAHMİN
    # ═══════════════════════════════════════════
    def predict_momentum(self, team: str,
                          match_history: list[dict]) -> dict:
        """Takımın momentum skorunu tahmin et.

        Returns:
            {
                "team": str,
                "momentum": float,      # 0.0-1.0
                "trend": str,           # "rising" / "stable" / "falling"
                "confidence": float,    # 0.0-1.0
                "method": str,          # "lstm" / "heuristic"
            }
        """
        if self._fitted and TORCH_OK and self._model:
            return self._predict_lstm(team, match_history)
        else:
            return self._predict_heuristic(team, match_history)

    def _predict_lstm(self, team: str, history: list[dict]) -> dict:
        """LSTM model ile tahmin."""
        seq = self._builder.build(history)
        seq = self._builder.normalize(seq)

        self._model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(seq).unsqueeze(0).to(self._device)
            momentum = self._model(X).item()

        trend = self._classify_trend(momentum)

        return {
            "team": team,
            "momentum": float(np.clip(momentum, 0.0, 1.0)),
            "trend": trend,
            "confidence": 0.7 + 0.3 * abs(momentum - 0.5),
            "method": "lstm",
            "seq_len": len(history),
        }

    def _predict_heuristic(self, team: str, history: list[dict]) -> dict:
        """PyTorch yokken ağırlıklı ortalama ile momentum hesapla."""
        if not history:
            return {
                "team": team, "momentum": 0.5, "trend": "stable",
                "confidence": 0.1, "method": "heuristic",
            }

        # Son maçlara daha fazla ağırlık (exponential)
        n = min(len(history), 10)
        recent = history[-n:]
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()

        # Puan bazlı momentum
        points = []
        for m in recent:
            result = m.get("result", "D")
            pts = {"W": 1.0, "D": 0.5, "L": 0.0}.get(result, 0.5)
            points.append(pts)

        momentum = float(np.average(points, weights=weights))

        # xG trendi
        xgs = [float(m.get("xg", 0)) for m in recent if m.get("xg")]
        if len(xgs) >= 3:
            xg_trend = np.polyfit(range(len(xgs)), xgs, 1)[0]
            momentum += np.clip(xg_trend * 0.1, -0.15, 0.15)

        momentum = float(np.clip(momentum, 0.0, 1.0))
        trend = self._classify_trend(momentum)

        return {
            "team": team,
            "momentum": momentum,
            "trend": trend,
            "confidence": 0.3 + 0.1 * n,
            "method": "heuristic",
            "seq_len": n,
        }

    @staticmethod
    def _classify_trend(momentum: float) -> str:
        if momentum > 0.65:
            return "rising"
        elif momentum < 0.35:
            return "falling"
        return "stable"

    # ═══════════════════════════════════════════
    #  TOPLU TAHMİN + ENSEMBLE ENTEGRASYONU
    # ═══════════════════════════════════════════
    def predict_for_match(self, home_team: str, away_team: str,
                           home_history: list[dict],
                           away_history: list[dict]) -> dict:
        """İki takımın momentumunu karşılaştır, maç tahmini üret."""
        home_m = self.predict_momentum(home_team, home_history)
        away_m = self.predict_momentum(away_team, away_history)

        home_mom = home_m["momentum"]
        away_mom = away_m["momentum"]
        diff = home_mom - away_mom

        # Momentum farkından olasılık tahmini
        base_home = 0.45 + diff * 0.2
        base_away = 0.30 - diff * 0.2
        base_draw = 1.0 - base_home - base_away

        # Normalize
        total = base_home + base_draw + base_away
        prob_home = max(base_home / total, 0.05)
        prob_draw = max(base_draw / total, 0.05)
        prob_away = max(base_away / total, 0.05)
        total2 = prob_home + prob_draw + prob_away

        return {
            "home_momentum": home_m,
            "away_momentum": away_m,
            "momentum_diff": float(diff),
            "prob_home": prob_home / total2,
            "prob_draw": prob_draw / total2,
            "prob_away": prob_away / total2,
            "confidence": (home_m["confidence"] + away_m["confidence"]) / 2,
        }

    def save_model(self, path: str = "models/lstm_trend.pt"):
        """Model ağırlıklarını kaydet."""
        if TORCH_OK and self._model and self._fitted:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self._model.state_dict(), path)
            logger.info(f"[LSTM] Model kaydedildi: {path}")

    def load_model(self, path: str = "models/lstm_trend.pt"):
        """Model ağırlıklarını yükle."""
        if TORCH_OK and self._model:
            try:
                self._model.load_state_dict(torch.load(path, map_location=self._device))
                self._fitted = True
                logger.info(f"[LSTM] Model yüklendi: {path}")
            except Exception as e:
                logger.warning(f"[LSTM] Model yüklenemedi: {e}")
