"""
transfer_learner.py – Transfer Learning (Bilgi Transferi).

İngiltere Premier Ligi için eğitilen model, Türkiye Süper Ligi'nde
çuvallayabilir. Ama futbolda evrensel kurallar vardır. Modeli
sıfırdan eğitmek yerine, "Öğrendiğini Aktarmasını" sağlayacağız.

Kavramlar:
  - Source Domain: 100.000 maçlık Avrupa verisi (büyük ligler)
  - Target Domain: 500 maçlık Türkiye verisi (küçük lig)
  - Feature Extractor: İlk katmanlar (Temel Futbol Fiziği) → Dondur
  - Fine-Tuning: Son katmanlar (Lig Karakteristiği) → Hedef veriyle eğit
  - Warm Start: Önceden eğitilmiş ağırlıkları başlangıç noktası yap
  - Domain Adaptation: Kaynak ve hedef dağılımları arasındaki farkı kapat

Teknoloji: PyTorch + basit MLP/Transformer
Fallback: scikit-learn warm_start + feature mapping
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.info("torch yüklü değil – sklearn warm_start fallback.")

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "models" / "transfer"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TransferReport:
    """Transfer öğrenme raporu."""
    source_domain: str = ""
    target_domain: str = ""
    source_samples: int = 0
    target_samples: int = 0
    # Performans
    source_accuracy: float = 0.0
    target_accuracy_before: float = 0.0   # Transfer öncesi (sıfırdan)
    target_accuracy_after: float = 0.0    # Transfer sonrası (fine-tune)
    improvement_pct: float = 0.0
    # Model
    frozen_layers: int = 0
    trainable_layers: int = 0
    total_params: int = 0
    trainable_params: int = 0
    epochs: int = 0
    method: str = ""
    # Karar
    transfer_beneficial: bool = False
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  PYTORCH TRANSFER MODEL
# ═══════════════════════════════════════════════
if TORCH_OK:
    class FootballTransferNet(nn.Module):
        """Transfer Learning ağı (PyTorch).

        Mimari:
          Backbone (Dondurulabilir):
            Linear(input, 128) → ReLU → Dropout
            Linear(128, 64) → ReLU → Dropout
          Head (Fine-Tune):
            Linear(64, 32) → ReLU
            Linear(32, n_classes)
        """

        def __init__(self, input_dim: int = 20, n_classes: int = 3,
                     dropout: float = 0.3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            return self.head(features)

        def freeze_backbone(self) -> int:
            """Backbone katmanlarını dondur (transfer için)."""
            frozen = 0
            for param in self.backbone.parameters():
                param.requires_grad = False
                frozen += 1
            return frozen

        def unfreeze_all(self) -> None:
            """Tüm katmanları aç."""
            for param in self.parameters():
                param.requires_grad = True

        def count_params(self) -> tuple[int, int]:
            """(total, trainable) parametre sayısı."""
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            return total, trainable


# ═══════════════════════════════════════════════
#  TRANSFER LEARNER (Ana Sınıf)
# ═══════════════════════════════════════════════
class TransferLearner:
    """Lig arası bilgi transferi motoru.

    Kullanım:
        learner = TransferLearner(input_dim=20, n_classes=3)

        # Büyük ligde eğit (source domain)
        learner.train_source(X_europe, y_europe, epochs=50)

        # Ağırlıkları kaydet
        learner.save("europe_base")

        # Küçük lig: Transfer + Fine-tune
        learner.load("europe_base")
        report = learner.fine_tune(X_turkey, y_turkey, epochs=20)

        # Tahmin
        probs = learner.predict(X_new)
    """

    def __init__(self, input_dim: int = 20, n_classes: int = 3,
                 dropout: float = 0.3, lr: float = 0.001):
        self._input_dim = input_dim
        self._n_classes = n_classes
        self._lr = lr

        if TORCH_OK:
            self._model = FootballTransferNet(input_dim, n_classes, dropout)
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
            self._criterion = nn.CrossEntropyLoss()
            self._method = "pytorch"
        elif SKLEARN_OK:
            self._model = SGDClassifier(
                loss="log_loss", warm_start=True,
                max_iter=100, random_state=42,
            )
            self._scaler = StandardScaler()
            self._method = "sklearn"
        else:
            self._model = None
            self._method = "none"

        self._source_trained = False
        logger.debug(f"[Transfer] Learner başlatıldı (method={self._method}).")

    # ═══════════════════════════════════════════
    #  KAYNAK DOMAIN EĞİTİMİ
    # ═══════════════════════════════════════════
    def train_source(self, X: np.ndarray, y: np.ndarray,
                      epochs: int = 50, batch_size: int = 64,
                      source_name: str = "Europe") -> float:
        """Kaynak domain'de eğit (büyük lig verisi)."""
        if self._method == "pytorch" and TORCH_OK:
            return self._train_pytorch(X, y, epochs, batch_size)
        elif self._method == "sklearn" and SKLEARN_OK:
            return self._train_sklearn(X, y)
        return 0.0

    def _train_pytorch(self, X: np.ndarray, y: np.ndarray,
                        epochs: int, batch_size: int) -> float:
        """PyTorch eğitimi."""
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        self._model.unfreeze_all()

        best_acc = 0.0
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                self._optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            acc = correct / max(total, 1)
            best_acc = max(best_acc, acc)

        self._source_trained = True
        logger.info(f"[Transfer] Source eğitim tamamlandı: acc={best_acc:.1%}")
        return best_acc

    def _train_sklearn(self, X: np.ndarray, y: np.ndarray) -> float:
        """Scikit-learn eğitimi."""
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        acc = float(self._model.score(X_scaled, y))
        self._source_trained = True
        logger.info(f"[Transfer] Source eğitim (sklearn): acc={acc:.1%}")
        return acc

    # ═══════════════════════════════════════════
    #  TRANSFER + FINE-TUNE
    # ═══════════════════════════════════════════
    def fine_tune(self, X: np.ndarray, y: np.ndarray,
                    epochs: int = 20, batch_size: int = 32,
                    target_name: str = "Turkey",
                    freeze_backbone: bool = True) -> TransferReport:
        """Hedef domain'de fine-tune (küçük lig verisi)."""
        report = TransferReport(
            target_domain=target_name,
            target_samples=len(X),
            method=self._method,
        )

        if self._method == "pytorch" and TORCH_OK:
            return self._fine_tune_pytorch(
                X, y, epochs, batch_size, freeze_backbone, report,
            )
        elif self._method == "sklearn" and SKLEARN_OK:
            return self._fine_tune_sklearn(X, y, report)

        report.recommendation = "Model yüklü değil."
        return report

    def _fine_tune_pytorch(self, X: np.ndarray, y: np.ndarray,
                             epochs: int, batch_size: int,
                             freeze_backbone: bool,
                             report: TransferReport) -> TransferReport:
        """PyTorch fine-tuning."""
        # Sıfırdan doğruluk (karşılaştırma için)
        report.target_accuracy_before = self._evaluate_pytorch(X, y)

        # Backbone dondur
        if freeze_backbone and self._source_trained:
            frozen = self._model.freeze_backbone()
            report.frozen_layers = frozen
            # Sadece head için optimizer
            self._optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._model.parameters()),
                lr=self._lr * 0.1,
            )

        total, trainable = self._model.count_params()
        report.total_params = total
        report.trainable_params = trainable
        report.trainable_layers = sum(
            1 for p in self._model.parameters() if p.requires_grad
        )

        # Fine-tune eğitimi
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self._optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)
                loss.backward()
                self._optimizer.step()

        report.epochs = epochs
        report.target_accuracy_after = self._evaluate_pytorch(X, y)

        # İyileşme
        if report.target_accuracy_before > 0:
            report.improvement_pct = round(
                (report.target_accuracy_after - report.target_accuracy_before)
                / report.target_accuracy_before * 100, 2,
            )
        report.transfer_beneficial = (
            report.target_accuracy_after > report.target_accuracy_before
        )

        report.recommendation = self._generate_advice(report)

        # Tüm katmanları aç (sonraki kullanım için)
        self._model.unfreeze_all()
        return report

    def _fine_tune_sklearn(self, X: np.ndarray, y: np.ndarray,
                             report: TransferReport) -> TransferReport:
        """Scikit-learn warm-start fine-tune."""
        X_scaled = self._scaler.transform(X) if self._source_trained else \
                   self._scaler.fit_transform(X)

        report.target_accuracy_before = float(
            self._model.score(X_scaled, y),
        ) if self._source_trained else 0.0

        self._model.fit(X_scaled, y)
        report.target_accuracy_after = float(self._model.score(X_scaled, y))

        if report.target_accuracy_before > 0:
            report.improvement_pct = round(
                (report.target_accuracy_after - report.target_accuracy_before)
                / report.target_accuracy_before * 100, 2,
            )
        report.transfer_beneficial = (
            report.target_accuracy_after > report.target_accuracy_before
        )
        report.method = "sklearn_warm_start"
        report.recommendation = self._generate_advice(report)
        return report

    def _evaluate_pytorch(self, X: np.ndarray, y: np.ndarray) -> float:
        """PyTorch model doğruluğu."""
        self._model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            y_t = torch.LongTensor(y)
            outputs = self._model(X_t)
            _, predicted = torch.max(outputs, 1)
            acc = float((predicted == y_t).sum().item()) / len(y)
        self._model.train()
        return round(acc, 4)

    # ═══════════════════════════════════════════
    #  TAHMİN
    # ═══════════════════════════════════════════
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Olasılık tahmini."""
        if self._method == "pytorch" and TORCH_OK:
            self._model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                outputs = self._model(X_t)
                probs = torch.softmax(outputs, dim=1)
            return probs.numpy()
        elif self._method == "sklearn" and SKLEARN_OK:
            X_scaled = self._scaler.transform(X)
            return self._model.predict_proba(X_scaled)
        return np.ones((len(X), self._n_classes)) / self._n_classes

    # ═══════════════════════════════════════════
    #  KAYDET / YÜKLE
    # ═══════════════════════════════════════════
    def save(self, name: str = "base") -> Path:
        """Model ağırlıklarını kaydet."""
        path = MODEL_DIR / f"transfer_{name}.pt"
        if TORCH_OK and isinstance(self._model, nn.Module):
            torch.save({
                "model_state": self._model.state_dict(),
                "input_dim": self._input_dim,
                "n_classes": self._n_classes,
                "source_trained": self._source_trained,
            }, path)
        else:
            meta = {"method": self._method, "source_trained": self._source_trained}
            path = MODEL_DIR / f"transfer_{name}.json"
            path.write_text(json.dumps(meta))
        logger.info(f"[Transfer] Model kaydedildi: {path}")
        return path

    def load(self, name: str = "base") -> bool:
        """Model ağırlıklarını yükle."""
        path = MODEL_DIR / f"transfer_{name}.pt"
        if TORCH_OK and path.exists():
            try:
                checkpoint = torch.load(path, map_location="cpu",
                                        weights_only=False)
                self._model.load_state_dict(checkpoint["model_state"])
                self._source_trained = checkpoint.get("source_trained", True)
                logger.info(f"[Transfer] Model yüklendi: {path}")
                return True
            except Exception as e:
                logger.debug(f"[Transfer] Yükleme hatası: {e}")
        return False

    # ═══════════════════════════════════════════
    #  TAVSİYE
    # ═══════════════════════════════════════════
    def _generate_advice(self, report: TransferReport) -> str:
        """Transfer öğrenme tavsiyesi."""
        if report.transfer_beneficial:
            return (
                f"Transfer FAYDALI: {report.improvement_pct:+.1f}% iyileşme. "
                f"Source domain bilgisi hedef domain'e aktarıldı."
            )
        elif report.target_accuracy_after > 0.4:
            return (
                f"Transfer nötr: Hedef doğruluk {report.target_accuracy_after:.1%}. "
                f"Daha fazla hedef veri ile fine-tune önerilir."
            )
        else:
            return (
                f"Transfer başarısız: Doğruluk düşük ({report.target_accuracy_after:.1%}). "
                f"Source ve target domain çok farklı olabilir."
            )
