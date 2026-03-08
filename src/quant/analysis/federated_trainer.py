"""
federated_trainer.py – Federated Learning (Dağıtık Sürü Eğitimi).

Tek bir devasa model eğitmek (Monolitik) risklidir. İngiltere Ligi
verisi, Brezilya Ligi modelini bozabilir (Catastrophic Forgetting).

Sürü Mimarisi:
  - Her lig için ayrı bir "Client" (Öğrenci Model) oluşturulur
  - Her client kendi yerel verisiyle (On-Device) eğitilir
  - Sadece ağırlıklar (Weights) merkeze gönderilir
  - Merkez, FedAvg ile ortalama alıp Global Model'i günceller
  - Güncellenmiş global ağırlıklar tüm client'lara geri dağıtılır

Avantajlar:
  - Veri gizliliği: Lig verisi asla merkeze gitmez
  - Catastrophic Forgetting engellenir
  - Yeni ligler kolayca eklenebilir
  - Her lig kendi karakteristiğini korur

Teknoloji: flower (flwr) veya manuel FedAvg implementasyonu
Fallback: Basit ağırlık ortalama (Weight Averaging) PyTorch/sklearn
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import flwr as fl
    FLOWER_OK = True
except ImportError:
    FLOWER_OK = False
    logger.debug("flwr yüklü değil – manuel FedAvg fallback.")

ROOT = Path(__file__).resolve().parent.parent.parent
FED_DIR = ROOT / "models" / "federated"
FED_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class ClientReport:
    """Bir lig client'ının eğitim raporu."""
    league: str = ""
    samples: int = 0
    epochs: int = 0
    accuracy: float = 0.0
    loss: float = 0.0
    round_num: int = 0


@dataclass
class FederatedReport:
    """Global federasyon raporu."""
    total_rounds: int = 0
    num_clients: int = 0
    leagues: list[str] = field(default_factory=list)
    global_accuracy: float = 0.0
    client_reports: list[ClientReport] = field(default_factory=list)
    improvement_pct: float = 0.0
    convergence_round: int = 0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  LİG MODELİ (Client)
# ═══════════════════════════════════════════════
if TORCH_OK:
    class LeagueNet(nn.Module):
        """Her lig için hafif yerel model."""

        def __init__(self, input_dim: int = 20, n_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class FederatedClient:
    """Tek bir ligin yerel eğitim client'ı."""

    def __init__(self, league: str, input_dim: int = 20,
                 n_classes: int = 3, lr: float = 0.01):
        self.league = league
        self._input_dim = input_dim
        self._n_classes = n_classes

        if TORCH_OK:
            self._model = LeagueNet(input_dim, n_classes)
            self._optimizer = optim.SGD(self._model.parameters(), lr=lr)
            self._criterion = nn.CrossEntropyLoss()
            self._method = "pytorch"
        elif SKLEARN_OK:
            self._model = SGDClassifier(
                loss="log_loss", warm_start=True,
                max_iter=50, random_state=42,
            )
            self._scaler = StandardScaler()
            self._method = "sklearn"
        else:
            self._model = None
            self._method = "none"

    def train_local(self, X: np.ndarray, y: np.ndarray,
                     epochs: int = 5, batch_size: int = 32) -> ClientReport:
        """Yerel veri ile eğit (veri merkeze gitmez)."""
        report = ClientReport(
            league=self.league, samples=len(X), epochs=epochs,
        )

        if self._method == "pytorch" and TORCH_OK:
            X_t = torch.FloatTensor(X)
            y_t = torch.LongTensor(y)
            ds = TensorDataset(X_t, y_t)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            self._model.train()
            for ep in range(epochs):
                total_loss = 0
                for bx, by in loader:
                    self._optimizer.zero_grad()
                    out = self._model(bx)
                    loss = self._criterion(out, by)
                    loss.backward()
                    self._optimizer.step()
                    total_loss += loss.item()

            self._model.eval()
            with torch.no_grad():
                preds = self._model(X_t).argmax(dim=1)
                report.accuracy = float((preds == y_t).float().mean())
            report.loss = total_loss / max(len(loader), 1)

        elif self._method == "sklearn" and SKLEARN_OK:
            X_s = self._scaler.fit_transform(X)
            self._model.fit(X_s, y)
            report.accuracy = float(self._model.score(X_s, y))

        return report

    def get_weights(self) -> list[np.ndarray]:
        """Model ağırlıklarını döndür (merkeze gönderilecek)."""
        if TORCH_OK and isinstance(self._model, nn.Module):
            return [
                p.detach().cpu().numpy().copy()
                for p in self._model.parameters()
            ]
        if SKLEARN_OK and hasattr(self._model, "coef_"):
            return [self._model.coef_.copy(), self._model.intercept_.copy()]
        return []

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Global ağırlıkları yükle (merkezden geldi)."""
        if TORCH_OK and isinstance(self._model, nn.Module):
            for param, w in zip(self._model.parameters(), weights):
                param.data = torch.FloatTensor(w)
        elif SKLEARN_OK and hasattr(self._model, "coef_") and len(weights) >= 2:
            self._model.coef_ = weights[0]
            self._model.intercept_ = weights[1]


# ═══════════════════════════════════════════════
#  FEDAVG – Merkezi Toplama (Aggregation)
# ═══════════════════════════════════════════════
def federated_average(client_weights: list[list[np.ndarray]],
                       sample_counts: list[int] | None = None
                       ) -> list[np.ndarray]:
    """FedAvg: Ağırlıklı ortalama ile global model güncelle.

    Her client'ın ağırlığı, örneklem sayısıyla orantılıdır.
    """
    if not client_weights:
        return []

    n_clients = len(client_weights)
    n_layers = len(client_weights[0])

    if sample_counts is None:
        sample_counts = [1] * n_clients

    total_samples = sum(sample_counts)
    if total_samples == 0:
        total_samples = 1

    averaged = []
    for layer_idx in range(n_layers):
        weighted_sum = np.zeros_like(client_weights[0][layer_idx])
        for c_idx in range(n_clients):
            if layer_idx < len(client_weights[c_idx]):
                weight = sample_counts[c_idx] / total_samples
                weighted_sum += client_weights[c_idx][layer_idx] * weight
        averaged.append(weighted_sum)

    return averaged


# ═══════════════════════════════════════════════
#  FEDERATED TRAINER (Ana Sınıf)
# ═══════════════════════════════════════════════
class FederatedTrainer:
    """Dağıtık Sürü Eğitimi orkestratörü.

    Kullanım:
        ft = FederatedTrainer(leagues=["premier_league", "super_lig", "bundesliga"])

        # Lig verilerini yükle
        ft.load_league_data("premier_league", X_pl, y_pl)
        ft.load_league_data("super_lig", X_sl, y_sl)

        # Federasyon eğitimi başlat
        report = ft.train(n_rounds=10, local_epochs=5)

        # Tahmin (global model)
        probs = ft.predict(X_new)
    """

    def __init__(self, leagues: list[str] | None = None,
                 input_dim: int = 20, n_classes: int = 3,
                 lr: float = 0.01):
        self._input_dim = input_dim
        self._n_classes = n_classes
        self._leagues = leagues or ["super_lig"]

        # Her lig için bir client
        self._clients: dict[str, FederatedClient] = {}
        for league in self._leagues:
            self._clients[league] = FederatedClient(
                league, input_dim, n_classes, lr,
            )

        # Lig verileri
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Global model ağırlıkları
        self._global_weights: list[np.ndarray] = []

        logger.debug(
            f"[Federated] Trainer başlatıldı: "
            f"{len(self._leagues)} lig client'ı"
        )

    def load_league_data(self, league: str,
                          X: np.ndarray, y: np.ndarray) -> None:
        """Lig verisini client'a yükle."""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        self._data[league] = (X, y)

        if league not in self._clients:
            self._clients[league] = FederatedClient(
                league, self._input_dim, self._n_classes,
            )
            self._leagues.append(league)

        logger.debug(f"[Federated] {league}: {len(X)} örnek yüklendi.")

    def train(self, n_rounds: int = 10,
              local_epochs: int = 5) -> FederatedReport:
        """Federasyon eğitimi çalıştır."""
        report = FederatedReport(
            total_rounds=n_rounds,
            num_clients=len(self._clients),
            leagues=list(self._clients.keys()),
        )

        best_acc = 0.0
        convergence_round = 0

        for rnd in range(1, n_rounds + 1):
            round_weights = []
            round_samples = []
            round_reports = []

            # Her client yerel eğitim yapar
            for league, client in self._clients.items():
                if league not in self._data:
                    continue

                # Global ağırlıkları dağıt (ilk round hariç)
                if self._global_weights:
                    client.set_weights(self._global_weights)

                X, y = self._data[league]
                cr = client.train_local(X, y, epochs=local_epochs)
                cr.round_num = rnd

                weights = client.get_weights()
                if weights:
                    round_weights.append(weights)
                    round_samples.append(len(X))
                round_reports.append(cr)

            # FedAvg: ağırlıkları ortala
            if round_weights:
                self._global_weights = federated_average(
                    round_weights, round_samples,
                )

            # Round doğruluğu
            round_acc = np.mean([r.accuracy for r in round_reports]) if round_reports else 0
            if round_acc > best_acc:
                best_acc = round_acc
                convergence_round = rnd

            if rnd == n_rounds:
                report.client_reports = round_reports

        report.global_accuracy = best_acc
        report.convergence_round = convergence_round
        report.method = "fedavg_pytorch" if TORCH_OK else "fedavg_sklearn"
        report.recommendation = self._advice(report)

        # Global ağırlıkları tüm client'lara dağıt
        if self._global_weights:
            for client in self._clients.values():
                client.set_weights(self._global_weights)

        logger.info(
            f"[Federated] Eğitim tamamlandı: {n_rounds} round, "
            f"global_acc={best_acc:.1%}, "
            f"yakınsama={convergence_round}. round"
        )

        return report

    def predict(self, X: np.ndarray,
                league: str | None = None) -> np.ndarray:
        """Tahmin yap (belirli lig veya global)."""
        X = np.array(X, dtype=np.float32)
        client = None

        if league and league in self._clients:
            client = self._clients[league]
        elif self._clients:
            client = next(iter(self._clients.values()))

        if client is None:
            return np.ones((len(X), self._n_classes)) / self._n_classes

        if TORCH_OK and isinstance(client._model, nn.Module):
            client._model.eval()
            with torch.no_grad():
                out = client._model(torch.FloatTensor(X))
                probs = torch.softmax(out, dim=1)
            return probs.numpy()

        if SKLEARN_OK and hasattr(client._model, "predict_proba"):
            X_s = client._scaler.transform(X) if hasattr(client, "_scaler") else X
            return client._model.predict_proba(X_s)

        return np.ones((len(X), self._n_classes)) / self._n_classes

    def save_global(self, name: str = "global") -> Path:
        """Global model ağırlıklarını kaydet."""
        path = FED_DIR / f"fed_{name}.npz"
        if self._global_weights:
            np.savez(path, *self._global_weights)
            logger.info(f"[Federated] Global model kaydedildi: {path}")
        return path

    def load_global(self, name: str = "global") -> bool:
        """Global model ağırlıklarını yükle."""
        path = FED_DIR / f"fed_{name}.npz"
        if path.exists():
            try:
                data = np.load(path, allow_pickle=True)
                self._global_weights = [data[k] for k in data.files]
                for client in self._clients.values():
                    client.set_weights(self._global_weights)
                logger.info(f"[Federated] Global model yüklendi: {path}")
                return True
            except Exception as e:
                logger.debug(f"[Federated] Yükleme hatası: {e}")
        return False

    def add_league(self, league: str) -> None:
        """Yeni lig client'ı ekle."""
        if league not in self._clients:
            self._clients[league] = FederatedClient(
                league, self._input_dim, self._n_classes,
            )
            self._leagues.append(league)
            if self._global_weights:
                self._clients[league].set_weights(self._global_weights)
            logger.info(f"[Federated] Yeni lig eklendi: {league}")

    def _advice(self, report: FederatedReport) -> str:
        if report.global_accuracy > 0.6:
            return (
                f"Federasyon başarılı: {report.global_accuracy:.1%} doğruluk. "
                f"{report.num_clients} lig modeli senkronize."
            )
        elif report.global_accuracy > 0.4:
            return (
                f"Orta düzey: {report.global_accuracy:.1%}. "
                f"Daha fazla round veya veri gerekebilir."
            )
        return (
            f"Düşük doğruluk: {report.global_accuracy:.1%}. "
            f"Lig verileri çok farklı olabilir – local-only modu deneyin."
        )
