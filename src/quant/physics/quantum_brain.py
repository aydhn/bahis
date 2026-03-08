"""
quantum_brain.py – Quantum Machine Learning (VQC / QSVM).

Klasik sinir ağları "Yerel Minimum" tuzaklarına düşer.
Kuantum devreleri ise Hilbert Uzayı'nda çözüm arar.

Kavramlar:
  - Qubit: Kuantum bilgi birimi (|0⟩ ve |1⟩ süperpozisyonu)
  - Quantum Gate: Qubit üzerinde dönüşüm (RX, RY, RZ, CNOT)
  - Entanglement: Dolanıklık (kuantum korelasyonu)
  - VQC: Variational Quantum Circuit (parametrik devre)
  - QSVM: Kuantum Destek Vektör Makinesi
  - Measurement: Ölçüm → klasik sonuç (0/1)

Mimari:
  1. Feature Encoding: Klasik veri → qubit durumları (RX rotasyonu)
  2. Ansatz: Parametrik kuantum devresi (CNOT + RY katmanları)
  3. Measurement: Ölçüm → beklenen değer (expectation)
  4. Optimization: Klasik optimizer parametreleri günceller

Teknoloji: PennyLane (Xanadu) – Python kuantum simülatörü
Fallback: Basit kernel trick (RBF) + sinüs kodlama
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_OK = True
except ImportError:
    PENNYLANE_OK = False
    logger.debug("pennylane yüklü değil – klasik kernel fallback.")

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class QuantumPrediction:
    """Kuantum tahmin sonucu."""
    match_id: str = ""
    prediction: int = 1              # 0=home, 1=draw, 2=away
    probabilities: list[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    confidence: float = 0.0
    method: str = ""
    n_qubits: int = 0
    circuit_depth: int = 0
    compute_time_ms: float = 0.0


@dataclass
class QuantumReport:
    """Kuantum modeli raporu."""
    accuracy: float = 0.0
    n_qubits: int = 4
    n_layers: int = 2
    n_params: int = 0
    training_epochs: int = 0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  KUANTUM KERNEL (Fallback)
# ═══════════════════════════════════════════════
def quantum_kernel_classical(X1: np.ndarray, X2: np.ndarray,
                               gamma: float = 1.0) -> np.ndarray:
    """Kuantum-esinlenmiş klasik kernel.

    Sinüs kodlama + RBF benzeri mesafe:
    K(x,y) = |⟨φ(x)|φ(y)⟩|² ≈ exp(-γ·Σ sin²((x_i - y_i)/2))
    """
    n1, d = X1.shape
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            diff = X1[i] - X2[j]
            sin_sq = np.sin(diff / 2) ** 2
            K[i, j] = np.exp(-gamma * np.sum(sin_sq))

    return K


# ═══════════════════════════════════════════════
#  PENNYLANE VQC
# ═══════════════════════════════════════════════
if PENNYLANE_OK:
    def create_vqc(n_qubits: int = 4, n_layers: int = 2):
        """Variational Quantum Circuit oluştur."""
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs, weights):
            # Feature Encoding: RX rotasyonu ile veri kodlama
            for i in range(n_qubits):
                idx = i % len(inputs)
                qml.RX(inputs[idx], wires=i)

            # Ansatz: Parametrik katmanlar
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Dolanıklık (Entanglement)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            # Ölçüm
            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, 3))]

        return circuit, dev


# ═══════════════════════════════════════════════
#  QUANTUM BRAIN (Ana Sınıf)
# ═══════════════════════════════════════════════
class QuantumBrain:
    """Kuantum Makine Öğrenmesi motoru.

    Kullanım:
        qb = QuantumBrain(n_qubits=4, n_layers=2)

        # Eğit
        qb.train(X_train, y_train, epochs=50)

        # Tahmin
        pred = qb.predict(X_test)

        # Tek maç
        result = qb.predict_match(features, match_id="gs_fb")
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 lr: float = 0.01):
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._lr = lr
        self._weights = None
        self._scaler = StandardScaler() if SKLEARN_OK else None
        self._fitted = False

        if PENNYLANE_OK:
            self._circuit, self._dev = create_vqc(n_qubits, n_layers)
            n_params = n_layers * n_qubits * 2
            self._weights = pnp.random.randn(n_layers, n_qubits, 2) * 0.1
            self._method = "pennylane_vqc"
        elif SKLEARN_OK:
            self._svc = SVC(kernel="precomputed", probability=True)
            self._method = "quantum_kernel_svc"
        else:
            self._method = "none"

        logger.debug(
            f"[Quantum] Brain başlatıldı: {n_qubits} qubit, "
            f"{n_layers} katman, method={self._method}"
        )

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50) -> QuantumReport:
        """Modeli eğit."""
        report = QuantumReport(
            n_qubits=self._n_qubits,
            n_layers=self._n_layers,
            training_epochs=epochs,
            method=self._method,
        )

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        if self._scaler:
            X = self._scaler.fit_transform(X)

        if self._method == "pennylane_vqc" and PENNYLANE_OK:
            report = self._train_vqc(X, y, epochs, report)
        elif self._method == "quantum_kernel_svc" and SKLEARN_OK:
            report = self._train_kernel(X, y, report)

        self._fitted = True
        report.recommendation = self._advice(report)
        return report

    def _train_vqc(self, X: np.ndarray, y: np.ndarray,
                     epochs: int, report: QuantumReport) -> QuantumReport:
        """PennyLane VQC eğitimi."""
        opt = qml.GradientDescentOptimizer(self._lr)
        n = len(X)
        report.n_params = self._n_layers * self._n_qubits * 2

        for epoch in range(epochs):
            for i in range(n):
                features = X[i, :self._n_qubits]

                def cost(w):
                    preds = self._circuit(features, w)
                    target = [1.0 if y[i] == c else -1.0 for c in range(min(self._n_qubits, 3))]
                    return sum((p - t) ** 2 for p, t in zip(preds, target))

                self._weights = opt.step(cost, self._weights)

        # Doğruluk hesapla
        correct = 0
        for i in range(n):
            pred = self._predict_single_vqc(X[i])
            if pred == y[i]:
                correct += 1
        report.accuracy = round(correct / max(n, 1), 4)

        return report

    def _train_kernel(self, X: np.ndarray, y: np.ndarray,
                        report: QuantumReport) -> QuantumReport:
        """Quantum kernel SVC eğitimi."""
        self._X_train = X
        K = quantum_kernel_classical(X, X)
        self._svc.fit(K, y)
        report.accuracy = round(float(self._svc.score(K, y)), 4)
        return report

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Toplu tahmin."""
        X = np.array(X, dtype=np.float64)
        if self._scaler and self._fitted:
            X = self._scaler.transform(X)

        if self._method == "pennylane_vqc" and PENNYLANE_OK:
            preds = [self._predict_single_vqc(X[i]) for i in range(len(X))]
            return np.array(preds)
        elif self._method == "quantum_kernel_svc" and SKLEARN_OK and hasattr(self, "_X_train"):
            K = quantum_kernel_classical(X, self._X_train)
            return self._svc.predict(K)

        return np.ones(len(X), dtype=int)

    def predict_match(self, features: list[float] | np.ndarray,
                        match_id: str = "") -> QuantumPrediction:
        """Tek maç tahmini."""
        t0 = time.perf_counter()
        result = QuantumPrediction(
            match_id=match_id,
            method=self._method,
            n_qubits=self._n_qubits,
            circuit_depth=self._n_layers * 3,
        )

        feat = np.array(features, dtype=np.float64).flatten()
        if self._scaler and self._fitted:
            feat = self._scaler.transform(feat.reshape(1, -1))[0]

        if self._method == "pennylane_vqc" and PENNYLANE_OK and self._weights is not None:
            pred = self._predict_single_vqc(feat)
            result.prediction = pred
            # Basit olasılık (raw output'tan)
            raw = self._circuit(feat[:self._n_qubits], self._weights)
            probs = np.array([(r + 1) / 2 for r in raw])
            probs = probs / max(probs.sum(), 1e-8)
            result.probabilities = [round(float(p), 4) for p in probs]
            result.confidence = round(float(max(probs)), 4)
        elif self._method == "quantum_kernel_svc" and SKLEARN_OK and hasattr(self, "_X_train"):
            K = quantum_kernel_classical(feat.reshape(1, -1), self._X_train)
            pred = int(self._svc.predict(K)[0])
            result.prediction = pred
            probs = self._svc.predict_proba(K)[0]
            result.probabilities = [round(float(p), 4) for p in probs]
            result.confidence = round(float(max(probs)), 4)
        else:
            result.probabilities = [0.33, 0.34, 0.33]
            result.confidence = 0.34

        result.compute_time_ms = round(
            (time.perf_counter() - t0) * 1000, 3,
        )
        return result

    def _predict_single_vqc(self, x: np.ndarray) -> int:
        """VQC ile tek tahmin."""
        features = x[:self._n_qubits]
        raw = self._circuit(features, self._weights)
        return int(np.argmax(raw))

    def _advice(self, report: QuantumReport) -> str:
        if report.accuracy > 0.5:
            return (
                f"Kuantum model çalışıyor: {report.accuracy:.1%} doğruluk, "
                f"{report.n_qubits} qubit, {report.n_layers} katman."
            )
        return (
            f"Düşük doğruluk: {report.accuracy:.1%}. "
            f"Daha fazla eğitim veya qubit sayısı artırın."
        )
