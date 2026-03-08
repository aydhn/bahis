"""
active_inference_agent.py – Active Inference (Serbest Enerji İlkesi).

Karl Friston'ın Free Energy Principle'ını futbol bahisine uygular.
Bot sürpriz yaşadığında "iç modelimde neresi hatalıydı?" diye sorarak
o bölgeyi otomatik olarak daha çok eğitir.

Kavramlar:
  - Free Energy Principle: Organizmalar (ve ajanlar) sürprizi (surprise)
    minimize etmeye çalışır
  - Surprisal: -log P(gözlem | model) → model beklentisiyle
    gerçekleşen arasındaki fark
  - Variational Free Energy: F = E_q[log q(s) - log p(o,s)]
    → KL divergence + belirsizlik
  - Active Sampling: En bilgi verici veriye odaklanma
    (expected information gain)
  - World Model: Botun "dünya anlayışı" — her modülün güvenilirliği
  - Precision Weighting: Güvenilir modüllere daha fazla ağırlık
  - Epistemic Value: "Bu veri ne kadar öğretici?" sorusunun cevabı

Akış:
  1. Her modülün (scraper, ml, poisson) tahmin doğruluğunu takip et
  2. Maç sonucu geldiğinde surprisal hesapla
  3. Surprisal yüksek modüller = dünya modeli hatalı
  4. Bu modüllere daha fazla kaynak ayır (Active Sampling)
  5. Precision'ı güncelle — güvenilir modüle daha çok güven

Teknoloji: pymdp veya özel numpy Active Inference
Fallback: Basit Bayesian belief update + surprisal takibi
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from pymdp import utils as pymdp_utils
    from pymdp.agent import Agent as PyMDPAgent
    PYMDP_OK = True
except ImportError:
    PYMDP_OK = False
    logger.debug("pymdp yüklü değil – numpy Active Inference fallback.")

try:
    from scipy.special import softmax, logsumexp
    from scipy.stats import entropy as sp_entropy
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class ModuleState:
    """Bir modülün iç durumu."""
    name: str = ""
    # Performans
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5
    # Serbest enerji
    surprisal_history: list[float] = field(default_factory=list)
    avg_surprisal: float = 0.0
    precision: float = 1.0        # Güvenilirlik (yüksek = güvenilir)
    # Kaynak tahsisi
    resource_weight: float = 1.0  # Eğitim kaynağı ağırlığı
    needs_retraining: bool = False
    last_update: float = 0.0


@dataclass
class ActiveInferenceReport:
    """Active Inference raporu."""
    # Genel
    total_free_energy: float = 0.0
    avg_surprisal: float = 0.0
    # Modül bazlı
    module_states: dict[str, ModuleState] = field(default_factory=dict)
    # Eylemler
    retrain_targets: list[str] = field(default_factory=list)
    resource_allocation: dict[str, float] = field(default_factory=dict)
    # Öneriler
    active_sampling_targets: list[str] = field(default_factory=list)
    recommendation: str = ""
    method: str = ""


@dataclass
class BeliefState:
    """Dünya modeli inancı."""
    # Her durum için olasılık vektörü
    beliefs: np.ndarray = field(default_factory=lambda: np.array([1 / 3] * 3))
    # Geçiş modeli
    transition_model: np.ndarray | None = None
    # Gözlem modeli
    observation_model: np.ndarray | None = None


# ═══════════════════════════════════════════════
#  SURPRISAL & FREE ENERGY
# ═══════════════════════════════════════════════
def surprisal(predicted_probs: np.ndarray, observed: int) -> float:
    """Surprisal: -log P(gözlem | model).

    Args:
        predicted_probs: Modelin verdiği olasılıklar [P(home), P(draw), P(away)]
        observed: Gerçekleşen sonuç (0=home, 1=draw, 2=away)
    """
    p = np.clip(predicted_probs, 1e-10, 1.0)
    if observed < len(p):
        return float(-np.log(p[observed]))
    return float(-np.log(1e-10))


def free_energy(q_beliefs: np.ndarray, log_joint: np.ndarray) -> float:
    """Variational Free Energy: F = E_q[log q - log p(o,s)].

    Args:
        q_beliefs: Approximate posterior q(s)
        log_joint: log p(o,s) for each state
    """
    q = np.clip(q_beliefs, 1e-10, 1.0)
    q = q / q.sum()
    log_q = np.log(q)
    return float(np.sum(q * (log_q - log_joint)))


def expected_information_gain(beliefs: np.ndarray,
                                observation_model: np.ndarray) -> np.ndarray:
    """Beklenen bilgi kazanımı (her eylem/gözlem için).

    Hangi veri en çok bilgi verir?
    """
    n_states = len(beliefs)
    n_obs = observation_model.shape[0] if observation_model.ndim > 1 else 1

    gains = np.zeros(n_obs)
    for o in range(n_obs):
        if observation_model.ndim > 1:
            likelihood = observation_model[o]
        else:
            likelihood = observation_model

        # Posterior (Bayes)
        posterior = likelihood * beliefs
        evidence = posterior.sum()
        if evidence > 0:
            posterior /= evidence

        # KL(posterior || prior) = bilgi kazanımı
        kl = 0.0
        for s in range(n_states):
            if posterior[s] > 1e-10 and beliefs[s] > 1e-10:
                kl += posterior[s] * np.log(posterior[s] / beliefs[s])
        gains[o] = kl

    return gains


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Basit Bayesian belief update."""
    posterior = prior * likelihood
    total = posterior.sum()
    if total > 0:
        posterior /= total
    else:
        posterior = np.ones_like(prior) / len(prior)
    return posterior


# ═══════════════════════════════════════════════
#  ACTIVE INFERENCE AGENT (Ana Sınıf)
# ═══════════════════════════════════════════════
class ActiveInferenceAgent:
    """Active Inference tabanlı otonom öğrenme ajanı.

    Kullanım:
        aia = ActiveInferenceAgent(modules=["poisson", "lgbm", "lstm", "rl"])

        # Tahmin yaptıktan sonra gerçek sonuç geldiğinde
        aia.observe(
            module="lgbm",
            predicted_probs=[0.6, 0.25, 0.15],
            observed=0,   # Ev sahibi kazandı
        )

        # Periyodik rapor
        report = aia.get_report()

        # Hangi modüller yeniden eğitilmeli?
        targets = aia.get_retrain_targets()
    """

    # Eşik değerleri
    SURPRISAL_THRESHOLD = 2.0     # Üstünde → modül hatalı
    PRECISION_DECAY = 0.95        # Her yanlışta güvenilirlik düşer
    PRECISION_BOOST = 1.02        # Her doğruda güvenilirlik artar
    RETRAIN_ACCURACY_THRESHOLD = 0.4  # Altında → yeniden eğit
    ACTIVE_SAMPLING_THRESHOLD = 1.5   # Surprisal yüksekse → daha çok veri topla

    def __init__(self, modules: list[str] | None = None):
        self._modules: dict[str, ModuleState] = {}

        default_modules = modules or [
            "poisson", "lightgbm", "lstm", "rl_trader",
            "ensemble", "sentiment", "xai",
        ]
        for name in default_modules:
            self._modules[name] = ModuleState(
                name=name,
                precision=1.0,
                resource_weight=1.0,
            )

        # Global dünya modeli
        self._world_beliefs = BeliefState()
        self._total_observations = 0
        self._global_surprisal_history: list[float] = []

        logger.debug(
            f"[ActiveInf] Başlatıldı: {len(self._modules)} modül"
        )

    def observe(self, module: str,
                  predicted_probs: list[float] | np.ndarray,
                  observed: int,
                  match_id: str = "") -> float:
        """Gözlem al, dünya modelini güncelle.

        Args:
            module: Modül adı (ör: "lightgbm")
            predicted_probs: Modülün tahmini [P(home), P(draw), P(away)]
            observed: Gerçekleşen sonuç (0, 1, 2)

        Returns:
            Surprisal değeri
        """
        probs = np.array(predicted_probs, dtype=np.float64)
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum()

        s = surprisal(probs, observed)

        # Modül durumu güncelle
        if module not in self._modules:
            self._modules[module] = ModuleState(name=module)

        state = self._modules[module]
        state.total_predictions += 1
        state.surprisal_history.append(s)
        state.avg_surprisal = float(np.mean(state.surprisal_history[-50:]))
        state.last_update = time.time()

        # Doğru mu?
        predicted_class = int(np.argmax(probs))
        if predicted_class == observed:
            state.correct_predictions += 1
            state.precision = min(5.0, state.precision * self.PRECISION_BOOST)
        else:
            state.precision = max(0.1, state.precision * self.PRECISION_DECAY)

        state.accuracy = (
            state.correct_predictions / max(state.total_predictions, 1)
        )

        # Yeniden eğitim gerekli mi?
        state.needs_retraining = (
            state.accuracy < self.RETRAIN_ACCURACY_THRESHOLD
            or state.avg_surprisal > self.SURPRISAL_THRESHOLD
        )

        # Kaynak ağırlığı (precision-weighted)
        state.resource_weight = max(0.1, 1.0 / max(state.precision, 0.1))

        # Global
        self._total_observations += 1
        self._global_surprisal_history.append(s)

        # Dünya modeli beliefs güncelle
        likelihood = np.zeros(3)
        likelihood[observed] = 1.0
        self._world_beliefs.beliefs = bayesian_update(
            self._world_beliefs.beliefs, probs,
        )

        return s

    def get_retrain_targets(self) -> list[str]:
        """Yeniden eğitilmesi gereken modüller."""
        targets = []
        for name, state in self._modules.items():
            if state.needs_retraining:
                targets.append(name)
        return sorted(targets, key=lambda n: -self._modules[n].avg_surprisal)

    def get_resource_allocation(self) -> dict[str, float]:
        """Kaynak tahsisi (eğitim zamanı/verisi)."""
        total_weight = sum(s.resource_weight for s in self._modules.values())
        if total_weight == 0:
            total_weight = 1.0
        return {
            name: round(s.resource_weight / total_weight, 3)
            for name, s in self._modules.items()
        }

    def get_precision_weights(self) -> dict[str, float]:
        """Ensemble ağırlıkları (precision-weighted)."""
        total_prec = sum(s.precision for s in self._modules.values())
        if total_prec == 0:
            total_prec = 1.0
        return {
            name: round(s.precision / total_prec, 3)
            for name, s in self._modules.items()
        }

    def get_active_sampling_targets(self) -> list[str]:
        """Daha fazla veriye ihtiyaç duyan alanlar."""
        targets = []
        for name, state in self._modules.items():
            if state.avg_surprisal > self.ACTIVE_SAMPLING_THRESHOLD:
                targets.append(name)
        return targets

    def get_report(self) -> ActiveInferenceReport:
        """Aktif çıkarım raporu."""
        report = ActiveInferenceReport()

        # Global free energy (yaklaşık)
        if self._global_surprisal_history:
            report.avg_surprisal = round(
                float(np.mean(self._global_surprisal_history[-100:])), 4,
            )
            beliefs = self._world_beliefs.beliefs
            log_joint = np.log(np.clip(beliefs, 1e-10, 1.0))
            report.total_free_energy = round(
                free_energy(beliefs, log_joint), 4,
            )

        report.module_states = dict(self._modules)
        report.retrain_targets = self.get_retrain_targets()
        report.resource_allocation = self.get_resource_allocation()
        report.active_sampling_targets = self.get_active_sampling_targets()
        report.method = "active_inference"
        report.recommendation = self._advice(report)
        return report

    def _advice(self, r: ActiveInferenceReport) -> str:
        n_retrain = len(r.retrain_targets)
        n_sampling = len(r.active_sampling_targets)

        if n_retrain > 0:
            modules_str = ", ".join(r.retrain_targets[:3])
            return (
                f"YENİDEN EĞİTİM GEREKLİ: {modules_str}. "
                f"Ort. surprisal: {r.avg_surprisal:.3f}. "
                f"Bu modüller dünya modeliyle uyumsuz."
            )
        if n_sampling > 0:
            return (
                f"AKTİF ÖRNEKLEME: {', '.join(r.active_sampling_targets[:3])} "
                f"alanlarında daha fazla veri gerekli."
            )
        return (
            f"STABIL: Tüm modüller uyumlu. "
            f"Ort. surprisal: {r.avg_surprisal:.3f}. "
            f"Serbest enerji: {r.total_free_energy:.3f}."
        )
