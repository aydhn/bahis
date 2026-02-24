"""
regime_switcher.py – Hidden Markov Models (Gizli Rejim Tespiti).

Takımların "Ruhu" vardır. Bazen "Atak Modu"nda, bazen "Otobüs
Çekme Modu"nda olurlar. Bu durumlar veride açıkça yazmaz,
gizlidir (Hidden States).

Kavramlar:
  - Observable: Şutlar, Paslar, Top kontrolü (görünen veri)
  - Hidden State: "Baskın", "Dengeli", "Pasif" (gizli rejim)
  - Transition Matrix: Rejimler arası geçiş olasılıkları
  - Emission Matrix: Her rejimde gözlem olasılıkları
  - Viterbi: En olası gizli durum dizisini bul
  - Baum-Welch: Model parametrelerini veriden öğren

Sinyaller:
  - Favori takım "Pasif" rejimde → gol olasılığı %40 düşür
  - "Baskın" → "Pasif" geçişi tespit → canlı bahis uyarısı
  - Ani rejim değişimi → momentum kırılması sinyali

Teknoloji: hmmlearn
Fallback: Manuel Viterbi + Baum-Welch (numpy)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM, MultinomialHMM
    HMM_OK = True
except ImportError:
    HMM_OK = False
    logger.debug("hmmlearn yüklü değil – manuel HMM fallback.")


# ═══════════════════════════════════════════════
#  REJIM TANIMLARI
# ═══════════════════════════════════════════════
REGIME_NAMES = {0: "Baskın", 1: "Dengeli", 2: "Pasif"}
REGIME_COLORS = {0: "🟢", 1: "🟡", 2: "🔴"}

# Default geçiş matrisi (takımlar genelde rejimde kalır)
DEFAULT_TRANSITION = np.array([
    [0.70, 0.20, 0.10],  # Baskın → Baskın %70
    [0.15, 0.70, 0.15],  # Dengeli → Dengeli %70
    [0.10, 0.20, 0.70],  # Pasif → Pasif %70
])

# Gol olasılığı çarpanları (rejime göre)
GOAL_MULTIPLIERS = {
    0: 1.30,   # Baskın → gol olasılığı %30 artır
    1: 1.00,   # Dengeli → değişme
    2: 0.60,   # Pasif → gol olasılığı %40 düşür
}


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class RegimeState:
    """Anlık rejim durumu."""
    regime_id: int = 1              # 0=Baskın, 1=Dengeli, 2=Pasif
    regime_name: str = "Dengeli"
    confidence: float = 0.0         # Bu rejimde olma olasılığı
    probabilities: list[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    goal_multiplier: float = 1.0


@dataclass
class RegimeReport:
    """Rejim analiz raporu."""
    team: str = ""
    match_id: str = ""
    # Mevcut rejim
    current: RegimeState = field(default_factory=RegimeState)
    # Rejim geçmişi
    regime_sequence: list[int] = field(default_factory=list)
    regime_names: list[str] = field(default_factory=list)
    # Geçiş analizi
    transition_detected: bool = False
    last_transition: str = ""        # "Baskın → Pasif"
    transition_minute: int = 0
    # Metrikler
    dominance_pct: float = 0.0      # Maçın kaçında baskındı
    passivity_pct: float = 0.0
    stability: float = 0.0          # Rejim stabilitesi (0–1)
    # Sinyal
    momentum_break: bool = False     # Momentum kırılması
    goal_adjustment: float = 1.0     # Gol olasılığı çarpanı
    recommendation: str = ""
    method: str = ""


# ═══════════════════════════════════════════════
#  MANUEL HMM (Fallback)
# ═══════════════════════════════════════════════
def _viterbi_manual(obs: np.ndarray, n_states: int,
                      trans: np.ndarray, means: np.ndarray,
                      variances: np.ndarray,
                      start_prob: np.ndarray) -> np.ndarray:
    """Manuel Viterbi algoritması (en olası durum dizisi).

    Gaussian emission ile.
    """
    T = len(obs)
    if T == 0:
        return np.array([])

    n_features = obs.shape[1] if obs.ndim == 2 else 1
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)

    # Log olasılıkları
    def log_emission(state, observation):
        total = 0.0
        for f in range(n_features):
            mu = means[state, f] if means.ndim == 2 else means[state]
            var = variances[state, f] if variances.ndim == 2 else variances[state]
            var = max(var, 1e-6)
            diff = observation[f] - mu
            total += -0.5 * (diff ** 2 / var + np.log(2 * np.pi * var))
        return total

    log_trans = np.log(trans + 1e-15)
    log_start = np.log(start_prob + 1e-15)

    # Viterbi tablosu
    V = np.full((T, n_states), -np.inf)
    backptr = np.zeros((T, n_states), dtype=int)

    # Başlangıç
    for s in range(n_states):
        V[0, s] = log_start[s] + log_emission(s, obs[0])

    # İleri adımlar
    for t in range(1, T):
        for s in range(n_states):
            candidates = V[t - 1] + log_trans[:, s]
            backptr[t, s] = int(np.argmax(candidates))
            V[t, s] = candidates[backptr[t, s]] + log_emission(s, obs[t])

    # Geri izleme
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(V[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path


# ═══════════════════════════════════════════════
#  REGIME SWITCHER (Ana Sınıf)
# ═══════════════════════════════════════════════
class RegimeSwitcher:
    """Gizli Markov Modeli ile rejim tespiti.

    Kullanım:
        rs = RegimeSwitcher(n_regimes=3)

        # Maç verisi: [şut, pas, top_kontrolü] dakika dakika
        match_data = np.array([
            [3, 45, 55],  # dk 0-5
            [5, 50, 60],  # dk 5-10
            [1, 30, 40],  # dk 10-15  ← Pasifleşme!
            ...
        ])

        report = rs.analyze_match(match_data, team="Galatasaray")
        if report.momentum_break:
            reduce_goal_probability()
    """

    def __init__(self, n_regimes: int = 3, n_iter: int = 50,
                 covariance_type: str = "diag"):
        self._n_regimes = n_regimes
        self._n_iter = n_iter
        self._cov_type = covariance_type

        if HMM_OK:
            self._model = GaussianHMM(
                n_components=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=42,
            )
            self._method = "hmmlearn"
        else:
            self._model = None
            self._method = "manual_viterbi"

        # Default parametreler (eğitim öncesi)
        self._means = np.array([
            [5.0, 55.0, 60.0],   # Baskın: yüksek şut, pas, kontrol
            [3.0, 45.0, 50.0],   # Dengeli: orta
            [1.0, 30.0, 35.0],   # Pasif: düşük
        ])
        self._variances = np.array([
            [2.0, 10.0, 8.0],
            [1.5, 8.0, 7.0],
            [1.0, 7.0, 6.0],
        ])
        self._start_prob = np.array([0.3, 0.5, 0.2])

        logger.debug(
            f"[Regime] Switcher başlatıldı: {n_regimes} rejim, "
            f"method={self._method}"
        )

    def fit(self, sequences: list[np.ndarray]) -> None:
        """Geçmiş verilerle model eğit."""
        if not sequences:
            return

        if HMM_OK and self._model:
            try:
                X = np.vstack(sequences)
                lengths = [len(s) for s in sequences]
                self._model.fit(X, lengths)
                self._means = self._model.means_
                logger.info(f"[Regime] Model eğitildi: {len(sequences)} maç")
            except Exception as e:
                logger.debug(f"[Regime] Eğitim hatası: {e}")

    def analyze_match(self, observations: np.ndarray,
                        team: str = "", match_id: str = ""
                        ) -> RegimeReport:
        """Maç verisi üzerinden rejim analizi."""
        report = RegimeReport(team=team, match_id=match_id)
        obs = np.array(observations, dtype=np.float64)

        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        if len(obs) < 3:
            report.recommendation = "Yetersiz veri (min 3 gözlem)."
            return report

        # Rejim dizisini bul
        if HMM_OK and self._model:
            try:
                self._model.fit(obs)
                states = self._model.predict(obs)
                probs = self._model.predict_proba(obs)
                report.method = "hmmlearn"
            except Exception:
                states = self._decode_manual(obs)
                probs = self._manual_probs(obs, states)
                report.method = "manual_viterbi"
        else:
            states = self._decode_manual(obs)
            probs = self._manual_probs(obs, states)
            report.method = "manual_viterbi"

        # Rejim dizisi
        report.regime_sequence = states.tolist()
        report.regime_names = [REGIME_NAMES.get(s, "?") for s in states]

        # Mevcut rejim (son gözlem)
        current_state = int(states[-1])
        report.current = RegimeState(
            regime_id=current_state,
            regime_name=REGIME_NAMES.get(current_state, "?"),
            confidence=round(float(probs[-1, current_state]) if probs is not None else 0.5, 4),
            probabilities=[round(float(p), 4) for p in probs[-1]] if probs is not None else [0.33, 0.34, 0.33],
            goal_multiplier=GOAL_MULTIPLIERS.get(current_state, 1.0),
        )

        # Geçiş tespiti
        if len(states) >= 2 and states[-1] != states[-2]:
            report.transition_detected = True
            prev_name = REGIME_NAMES.get(int(states[-2]), "?")
            curr_name = REGIME_NAMES.get(int(states[-1]), "?")
            report.last_transition = f"{prev_name} → {curr_name}"
            report.transition_minute = len(states) * 5  # ~5dk periyot varsayımı

        # Momentum kırılması (Baskın → Pasif geçişi)
        if len(states) >= 2:
            if int(states[-2]) == 0 and int(states[-1]) == 2:
                report.momentum_break = True

        # Oranlar
        n = len(states)
        report.dominance_pct = round(float(np.sum(states == 0)) / n * 100, 1)
        report.passivity_pct = round(float(np.sum(states == 2)) / n * 100, 1)

        # Stabilite (rejim değişim sıklığının tersi)
        transitions = sum(1 for i in range(1, n) if states[i] != states[i - 1])
        report.stability = round(1.0 - transitions / max(n - 1, 1), 4)

        # Gol çarpanı
        report.goal_adjustment = report.current.goal_multiplier

        report.recommendation = self._advice(report)
        return report

    def get_current_regime(self, observations: np.ndarray) -> RegimeState:
        """Sadece mevcut rejimi döndür (hızlı)."""
        obs = np.array(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        if len(obs) == 0:
            return RegimeState()

        if HMM_OK and self._model:
            try:
                self._model.fit(obs)
                states = self._model.predict(obs)
                probs = self._model.predict_proba(obs)
                s = int(states[-1])
                return RegimeState(
                    regime_id=s,
                    regime_name=REGIME_NAMES.get(s, "?"),
                    confidence=round(float(probs[-1, s]), 4),
                    goal_multiplier=GOAL_MULTIPLIERS.get(s, 1.0),
                )
            except Exception:
                pass

        states = self._decode_manual(obs)
        s = int(states[-1]) if len(states) > 0 else 1
        return RegimeState(
            regime_id=s,
            regime_name=REGIME_NAMES.get(s, "?"),
            goal_multiplier=GOAL_MULTIPLIERS.get(s, 1.0),
        )

    def _decode_manual(self, obs: np.ndarray) -> np.ndarray:
        """Manuel Viterbi ile rejim dizisi."""
        return _viterbi_manual(
            obs, self._n_regimes,
            DEFAULT_TRANSITION, self._means, self._variances,
            self._start_prob,
        )

    def _manual_probs(self, obs: np.ndarray, states: np.ndarray
                       ) -> np.ndarray:
        """Manuel durum olasılıkları (basit Gaussian)."""
        T = len(obs)
        probs = np.zeros((T, self._n_regimes))

        for t in range(T):
            for s in range(self._n_regimes):
                diff = obs[t] - self._means[s, :obs.shape[1]] if obs.ndim == 2 else obs[t] - self._means[s, 0]
                if np.ndim(diff) == 0:
                    dist = float(diff) ** 2
                else:
                    dist = float(np.sum(diff ** 2))
                probs[t, s] = np.exp(-dist / (2 * 100))

            total = probs[t].sum()
            if total > 0:
                probs[t] /= total
            else:
                probs[t] = 1.0 / self._n_regimes

        return probs

    def _advice(self, report: RegimeReport) -> str:
        if report.momentum_break:
            return (
                f"MOMENTUM KIRILMASI: {report.team} Baskın→Pasif geçişi! "
                f"Gol olasılığı x{report.goal_adjustment:.2f}. "
                f"Canlı bahis uyarısı."
            )
        if report.current.regime_id == 2 and report.current.confidence > 0.6:
            return (
                f"{report.team} PASİF rejimde (güven={report.current.confidence:.0%}). "
                f"Gol olasılığı x{report.goal_adjustment:.2f}."
            )
        if report.current.regime_id == 0 and report.current.confidence > 0.6:
            return (
                f"{report.team} BASKIN rejimde (güven={report.current.confidence:.0%}). "
                f"Gol olasılığı x{report.goal_adjustment:.2f}."
            )
        return (
            f"{report.team}: {report.current.regime_name} rejim, "
            f"stabilite={report.stability:.2f}."
        )
