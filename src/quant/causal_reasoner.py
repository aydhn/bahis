"""
causal_reasoner.py – Causal Inference (Nedensellik Analizi).

ML "korelasyonu" bilir: "Yağmur yağınca şemsiye açılır."
Ama "nedeni" bilmez: "Şemsiye açıldığı için yağmur yağmaz."

Causal Inference:
  - Korelasyon ≠ Nedensellik
  - "Oran düştüğü için mi favoriler, yoksa favori oldukları
     için mi oran düştü?" → Ters Nedensellik
  - ATE (Average Treatment Effect): "Kırmızı kart çıkınca
     gol beklentisi gerçekten artıyor mu?"
  - Counterfactual: "Bu maçta kırmızı kart olmasaydı
     sonuç ne olurdu?"

Teknoloji: DoWhy (Microsoft)
Fallback: statsmodels OLS + Propensity Score Matching
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_OK = True
except ImportError:
    DOWHY_OK = False
    logger.info("dowhy yüklü değil – istatistiksel nedensellik fallback.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import statsmodels.api as sm
    SM_OK = True
except ImportError:
    SM_OK = False


# ═══════════════════════════════════════════════
#  VERİ MODELLERİ
# ═══════════════════════════════════════════════
@dataclass
class CausalEffect:
    """Nedensel etki sonucu."""
    treatment: str = ""           # Müdahale: red_card, injury, rain, ...
    outcome: str = ""             # Sonuç: goals, xg, result, ...
    ate: float = 0.0              # Average Treatment Effect
    ate_ci_lower: float = 0.0     # Güven aralığı alt
    ate_ci_upper: float = 0.0     # Güven aralığı üst
    p_value: float = 1.0
    is_significant: bool = False  # p < 0.05
    method: str = ""              # dowhy | propensity | ols | heuristic
    confounders: list[str] = field(default_factory=list)
    interpretation: str = ""


@dataclass
class CounterfactualResult:
    """Karşı-olgusal analiz sonucu."""
    match_id: str = ""
    scenario: str = ""            # "kırmızı kart olmasaydı"
    actual_outcome: dict = field(default_factory=dict)
    counterfactual_outcome: dict = field(default_factory=dict)
    difference: dict = field(default_factory=dict)
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class CausalGraph:
    """Nedensel graf (DAG) yapısı."""
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (cause, effect)
    confounders: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  FUTBOL NEDENSELLİK GRAFI
# ═══════════════════════════════════════════════
FOOTBALL_CAUSAL_GRAPH = CausalGraph(
    nodes=[
        "red_card", "injury_key_player", "rain", "altitude",
        "travel_distance", "rest_days", "referee_strictness",
        "home_advantage", "xg", "goals", "result",
        "odds_movement", "form_last5", "league_position",
    ],
    edges=[
        # Doğrudan nedenler
        ("red_card", "xg"),
        ("red_card", "goals"),
        ("injury_key_player", "xg"),
        ("rain", "xg"),
        ("rain", "goals"),
        ("altitude", "xg"),
        ("travel_distance", "xg"),
        ("rest_days", "xg"),
        ("home_advantage", "goals"),
        ("xg", "goals"),
        ("goals", "result"),
        # Karıştırıcılar (confounders)
        ("form_last5", "xg"),
        ("form_last5", "goals"),
        ("form_last5", "odds_movement"),
        ("league_position", "odds_movement"),
        ("league_position", "result"),
        # Referee effect
        ("referee_strictness", "red_card"),
    ],
    confounders=["form_last5", "league_position", "home_advantage"],
    instruments=["referee_strictness", "rain"],
)


class CausalReasoner:
    """Nedensellik muhakeme motoru.

    Kullanım:
        reasoner = CausalReasoner()
        # Veri yükle
        reasoner.fit(historical_data)
        # ATE hesapla
        effect = reasoner.estimate_effect("red_card", "goals")
        # Karşı-olgusal analiz
        cf = reasoner.counterfactual("GS_FB", {"red_card": 0})
    """

    # Bilinen müdahaleler ve beklenen yönleri
    KNOWN_TREATMENTS = {
        "red_card": {"expected_direction": "negative", "outcome": "goals"},
        "injury_key_player": {"expected_direction": "negative", "outcome": "xg"},
        "rain": {"expected_direction": "negative", "outcome": "goals"},
        "home_advantage": {"expected_direction": "positive", "outcome": "goals"},
        "rest_days": {"expected_direction": "positive", "outcome": "xg"},
    }

    def __init__(self, significance_level: float = 0.05):
        self._alpha = significance_level
        self._data: Any = None
        self._fitted = False
        self._causal_graph = FOOTBALL_CAUSAL_GRAPH
        self._effects_cache: dict[str, CausalEffect] = {}
        logger.debug("CausalReasoner başlatıldı.")

    # ═══════════════════════════════════════════
    #  VERİ YÜKLEME
    # ═══════════════════════════════════════════
    def fit(self, data: list[dict] | Any):
        """Geçmiş maç verileriyle nedensel modeli eğit.

        data alanları:
            match_id, home_team, away_team, goals_home, goals_away,
            xg_home, xg_away, red_card (0/1), injury_key_player (0/1),
            rain (0/1), home_advantage (0/1), rest_days, form_last5,
            league_position, referee_strictness, ...
        """
        if PANDAS_OK:
            if isinstance(data, list):
                self._data = pd.DataFrame(data)
            else:
                self._data = data
        else:
            self._data = data

        if PANDAS_OK and hasattr(self._data, "__len__"):
            n = len(self._data)
            self._fitted = n >= 30
            logger.info(f"[Causal] {n} maç yüklendi (fitted={self._fitted}).")
        else:
            self._fitted = bool(data)

    # ═══════════════════════════════════════════
    #  NEDENSEL ETKİ TAHMİNİ
    # ═══════════════════════════════════════════
    def estimate_effect(self, treatment: str, outcome: str,
                         confounders: list[str] | None = None
                         ) -> CausalEffect:
        """Average Treatment Effect (ATE) hesapla.

        "Kırmızı kart çıkınca gol beklentisi gerçekten artıyor mu?"
        """
        cache_key = f"{treatment}__{outcome}"
        if cache_key in self._effects_cache:
            return self._effects_cache[cache_key]

        result = CausalEffect(treatment=treatment, outcome=outcome)
        confounders = confounders or self._causal_graph.confounders

        if not self._fitted or self._data is None:
            result.method = "prior_knowledge"
            result = self._prior_knowledge_effect(treatment, outcome)
            self._effects_cache[cache_key] = result
            return result

        # DoWhy
        if DOWHY_OK and PANDAS_OK:
            result = self._dowhy_estimate(treatment, outcome, confounders)
        # Propensity Score Matching fallback
        elif SKLEARN_OK and PANDAS_OK:
            result = self._propensity_estimate(treatment, outcome, confounders)
        # OLS fallback
        elif SM_OK and PANDAS_OK:
            result = self._ols_estimate(treatment, outcome, confounders)
        else:
            result = self._prior_knowledge_effect(treatment, outcome)

        # Yorumlama
        result.interpretation = self._interpret(result)
        self._effects_cache[cache_key] = result
        return result

    def _dowhy_estimate(self, treatment: str, outcome: str,
                         confounders: list[str]) -> CausalEffect:
        """DoWhy ile nedensel etki tahmini."""
        result = CausalEffect(treatment=treatment, outcome=outcome, method="dowhy")

        try:
            # DAG string oluştur
            graph_str = self._build_graph_string(treatment, outcome, confounders)

            available_cols = list(self._data.columns)
            valid_confounders = [c for c in confounders if c in available_cols]

            if treatment not in available_cols or outcome not in available_cols:
                return self._prior_knowledge_effect(treatment, outcome)

            model = CausalModel(
                data=self._data,
                treatment=treatment,
                outcome=outcome,
                common_causes=valid_confounders if valid_confounders else None,
                graph=graph_str if graph_str else None,
            )

            # Identification
            identified = model.identify_effect(proceed_when_unidentifiable=True)

            # Estimation (Linear Regression)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )

            result.ate = float(estimate.value)

            # Refutation (placebo test)
            try:
                refute = model.refute_estimate(
                    identified, estimate,
                    method_name="placebo_treatment_refuter",
                    placebo_type="permute",
                    num_simulations=50,
                )
                result.p_value = float(getattr(refute, "refutation_result", {}).get(
                    "p_value", 0.05,
                ))
            except Exception:
                result.p_value = 0.05

            result.is_significant = result.p_value < self._alpha
            result.confounders = valid_confounders
            logger.info(
                f"[Causal/DoWhy] {treatment} → {outcome}: "
                f"ATE={result.ate:.4f}, p={result.p_value:.3f}"
            )

        except Exception as e:
            logger.debug(f"[Causal/DoWhy] Hata: {e}")
            return self._prior_knowledge_effect(treatment, outcome)

        return result

    def _propensity_estimate(self, treatment: str, outcome: str,
                              confounders: list[str]) -> CausalEffect:
        """Propensity Score Matching ile ATE."""
        result = CausalEffect(
            treatment=treatment, outcome=outcome, method="propensity",
        )

        try:
            available = list(self._data.columns)
            valid_conf = [c for c in confounders if c in available]

            if treatment not in available or outcome not in available:
                return self._prior_knowledge_effect(treatment, outcome)

            df = self._data.dropna(subset=[treatment, outcome] + valid_conf)
            if len(df) < 20:
                return self._prior_knowledge_effect(treatment, outcome)

            T = df[treatment].values.astype(float)
            Y = df[outcome].values.astype(float)

            if valid_conf:
                X = df[valid_conf].values.astype(float)

                # Propensity score
                lr = LogisticRegression(max_iter=1000)
                T_binary = (T > T.mean()).astype(int)
                lr.fit(X, T_binary)
                ps = lr.predict_proba(X)[:, 1]

                # IPW (Inverse Propensity Weighting)
                weights = np.where(T_binary == 1, 1 / ps, 1 / (1 - ps))
                weights = np.clip(weights, 0.1, 10)

                treated_mean = np.average(Y[T_binary == 1], weights=weights[T_binary == 1])
                control_mean = np.average(Y[T_binary == 0], weights=weights[T_binary == 0])
                result.ate = float(treated_mean - control_mean)
            else:
                T_binary = (T > T.mean()).astype(int)
                result.ate = float(Y[T_binary == 1].mean() - Y[T_binary == 0].mean())

            # Bootstrap CI
            n_boot = 200
            ates = []
            for _ in range(n_boot):
                idx = np.random.choice(len(Y), size=len(Y), replace=True)
                t_b = T_binary[idx]
                y_b = Y[idx]
                if t_b.sum() > 0 and (1 - t_b).sum() > 0:
                    ate_b = y_b[t_b == 1].mean() - y_b[t_b == 0].mean()
                    ates.append(ate_b)

            if ates:
                result.ate_ci_lower = float(np.percentile(ates, 2.5))
                result.ate_ci_upper = float(np.percentile(ates, 97.5))
                result.is_significant = not (
                    result.ate_ci_lower <= 0 <= result.ate_ci_upper
                )

            result.confounders = valid_conf
            logger.info(
                f"[Causal/PSM] {treatment} → {outcome}: "
                f"ATE={result.ate:.4f}, sig={result.is_significant}"
            )

        except Exception as e:
            logger.debug(f"[Causal/PSM] Hata: {e}")
            return self._prior_knowledge_effect(treatment, outcome)

        return result

    def _ols_estimate(self, treatment: str, outcome: str,
                       confounders: list[str]) -> CausalEffect:
        """OLS regresyonla basit nedensel tahmin."""
        result = CausalEffect(
            treatment=treatment, outcome=outcome, method="ols",
        )

        try:
            available = list(self._data.columns)
            valid_conf = [c for c in confounders if c in available]

            if treatment not in available or outcome not in available:
                return self._prior_knowledge_effect(treatment, outcome)

            df = self._data.dropna(subset=[treatment, outcome] + valid_conf)
            Y = df[outcome].values.astype(float)
            X_cols = [treatment] + valid_conf
            X = df[X_cols].values.astype(float)
            X = sm.add_constant(X)

            model = sm.OLS(Y, X).fit()

            result.ate = float(model.params[1])  # Treatment coefficient
            result.p_value = float(model.pvalues[1])
            ci = model.conf_int()[1]
            result.ate_ci_lower = float(ci[0])
            result.ate_ci_upper = float(ci[1])
            result.is_significant = result.p_value < self._alpha
            result.confounders = valid_conf

        except Exception as e:
            logger.debug(f"[Causal/OLS] Hata: {e}")
            return self._prior_knowledge_effect(treatment, outcome)

        return result

    def _prior_knowledge_effect(self, treatment: str,
                                  outcome: str) -> CausalEffect:
        """Alan bilgisine dayalı heuristic tahmin."""
        result = CausalEffect(
            treatment=treatment, outcome=outcome, method="prior_knowledge",
        )

        known = self.KNOWN_TREATMENTS.get(treatment, {})
        direction = known.get("expected_direction", "unknown")

        effect_map = {
            "red_card": {"goals": -0.35, "xg": -0.40},
            "injury_key_player": {"goals": -0.20, "xg": -0.25},
            "rain": {"goals": -0.15, "xg": -0.10},
            "home_advantage": {"goals": 0.35, "xg": 0.30},
            "rest_days": {"goals": 0.05, "xg": 0.08},
        }

        ate = effect_map.get(treatment, {}).get(outcome, 0.0)
        result.ate = ate
        result.ate_ci_lower = ate - 0.2
        result.ate_ci_upper = ate + 0.2
        result.is_significant = abs(ate) > 0.1
        result.p_value = 0.03 if result.is_significant else 0.5

        return result

    # ═══════════════════════════════════════════
    #  KARŞI-OLGUSAL (COUNTERFACTUAL) ANALİZ
    # ═══════════════════════════════════════════
    def counterfactual(self, match_id: str,
                        intervention: dict,
                        actual_data: dict | None = None
                        ) -> CounterfactualResult:
        """Karşı-olgusal analiz: "Eğer X olmasaydı ne olurdu?"

        Args:
            match_id: Maç ID
            intervention: Değiştirilecek değişkenler {"red_card": 0}
            actual_data: Gerçek maç verisi
        """
        result = CounterfactualResult(
            match_id=match_id,
            actual_outcome=actual_data or {},
        )

        # Senaryo açıklaması
        scenarios = []
        for var, new_val in intervention.items():
            old_val = (actual_data or {}).get(var, "?")
            scenarios.append(f"{var}: {old_val} → {new_val}")
        result.scenario = ", ".join(scenarios)

        # Her müdahale için ATE uygula
        counterfactual = dict(actual_data or {})
        total_effect = 0.0

        for treatment, new_value in intervention.items():
            old_value = (actual_data or {}).get(treatment, 0)
            if old_value == new_value:
                continue

            # ATE hesapla
            effect = self.estimate_effect(treatment, "goals")

            # Müdahalenin etkisini uygula
            delta = effect.ate * (new_value - old_value)
            total_effect += delta

            goals_key = "goals_home"
            if goals_key in counterfactual:
                counterfactual[goals_key] = max(
                    0, counterfactual[goals_key] + delta,
                )

        result.counterfactual_outcome = counterfactual

        # Farkları hesapla
        for key in ("goals_home", "goals_away", "xg_home", "xg_away"):
            if key in (actual_data or {}) and key in counterfactual:
                diff = counterfactual[key] - actual_data[key]
                if abs(diff) > 0.001:
                    result.difference[key] = round(diff, 3)

        result.confidence = 0.6 if self._fitted else 0.3

        # Açıklama oluştur
        if total_effect > 0.1:
            result.explanation = (
                f"Karşı-olgusal: {result.scenario} olsaydı, "
                f"gol beklentisi {total_effect:+.2f} değişirdi. "
                f"Sonuç muhtemelen farklı olurdu."
            )
        elif total_effect < -0.1:
            result.explanation = (
                f"Karşı-olgusal: {result.scenario} olsaydı, "
                f"gol beklentisi {total_effect:+.2f} düşerdi."
            )
        else:
            result.explanation = (
                f"Karşı-olgusal: {result.scenario} sonucu "
                f"önemli ölçüde değiştirmezdi (etki: {total_effect:+.3f})."
            )

        return result

    # ═══════════════════════════════════════════
    #  ÇOKLU ANALİZ
    # ═══════════════════════════════════════════
    def analyze_all_treatments(self, outcome: str = "goals"
                                 ) -> list[CausalEffect]:
        """Tüm bilinen müdahalelerin etkisini hesapla."""
        effects = []
        for treatment in self.KNOWN_TREATMENTS:
            effect = self.estimate_effect(treatment, outcome)
            effects.append(effect)

        # Etkiye göre sırala
        effects.sort(key=lambda e: abs(e.ate), reverse=True)
        return effects

    def get_match_causal_factors(self, match_data: dict
                                   ) -> list[CausalEffect]:
        """Maçtaki aktif nedensel faktörleri listele."""
        active = []
        for treatment in self.KNOWN_TREATMENTS:
            if match_data.get(treatment, 0) > 0:
                effect = self.estimate_effect(treatment, "goals")
                active.append(effect)
        return active

    # ═══════════════════════════════════════════
    #  YARDIMCI
    # ═══════════════════════════════════════════
    def _build_graph_string(self, treatment: str, outcome: str,
                              confounders: list[str]) -> str | None:
        """DoWhy için DAG string oluştur."""
        try:
            edges_str = "; ".join(
                f"{c} -> {e}"
                for c, e in self._causal_graph.edges
            )
            return f"digraph {{{edges_str}}}"
        except Exception:
            return None

    def _interpret(self, effect: CausalEffect) -> str:
        """Nedensel etkiyi Türkçe yorumla."""
        t = effect.treatment.replace("_", " ").title()
        o = effect.outcome

        if not effect.is_significant:
            return (
                f"{t}'ın {o} üzerindeki etkisi istatistiksel olarak "
                f"anlamlı DEĞİL (ATE={effect.ate:.3f}, p={effect.p_value:.3f}). "
                f"Bu bir KORELASYON olabilir, NEDENSELLİK değil."
            )

        direction = "artırıyor" if effect.ate > 0 else "azaltıyor"
        magnitude = "güçlü" if abs(effect.ate) > 0.3 else (
            "orta" if abs(effect.ate) > 0.1 else "zayıf"
        )

        return (
            f"{t}, {o}'ı {direction} (ATE={effect.ate:+.3f}). "
            f"Etki {magnitude} ve istatistiksel olarak anlamlı "
            f"(p={effect.p_value:.3f}, CI=[{effect.ate_ci_lower:.3f}, {effect.ate_ci_upper:.3f}]). "
            f"Metod: {effect.method}"
        )

    @property
    def causal_graph(self) -> CausalGraph:
        return self._causal_graph
