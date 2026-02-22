"""
probabilistic_engine.py – Probabilistic Programming (PyMC).

Sabit rakamlardan kurtuluyoruz. "Takımın gücü 80" demek yerine,
"Takımın gücü %90 ihtimalle 75-85 arasındadır" diyoruz.

Kavramlar:
  - Probabilistic Programming: Kodun her satırında belirsizliği taşıma
  - Random Variable: x = Normal(5, 1) → x bir dağılımdır, sabit değil
  - Bayesian Inference: Prior + Likelihood → Posterior
  - MCMC (Markov Chain Monte Carlo): Posterior dağılımı örnekleme
  - No-U-Turn Sampler (NUTS): PyMC'nin varsayılan MCMC yöntemi
  - Credible Interval: %95 güvenilir aralık (frequentist CI'den farklı)
  - Posterior Predictive: Gelecek veri noktalarının olasılık dağılımı

Model:
  - attack_i ~ Normal(μ_att, σ_att)     → Takım i'nin hücum gücü
  - defense_j ~ Normal(μ_def, σ_def)    → Takım j'nin savunma gücü
  - home_adv ~ Normal(0.3, 0.1)         → Ev sahibi avantajı
  - goals_ij ~ Poisson(exp(attack_i - defense_j + home_adv))
  - Posterior: P(attack_i | gözlenen goller)

Akış:
  1. Takım gücü parametrelerini prior dağılımlarla tanımla
  2. Gözlenen maç sonuçlarıyla likelihood oluştur
  3. MCMC ile posterior dağılımı örnekle (2000 draw, 1000 tune)
  4. Posterior predictive → gelecek maç olasılık dağılımı
  5. Credible interval → bahis kararına risk aralığı ekle

Teknoloji: PyMC (v5+)
Fallback: NumPy tabanlı Metropolis-Hastings + conjugate priors
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    import pymc as pm
    import arviz as az
    PYMC_OK = True
except ImportError:
    PYMC_OK = False
    logger.debug("pymc/arviz yüklü değil – numpy MCMC fallback.")

try:
    from scipy.stats import poisson as sp_poisson, norm as sp_norm
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class PosteriorSummary:
    """Bir parametrenin posterior özeti."""
    name: str = ""
    mean: float = 0.0
    std: float = 0.0
    hdi_low: float = 0.0        # %95 HDI alt sınır
    hdi_high: float = 0.0       # %95 HDI üst sınır
    ess: float = 0.0            # Effective Sample Size
    rhat: float = 1.0           # Convergence diagnostic


@dataclass
class MatchPrediction:
    """Olasılıksal maç tahmini."""
    match_id: str = ""
    home_team: str = ""
    away_team: str = ""
    # Gol dağılımları
    home_goals_mean: float = 0.0
    home_goals_std: float = 0.0
    away_goals_mean: float = 0.0
    away_goals_std: float = 0.0
    # Olasılıklar (posterior'dan hesaplanmış)
    p_home: float = 0.0
    p_draw: float = 0.0
    p_away: float = 0.0
    p_over25: float = 0.0
    p_under25: float = 0.0
    p_btts: float = 0.0         # Both Teams To Score
    # Güvenilir aralık
    home_goals_hdi: tuple[float, float] = (0.0, 0.0)
    away_goals_hdi: tuple[float, float] = (0.0, 0.0)
    # En olası skorlar
    most_likely_scores: list[tuple[int, int, float]] = field(default_factory=list)
    # Meta
    n_samples: int = 0
    convergence_ok: bool = True
    method: str = ""
    recommendation: str = ""


@dataclass
class ProbabilisticReport:
    """Model raporu."""
    n_teams: int = 0
    n_matches_trained: int = 0
    parameters: list[PosteriorSummary] = field(default_factory=list)
    home_advantage: PosteriorSummary = field(default_factory=PosteriorSummary)
    fit_time_sec: float = 0.0
    method: str = ""


# ═══════════════════════════════════════════════
#  NUMPY MCMC FALLBACK (Metropolis-Hastings)
# ═══════════════════════════════════════════════
def metropolis_hastings(log_posterior_fn, init: np.ndarray,
                          n_samples: int = 2000,
                          proposal_std: float = 0.1) -> np.ndarray:
    """Basit Metropolis-Hastings örnekleyici."""
    n_params = len(init)
    samples = np.zeros((n_samples, n_params))
    current = init.copy()
    current_lp = log_posterior_fn(current)
    accepted = 0

    for i in range(n_samples):
        proposal = current + np.random.randn(n_params) * proposal_std
        proposal_lp = log_posterior_fn(proposal)
        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accepted += 1

        samples[i] = current

    logger.debug(
        f"[ProbEngine] MH: kabul oranı = {accepted / n_samples:.1%}"
    )
    return samples


# ═══════════════════════════════════════════════
#  Numba JIT – Monte Carlo skor simülasyonu
# ═══════════════════════════════════════════════
if NUMBA_OK:
    @njit(cache=True, parallel=True)
    def _mc_score_probs(home_rate: np.ndarray, away_rate: np.ndarray):
        """JIT-hızlandırılmış Monte Carlo skor simülasyonu.

        Returns: (p_home, p_draw, p_away, p_over25, p_btts, n)
        """
        n = len(home_rate)
        home_wins = 0
        draws = 0
        away_wins = 0
        over25 = 0
        btts = 0
        for i in prange(n):
            hg = np.random.poisson(home_rate[i])
            ag = np.random.poisson(away_rate[i])
            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1
            if hg + ag > 2:
                over25 += 1
            if hg > 0 and ag > 0:
                btts += 1
        fn = float(n)
        return (home_wins / fn, draws / fn, away_wins / fn,
                over25 / fn, btts / fn, n)
else:
    _mc_score_probs = None


def hdi(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Highest Density Interval."""
    sorted_pts = np.sort(samples)
    n = len(sorted_pts)
    interval_width = int(np.ceil(prob * n))
    if interval_width >= n:
        return float(sorted_pts[0]), float(sorted_pts[-1])

    widths = sorted_pts[interval_width:] - sorted_pts[:n - interval_width]
    best = int(np.argmin(widths))
    return float(sorted_pts[best]), float(sorted_pts[best + interval_width])


# ═══════════════════════════════════════════════
#  PROBABILISTIC ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class ProbabilisticEngine:
    """Olasılıksal futbol modeli.

    Kullanım:
        pe = ProbabilisticEngine()

        # Eğitim (geçmiş maçlar)
        pe.fit(home_goals=[2,1,0,3], away_goals=[1,0,2,1],
               home_teams=["GS","FB","BJK","TS"],
               away_teams=["FB","GS","TS","BJK"])

        # Tahmin
        pred = pe.predict("GS", "FB", match_id="derbi")
    """

    def __init__(self, n_samples: int = 2000, n_tune: int = 1000,
                 target_accept: float = 0.9):
        self._n_samples = n_samples
        self._n_tune = n_tune
        self._target_accept = target_accept
        self._trace: Any = None
        self._model: Any = None
        self._team_index: dict[str, int] = {}
        # Posterior parametreleri (fallback)
        self._attack_samples: dict[str, np.ndarray] = {}
        self._defense_samples: dict[str, np.ndarray] = {}
        self._home_adv_samples: np.ndarray = np.array([0.3])
        self._fitted = False

        logger.debug(
            f"[ProbEngine] Başlatıldı: samples={n_samples}, "
            f"tune={n_tune}, pymc={'OK' if PYMC_OK else 'fallback'}"
        )

    def fit(self, home_goals: list[int], away_goals: list[int],
            home_teams: list[str], away_teams: list[str]) -> ProbabilisticReport:
        """Olasılıksal modeli eğit."""
        report = ProbabilisticReport()
        t0 = time.perf_counter()

        # Takım indeksleme
        all_teams = sorted(set(home_teams + away_teams))
        self._team_index = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)
        report.n_teams = n_teams
        report.n_matches_trained = len(home_goals)

        home_idx = np.array([self._team_index[t] for t in home_teams])
        away_idx = np.array([self._team_index[t] for t in away_teams])
        hg = np.array(home_goals, dtype=np.int64)
        ag = np.array(away_goals, dtype=np.int64)

        if PYMC_OK:
            report = self._fit_pymc(
                hg, ag, home_idx, away_idx, n_teams, report,
            )
        else:
            report = self._fit_fallback(
                hg, ag, home_idx, away_idx, n_teams, all_teams, report,
            )

        report.fit_time_sec = round(time.perf_counter() - t0, 2)
        self._fitted = True
        return report

    def _fit_pymc(self, hg, ag, home_idx, away_idx,
                    n_teams, report) -> ProbabilisticReport:
        """PyMC ile Bayesian fit."""
        try:
            with pm.Model() as model:
                # Priors
                home_adv = pm.Normal("home_adv", mu=0.3, sigma=0.2)
                attack = pm.Normal("attack", mu=0, sigma=1, shape=n_teams)
                defense = pm.Normal("defense", mu=0, sigma=1, shape=n_teams)
                intercept = pm.Normal("intercept", mu=0.3, sigma=0.5)

                # Beklenen gol sayıları (log-link)
                home_rate = pm.math.exp(
                    intercept + home_adv + attack[home_idx] - defense[away_idx]
                )
                away_rate = pm.math.exp(
                    intercept + attack[away_idx] - defense[home_idx]
                )

                # Likelihood
                pm.Poisson("home_goals", mu=home_rate, observed=hg)
                pm.Poisson("away_goals", mu=away_rate, observed=ag)

                # MCMC
                self._trace = pm.sample(
                    draws=self._n_samples,
                    tune=self._n_tune,
                    target_accept=self._target_accept,
                    return_inferencedata=True,
                    progressbar=False,
                    cores=1,
                )
                self._model = model

            # Posterior özetlerini kaydet
            summary = az.summary(self._trace, var_names=["home_adv", "attack", "defense"])
            report.home_advantage = PosteriorSummary(
                name="home_adv",
                mean=float(summary.loc["home_adv", "mean"]),
                std=float(summary.loc["home_adv", "sd"]),
                hdi_low=float(summary.loc["home_adv", "hdi_3%"]),
                hdi_high=float(summary.loc["home_adv", "hdi_97%"]),
                rhat=float(summary.loc["home_adv", "r_hat"]),
            )
            report.method = "pymc_nuts"
            logger.info(
                f"[ProbEngine] PyMC fit: {n_teams} takım, "
                f"home_adv={report.home_advantage.mean:.3f}"
            )

        except Exception as e:
            logger.warning(f"[ProbEngine] PyMC hatası: {e}")
            teams = sorted(self._team_index.keys())
            report = self._fit_fallback(
                hg, ag,
                np.array([self._team_index[t] for t in teams[:len(hg)]]),
                np.array([self._team_index[t] for t in teams[:len(ag)]]),
                n_teams, teams, report,
            )

        return report

    def _fit_fallback(self, hg, ag, home_idx, away_idx,
                        n_teams, all_teams, report) -> ProbabilisticReport:
        """NumPy Metropolis-Hastings fallback."""
        # n_params = n_teams (attack) + n_teams (defense) + 1 (home_adv) + 1 (intercept)
        n_params = 2 * n_teams + 2

        def log_posterior(params):
            intercept = params[0]
            home_adv_val = params[1]
            attack_vals = params[2:2 + n_teams]
            defense_vals = params[2 + n_teams:]

            # Prior
            lp = -0.5 * np.sum(attack_vals ** 2)   # N(0,1)
            lp += -0.5 * np.sum(defense_vals ** 2)
            lp += -0.5 * ((home_adv_val - 0.3) / 0.2) ** 2
            lp += -0.5 * ((intercept - 0.3) / 0.5) ** 2

            # Likelihood (Poisson)
            home_rate = np.exp(
                intercept + home_adv_val
                + attack_vals[home_idx] - defense_vals[away_idx]
            )
            away_rate = np.exp(
                intercept + attack_vals[away_idx] - defense_vals[home_idx]
            )

            home_rate = np.clip(home_rate, 0.01, 10)
            away_rate = np.clip(away_rate, 0.01, 10)

            if SCIPY_OK:
                lp += np.sum(sp_poisson.logpmf(hg, home_rate))
                lp += np.sum(sp_poisson.logpmf(ag, away_rate))
            else:
                lp += np.sum(hg * np.log(home_rate) - home_rate)
                lp += np.sum(ag * np.log(away_rate) - away_rate)

            return lp if np.isfinite(lp) else -1e10

        init = np.zeros(n_params)
        init[0] = 0.3   # intercept
        init[1] = 0.3   # home_adv

        burnin = min(500, self._n_tune)
        total = burnin + self._n_samples
        raw_samples = metropolis_hastings(
            log_posterior, init,
            n_samples=total, proposal_std=0.05,
        )
        samples = raw_samples[burnin:]

        # Posterior saklama
        self._home_adv_samples = samples[:, 1]
        for team, idx in self._team_index.items():
            self._attack_samples[team] = samples[:, 2 + idx]
            self._defense_samples[team] = samples[:, 2 + n_teams + idx]

        ha_mean = float(np.mean(self._home_adv_samples))
        ha_std = float(np.std(self._home_adv_samples))
        ha_hdi = hdi(self._home_adv_samples)

        report.home_advantage = PosteriorSummary(
            name="home_adv", mean=round(ha_mean, 4),
            std=round(ha_std, 4),
            hdi_low=round(ha_hdi[0], 4),
            hdi_high=round(ha_hdi[1], 4),
        )
        report.method = "numpy_metropolis_hastings"
        logger.info(
            f"[ProbEngine] MH fit: {n_teams} takım, "
            f"home_adv={ha_mean:.3f}±{ha_std:.3f}"
        )
        return report

    def predict(self, home_team: str, away_team: str,
                  match_id: str = "") -> MatchPrediction:
        """Olasılıksal maç tahmini."""
        pred = MatchPrediction(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
        )

        if not self._fitted:
            pred.recommendation = "Model henüz eğitilmedi."
            pred.method = "not_fitted"
            return pred

        # PyMC posterior
        if PYMC_OK and self._trace is not None:
            pred = self._predict_pymc(home_team, away_team, pred)
        else:
            pred = self._predict_fallback(home_team, away_team, pred)

        pred.recommendation = self._advice(pred)
        return pred

    def _predict_pymc(self, home: str, away: str,
                        pred: MatchPrediction) -> MatchPrediction:
        """PyMC posterior predictive."""
        try:
            post = self._trace.posterior
            hi = self._team_index.get(home, 0)
            ai = self._team_index.get(away, 0)

            intercept = post["intercept"].values.flatten()
            home_adv = post["home_adv"].values.flatten()
            attack = post["attack"].values
            defense = post["defense"].values

            att_h = attack[:, :, hi].flatten()
            def_h = defense[:, :, hi].flatten()
            att_a = attack[:, :, ai].flatten()
            def_a = defense[:, :, ai].flatten()

            home_rate = np.exp(intercept + home_adv + att_h - def_a)
            away_rate = np.exp(intercept + att_a - def_h)

            home_rate = np.clip(home_rate, 0.01, 10)
            away_rate = np.clip(away_rate, 0.01, 10)

            pred = self._compute_predictions(home_rate, away_rate, pred)
            pred.method = "pymc_posterior"

        except Exception:
            pred = self._predict_fallback(home, away, pred)

        return pred

    def _predict_fallback(self, home: str, away: str,
                            pred: MatchPrediction) -> MatchPrediction:
        """Fallback posterior predictive."""
        ha_samples = self._home_adv_samples
        att_h = self._attack_samples.get(home, np.zeros(len(ha_samples)))
        def_h = self._defense_samples.get(home, np.zeros(len(ha_samples)))
        att_a = self._attack_samples.get(away, np.zeros(len(ha_samples)))
        def_a = self._defense_samples.get(away, np.zeros(len(ha_samples)))

        # Boyut eşitle
        n = min(len(ha_samples), len(att_h), len(def_h), len(att_a), len(def_a))
        intercept = 0.3  # sabit yaklaşım

        home_rate = np.exp(
            intercept + ha_samples[:n] + att_h[:n] - def_a[:n]
        )
        away_rate = np.exp(
            intercept + att_a[:n] - def_h[:n]
        )

        home_rate = np.clip(home_rate, 0.01, 10)
        away_rate = np.clip(away_rate, 0.01, 10)

        pred = self._compute_predictions(home_rate, away_rate, pred)
        pred.method = "numpy_posterior"
        return pred

    def _compute_predictions(self, home_rate: np.ndarray,
                               away_rate: np.ndarray,
                               pred: MatchPrediction) -> MatchPrediction:
        """Posterior rate'lerden olasılıkları hesapla. Numba JIT ile hızlandırılmış."""
        pred.n_samples = len(home_rate)

        # Gol dağılımı
        pred.home_goals_mean = round(float(np.mean(home_rate)), 3)
        pred.home_goals_std = round(float(np.std(home_rate)), 3)
        pred.away_goals_mean = round(float(np.mean(away_rate)), 3)
        pred.away_goals_std = round(float(np.std(away_rate)), 3)

        pred.home_goals_hdi = tuple(
            round(x, 2) for x in hdi(home_rate)
        )
        pred.away_goals_hdi = tuple(
            round(x, 2) for x in hdi(away_rate)
        )

        # Monte Carlo skor simülasyonu (Numba JIT fast path)
        if _mc_score_probs is not None:
            hr = np.ascontiguousarray(home_rate, dtype=np.float64)
            ar = np.ascontiguousarray(away_rate, dtype=np.float64)
            ph, pd_, pa, po25, pbtts, _ = _mc_score_probs(hr, ar)
            pred.p_home = round(ph, 4)
            pred.p_draw = round(pd_, 4)
            pred.p_away = round(pa, 4)
            pred.p_over25 = round(po25, 4)
            pred.p_under25 = round(1.0 - po25, 4)
            pred.p_btts = round(pbtts, 4)
        else:
            home_goals_sim = np.random.poisson(home_rate)
            away_goals_sim = np.random.poisson(away_rate)
            pred.p_home = round(float(np.mean(home_goals_sim > away_goals_sim)), 4)
            pred.p_draw = round(float(np.mean(home_goals_sim == away_goals_sim)), 4)
            pred.p_away = round(float(np.mean(home_goals_sim < away_goals_sim)), 4)
            pred.p_over25 = round(float(np.mean(
                (home_goals_sim + away_goals_sim) > 2.5,
            )), 4)
            pred.p_under25 = round(1 - pred.p_over25, 4)
            pred.p_btts = round(float(np.mean(
                (home_goals_sim > 0) & (away_goals_sim > 0),
            )), 4)

        # En olası skorlar (always use numpy for score counting)
        home_goals_sim = np.random.poisson(home_rate)
        away_goals_sim = np.random.poisson(away_rate)
        score_counts: dict[tuple[int, int], int] = {}
        for hg, ag in zip(home_goals_sim, away_goals_sim):
            key = (int(hg), int(ag))
            score_counts[key] = score_counts.get(key, 0) + 1

        total = len(home_goals_sim)
        sorted_scores = sorted(
            score_counts.items(), key=lambda x: -x[1],
        )[:5]
        pred.most_likely_scores = [
            (s[0], s[1], round(c / total, 3))
            for (s, c) in sorted_scores
        ]

        return pred

    def predict_for_dataframe(self, df) -> list[dict]:
        """DataFrame veya numpy array üzerinden toplu tahmin üretir.

        Her satır bir maçı temsil eder. home_team, away_team ve match_id
        sütunları varsa kullanır, yoksa varsayılan isimlendirme yapar.
        """
        results = []
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                rows = df.to_dicts()
            elif hasattr(df, "tolist"):
                rows = [{"match_id": f"m{i}", "home_team": f"team_h{i}", "away_team": f"team_a{i}"}
                        for i in range(len(df))]
            else:
                rows = list(df) if hasattr(df, "__iter__") else []
        except Exception:
            rows = []

        for row in rows:
            home = row.get("home_team", "unknown_home")
            away = row.get("away_team", "unknown_away")
            mid = row.get("match_id", f"{home}_vs_{away}")
            pred = self.predict(home, away, mid)
            results.append({
                "match_id": mid,
                "home_team": home,
                "away_team": away,
                "p_home": pred.p_home,
                "p_draw": pred.p_draw,
                "p_away": pred.p_away,
                "p_over25": pred.p_over25,
                "p_under25": pred.p_under25,
                "p_btts": pred.p_btts,
                "home_goals_mean": pred.home_goals_mean,
                "away_goals_mean": pred.away_goals_mean,
                "method": pred.method,
                "recommendation": pred.recommendation,
            })
        return results

    def ensemble(self, **kwargs) -> list[dict]:
        """Farklı model tahminlerini log-odds Bayesian ensemble ile birleştirir.

        Log-odds uzayında birleştirme ile olasılık tutarlılığı sağlar.
        Dinamik ağırlıklar: model çeşitliliği ve uyum analizi.

        Args:
            **kwargs: model_name=predictions şeklinde her modelin çıktısı.

        Returns:
            Birleştirilmiş tahmin listesi.
        """
        base_weights = {
            "bayesian": 0.25, "mtl": 0.12, "kan": 0.08,
            "poisson": 0.10, "dixon_coles": 0.08, "gradient": 0.07,
            "glm": 0.05, "tail": 0.05, "causal": 0.05,
            "intervals": 0.03, "signatures": 0.03, "jumps": 0.03,
            "geometric": 0.03, "monte_carlo": 0.03,
        }

        CLIP_MIN, CLIP_MAX = 0.01, 0.99

        def _to_logodds(p: float) -> float:
            p = max(CLIP_MIN, min(p, CLIP_MAX))
            return float(np.log(p / (1.0 - p)))

        def _from_logodds(lo: float) -> float:
            return float(1.0 / (1.0 + np.exp(-lo)))

        all_predictions: dict[str, dict] = {}

        for model_name, preds in kwargs.items():
            if preds is None:
                continue
            w = base_weights.get(model_name, 0.02)

            pred_list = []
            if isinstance(preds, list):
                pred_list = preds
            elif hasattr(preds, "to_dicts"):
                try:
                    pred_list = preds.to_dicts()
                except Exception:
                    continue
            elif isinstance(preds, dict):
                pred_list = [preds]

            for pred in pred_list:
                if not isinstance(pred, dict):
                    continue
                mid = pred.get("match_id", "")
                if not mid:
                    continue

                if mid not in all_predictions:
                    all_predictions[mid] = {
                        "match_id": mid,
                        "home_team": pred.get("home_team", ""),
                        "away_team": pred.get("away_team", ""),
                        "logodds_home": [],
                        "logodds_draw": [],
                        "logodds_away": [],
                        "logodds_over25": [],
                        "weights": [],
                        "sources": [],
                        "recommendations": [],
                    }

                entry = all_predictions[mid]
                ph = pred.get("p_home", 0)
                pd_ = pred.get("p_draw", 0)
                pa = pred.get("p_away", 0)
                po25 = pred.get("p_over25", 0)

                if isinstance(ph, (int, float)) and ph > 0:
                    entry["logodds_home"].append(_to_logodds(ph))
                    entry["logodds_draw"].append(_to_logodds(pd_ or 0.33))
                    entry["logodds_away"].append(_to_logodds(pa or 0.33))
                    entry["logodds_over25"].append(_to_logodds(po25 or 0.5))
                    entry["weights"].append(w)
                    entry["sources"].append(model_name)

                rec = pred.get("recommendation", "")
                if rec:
                    entry["recommendations"].append(f"[{model_name}] {rec}")

        result = []
        for mid, entry in all_predictions.items():
            wts = np.array(entry["weights"])
            n_sources = len(entry["sources"])

            if n_sources == 0 or wts.sum() <= 0:
                result.append({
                    "match_id": mid, "home_team": entry["home_team"],
                    "away_team": entry["away_team"],
                    "p_home": 0.33, "p_draw": 0.34, "p_away": 0.33,
                    "p_over25": 0.5, "confidence": 0.0, "sources": [],
                    "n_models": 0,
                })
                continue

            wts_norm = wts / wts.sum()

            # Log-odds uzayında ağırlıklı ortalama
            lo_h = float(np.dot(wts_norm, entry["logodds_home"]))
            lo_d = float(np.dot(wts_norm, entry["logodds_draw"]))
            lo_a = float(np.dot(wts_norm, entry["logodds_away"]))
            lo_o25 = float(np.dot(wts_norm, entry["logodds_over25"]))

            ph = _from_logodds(lo_h)
            pd_ = _from_logodds(lo_d)
            pa = _from_logodds(lo_a)
            p_o25 = _from_logodds(lo_o25)

            # Normalizasyon (1X2 olasılıkları toplamı = 1.0)
            total = ph + pd_ + pa
            if total > 0:
                ph, pd_, pa = ph / total, pd_ / total, pa / total
            else:
                ph, pd_, pa = 0.33, 0.34, 0.33

            # Gelişmiş güven skoru:
            # 1) Model sayısı (daha fazla = daha güvenilir)
            diversity_score = min(n_sources / 6.0, 1.0)
            # 2) Model uyumu (log-odds varyansı düşükse modeller hemfikir)
            lo_std = float(np.std(entry["logodds_home"])) if n_sources > 1 else 1.0
            agreement_score = max(0.0, 1.0 - lo_std / 3.0)
            # 3) Dominant olasılık netliği (daha belirgin = daha güvenilir)
            clarity = max(ph, pd_, pa)
            clarity_score = (clarity - 0.33) / 0.67

            confidence = (
                0.40 * diversity_score
                + 0.35 * agreement_score
                + 0.25 * clarity_score
            )
            confidence = max(0.05, min(confidence, 0.99))

            result.append({
                "match_id": mid,
                "home_team": entry["home_team"],
                "away_team": entry["away_team"],
                "p_home": round(ph, 4),
                "p_draw": round(pd_, 4),
                "p_away": round(pa, 4),
                "p_over25": round(p_o25, 4),
                "confidence": round(confidence, 4),
                "sources": entry["sources"],
                "n_models": n_sources,
                "model_agreement": round(agreement_score, 3),
                "recommendation": "; ".join(entry["recommendations"][:3]),
            })

        if not result:
            result = [{"match_id": "none", "p_home": 0.33, "p_draw": 0.34,
                        "p_away": 0.33, "confidence": 0.0, "sources": [],
                        "n_models": 0}]
        return result

    def _advice(self, p: MatchPrediction) -> str:
        winner = "Ev" if p.p_home > p.p_away else "Dep"
        dominant_p = max(p.p_home, p.p_away)
        top_score = p.most_likely_scores[0] if p.most_likely_scores else (0, 0, 0)

        return (
            f"Tahmin: {winner} ({dominant_p:.0%}), Beraberlik {p.p_draw:.0%}. "
            f"Ev gol: {p.home_goals_mean:.1f} [{p.home_goals_hdi[0]:.1f}-{p.home_goals_hdi[1]:.1f}], "
            f"Dep gol: {p.away_goals_mean:.1f} [{p.away_goals_hdi[0]:.1f}-{p.away_goals_hdi[1]:.1f}]. "
            f"Ü2.5: {p.p_over25:.0%}, KG: {p.p_btts:.0%}. "
            f"En olası skor: {top_score[0]}-{top_score[1]} ({top_score[2]:.0%})."
        )

    def bayesian_update(self, team: str, actual_score: int, expected_score: float):
        """Maç sonrası takım gücünü Bayesian yöntemiyle günceller."""
        # Prior: Mevcut güç (Mock: 1.0)
        # Gerçekte posterior_summary'den çekilmeli
        prior_strength = 1.0 
        
        # Likelihood: Poisson(actual | expected)
        # Basit bir gradient update (Bayesian approximation)
        alpha = 0.05 # Learning rate
        innovation = actual_score - expected_score
        posterior_strength = prior_strength + alpha * innovation
        
        logger.info(f"Bayesian Update: {team} | New Strength: {posterior_strength:.3f}")
        return posterior_strength
