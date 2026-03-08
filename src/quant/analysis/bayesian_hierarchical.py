"""
bayesian_hierarchical.py – Bayesyen Hiyerarşik Model.

Klasik istatistik: "Veri ne diyorsa odur."
Bayesyen yaklaşım: "Önceki inancım bu, yeni veriyle güncelliyorum."

Lig başında bir takımın verisi yoksa → Lig ortalamasını prior kabul et.
Maç oynadıkça → takımın kendi karakterine doğru evrilir.
Bu, sezon başı sürprizlerini yakalar.

Model: Hiyerarşik Poisson
  - Lig seviyesi: genel gol ortalaması (hyper-prior)
  - Takım seviyesi: takıma özel atak/savunma (posterior)
  - Her yeni maç → posterior güncellenir (Bayesian updating)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from loguru import logger


@dataclass
class TeamPrior:
    """Takımın Bayesyen parametreleri."""
    team: str
    # Atak: Gamma dağılımı parametreleri (α, β)
    attack_alpha: float = 2.0    # shape
    attack_beta: float = 2.0     # rate  → ortalama = α/β
    # Savunma: Gamma dağılımı parametreleri
    defence_alpha: float = 2.0
    defence_beta: float = 2.0
    # Gözlemlenen maç sayısı
    n_matches: int = 0
    # Posterior ortalamalar
    attack_mean: float = 1.0     # α/β
    defence_mean: float = 1.0

    def update_attack_mean(self):
        self.attack_mean = self.attack_alpha / max(self.attack_beta, 0.01)

    def update_defence_mean(self):
        self.defence_mean = self.defence_alpha / max(self.defence_beta, 0.01)


@dataclass
class LeagueHyperPrior:
    """Lig seviyesi hyper-prior. Tüm takımların ortak prior'u."""
    avg_goals: float = 1.35          # Lig genel gol ortalaması
    home_advantage: float = 0.25     # Ev sahibi avantajı (log-scale)
    avg_attack_alpha: float = 2.0    # Takımların başlangıç atak shape
    avg_attack_beta: float = 2.0     # Takımların başlangıç atak rate
    total_matches: int = 0


class BayesianHierarchicalModel:
    """Hiyerarşik Bayesyen model – lig ortalamasından takıma evrilir.

    Avantajları:
    1. Sezon başında veri azken lig ortalamasını kullanır (regularization)
    2. Maç oynadıkça takımın gerçek gücüne yakınsar
    3. Yeni takımlar (çıkan takımlar) için bile tahmin yapabilir
    4. Overfit'e karşı doğal koruma (prior shrinkage)
    """

    # Önceden bilinen lig ortalamaları
    LEAGUE_PRIORS: dict[str, LeagueHyperPrior] = {
        "super_lig": LeagueHyperPrior(avg_goals=1.35, home_advantage=0.30),
        "premier_league": LeagueHyperPrior(avg_goals=1.40, home_advantage=0.20),
        "la_liga": LeagueHyperPrior(avg_goals=1.30, home_advantage=0.25),
        "bundesliga": LeagueHyperPrior(avg_goals=1.55, home_advantage=0.20),
        "serie_a": LeagueHyperPrior(avg_goals=1.30, home_advantage=0.22),
        "ligue_1": LeagueHyperPrior(avg_goals=1.35, home_advantage=0.23),
        "default": LeagueHyperPrior(avg_goals=1.35, home_advantage=0.25),
    }

    def __init__(self, league: str = "super_lig", shrinkage_strength: float = 5.0):
        self._league = league
        self._hyper = self.LEAGUE_PRIORS.get(league, self.LEAGUE_PRIORS["default"])
        self._shrinkage = shrinkage_strength  # Yüksek → prior'a daha çok bağlı
        self._teams: dict[str, TeamPrior] = {}
        logger.debug(f"BayesianHierarchical başlatıldı: league={league}, "
                     f"avg_goals={self._hyper.avg_goals}")

    def _get_or_create_team(self, team: str) -> TeamPrior:
        """Takımın prior'ını al, yoksa lig ortalamasından oluştur."""
        if team not in self._teams:
            self._teams[team] = TeamPrior(
                team=team,
                attack_alpha=self._hyper.avg_attack_alpha,
                attack_beta=self._hyper.avg_attack_beta,
                defence_alpha=self._hyper.avg_attack_alpha,
                defence_beta=self._hyper.avg_attack_beta,
            )
            logger.debug(f"Yeni takım prior: {team} (lig ortalaması)")
        return self._teams[team]

    # ═══════════════════════════════════════════
    #  BAYESIAN UPDATING – Her maç sonrası güncelle
    # ═══════════════════════════════════════════
    def update(self, home: str, away: str,
               home_goals: int, away_goals: int):
        """Maç sonucuyla posterior'u güncelle.

        Gamma-Poisson conjugate:
        Prior: Gamma(α, β)
        Likelihood: Poisson(λ)
        Posterior: Gamma(α + Σx, β + n)
        """
        h = self._get_or_create_team(home)
        a = self._get_or_create_team(away)

        # Ev sahibi atak gücü güncelleme
        h.attack_alpha += home_goals
        h.attack_beta += 1
        h.update_attack_mean()

        # Ev sahibi savunma güncelleme (yediği gol)
        h.defence_alpha += away_goals
        h.defence_beta += 1
        h.update_defence_mean()

        # Deplasman atak güncelleme
        a.attack_alpha += away_goals
        a.attack_beta += 1
        a.update_attack_mean()

        # Deplasman savunma güncelleme
        a.defence_alpha += home_goals
        a.defence_beta += 1
        a.update_defence_mean()

        h.n_matches += 1
        a.n_matches += 1
        self._hyper.total_matches += 1

    def batch_update(self, matches: list[dict]):
        """Birden fazla maçı toplu güncelle.

        matches: [{home, away, home_goals, away_goals}, ...]
        """
        for m in matches:
            self.update(
                m["home"], m["away"],
                m.get("home_goals", 0), m.get("away_goals", 0),
            )
        logger.info(f"Bayesian batch update: {len(matches)} maç işlendi.")

    # ═══════════════════════════════════════════
    #  TAHMİN
    # ═══════════════════════════════════════════
    def predict(self, home: str, away: str) -> dict:
        """Bayesyen posterior ile maç tahmini."""
        h = self._get_or_create_team(home)
        a = self._get_or_create_team(away)

        # Shrinkage: takımın verisi azsa lig ortalamasına yaklaşır
        n_h = h.n_matches
        n_a = a.n_matches

        # Bayesyen shrinkage formülü:
        # λ_team = (n/(n+k)) * data_mean + (k/(n+k)) * prior_mean
        k = self._shrinkage

        attack_h = (n_h / (n_h + k)) * h.attack_mean + (k / (n_h + k)) * self._hyper.avg_goals
        defence_h = (n_h / (n_h + k)) * h.defence_mean + (k / (n_h + k)) * self._hyper.avg_goals
        attack_a = (n_a / (n_a + k)) * a.attack_mean + (k / (n_a + k)) * self._hyper.avg_goals
        defence_a = (n_a / (n_a + k)) * a.defence_mean + (k / (n_a + k)) * self._hyper.avg_goals

        # Beklenen gol sayıları
        lambda_h = attack_h * defence_a * math.exp(self._hyper.home_advantage)
        mu_a = attack_a * defence_h

        lambda_h = max(min(lambda_h, 6.0), 0.1)
        mu_a = max(min(mu_a, 6.0), 0.1)

        # Poisson olasılıkları
        from scipy.stats import poisson
        max_g = 8

        p_home = sum(
            poisson.pmf(i, lambda_h) * poisson.pmf(j, mu_a)
            for i in range(max_g) for j in range(max_g) if i > j
        )
        p_draw = sum(
            poisson.pmf(i, lambda_h) * poisson.pmf(i, mu_a)
            for i in range(max_g)
        )
        p_away = 1.0 - p_home - p_draw
        p_away = max(p_away, 0.0)

        over25 = sum(
            poisson.pmf(i, lambda_h) * poisson.pmf(j, mu_a)
            for i in range(max_g) for j in range(max_g) if i + j > 2
        )
        btts = sum(
            poisson.pmf(i, lambda_h) * poisson.pmf(j, mu_a)
            for i in range(1, max_g) for j in range(1, max_g)
        )

        # Credible interval (posterior belirsizliği)
        from scipy.stats import gamma as gamma_dist
        attack_ci = gamma_dist.interval(0.90, h.attack_alpha, scale=1/max(h.attack_beta, 0.01))
        defence_ci = gamma_dist.interval(0.90, h.defence_alpha, scale=1/max(h.defence_beta, 0.01))

        return {
            "home_team": home,
            "away_team": away,
            "lambda_home": float(lambda_h),
            "mu_away": float(mu_a),
            "prob_home": float(p_home),
            "prob_draw": float(p_draw),
            "prob_away": float(p_away),
            "prob_over25": float(over25),
            "prob_btts": float(btts),
            "home_attack_posterior": float(attack_h),
            "home_defence_posterior": float(defence_h),
            "away_attack_posterior": float(attack_a),
            "away_defence_posterior": float(defence_a),
            "shrinkage_home": float(k / (n_h + k)),  # 1.0 = tamamen prior
            "shrinkage_away": float(k / (n_a + k)),
            "home_matches": n_h,
            "away_matches": n_a,
            "credible_interval_attack": [float(attack_ci[0]), float(attack_ci[1])],
            "credible_interval_defence": [float(defence_ci[0]), float(defence_ci[1])],
        }

    def team_profile(self, team: str) -> dict:
        """Takımın Bayesyen profili."""
        t = self._get_or_create_team(team)
        k = self._shrinkage
        n = t.n_matches
        shrinkage = k / (n + k)

        return {
            "team": team,
            "n_matches": n,
            "attack_posterior": float(t.attack_mean),
            "defence_posterior": float(t.defence_mean),
            "shrinkage": float(shrinkage),
            "prior_weight": f"{shrinkage:.0%}",
            "data_weight": f"{1-shrinkage:.0%}",
            "interpretation": (
                "Prior ağırlıklı (yeterli veri yok)" if shrinkage > 0.5
                else "Veri ağırlıklı (güvenilir posterior)"
            ),
        }


class NPxGFilter:
    """Non-Penalty xG (npxG) filtresi.

    Penaltıdan gelen 0.76 xG ile akan oyundan gelen 0.76 xG
    aynı değildir. Penaltı xG'sini ayrıştırarak gerçek atak
    gücünü daha doğru ölçeriz.

    Formül: npxG = xG - (penaltı_sayısı × 0.76)
    Penaltı xG ortalaması ≈ 0.76
    """

    PENALTY_XG = 0.76  # Ortalama penaltı xG değeri

    def __init__(self):
        logger.debug("NPxGFilter başlatıldı.")

    def calculate_npxg(self, total_xg: float, penalties_taken: int) -> float:
        """Non-Penalty xG hesapla."""
        penalty_xg = penalties_taken * self.PENALTY_XG
        npxg = max(total_xg - penalty_xg, 0.0)
        return npxg

    def npxg_per_match(self, total_xg: float, penalties: int,
                       matches: int) -> float:
        """Maç başı npxG."""
        npxg = self.calculate_npxg(total_xg, penalties)
        return npxg / max(matches, 1)

    def filter_features(self, features: dict) -> dict:
        """Feature dict'ine npxG sütunları ekle."""
        home_xg = features.get("home_xg", 0.0) or 0.0
        away_xg = features.get("away_xg", 0.0) or 0.0
        home_penalties = features.get("home_penalties_taken", 0) or 0
        away_penalties = features.get("away_penalties_taken", 0) or 0

        home_npxg = self.calculate_npxg(home_xg, home_penalties)
        away_npxg = self.calculate_npxg(away_xg, away_penalties)

        features["home_npxg"] = home_npxg
        features["away_npxg"] = away_npxg
        features["npxg_diff"] = home_npxg - away_npxg
        features["penalty_xg_share_home"] = (
            (home_penalties * self.PENALTY_XG) / max(home_xg, 0.01)
        )
        features["penalty_xg_share_away"] = (
            (away_penalties * self.PENALTY_XG) / max(away_xg, 0.01)
        )
        return features

    def assess_quality(self, xg: float, npxg: float) -> dict:
        """xG kalitesini değerlendir."""
        if xg < 0.01:
            return {"quality": "unknown", "penalty_dependency": 0}

        penalty_share = (xg - npxg) / xg
        if penalty_share > 0.4:
            quality = "penalty_dependent"
            warning = "xG'nin >%40'ı penaltıdan – gerçek atak gücü abartılıyor!"
        elif penalty_share > 0.2:
            quality = "moderate_penalty"
            warning = "Dikkate değer penaltı katkısı."
        else:
            quality = "organic"
            warning = "xG organik – güvenilir."

        return {
            "quality": quality,
            "penalty_share": float(penalty_share),
            "warning": warning,
            "npxg": float(npxg),
            "total_xg": float(xg),
        }
