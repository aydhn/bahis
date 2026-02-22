"""
bayesian_elo.py – Bayesyen Hiyerarşik ELO Modeli.

Geleneksel ELO, takımları birbirinden bağımsız görür. 
Hiyerarşik model ise ligleri ve takımları bir ağaç yapısında toplar. 
Örn: Premier Lig'deki bir sonuç, Şampiyonlar Ligi üzerinden diğer liglerin 
belirsizlik (uncertainty) parametrelerini etkileyebilir.

Matematik:
  R_i ~ Normal(mu_league, sigma_league)
  P(i wins) = Logistic(R_i - R_j)
"""
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple
from loguru import logger

class BayesianEloEngine:
    def __init__(self, db: Any = None):
        self.db = db
        # ratings[team_id] = (mean, std)
        self.ratings: Dict[str, Tuple[float, float]] = {}
        # league_hyperparameters[league_id] = (mu, sigma)
        self.league_meta: Dict[str, Tuple[float, float]] = {}
        self.base_rating = 1500.0
        self.base_sigma = 200.0

    def get_rating(self, team: str, league: str = "global") -> Tuple[float, float]:
        """Takımın mevcut rating ve belirsizlik değerini döner."""
        if team not in self.ratings:
            # Hiyerarşik başlangıç: Ligin ortalamasından başla
            mu_l, sigma_l = self.league_meta.get(league, (self.base_rating, self.base_sigma))
            self.ratings[team] = (mu_l, sigma_l)
        return self.ratings[team]

    def update_rating(self, team_a: str, team_b: str, result: float, league: str = "global"):
        """
        Bayesyen güncelleme (Kalman-like).
        result: 1.0 (A wins), 0.5 (Draw), 0.0 (B wins)
        """
        mu_a, sigma_a = self.get_rating(team_a, league)
        mu_b, sigma_b = self.get_rating(team_b, league)

        # 1. Beklenen Sonuç (Logistic Likelihood)
        diff = mu_a - mu_b
        expected_a = 1 / (1 + 10**(-diff / 400))
        
        # 2. Uncertainty (K-Factor yerine Belirsizlik Payı)
        # Maç yapıldıkça sigma azalır, ama sürpriz olursa mu kayar
        innovation_a = (result - expected_a)
        
        # Ssigma güncelleme (Shrinkage)
        new_sigma_a = np.sqrt(sigma_a**2 * 0.95) # Bilgi arttıkça belirsizlik düşer
        new_mu_a = mu_a + (sigma_a / 100) * innovation_a # Belirsizlik kadar esneklik
        
        self.ratings[team_a] = (float(new_mu_a), float(new_sigma_a))
        
        # Geri besleme: Lig hiperparametrelerini güncelle (Hiyejarşik öğrenme)
        if league != "global":
            mu_l, sigma_l = self.league_meta.get(league, (self.base_rating, self.base_sigma))
            new_mu_l = mu_l * 0.99 + new_mu_a * 0.01
            self.league_meta[league] = (new_mu_l, sigma_l)

    def predict_probs(self, team_a: str, team_b: str, league: str = "global") -> Dict[str, float]:
        """Galibiyet ihtimallerini belirsizliği hesaba katarak hesaplar."""
        mu_a, sigma_a = self.get_rating(team_a, league)
        mu_b, sigma_b = self.get_rating(team_b, league)
        
        # Monte Carlo Integration (Simple approximation)
        # Fark dağılımı: Normal(mu_a - mu_b, sqrt(sigma_a^2 + sigma_b^2))
        diff_mu = mu_a - mu_b
        diff_sigma = np.sqrt(sigma_a**2 + sigma_b**2)
        
        # Win prob = Integral of Logistic over the normal distribution
        # Basitleştirme: Normal Approximation
        prob_a = stats.norm.cdf(diff_mu / (diff_sigma + 100)) # 100 scaling factor
        
        return {
            "HOME_WIN": float(prob_a),
            "AWAY_WIN": float(1 - prob_a),
            "DRAW": 0.25 # Basit beraberlik varsayımı
        }

    async def run_batch(self, **kwargs):
        """DB'deki geçmiş maçlardan tüm ratingleri sıfırdan inşa eder."""
        if self.db is None: return
        
        logger.info("[BayesianELO] Ratingler yeniden hesaplanıyor...")
        query = "SELECT home_team, away_team, home_score, away_score, league FROM matches WHERE status='finished' ORDER BY match_date"
        try:
            df = self.db.query(query).to_dicts()
            for m in df:
                score_a, score_b = m["home_score"], m["away_score"]
                res = 1.0 if score_a > score_b else 0.5 if score_a == score_b else 0.0
                self.update_rating(m["home_team"], m["away_team"], res, m.get("league", "global"))
            
            logger.success(f"[BayesianELO] {len(self.ratings)} takım için ratingler güncellendi.")
        except Exception as e:
            logger.error(f"BayesianELO batch hatası: {e}")
