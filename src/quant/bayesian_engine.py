"""
bayesian_engine.py – Bayesçi İnanç Güncelleyici.

Amacı:
Takımların gücünü (Win Probability) statik bir sayı olarak değil,
bir Olasılık Dağılımı (Beta Distribution) olarak modellemek.

Özellikler:
- Prior (Önsel İnanç): Sezon başı tahmini veya lig ortalaması.
- Likelihood (Kanıt): Maç sonucu (Galibiyet/Mağlubiyet).
- Posterior (Sonsal İnanç): Güncellenmiş güç dağılımı.

Avantajı:
Az veriyle bile çalışır. Belirsizliği (Varyans) ölçer.
"Bu takım %60 kazanır" yerine "Bu takım %55-%65 arasında kazanır" der.
"""
import numpy as np
from scipy.stats import beta
from dataclasses import dataclass

@dataclass
class TeamBelief:
    team: str
    alpha: float = 2.0  # Başarı sayısı (sanal)
    beta: float = 2.0   # Başarısızlık sayısı (sanal)
    
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        return (self.alpha * self.beta) / (
            (self.alpha + self.beta)**2 * (self.alpha + self.beta + 1)
        )
    
    @property
    def confidence_interval(self) -> tuple[float, float]:
        """%95 Güven Aralığı"""
        return beta.interval(0.95, self.alpha, self.beta)

class BayesianEngine:
    def __init__(self):
        self.beliefs: dict[str, TeamBelief] = {}

    def get_belief(self, team: str) -> TeamBelief:
        if team not in self.beliefs:
            # Default: Nötr inanç (alpha=2, beta=2 -> Mean=0.5)
            # Lig ortalaması genellikle ev sahibi için 0.45, deplasman 0.30 vb.
            # Şimdilik nötrden başlatalım.
            self.beliefs[team] = TeamBelief(team, 2.0, 2.0)
        return self.beliefs[team]

    def update_belief(self, team: str, result: str):
        """
        Maç sonucuna göre inancı güncelle (Bayes Kuralı).
        Conjugate Prior (Beta-Binomial) sayesinde analitik güncelleme basittir:
        Posterior Alpha = Prior Alpha + Success
        Posterior Beta = Prior Beta + Failure
        """
        b = self.get_belief(team)
        
        if result == "WIN":
            b.alpha += 1
        elif result == "LOSS":
            b.beta += 1
        elif result == "DRAW":
            # Beraberlik yarım galibiyet, yarım mağlubiyet gibi
            b.alpha += 0.5
            b.beta += 0.5
            
        self.beliefs[team] = b # Gerekli değil (ref) ama temiz

    def calculate_win_prob(self, home: str, away: str, 
                           simulations: int = 1000) -> float:
        """
        İki takımın Beta dağılımlarından örneklem çekerek
        Home > Away olasılığını simüle eder.
        """
        home_b = self.get_belief(home)
        away_b = self.get_belief(away)
        
        # Monte Carlo örneklemesi
        home_samples = np.random.beta(home_b.alpha, home_b.beta, simulations)
        away_samples = np.random.beta(away_b.alpha, away_b.beta, simulations)
        
        # Home'un Away'den daha güçlü olduğu durumların oranı
        # Dikkat: Bu "Home maçı kazanır" demek değil, "Home daha güçlüdür" demek.
        # Maç sonucu için bu gücü bir fonksiyona sokmak gerekir (örn. log5).
        
        better_count = np.sum(home_samples > away_samples)
        return float(better_count / simulations)
