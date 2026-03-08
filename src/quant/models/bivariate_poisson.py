"""
bivariate_poisson.py – Karlis & Ntzoufras (2003) Bivariate Poisson Model.

Futbol skor tahmini için standart Poisson dağılımı (X, Y bağımsız)
yerine, skorlar arasındaki korelasyonu (λ3) dikkate alan gelişmiş model.

Formülasyon:
  X = X1 + X3,  Y = Y2 + X3
  X1 ~ Po(λ1), Y2 ~ Po(λ2), X3 ~ Po(λ3)
  Cov(X, Y) = λ3

Avantajı:
  - 0-0, 1-1 gibi beraberlik olasılıklarını daha doğru modeller.
  - Bağımsız Poisson'un "beraberlikleri düşük tahmin etme" sorununu çözer.
"""
import numpy as np
from scipy.special import factorial

class BivariatePoisson:
    """Bivariate Poisson Dağılımı Hesaplayıcısı."""

    def __init__(self):
        pass

    @staticmethod
    def pmf(x: int, y: int, lambda1: float, lambda2: float, lambda3: float) -> float:
        """
        P(X=x, Y=y) olasılığını hesaplar.

        Parametreler:
          x, y: Skorlar (örn: 2, 1)
          lambda1: Ev sahibi atak gücü (bağımsız kısım)
          lambda2: Deplasman atak gücü (bağımsız kısım)
          lambda3: Kovaryans (ortak şok/korelasyon)
        """
        # Negatif input kontrolü
        if x < 0 or y < 0:
            return 0.0

        # λ3 = 0 ise bağımsız Poisson çarpımı
        if lambda3 <= 1e-9:
            p_x = (np.exp(-(lambda1)) * (lambda1**x)) / factorial(x)
            p_y = (np.exp(-(lambda2)) * (lambda2**y)) / factorial(y)
            return float(p_x * p_y)

        # Bivariate Poisson formülü (Karlis & Ntzoufras 2003)
        # P(x,y) = exp(-(L1+L2+L3)) * Sum_{k=0 to min(x,y)} ...

        term1 = np.exp(-(lambda1 + lambda2 + lambda3))

        # Summation
        total_sum = 0.0
        min_k = min(x, y)

        for k in range(min_k + 1):
            # Binomial coefficients: binom(n, k) = n! / (k! * (n-k)!)
            # Formula: (L1^x * L2^y / x!y!) * binom(x,k)*binom(y,k)*k! * (L3/L1L2)^k
            # Simplifies to:
            # (L1^(x-k) * L2^(y-k) * L3^k) / ((x-k)! * (y-k)! * k!)

            num = (lambda1**(x-k)) * (lambda2**(y-k)) * (lambda3**k)
            den = factorial(x-k) * factorial(y-k) * factorial(k)
            total_sum += num / den

        return float(term1 * total_sum)

    def predict_score_matrix(self, lambda_home: float, lambda_away: float, correlation: float, max_goals: int = 10) -> np.ndarray:
        """
        Skor matrisi (0-0'dan max_goals-max_goals'a kadar) döndürür.

        Input:
          lambda_home: Beklenen ev golü (E[X] = L1 + L3)
          lambda_away: Beklenen dep golü (E[Y] = L2 + L3)
          correlation: Cov(X,Y) = L3. (Genelde 0 ile min(LH, LA) arasında)

        Dönüşüm:
          L3 = correlation
          L1 = lambda_home - L3
          L2 = lambda_away - L3
        """
        # Parametre dönüşümü
        l3 = max(0.0, correlation)
        # L1 ve L2 negatif olamaz
        l1 = max(0.001, lambda_home - l3)
        l2 = max(0.001, lambda_away - l3)

        # Grid oluştur
        grid = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                grid[i, j] = self.pmf(i, j, l1, l2, l3)

        # Normalizasyon (Truncation hatasını düzeltmek için)
        total_prob = np.sum(grid)
        if total_prob > 0:
            grid /= total_prob

        return grid

    def get_market_probabilities(self, grid: np.ndarray) -> dict:
        """Skor matrisinden bahis market olasılıklarını türetir."""
        prob_home_win = np.sum(np.tril(grid, -1))
        prob_draw = np.sum(np.diag(grid))
        prob_away_win = np.sum(np.triu(grid, 1))

        prob_over_25 = np.sum([grid[i, j] for i in range(grid.shape[0]) for j in range(grid.shape[1]) if i+j > 2.5])
        prob_bts = 1.0 - (np.sum(grid[0, :]) + np.sum(grid[:, 0]) - grid[0, 0])

        return {
            "home_win": float(prob_home_win),
            "draw": float(prob_draw),
            "away_win": float(prob_away_win),
            "over_2.5": float(prob_over_25),
            "bts_yes": float(prob_bts)
        }
