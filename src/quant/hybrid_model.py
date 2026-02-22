"""
hybrid_model.py – Poisson + Weibull Gol Modeli.

Amaç:
Standart Poisson dağılımı "maçta kaç gol olur" sorusuna yanıt verir.
Ancak "Gol NE ZAMAN gelir?" sorusuna yanıt veremez.

Weibull dağılımı, 'Time-to-Event' (Olay gerçekleşme süresi) analizlerinde kullanılır.
Bu hibrit model ikisini birleştirir:
1. Poisson -> Beklenen Toplam Gol (Lambda)
2. Weibull -> Gollerin zamanlaması (Shape parameter -> Increasing hazard rate)

Kullanım: "LATE GOAL" (Son dakika golü) ihtimalini hesaplamak.
"""
import numpy as np
from scipy.stats import poisson, weibull_min
from dataclasses import dataclass

@dataclass
class TimeProb:
    minute: int
    prob_goal_home: float
    prob_goal_away: float

class HybridGoalModel:
    def __init__(self, weibull_shape: float = 1.5):
        # Shape (k) > 1: Zaman geçtikçe gol olma ihtimali artar (Yorgunluk etkisi)
        # Shape (k) = 1: Sabit ihtimal (Exponential)
        self.shape = weibull_shape

    def predict_late_goal_prob(self, home_xg: float, away_xg: float, current_minute: int = 70) -> dict:
        """Kalan sürede gol olma ihtimalini hesaplar."""
        
        remaining_minutes = 90 - current_minute
        if remaining_minutes <= 0:
            return {"prob_late_home": 0.0, "prob_late_away": 0.0}
            
        # 1. Poisson: Kalan süre için XG'yi ölçekle (Basit yaklaşım)
        # Ancak Weibull ile bunu modüle edeceğiz.
        
        # Weibull Cumulative Distribution Function (CDF): P(T <= t)
        # Scale (lambda) parametresini XG'den türetelim.
        # Mean of Weibull = scale * gamma(1 + 1/shape)
        # Biz ortalama gol zamanını 45. dakikaya (veya XG yoğunluğuna göre) oturtmalıyız.
        # Basitleştirme:
        # Bir devrede gol olma ihtimalini Poisson belirliyor.
        # Bu ihtimalin zaman içindeki dağılımını Weibull belirliyor.
        
        # Poisson: Maç sonuna kadar hiç gol OLMAMA ihtimali
        # P(X >= 1) = 1 - P(X=0) = 1 - exp(-lambda)
        
        # Kalan süredeki "efektif lambda"yı bulalım.
        # Normalde XG tüm maça yayılır.
        # Lineer varsayımda: rem_lambda = total_xg * (rem_min / 90)
        # Weibull varsayımda: Gollerin sonlara doğru sıklaşmasını istiyoruz.
        
        # Weibull Hazard Function h(t) = (k/scale) * (t/scale)^(k-1)
        # Artan hazard (k > 1) -> Maç sonlarına doğru gol riski artar.
        
        scale = 90.0 # Gollerin ortalama dağılım ölçeği
        
        # CDF farkı: P(T <= 90) - P(T <= current)
        # Bu bize kalan süredeki "olay yoğunluğunu" verir.
        cdf_90 = weibull_min.cdf(90, self.shape, scale=scale)
        cdf_curr = weibull_min.cdf(current_minute, self.shape, scale=scale)
        
        intensity_factor = (cdf_90 - cdf_curr) / cdf_90 if cdf_90 > 0 else 0
        
        # Şimdi Poisson lambda'yı bu faktörle güncelle
        rem_lambda_home = home_xg * intensity_factor
        rem_lambda_away = away_xg * intensity_factor
        
        # Kalan sürede en az 1 gol olma ihtimali
        prob_home = 1.0 - np.exp(-rem_lambda_home)
        prob_away = 1.0 - np.exp(-rem_lambda_away)
        
        return {
            "current_minute": current_minute,
            "intensity_factor": float(intensity_factor), # Lineer olsaydı rem/90 olurdu
            "prob_late_goal_home": float(prob_home),
            "prob_late_goal_away": float(prob_away),
            "prob_any_late_goal": float(1.0 - (1.0 - prob_home) * (1.0 - prob_away))
        }

    def simulate_match_timeline(self, home_xg: float, away_xg: float) -> list[TimeProb]:
        """Maçın 10'ar dakikalık dilimlerindeki gol ihtimalleri."""
        timeline = []
        for m in range(0, 90, 10):
            p = self.predict_late_goal_prob(home_xg, away_xg, current_minute=m)
            # Bu fonksiyon "kalan süre" veriyor, biz sadece o 10 dk'lık dilimi istiyoruz.
            # O yüzden predict metodunu biraz hackleyebiliriz veya basitçe CDF farkı alabiliriz.
            
            # Dilim için:
            start = m
            end = m + 10
            scale = 90.0
            
            cdf_end = weibull_min.cdf(end, self.shape, scale=scale)
            cdf_start = weibull_min.cdf(start, self.shape, scale=scale)
            
            factor = (cdf_end - cdf_start)
            
            l_home = home_xg * factor
            l_away = away_xg * factor
            
            prob_h = 1.0 - np.exp(-l_home)
            prob_a = 1.0 - np.exp(-l_away)
            
            timeline.append(TimeProb(minute=end, prob_goal_home=prob_h, prob_goal_away=prob_a))
            
        return timeline
