"""
kalman_filter.py – Kalman Filtresi ile Dinamik Takım Gücü Takibi.

Geleneksel Elo modelleri "gürültülü" (noise) sonuçlara karşı çok hassastır.
Kalman filtresi, bir takımın "gerçek gücünü" gizli bir durum (hidden state) 
olarak görür ve her maç sonucuyla bu durumu günceller.

Özellikler:
  - State: [Hücum Gücü, Savunma Gücü]
  - Transition: Güç zamanla yavaşça değişir (Process Noise).
  - Measurement: Maç skorları (Observation Noise).
"""
import numpy as np
from typing import Dict, Tuple, Any
from loguru import logger

class KalmanStrengthFilter:
    def __init__(self, db: Any = None):
        self.db = db
        # Takım -> { 'state': np.array([atk, def]), 'P': covariance_matrix }
        self._states: Dict[str, Dict] = {}
        
        # Filtre Parametreleri
        self.Q = np.eye(2) * 0.01  # Process Noise (Güç ne kadar hızlı değişebilir?)
        self.R = np.eye(1) * 1.5   # Measurement Noise (Maç skorları ne kadar gürültülü?)
        self.H = np.array([[1, -1]]) # Gözlem matrisi (Atak - Defans farkı skoru belirler gibi)

    def update_team(self, team_name: str, score_diff: float):
        """Yeni bir maç sonucuyla takım gücünü günceller."""
        if team_name not in self._states:
            self._states[team_name] = {
                'state': np.array([0.0, 0.0]), # Başlangıç: Ortalama
                'P': np.eye(2) * 1.0           # Başlangıç belirsizliği
            }

        s = self._states[team_name]
        x = s['state']
        P = s['P']

        # 1. Tahmin (Predict)
        x_pred = x # Sabit hız modeli (güç değişmez varsayımı)
        P_pred = P + self.Q

        # 2. Güncelleme (Update)
        # İnovasyon (Ölçüm - Tahmin)
        z = np.array([score_diff])
        y = z - (self.H @ x_pred)

        # Kalman Kazancı (Kalman Gain)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Yeni Durum
        x_new = x_pred + (K @ y).flatten()
        P_new = (np.eye(2) - K @ self.H) @ P_pred

        self._states[team_name] = {'state': x_new, 'P': P_new}
        return x_new

    def get_strength(self, team_name: str) -> Tuple[float, float]:
        """Takımın mevcut atak ve defans gücünü döndürür."""
        state = self._states.get(team_name, {}).get('state', np.array([0.0, 0.0]))
        return float(state[0]), float(state[1])

    def run_batch(self, days: int = 365):
        """DB'deki geçmiş maçları tarayarak tüm takımların güçlerini hesaplar."""
        if self.db is None:
            logger.warning("KalmanFilter: DB bağlantısı yok.")
            return

        logger.info(f"Kalman Filtresi çalıştırılıyor (Son {days} gün)...")
        query = f"""
        SELECT home_team, away_team, home_score, away_score
        FROM matches
        WHERE status = 'finished' AND match_date >= CURRENT_DATE - INTERVAL '{days} DAY'
        ORDER BY match_date ASC
        """
        try:
            matches = self.db.query(query).to_dicts()
            for m in matches:
                diff = m['home_score'] - m['away_score']
                self.update_team(m['home_team'], diff)
                self.update_team(m['away_team'], -diff)
            
            logger.success(f"Kalman Filtresi tamamlandı. {len(self._states)} takım güncellendi.")
        except Exception as e:
            logger.error(f"Kalman batch hatası: {e}")
