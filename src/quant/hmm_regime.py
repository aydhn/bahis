"""
hmm_regime.py – Hidden Markov Model (HMM) Tabanlı Rejim Tespiti.

Piyasa (bahis piyasası dahil) zamanla farklı 'modlara' girer. 
Bu modlar doğrudan gözlenemez (hidden state), ancak gözlenen veriler (pnl, hacim, volatilite) 
üzerinden tahmin edilebilir.

Rejimler:
  0: BULL (Karlı, düşük volatilite)
  1: BEAR (Zararlı veya durağan)
  2: CHAOTIC (Yüksek oynaklık, tahmin edilemez)
"""
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
try:
    from hmmlearn import hmm
    HMM_OK = True
except ImportError:
    HMM_OK = False
    logger.warning("hmmlearn kütüphanesi yüklü değil. HMMRegimeSwitcher mock modda çalışacak.")

class HMMRegimeSwitcher:
    def __init__(self, n_states: int = 3, db: Any = None):
        self.n_states = n_states
        self.db = db
        self.model = None
        if HMM_OK:
            self.model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

    def fit(self, data: np.ndarray):
        """HMM modelini geçmiş verilerle eğitir."""
        if not HMM_OK or self.model is None:
            return
        
        try:
            # Data format: (n_samples, n_features)
            # Features: [PnL_Change, Volatility_Change]
            self.model.fit(data)
            logger.info("HMM Modeli başarıyla eğitildi.")
        except Exception as e:
            logger.error(f"HMM fit hatası: {e}")

    def predict_state(self, current_data: np.ndarray) -> int:
        """Mevcut verilere göre gizli durumu (regime) tahmin eder."""
        if not HMM_OK or self.model is None or not hasattr(self.model, "means_"):
            return 0 # Default: Bull

        try:
            state = self.model.predict(current_data.reshape(1, -1))[0]
            return int(state)
        except Exception as e:
            logger.error(f"HMM predict hatası: {e}")
            return 0

    async def run_batch(self, **kwargs):
        """DB'den veri çekip modeli update eder."""
        if self.db is None:
            return

        logger.info("[HMM] Rejim analizi güncelleniyor...")
        try:
            # Son 200 bahisin PnL ve odds hareketlerini çek
            query = "SELECT pnl, odds FROM bets ORDER BY timestamp DESC LIMIT 200"
            df = self.db.query(query)
            if len(df) < 50:
                logger.debug("HMM: Yetersiz veri.")
                return

            # Feature Engineering
            pnl_diff = np.diff(df["pnl"].to_numpy())
            volatility = np.abs(pnl_diff)
            features = np.column_stack([pnl_diff, volatility[:-1] if len(volatility) > len(pnl_diff) else volatility])
            
            # Eğit
            self.fit(features)
            
            # Mevcut durum
            current_state = self.predict_state(features[-1])
            logger.success(f"[HMM] Mevcut Piyasa Rejimi: {current_state}")
            
        except Exception as e:
            logger.error(f"HMM batch hatası: {e}")
