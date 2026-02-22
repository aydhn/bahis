"""
drift_detector.py – Wasserstein Distance bazlı Model Drift Tespiti.

Veri dağılımları arasındaki 'dünya taşıma mesafesini' (Wasserstein Distance) 
hesaplayarak modelin eskidiği anları otonom tespit eder.
"""
import numpy as np
from scipy.stats import wasserstein_distance
from loguru import logger
from typing import List, Optional

class DriftDetector:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_distribution = None
        logger.debug(f"[Drift] Wasserstein detektörü başlatıldı (threshold={threshold})")

    def set_reference(self, data: np.ndarray):
        """Eğitim anındaki veri dağılımını referans olarak kaydet."""
        self.reference_distribution = data
        logger.info(f"[Drift] Referans dağılım kaydedildi. (n={len(data)})")

    def check_drift(self, current_data: np.ndarray) -> dict:
        """Canlı veri ile referans arasındaki sapmayı ölçer."""
        if self.reference_distribution is None:
            return {"drift_detected": False, "distance": 0.0, "reason": "No reference"}

        # Wasserstein Distance hesapla
        distance = wasserstein_distance(self.reference_distribution, current_data)
        
        result = {
            "distance": round(float(distance), 4),
            "drift_detected": distance > self.threshold,
            "threshold": self.threshold
        }
        
        if result["drift_detected"]:
            logger.warning(f"[Drift] ALERT! Veri sapması tespit edildi: {distance:.4f} > {self.threshold}")
        else:
            logger.debug(f"[Drift] Veri dağılımı stabil: {distance:.4f}")
            
        return result

    def run_batch(self, features_list: List[np.ndarray]) -> List[dict]:
        """Toplu drift kontrolü."""
        results = []
        for feat in features_list:
            results.append(self.check_drift(feat))
        return results
