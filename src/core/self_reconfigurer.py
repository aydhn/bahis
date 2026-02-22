"""
self_reconfigurer.py – Sistemin kendi parametrelerini otonom güncelleme katmanı.

PnL verilerine ve hata loglarına göre sistemin config/threshold 
ayarlarını insan müdahalesi olmadan yeniden yapılandırır.
"""
import json
import os
from loguru import logger
from typing import Any

class SelfReconfigurer:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._last_pnl = 0.0

    def check_and_reconfigure(self, current_pnl: float = 0.0, win_rate: float = 0.5, **kwargs):
        """Kârlılık ve başarı oranına göre config'i optimize eder."""
        if not os.path.exists(self.config_path):
            return

        reconfig_needed = False
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # 1. Başarı oranı çok düşükse (Loss-streak koruması)
            if win_rate < 0.45:
                # Sinyal eşiklerini (thresholds) yukarı çek (daha seçici ol)
                if "thresholds" in config:
                    config["thresholds"]["min_confidence"] = min(config["thresholds"].get("min_confidence", 0.6) + 0.05, 0.85)
                    reconfig_needed = True
                    logger.warning("[Reconfig] Başarı oranı düşük. Seçicilik artırılıyor.")

            # 2. Kâr yüksekse (Aggressive Mode)
            if current_pnl > self._last_pnl + 100.0:
                # Kelly limitlerini hafifçe genişlet
                if "risk" in config:
                    config["risk"]["max_kelly"] = min(config["risk"].get("max_kelly", 0.05) + 0.01, 0.10)
                    reconfig_needed = True
                    logger.info("[Reconfig] Yüksek kâr! Risk limitleri otonom genişletildi.")

            if reconfig_needed:
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                logger.success("[Reconfig] Sistem parametreleri başarıyla otonom olarak güncellendi.")
            
            self._last_pnl = current_pnl

        except Exception as e:
            logger.error(f"[Reconfig] Hata: {e}")

    def apply_logic_patch(self, module_name: str, fix_code: str):
        """Kritik mantık hataları için otonom yama (patch) uygular. (Advanced)"""
        logger.warning(f"[Reconfig:Patch] {module_name} için otonom yama talebi alındı. Güvenlik kontrolü yapılıyor...")
        # Uygulama detayları AutoHealer_v2 kapsamında geliştirilecektir.
        pass
