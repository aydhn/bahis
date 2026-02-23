"""
model_registry.py – Quant Modellerinin dinamik kaydı ve yönetimi.
"""
from typing import Dict, Type, List, Optional
from loguru import logger
from src.core.interfaces import QuantModel

class ModelRegistry:
    """
    Sistemdeki tüm quant modellerini yönetir.
    Plugin mimarisi sağlar (yeni modeller kolayca eklenebilir).
    """
    _registry: Dict[str, Type[QuantModel]] = {}
    _instances: Dict[str, QuantModel] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[QuantModel]):
        """Yeni bir model sınıfı kaydet."""
        cls._registry[name] = model_class
        logger.debug(f"Model kaydedildi: {name}")

    @classmethod
    def get_model(cls, name: str) -> Optional[QuantModel]:
        """Model örneğini getir (Singleton pattern)."""
        if name in cls._instances:
            return cls._instances[name]

        if name not in cls._registry:
            logger.error(f"Model bulunamadı: {name}")
            return None

        try:
            instance = cls._registry[name]()
            cls._instances[name] = instance
            return instance
        except Exception as e:
            logger.error(f"Model başlatılamadı ({name}): {e}")
            return None

    @classmethod
    def load_defaults(cls):
        """Varsayılan modelleri yükle (Auto-Discovery)."""
        try:
            from src.quant.adapters import BenterAdapter, LSTMAdapter
            cls.register("benter", BenterAdapter)
            cls.register("lstm", LSTMAdapter)
            logger.info("Varsayılan modeller yüklendi: benter, lstm")
        except ImportError as e:
            logger.warning(f"Varsayılan modeller yüklenemedi: {e}")

    @classmethod
    def get_all_models(cls) -> List[QuantModel]:
        """Kayıtlı tüm modellerin örneklerini döndür."""
        models = []
        for name in cls._registry:
            model = cls.get_model(name)
            if model:
                models.append(model)
        return models
