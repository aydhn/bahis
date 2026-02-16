"""
dependency_container.py – Dependency Injection (Bağımlılık Enjeksiyonu).
Veritabanı, scraper ve model sınıflarını doğrudan "new"lemek yerine
dışarıdan parametre olarak geçirir. Kodun test edilebilirliğini artırır.

Ileride veritabanını veya scraper'ı değiştirmek isterseniz, sadece
container kaydını değiştirirsiniz – kodun geri kalanı etkilenmez.
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar
from loguru import logger

T = TypeVar("T")


class DependencyContainer:
    """Basit ama etkili bağımlılık enjeksiyon konteyneri.

    Kullanım:
        container = DependencyContainer()
        container.register("db", DBManager, db_path="data/bahis.duckdb")
        db = container.resolve("db")
    """

    def __init__(self):
        self._factories: dict[str, tuple[Callable, dict]] = {}
        self._singletons: dict[str, Any] = {}
        self._singleton_flags: dict[str, bool] = {}
        logger.debug("DependencyContainer başlatıldı.")

    def register(self, name: str, factory: Callable, *,
                 singleton: bool = True, **kwargs):
        """Bağımlılık kaydeder.

        Args:
            name: Bağımlılık adı (ör. "db", "cache")
            factory: Sınıf veya factory fonksiyonu
            singleton: True ise tek instance oluşturulur (varsayılan)
            **kwargs: factory'ye geçilecek parametreler
        """
        self._factories[name] = (factory, kwargs)
        self._singleton_flags[name] = singleton
        logger.debug(f"Kayıt: {name} → {factory.__name__} (singleton={singleton})")

    def resolve(self, name: str) -> Any:
        """Bağımlılığı çözümler (oluşturur veya cache'ten döndürür)."""
        if name in self._singletons:
            return self._singletons[name]

        if name not in self._factories:
            raise KeyError(f"Bağımlılık bulunamadı: '{name}'. Önce register() ile kaydedin.")

        factory, kwargs = self._factories[name]

        # kwargs içinde container referansı gerekebilir
        resolved_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and v.startswith("@"):
                dep_name = v[1:]
                resolved_kwargs[k] = self.resolve(dep_name)
            else:
                resolved_kwargs[k] = v

        try:
            instance = factory(**resolved_kwargs)
        except Exception as e:
            logger.error(f"Bağımlılık oluşturma hatası [{name}]: {e}")
            raise

        if self._singleton_flags.get(name, True):
            self._singletons[name] = instance

        return instance

    def override(self, name: str, instance: Any):
        """Test için: mevcut kaydı mock ile değiştir."""
        self._singletons[name] = instance
        logger.info(f"Override: {name} → {type(instance).__name__}")

    def reset(self, name: str | None = None):
        """Singleton cache'ini temizler."""
        if name:
            self._singletons.pop(name, None)
        else:
            self._singletons.clear()

    def is_registered(self, name: str) -> bool:
        return name in self._factories

    @property
    def registered(self) -> list[str]:
        return list(self._factories.keys())

    def __getitem__(self, name: str) -> Any:
        return self.resolve(name)

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)


def create_default_container() -> DependencyContainer:
    """Varsayılan bağımlılıkları kayıtlı container oluşturur."""
    from src.memory.db_manager import DBManager
    from src.memory.feature_cache import FeatureCache
    from src.memory.lance_memory import LanceMemory
    from src.memory.graph_rag import GraphRAG
    from src.core.data_validator import DataValidator

    c = DependencyContainer()

    # Layer 2 – Memory
    c.register("db", DBManager)
    c.register("cache", FeatureCache)
    c.register("lance", LanceMemory)
    c.register("graph", GraphRAG)

    # Core
    c.register("validator", DataValidator)

    return c
