"""
circuit_breaker.py – Gelişmiş Circuit Breaker pattern.
Üst üste N hata → 1 saat devre dışı → yarım açık test → karar.

Scraper'lar ile entegre: mackolik çökerse sofascore devam eder.
Telegram'a otomatik bildirim + metrik toplama.
"""
from __future__ import annotations

import asyncio
import importlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from loguru import logger


class CBState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitStats:
    total_calls: int = 0
    failures: int = 0
    successes: int = 0
    last_failure_time: float = 0.0
    last_error: str = ""
    consecutive_failures: int = 0
    open_count: int = 0             # kaç kez devre açıldı
    last_state_change: float = 0.0
    total_rejected: int = 0         # devre açıkken reddedilen çağrı


@dataclass
class CBConfig:
    """Her scraper/modül için ayrı konfigürasyon."""
    failure_threshold: int = 3        # üst üste 3 hatada aç
    recovery_timeout: float = 3600.0  # 1 saat bekle
    half_open_max_calls: int = 2      # yarım açıkta en fazla 2 test
    success_threshold: int = 2        # yarım açıkta 2 başarı → kapat
    backoff_multiplier: float = 1.5   # her açılmada bekleme çarpanı
    max_recovery_timeout: float = 14400.0  # maks 4 saat


class CircuitBreaker:
    """Scraper ve modülleri izole eden hata tolerans mekanizması.

    Senaryolar:
    - Mackolik IP engeli → 3 hata → 1 saat devre dışı → sofascore devam
    - 2. kez açılma → 1.5 saat bekleme (exponential backoff)
    - Yarım açık test başarılı → normal moda dön
    """

    def __init__(self, name: str, config: CBConfig | None = None):
        self.name = name
        self._config = config or CBConfig()
        self._state = CBState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._current_recovery = self._config.recovery_timeout
        self._on_open_callbacks: list[Callable] = []
        self._on_close_callbacks: list[Callable] = []

    @property
    def state(self) -> CBState:
        if self._state == CBState.OPEN:
            elapsed = time.time() - self._stats.last_failure_time
            if elapsed >= self._current_recovery:
                self._transition(CBState.HALF_OPEN)
        return self._state

    @property
    def is_available(self) -> bool:
        return self.state != CBState.OPEN

    @property
    def time_until_recovery(self) -> float:
        """Devre açıksa kaç saniye kaldı."""
        if self._state != CBState.OPEN:
            return 0.0
        elapsed = time.time() - self._stats.last_failure_time
        return max(self._current_recovery - elapsed, 0.0)

    # ───────────────────────────────────────────
    #  Senkron çağrı
    # ───────────────────────────────────────────
    def call(self, fn: Callable, *args, **kwargs) -> Any:
        current = self.state

        if current == CBState.OPEN:
            self._stats.total_rejected += 1
            remaining = self.time_until_recovery
            logger.warning(
                f"[CB:{self.name}] Devre AÇIK – çağrı reddedildi. "
                f"Kalan: {remaining/60:.0f} dk"
            )
            return None

        if current == CBState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls > self._config.half_open_max_calls:
                logger.warning(f"[CB:{self.name}] HALF_OPEN test limiti aşıldı.")
                return None

        self._stats.total_calls += 1
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            return None

    # ───────────────────────────────────────────
    #  Asenkron çağrı
    # ───────────────────────────────────────────
    async def call_async(self, fn: Callable, *args, **kwargs) -> Any:
        current = self.state

        if current == CBState.OPEN:
            self._stats.total_rejected += 1
            remaining = self.time_until_recovery
            logger.warning(
                f"[CB:{self.name}] Devre AÇIK – async çağrı reddedildi. "
                f"Kalan: {remaining/60:.0f} dk"
            )
            return None

        if current == CBState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls > self._config.half_open_max_calls:
                return None

        self._stats.total_calls += 1
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            return None

    # ───────────────────────────────────────────
    #  Durum geçişleri
    # ───────────────────────────────────────────
    def _transition(self, new_state: CBState):
        old = self._state
        self._state = new_state
        self._stats.last_state_change = time.time()
        logger.info(f"[CB:{self.name}] {old.value} → {new_state.value}")

        if new_state == CBState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0

        if new_state == CBState.OPEN:
            self._stats.open_count += 1
            for cb in self._on_open_callbacks:
                try:
                    cb(self.name, self._stats.last_error)
                except Exception:
                    pass

        if new_state == CBState.CLOSED:
            for cb in self._on_close_callbacks:
                try:
                    cb(self.name)
                except Exception:
                    pass

    def _on_success(self):
        self._stats.successes += 1
        self._stats.consecutive_failures = 0

        if self._state == CBState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._config.success_threshold:
                self._current_recovery = self._config.recovery_timeout
                self._transition(CBState.CLOSED)
                logger.success(
                    f"[CB:{self.name}] HALF_OPEN → CLOSED (test başarılı, "
                    f"recovery sıfırlandı)"
                )

    def _on_failure(self, error: Exception):
        self._stats.failures += 1
        self._stats.consecutive_failures += 1
        self._stats.last_failure_time = time.time()
        self._stats.last_error = str(error)[:300]
        logger.error(
            f"[CB:{self.name}] Hata "
            f"({self._stats.consecutive_failures}/{self._config.failure_threshold}): "
            f"{error}"
        )

        if self._state == CBState.HALF_OPEN:
            self._current_recovery = min(
                self._current_recovery * self._config.backoff_multiplier,
                self._config.max_recovery_timeout,
            )
            self._transition(CBState.OPEN)
            logger.critical(
                f"[CB:{self.name}] HALF_OPEN → OPEN "
                f"(test başarısız, yeni bekleme: {self._current_recovery/60:.0f} dk)"
            )
            return

        if self._stats.consecutive_failures >= self._config.failure_threshold:
            self._transition(CBState.OPEN)
            logger.critical(
                f"[CB:{self.name}] CLOSED → OPEN "
                f"(eşik aşıldı, bekleme: {self._current_recovery/60:.0f} dk)"
            )

    # ───────────────────────────────────────────
    #  Callback kayıt
    # ───────────────────────────────────────────
    def on_open(self, callback: Callable):
        """Devre açıldığında çağrılacak callback: fn(name, error)."""
        self._on_open_callbacks.append(callback)

    def on_close(self, callback: Callable):
        """Devre kapandığında çağrılacak callback: fn(name)."""
        self._on_close_callbacks.append(callback)

    def reset(self):
        self._state = CBState.CLOSED
        self._stats = CircuitStats()
        self._current_recovery = self._config.recovery_timeout
        logger.info(f"[CB:{self.name}] Manuel sıfırlandı.")

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self._stats.total_calls,
            "failures": self._stats.failures,
            "successes": self._stats.successes,
            "consecutive_failures": self._stats.consecutive_failures,
            "open_count": self._stats.open_count,
            "total_rejected": self._stats.total_rejected,
            "last_error": self._stats.last_error,
            "recovery_timeout_min": self._current_recovery / 60,
            "time_until_recovery_min": self.time_until_recovery / 60,
        }


# ═══════════════════════════════════════════════════════
#  CIRCUIT BREAKER REGISTRY – tüm modüller için merkezi yönetim
# ═══════════════════════════════════════════════════════
class CircuitBreakerRegistry:
    """Tüm circuit breaker'ları merkezi olarak yönetir."""

    # Ön tanımlı konfigürasyonlar
    PRESETS: dict[str, CBConfig] = {
        "scraper": CBConfig(
            failure_threshold=3,
            recovery_timeout=3600.0,   # 1 saat
            half_open_max_calls=2,
            success_threshold=2,
            backoff_multiplier=1.5,
            max_recovery_timeout=14400.0,  # 4 saat
        ),
        "api": CBConfig(
            failure_threshold=5,
            recovery_timeout=300.0,     # 5 dakika
            half_open_max_calls=3,
            success_threshold=2,
            backoff_multiplier=2.0,
            max_recovery_timeout=3600.0,
        ),
        "model": CBConfig(
            failure_threshold=3,
            recovery_timeout=60.0,      # 1 dakika
            half_open_max_calls=1,
            success_threshold=1,
            backoff_multiplier=2.0,
            max_recovery_timeout=600.0,
        ),
    }

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, preset: str = "scraper") -> CircuitBreaker:
        if name not in self._breakers:
            config = self.PRESETS.get(preset, CBConfig())
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def all_statuses(self) -> list[dict]:
        return [cb.status() for cb in self._breakers.values()]

    def healthy_count(self) -> int:
        return sum(1 for cb in self._breakers.values() if cb.state != CBState.OPEN)

    def open_breakers(self) -> list[str]:
        return [name for name, cb in self._breakers.items() if cb.state == CBState.OPEN]

    def reset_all(self):
        for cb in self._breakers.values():
            cb.reset()


class ModuleLoader:
    """importlib ile dinamik modül yükleme + Circuit Breaker koruması."""

    def __init__(self, registry: CircuitBreakerRegistry | None = None):
        self._registry = registry or CircuitBreakerRegistry()
        self._modules: dict[str, Any] = {}

    def load(self, module_path: str, class_name: str,
             preset: str = "model", **init_kwargs) -> Any:
        cb_name = f"{module_path}.{class_name}"
        cb = self._registry.get_or_create(cb_name, preset)

        def _do_load():
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(**init_kwargs)

        instance = cb.call(_do_load)
        if instance is not None:
            self._modules[cb_name] = instance
        return instance

    @property
    def registry(self) -> CircuitBreakerRegistry:
        return self._registry

    def all_statuses(self) -> list[dict]:
        return self._registry.all_statuses()

    def healthy_count(self) -> int:
        return self._registry.healthy_count()
