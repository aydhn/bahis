"""
exception_guardian.py – Global Exception Guardian + Deep Audit Logger.

Tüm `except Exception: pass` bloklarını merkezi bir koruma
katmanıyla değiştirir. Her exception yakalanır, sınıflandırılır,
loglanır ve gerekirse alarm gönderilir. Hiçbir hata sessizce yutulmaz.

Kavramlar:
  - Guardian Context Manager: Her modül `with guardian(module):` içinde çalışır
  - Exception Taxonomy: Hatalar sınıflandırılır (data, network, model, system)
  - Circuit Breaker: Aynı modülden çok hata → devre kırıcı
  - Error Budget: Her modülün saatlik hata bütçesi var
  - Forensics Logger: Her hata tam stack trace + context ile kaydedilir
  - Aggregation: Benzer hatalar gruplandırılır, spam önlenir
  - Heartbeat: Modüllerin yaşam sinyali — 5 dk sinyal yoksa alarm
  - Severity Levels: DEBUG/INFO/WARN/ERROR/CRITICAL → farklı eylemler

Akış:
  1. Modül guardian context'ine girer
  2. Exception olursa Guardian yakalar
  3. Sınıflandırır (taxonomy)
  4. Circuit breaker kontrolü
  5. Forensics log yazar (tam context)
  6. Gerekirse Telegram alertı gönderir
  7. Modül güvenli şekilde devam eder
"""
from __future__ import annotations

import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


# ═══════════════════════════════════════════════
#  EXCEPTION TAKSONOMİSİ
# ═══════════════════════════════════════════════
class ExceptionTaxonomy:
    """Hata sınıflandırma."""

    CATEGORIES = {
        # Network / IO
        ConnectionError: "network",
        TimeoutError: "network",
        OSError: "io",
        FileNotFoundError: "io",
        PermissionError: "io",
        # Data
        ValueError: "data",
        TypeError: "data",
        KeyError: "data",
        IndexError: "data",
        # Model / Math
        ArithmeticError: "math",
        ZeroDivisionError: "math",
        OverflowError: "math",
        FloatingPointError: "math",
        # System
        MemoryError: "system",
        RecursionError: "system",
        SystemError: "system",
        # Import
        ImportError: "import",
        ModuleNotFoundError: "import",
    }

    @classmethod
    def classify(cls, exc: BaseException) -> str:
        for exc_type, category in cls.CATEGORIES.items():
            if isinstance(exc, exc_type):
                return category
        return "unknown"

    @classmethod
    def severity(cls, exc: BaseException) -> str:
        """Ciddiyet seviyesi."""
        cat = cls.classify(exc)
        if cat in ("system", "math"):
            return "CRITICAL"
        if cat == "network":
            return "WARNING"
        if cat in ("data", "import"):
            return "ERROR"
        return "ERROR"


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class ErrorRecord:
    """Tek bir hata kaydı."""
    module: str = ""
    category: str = ""
    severity: str = ""
    exception_type: str = ""
    message: str = ""
    stack_trace: str = ""
    context: dict = field(default_factory=dict)
    timestamp: float = 0.0
    occurrence: int = 1


@dataclass
class ModuleHealth:
    """Modül sağlık durumu."""
    module: str = ""
    total_errors: int = 0
    errors_last_hour: int = 0
    last_error_time: float = 0.0
    last_heartbeat: float = 0.0
    circuit_open: bool = False     # True → modül devre dışı
    error_budget: int = 50         # Saatlik max hata
    categories: dict = field(default_factory=lambda: defaultdict(int))


# ═══════════════════════════════════════════════
#  EXCEPTION GUARDIAN (Ana Sınıf)
# ═══════════════════════════════════════════════
class ExceptionGuardian:
    """Global exception yönetim sistemi.

    Kullanım:
        guardian = ExceptionGuardian()

        # Context manager
        with guardian.protect("poisson_model"):
            result = poisson.predict(data)

        # Dekoratör
        @guardian.guard("ensemble")
        async def run_ensemble(data):
            ...

        # Heartbeat
        guardian.heartbeat("data_factory")

        # Rapor
        report = guardian.health_report()
    """

    def __init__(self, error_budget_per_hour: int = 50,
                 circuit_threshold: int = 10,
                 circuit_reset_seconds: float = 300,
                 alert_callback=None):
        self._budget = error_budget_per_hour
        self._circuit_thresh = circuit_threshold
        self._circuit_reset = circuit_reset_seconds
        self._alert_cb = alert_callback

        self._modules: dict[str, ModuleHealth] = {}
        self._error_log: list[ErrorRecord] = []
        self._aggregated: dict[str, int] = defaultdict(int)

        logger.info(
            f"[Guardian] Başlatıldı: budget={error_budget_per_hour}/h, "
            f"circuit_thresh={circuit_threshold}"
        )

    @contextmanager
    def protect(self, module: str, context: dict | None = None,
                  suppress: bool = True):
        """Exception koruma context manager'ı.

        Args:
            module: Modül adı
            context: Ek bağlam bilgisi (match_id, vb.)
            suppress: True → hata yutulur ama loglanır.
                      False → loglanır VE yeniden raise edilir.
        """
        health = self._ensure_module(module)
        health.last_heartbeat = time.time()

        # Circuit breaker kontrolü
        if health.circuit_open:
            if time.time() - health.last_error_time > self._circuit_reset:
                health.circuit_open = False
                health.errors_last_hour = 0
                logger.info(f"[Guardian] {module}: Circuit breaker sıfırlandı")
            else:
                logger.warning(f"[Guardian] {module}: Circuit OPEN — atlandı")
                yield
                return

        try:
            yield
        except Exception as exc:
            self._handle_exception(module, exc, context or {}, suppress)

    def guard(self, module: str, suppress: bool = True):
        """Exception koruma dekoratörü."""
        def decorator(func):
            if hasattr(func, '__call__') and not hasattr(func, '__wrapped__'):
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    async def async_wrapper(*args, **kwargs):
                        with self.protect(module, suppress=suppress):
                            return await func(*args, **kwargs)
                    async_wrapper.__name__ = func.__name__
                    async_wrapper.__wrapped__ = func
                    return async_wrapper
                else:
                    def sync_wrapper(*args, **kwargs):
                        with self.protect(module, suppress=suppress):
                            return func(*args, **kwargs)
                    sync_wrapper.__name__ = func.__name__
                    sync_wrapper.__wrapped__ = func
                    return sync_wrapper
            return func
        return decorator

    def heartbeat(self, module: str) -> None:
        """Modül yaşam sinyali."""
        health = self._ensure_module(module)
        health.last_heartbeat = time.time()

    def check_heartbeats(self, timeout: float = 300) -> list[str]:
        """Heartbeat kontrolü — sessiz modülleri tespit et."""
        now = time.time()
        silent = []
        for name, health in self._modules.items():
            if health.last_heartbeat > 0 and (now - health.last_heartbeat) > timeout:
                silent.append(name)
                logger.warning(
                    f"[Guardian] {name}: Heartbeat yok "
                    f"({now - health.last_heartbeat:.0f}s)"
                )
        return silent

    def health_report(self) -> dict:
        """Tüm modüllerin sağlık raporu."""
        report = {
            "total_errors": sum(m.total_errors for m in self._modules.values()),
            "open_circuits": [
                n for n, m in self._modules.items() if m.circuit_open
            ],
            "modules": {},
        }
        for name, health in self._modules.items():
            report["modules"][name] = {
                "total_errors": health.total_errors,
                "errors_last_hour": health.errors_last_hour,
                "circuit_open": health.circuit_open,
                "last_heartbeat": health.last_heartbeat,
                "categories": dict(health.categories),
            }
        return report

    def get_recent_errors(self, limit: int = 20) -> list[dict]:
        """Son N hata kaydı."""
        return [
            {
                "module": e.module,
                "category": e.category,
                "severity": e.severity,
                "type": e.exception_type,
                "message": e.message[:200],
                "timestamp": e.timestamp,
                "occurrence": e.occurrence,
            }
            for e in self._error_log[-limit:]
        ]

    # ═══════════════════════════════════════════
    #  İÇ YÖNTEMLER
    # ═══════════════════════════════════════════
    def _handle_exception(self, module: str, exc: Exception,
                            context: dict, suppress: bool) -> None:
        """Exception'ı işle."""
        health = self._ensure_module(module)
        now = time.time()

        # Sınıflandır
        category = ExceptionTaxonomy.classify(exc)
        severity = ExceptionTaxonomy.severity(exc)

        # Stack trace
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        stack = "".join(tb)

        # Aggregation anahtarı
        agg_key = f"{module}:{type(exc).__name__}:{str(exc)[:80]}"
        self._aggregated[agg_key] += 1
        occurrence = self._aggregated[agg_key]

        # Kayıt oluştur
        record = ErrorRecord(
            module=module,
            category=category,
            severity=severity,
            exception_type=type(exc).__name__,
            message=str(exc),
            stack_trace=stack,
            context=context,
            timestamp=now,
            occurrence=occurrence,
        )
        self._error_log.append(record)

        # Log boyutunu sınırla
        if len(self._error_log) > 5000:
            self._error_log = self._error_log[-3000:]

        # Modül sağlığını güncelle
        health.total_errors += 1
        health.errors_last_hour += 1
        health.last_error_time = now
        health.categories[category] += 1

        # Saatlik reset
        if now - getattr(health, '_hour_start', now) > 3600:
            health.errors_last_hour = 1
            health._hour_start = now

        # Circuit breaker
        if health.errors_last_hour >= self._circuit_thresh:
            health.circuit_open = True
            logger.error(
                f"[Guardian] {module}: CIRCUIT BREAKER OPEN — "
                f"{health.errors_last_hour} hata/saat"
            )

        # Loglama (spam filtresi: ilk 3 + her 10. tekrar)
        if occurrence <= 3 or occurrence % 10 == 0:
            log_level = {
                "CRITICAL": "critical",
                "ERROR": "error",
                "WARNING": "warning",
            }.get(severity, "error")

            safe_exc = str(exc)[:200].replace("<", "[").replace(">", "]")
            msg = (
                f"[Guardian] {module} [{category}/{severity}]: "
                f"{type(exc).__name__}: {safe_exc}"
            )
            if occurrence > 1:
                msg += f" (#{occurrence})"
            if context:
                safe_ctx = str(context)[:200].replace("<", "[").replace(">", "]")
                msg += f" | ctx: {safe_ctx}"

            getattr(logger, log_level)(msg)

            if severity == "CRITICAL" or occurrence == 1:
                safe_stack = stack[:1000].replace("<", "[").replace(">", "]")
                logger.debug(f"[Guardian] Stack:\n{safe_stack}")

        # Alarm callback
        if severity == "CRITICAL" and self._alert_cb and occurrence <= 3:
            try:
                self._alert_cb(record)
            except Exception:
                pass

        if not suppress:
            raise

    def _ensure_module(self, module: str) -> ModuleHealth:
        if module not in self._modules:
            self._modules[module] = ModuleHealth(
                module=module,
                error_budget=self._budget,
            )
            self._modules[module]._hour_start = time.time()
        return self._modules[module]


# ═══════════════════════════════════════════════
#  GLOBAL EXCEPTION HOOK
# ═══════════════════════════════════════════════
def install_global_hook(guardian: ExceptionGuardian | None = None) -> None:
    """Yakalanmamış exception'ları da Guardian'a yönlendir."""
    original_hook = sys.excepthook

    def guardian_hook(exc_type, exc_value, exc_tb):
        if guardian:
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            # Loguru'nun colorizer'ı <module> gibi etiketleri renk yönergesi
            # olarak yorumlar – açı parantezlerini [] ile değiştir
            safe_tb = tb[:2000].replace("<", "[").replace(">", "]")
            safe_val = str(exc_value).replace("<", "[").replace(">", "]")
            logger.critical(
                f"[Guardian] UNCAUGHT: {exc_type.__name__}: {safe_val}\n{safe_tb}"
            )
        original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = guardian_hook
    logger.info("[Guardian] Global exception hook kuruldu")
