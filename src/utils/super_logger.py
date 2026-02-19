"""
super_logger.py – Structured Deep Logging (Loguru & JSON).

Sistem karmaşıklaştıkça standart loglar yetersiz kalır. Bu modül
her modülün ne yaptığını, hangi veriyi aldığını ve neden o kararı
verdiğini makine tarafından okunabilir (JSON) şekilde kaydeder.

Kavramlar:
  - Structured Logging: Log satırları JSON formatında — makine okunabilir
  - Module-Scoped Sinks: Her modül kendi log dosyasına yazar
  - Log Rotation: Dosya 100MB'a ulaşınca otomatik yeni dosya açar
  - Log Compression: 1 aydan eski loglar gzip ile sıkıştırılır
  - Context Enrichment: Her log satırına timestamp, module, duration,
    cpu_pct, memory_mb, input_summary, output_summary eklenir
  - Decision Audit Trail: Model kararlarının gerekçeli kaydı
  - Performance Profiling: İşlem süresi, CPU ve bellek kullanımı

Akış:
  1. Modül init sırasında SuperLogger.get_module_logger("quant.poisson")
  2. Logger JSON sink'e bağlı — data/logs/quant_poisson.jsonl
  3. logger.bind(module="poisson", match_id="gs_fb").info(...)
  4. Her kayıt: {timestamp, level, module, message, extra, duration_ms, ...}
  5. Eski loglar sıkıştırılır: quant_poisson.jsonl.1.gz

Teknoloji: loguru (Python'un en gelişmiş log kütüphanesi)
"""
from __future__ import annotations

import json
import logging
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class LogEntry:
    """Yapılandırılmış log kaydı."""
    timestamp: str = ""
    level: str = "INFO"
    module: str = ""
    function: str = ""
    message: str = ""
    # Performans
    duration_ms: float = 0.0
    cpu_pct: float = 0.0
    memory_mb: float = 0.0
    # Bağlam
    match_id: str = ""
    team: str = ""
    cycle: int = 0
    # Girdi / çıktı
    input_summary: dict = field(default_factory=dict)
    output_summary: dict = field(default_factory=dict)
    # Hata
    error: str = ""
    traceback: str = ""
    # Ekstra
    extra: dict = field(default_factory=dict)


@dataclass
class ModuleLogStats:
    """Modül bazlı log istatistikleri."""
    module: str = ""
    total_entries: int = 0
    errors: int = 0
    warnings: int = 0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    last_entry_time: str = ""


# ═══════════════════════════════════════════════
#  JSON FORMATTER
# ═══════════════════════════════════════════════
def _json_sink_format(record: dict) -> str:
    """Loguru kaydını JSON satırına çevirir."""
    extra = record.get("extra", {})
    entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": record["level"].name,
        "module": extra.get("module", record.get("name", "")),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "message": record["message"],
    }

    # Performans bilgisi
    for key in ("duration_ms", "cpu_pct", "memory_mb",
                "match_id", "team", "cycle"):
        if key in extra:
            entry[key] = extra[key]

    # Girdi/çıktı özeti
    if "input_summary" in extra:
        entry["input"] = extra["input_summary"]
    if "output_summary" in extra:
        entry["output"] = extra["output_summary"]

    # Hata bilgisi
    if record.get("exception"):
        exc = record["exception"]
        if exc:
            entry["error"] = str(exc.value) if exc.value else ""
            entry["traceback"] = (
                "".join(exc.traceback.format())
                if hasattr(exc.traceback, "format") else str(exc.traceback)
            ) if exc.traceback else ""

    # Ekstra alanlar
    skip_keys = {
        "module", "duration_ms", "cpu_pct", "memory_mb",
        "match_id", "team", "cycle",
        "input_summary", "output_summary",
    }
    remaining = {
        k: v for k, v in extra.items()
        if k not in skip_keys and not k.startswith("_")
    }
    if remaining:
        entry["extra"] = remaining

    json_str = json.dumps(entry, ensure_ascii=False, default=str)
    # loguru format_map() çakışmasını önlemek için braces escape
    return json_str.replace("{", "{{").replace("}", "}}") + "\n"


# ═══════════════════════════════════════════════
#  SUPER LOGGER (Ana Sınıf)
# ═══════════════════════════════════════════════
class SuperLogger:
    """Yapılandırılmış derin loglama sistemi.

    Kullanım:
        sl = SuperLogger(log_dir="data/logs")

        # Modül logger'ı al
        lg = sl.get_module_logger("quant.poisson")

        # Yapılandırılmış log
        lg.bind(match_id="gs_fb", duration_ms=45.2).info("Tahmin üretildi")

        # Zamanlama context manager
        with sl.timed("quant.poisson", match_id="gs_fb"):
            result = poisson.predict(...)

        # Karar logu
        sl.log_decision(
            module="ensemble",
            match_id="gs_fb",
            decision="BET",
            confidence=0.82,
            reason="EV=+8.5%, Kelly=3.2%",
            inputs={"xG": 1.82, "form": 0.78},
            outputs={"prob_home": 0.55, "fair_odds": 1.82},
        )

        # İstatistikler
        stats = sl.get_module_stats("quant.poisson")
    """

    # Varsayılan yapılandırma
    DEFAULT_LOG_DIR = "data/logs"
    DEFAULT_ROTATION = "100 MB"
    DEFAULT_RETENTION = "30 days"
    DEFAULT_COMPRESSION = "gz"

    _instance: SuperLogger | None = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR,
                 rotation: str = DEFAULT_ROTATION,
                 retention: str = DEFAULT_RETENTION,
                 compression: str = DEFAULT_COMPRESSION,
                 console_level: str = "INFO",
                 json_level: str = "DEBUG"):
        if self._initialized:
            return
        self._initialized = True

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._rotation = rotation
        self._retention = retention
        self._compression = compression
        self._console_level = console_level
        self._json_level = json_level

        # Modül bazlı sink ID'leri
        self._module_sinks: dict[str, int] = {}

        # İstatistikler
        self._module_stats: dict[str, ModuleLogStats] = {}

        # Ana JSON sink (tüm modüller)
        self._master_sink_id = logger.add(
            str(self._log_dir / "master.jsonl"),
            format=_json_sink_format,
            level=json_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=False,
            enqueue=True,  # Thread-safe
            colorize=False,
        )

        # Hata-özel sink
        self._error_sink_id = logger.add(
            str(self._log_dir / "errors.jsonl"),
            format=_json_sink_format,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=False,
            enqueue=True,
            colorize=False,
        )

        # Kütüphane uyarılarını Loguru'ya yönlendir
        self._intercept_stdlib_logging()
        self._intercept_warnings()

        # Kütüphane etkinlik logu (ayrı sink)
        self._lib_sink_id = logger.add(
            str(self._log_dir / "library_events.jsonl"),
            format=_json_sink_format,
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=False,
            enqueue=True,
            colorize=False,
            filter=lambda record: record["extra"].get("module", "").startswith("lib."),
        )

        logger.debug(
            f"[SuperLogger] Başlatıldı: dir={log_dir}, "
            f"rotation={rotation}, retention={retention}, "
            f"stdlib_bridge=aktif, warnings_capture=aktif"
        )

    # ─────────────────────────────────────────────
    #  STDLIB / KÜTÜPHANE LOGGING KÖPRÜSÜ
    # ─────────────────────────────────────────────
    def _intercept_stdlib_logging(self):
        """Python stdlib logging → Loguru köprüsü.

        Tüm kütüphanelerin (PyTensor, arviz, torch, JAX vb.)
        standart logging çıktılarını yakalar ve Loguru'ya yönlendirir.
        """
        class _InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord):
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = logging.currentframe(), 0
                while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                    frame = frame.f_back
                    depth += 1
                # Kütüphane mesajlarındaki <TAG> ifadelerini [] ile değiştir
                # (Loguru colorizer bunları renk yönergesi olarak yorumlar ve çöker)
                msg = record.getMessage().replace("<", "[").replace(">", "]")
                logger.bind(module=f"lib.{record.name}").opt(
                    depth=depth, exception=record.exc_info,
                ).log(level, msg)

        logging.basicConfig(handlers=[_InterceptHandler()], level=logging.DEBUG, force=True)

        # Gürültülü kütüphaneleri WARNING seviyesine çek
        for noisy in (
            "urllib3", "httpx", "httpcore", "asyncio", "charset_normalizer",
            "filelock", "PIL", "matplotlib.font_manager",
            "prefect.flow_runs", "prefect.task_runs",
            # JAX: compile cache, dispatch, XLA tracing mesajları çok gürültülü
            "jax", "jax._src", "jax._src.dispatch", "jax._src.compiler",
            "jax._src.interpreters", "jax._src.cache_key",
            "jax._src.interpreters.pxla", "jax._src.compilation_cache",
            # Neo4j driver debug mesajları
            "neo4j",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def _intercept_warnings(self):
        """Python warnings modülünü yakalar.

        FutureWarning, DeprecationWarning gibi uyarıları log'a yazar.
        """
        _original_showwarning = warnings.showwarning

        def _loguru_showwarning(message, category, filename, lineno,
                                file=None, line=None):
            safe_msg = str(message).replace("<", "[").replace(">", "]")
            logger.bind(module="lib.warnings").warning(
                f"{category.__name__}: {safe_msg} "
                f"({filename}:{lineno})"
            )

        warnings.showwarning = _loguru_showwarning
        # DeprecationWarning'leri de göster (normalde gizlenir)
        warnings.filterwarnings("always", category=DeprecationWarning)
        warnings.filterwarnings("always", category=FutureWarning)

    def get_module_logger(self, module_name: str) -> logger.__class__:
        """Modül-özel logger döndürür.

        Her modül kendi .jsonl dosyasına yazar.
        """
        if module_name not in self._module_sinks:
            safe_name = module_name.replace(".", "_").replace("/", "_")
            log_path = str(self._log_dir / f"{safe_name}.jsonl")

            sink_id = logger.add(
                log_path,
                format=_json_sink_format,
                level=self._json_level,
                rotation=self._rotation,
                retention=self._retention,
                compression=self._compression,
                serialize=False,
                enqueue=True,
                colorize=False,
                filter=lambda record, mod=module_name: (
                    record["extra"].get("module", "") == mod
                ),
            )
            self._module_sinks[module_name] = sink_id
            self._module_stats[module_name] = ModuleLogStats(
                module=module_name,
            )

        return logger.bind(module=module_name)

    @contextmanager
    def timed(self, module: str, **bind_kwargs):
        """İşlem süresini ölçen context manager.

        Kullanım:
            with super_logger.timed("quant.poisson", match_id="gs_fb"):
                result = poisson.predict(...)
        """
        t0 = time.perf_counter()
        cpu_before = psutil.cpu_percent() if PSUTIL_OK else 0.0
        mem_before = (
            psutil.Process().memory_info().rss / (1024 * 1024)
            if PSUTIL_OK else 0.0
        )

        bound = logger.bind(module=module, **bind_kwargs)
        try:
            yield bound
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            bound.bind(
                duration_ms=round(elapsed, 2),
                cpu_pct=round(psutil.cpu_percent() - cpu_before, 1) if PSUTIL_OK else 0,
            ).error(f"Hata: {e.__class__.__name__}: {str(e)[:200]}")
            self._update_stats(module, elapsed, is_error=True)
            raise
        else:
            elapsed = (time.perf_counter() - t0) * 1000
            cpu_after = psutil.cpu_percent() if PSUTIL_OK else 0.0
            mem_after = (
                psutil.Process().memory_info().rss / (1024 * 1024)
                if PSUTIL_OK else 0.0
            )
            bound.bind(
                duration_ms=round(elapsed, 2),
                cpu_pct=round(cpu_after - cpu_before, 1) if PSUTIL_OK else 0,
                memory_mb=round(mem_after - mem_before, 2),
            ).debug(f"Tamamlandı ({elapsed:.1f}ms)")
            self._update_stats(module, elapsed, is_error=False)

    def log_decision(self, module: str, match_id: str,
                     decision: str, confidence: float,
                     reason: str = "",
                     inputs: dict | None = None,
                     outputs: dict | None = None,
                     **extra) -> None:
        """Karar audit trail logu.

        Her model kararı; gerekçesi, girdileri ve çıktıları ile
        birlikte kalıcı olarak kaydedilir.
        """
        lg = logger.bind(
            module=module,
            match_id=match_id,
            input_summary=inputs or {},
            output_summary=outputs or {},
            decision=decision,
            confidence=round(confidence, 4),
            reason=reason,
            **extra,
        )
        lg.info(
            f"KARAR: {decision} | Güven: {confidence:.1%} | "
            f"Gerekçe: {reason}"
        )

    def log_model_output(self, module: str, match_id: str,
                         model_name: str,
                         predictions: dict,
                         features_used: list[str] | None = None,
                         duration_ms: float = 0.0,
                         **extra) -> None:
        """Model çıktısı logu."""
        lg = logger.bind(
            module=module,
            match_id=match_id,
            model_name=model_name,
            output_summary=predictions,
            features_used=features_used or [],
            duration_ms=round(duration_ms, 2),
            **extra,
        )
        lg.debug(
            f"Model: {model_name} → "
            f"{json.dumps(predictions, default=str)[:120]}"
        )

    def log_pipeline_step(self, step_name: str, step_num: int,
                          cycle: int = 0,
                          status: str = "ok",
                          detail: str = "",
                          duration_ms: float = 0.0) -> None:
        """Pipeline adım logu."""
        lg = logger.bind(
            module="pipeline",
            step=step_name,
            step_num=step_num,
            cycle=cycle,
            duration_ms=round(duration_ms, 2),
        )
        if status == "ok":
            lg.debug(f"Step {step_num}: {step_name} ✓ ({duration_ms:.1f}ms)")
        elif status == "warn":
            lg.warning(f"Step {step_num}: {step_name} ⚠ {detail}")
        else:
            lg.error(f"Step {step_num}: {step_name} ✗ {detail}")

    # ─────────────────────────────────────────────
    #  İSTATİSTİKLER
    # ─────────────────────────────────────────────
    def _update_stats(self, module: str, elapsed_ms: float,
                      is_error: bool = False) -> None:
        stats = self._module_stats.get(module)
        if stats is None:
            stats = ModuleLogStats(module=module)
            self._module_stats[module] = stats

        stats.total_entries += 1
        stats.total_duration_ms += elapsed_ms
        stats.avg_duration_ms = round(
            stats.total_duration_ms / stats.total_entries, 2,
        )
        if elapsed_ms > stats.max_duration_ms:
            stats.max_duration_ms = round(elapsed_ms, 2)
        if is_error:
            stats.errors += 1

        from datetime import datetime
        stats.last_entry_time = datetime.utcnow().isoformat()

    def get_module_stats(self, module: str) -> ModuleLogStats:
        return self._module_stats.get(
            module, ModuleLogStats(module=module),
        )

    def get_all_stats(self) -> dict[str, ModuleLogStats]:
        return dict(self._module_stats)

    def get_slowest_modules(self, top_n: int = 10) -> list[ModuleLogStats]:
        """En yavaş modüller."""
        all_stats = list(self._module_stats.values())
        all_stats.sort(key=lambda s: s.avg_duration_ms, reverse=True)
        return all_stats[:top_n]

    def get_error_prone_modules(self, top_n: int = 10) -> list[ModuleLogStats]:
        """En çok hata veren modüller."""
        all_stats = list(self._module_stats.values())
        all_stats.sort(key=lambda s: s.errors, reverse=True)
        return [s for s in all_stats if s.errors > 0][:top_n]

    # ─────────────────────────────────────────────
    #  YARDIMCI
    # ─────────────────────────────────────────────
    def get_log_dir(self) -> Path:
        return self._log_dir

    def get_log_files(self) -> list[str]:
        """Tüm log dosyalarını listele."""
        return sorted(str(p) for p in self._log_dir.glob("*.jsonl*"))

    def get_module_log_path(self, module: str) -> str:
        safe_name = module.replace(".", "_").replace("/", "_")
        return str(self._log_dir / f"{safe_name}.jsonl")

    def check_disk_usage(self, max_log_size_mb: float = 500.0) -> dict:
        """Log dizin boyutunu kontrol eder. Limit aşılırsa uyarı verir."""
        total = sum(
            f.stat().st_size for f in self._log_dir.rglob("*") if f.is_file()
        )
        total_mb = total / (1024 * 1024)
        over_limit = total_mb > max_log_size_mb

        if over_limit:
            logger.warning(
                f"[SuperLogger] Log dizini {total_mb:.0f}MB – "
                f"limit {max_log_size_mb:.0f}MB aşıldı!"
            )

        return {
            "total_size_mb": round(total_mb, 2),
            "max_size_mb": max_log_size_mb,
            "over_limit": over_limit,
            "file_count": sum(1 for _ in self._log_dir.rglob("*") if _.is_file()),
        }

    def log_library_event(self, library: str, event: str,
                          level: str = "DEBUG", **extra) -> None:
        """Kütüphane olaylarını özel sink'e loglar.

        PyTensor compile, JAX trace, Neo4j query gibi
        kütüphane seviyesi olayları merkezi olarak kaydeder.
        """
        lg = logger.bind(module=f"lib.{library}", **extra)
        safe_event = event.replace("<", "[").replace(">", "]")
        getattr(lg, level.lower(), lg.debug)(safe_event)

    def summarize_session(self) -> dict:
        """Mevcut oturum özet istatistiklerini döndürür."""
        stats = self.get_all_stats()
        total_entries = sum(s.total_entries for s in stats.values())
        total_errors = sum(s.errors for s in stats.values())
        total_warnings = sum(s.warnings for s in stats.values())
        slowest = sorted(stats.values(), key=lambda s: s.max_duration_ms, reverse=True)

        return {
            "total_modules": len(stats),
            "total_entries": total_entries,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "error_rate": total_errors / max(total_entries, 1),
            "top_3_slowest": [
                {"module": s.module, "max_ms": s.max_duration_ms, "avg_ms": s.avg_duration_ms}
                for s in slowest[:3]
            ],
            "top_3_error_prone": [
                {"module": s.module, "errors": s.errors}
                for s in sorted(stats.values(), key=lambda s: s.errors, reverse=True)[:3]
                if s.errors > 0
            ],
        }
