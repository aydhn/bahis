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
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

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
class DecisionContext:
    """Model karar bağlamı."""
    module: str
    match_id: str
    decision: str
    confidence: float
    reason: str = ""
    inputs: dict | None = None
    outputs: dict | None = None


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

    return json.dumps(entry, ensure_ascii=False, default=str) + "\n"


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
        ctx = DecisionContext(
            module="ensemble",
            match_id="gs_fb",
            decision="BET",
            confidence=0.82,
            reason="EV=+8.5%, Kelly=3.2%",
            inputs={"xG": 1.82, "form": 0.78},
            outputs={"prob_home": 0.55, "fair_odds": 1.82},
        )
        sl.log_decision(ctx)

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
        )

        logger.debug(
            f"[SuperLogger] Başlatıldı: dir={log_dir}, "
            f"rotation={rotation}, retention={retention}"
        )

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

    def log_decision(self, context: DecisionContext, **extra) -> None:
        """Karar audit trail logu.

        Her model kararı; gerekçesi, girdileri ve çıktıları ile
        birlikte kalıcı olarak kaydedilir.
        """
        lg = logger.bind(
            module=context.module,
            match_id=context.match_id,
            input_summary=context.inputs or {},
            output_summary=context.outputs or {},
            decision=context.decision,
            confidence=round(context.confidence, 4),
            reason=context.reason,
            **extra,
        )
        lg.info(
            f"KARAR: {context.decision} | Güven: {context.confidence:.1%} | "
            f"Gerekçe: {context.reason}"
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

        from datetime import datetime, timezone
        stats.last_entry_time = datetime.now(timezone.utc).isoformat()

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
