"""
telemetry_tracer.py – OpenTelemetry (Tam Gözlemlenebilirlik).

Distributed Tracing: bahis.py'den çıkan bir "Analiz İsteği"nin,
vision_tracker'a gidişini, oradan neo4j'e uğrayışını ve
rl_trader'dan dönüşünü milisaniye milisaniye izler.

Kavramlar:
  - Trace: Uçtan uca bir isteğin yaşam döngüsü
  - Span: Bir modülün veya fonksiyonun çalışma süresi
  - Context Propagation: Span'ler arası bağlam aktarımı
  - Baggage: Span'e eklenecek metadata (match_id, cycle)
  - Exporter: Verilerin dışarıya aktarılması (Jaeger, konsol, dosya)
  - Metrics: Latency histogram, throughput counter, error rate

Root Cause Analysis:
  "Sistem neden yavaşladı?" → "Neo4j sorgusu 500ms sürdü"

Teknoloji: opentelemetry-api + opentelemetry-sdk
Fallback: Basit time-based profiler (decorator)
"""
from __future__ import annotations

import functools
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator

from loguru import logger

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    OTEL_OK = True
except ImportError:
    OTEL_OK = False
    logger.debug("opentelemetry yüklü değil – basit profiler fallback.")

ROOT = Path(__file__).resolve().parent.parent.parent
TRACE_LOG = ROOT / "data" / "traces.jsonl"
TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class SpanRecord:
    """Tek bir span kaydı (fallback profiler)."""
    name: str = ""
    module: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"           # "ok" | "error"
    error_type: str = ""
    attributes: dict = field(default_factory=dict)
    parent_span: str = ""
    trace_id: str = ""


@dataclass
class BottleneckReport:
    """Darboğaz analiz raporu."""
    total_spans: int = 0
    total_duration_ms: float = 0.0
    # Modül bazlı
    module_stats: dict = field(default_factory=dict)
    # En yavaş operasyonlar
    slowest_spans: list[SpanRecord] = field(default_factory=list)
    # Hata oranları
    error_rate: float = 0.0
    error_modules: list[str] = field(default_factory=list)
    # Darboğaz
    bottleneck_module: str = ""
    bottleneck_avg_ms: float = 0.0
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  FALLBACK PROFILER (OpenTelemetry yoksa)
# ═══════════════════════════════════════════════
class SimpleProfiler:
    """Basit zamanlama profiler'ı."""

    def __init__(self):
        self._spans: list[SpanRecord] = []
        self._active_stack: list[SpanRecord] = []
        self._counters: dict[str, int] = defaultdict(int)
        self._durations: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def span(self, name: str, module: str = "",
             attributes: dict | None = None) -> Generator[SpanRecord, None, None]:
        """Span context manager."""
        record = SpanRecord(
            name=name,
            module=module,
            start_time=time.perf_counter(),
            attributes=attributes or {},
            parent_span=self._active_stack[-1].name if self._active_stack else "",
            trace_id=f"t_{int(time.time())}_{len(self._spans)}",
        )
        self._active_stack.append(record)

        try:
            yield record
            record.status = "ok"
        except Exception as e:
            record.status = "error"
            record.error_type = type(e).__name__
            raise
        finally:
            record.end_time = time.perf_counter()
            record.duration_ms = round(
                (record.end_time - record.start_time) * 1000, 3,
            )
            self._active_stack.pop()
            self._spans.append(record)
            self._counters[module] += 1
            self._durations[module].append(record.duration_ms)

    def record_span(self, name: str, module: str, duration_ms: float,
                      status: str = "ok", attributes: dict | None = None) -> None:
        """Manuel span kaydı."""
        record = SpanRecord(
            name=name, module=module,
            duration_ms=duration_ms, status=status,
            attributes=attributes or {},
        )
        self._spans.append(record)
        self._counters[module] += 1
        self._durations[module].append(duration_ms)

    def get_report(self, top_n: int = 10) -> BottleneckReport:
        """Darboğaz raporu."""
        report = BottleneckReport(
            total_spans=len(self._spans),
            total_duration_ms=round(sum(
                s.duration_ms for s in self._spans
            ), 3),
        )

        # Modül istatistikleri
        for module, durations in self._durations.items():
            if not durations:
                continue
            report.module_stats[module] = {
                "count": len(durations),
                "total_ms": round(sum(durations), 3),
                "avg_ms": round(sum(durations) / len(durations), 3),
                "max_ms": round(max(durations), 3),
                "min_ms": round(min(durations), 3),
                "p99_ms": round(
                    sorted(durations)[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0], 3,
                ),
            }

        # En yavaş span'ler
        sorted_spans = sorted(self._spans, key=lambda s: -s.duration_ms)
        report.slowest_spans = sorted_spans[:top_n]

        # Hata oranları
        errors = [s for s in self._spans if s.status == "error"]
        report.error_rate = round(
            len(errors) / max(len(self._spans), 1), 4,
        )
        report.error_modules = list(set(s.module for s in errors))

        # Darboğaz tespiti
        if report.module_stats:
            bottleneck = max(
                report.module_stats.items(),
                key=lambda x: x[1]["avg_ms"],
            )
            report.bottleneck_module = bottleneck[0]
            report.bottleneck_avg_ms = bottleneck[1]["avg_ms"]

        report.recommendation = self._advice(report)
        return report

    def flush_to_file(self) -> int:
        """Span'leri dosyaya yaz."""
        count = 0
        with open(TRACE_LOG, "a", encoding="utf-8") as f:
            for span in self._spans:
                f.write(json.dumps({
                    "name": span.name,
                    "module": span.module,
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                    "error_type": span.error_type,
                    "attributes": span.attributes,
                    "parent": span.parent_span,
                    "trace_id": span.trace_id,
                    "ts": span.start_time,
                }) + "\n")
                count += 1
        self._spans.clear()
        return count

    def reset(self) -> None:
        """Span'leri temizle."""
        self._spans.clear()
        self._counters.clear()
        self._durations.clear()

    def _advice(self, report: BottleneckReport) -> str:
        if report.bottleneck_avg_ms > 1000:
            return (
                f"KRİTİK DARBOĞAZ: {report.bottleneck_module} "
                f"(ort. {report.bottleneck_avg_ms:.0f}ms). "
                f"Bu modülü optimize edin veya cache ekleyin."
            )
        if report.bottleneck_avg_ms > 200:
            return (
                f"Yavaş modül: {report.bottleneck_module} "
                f"(ort. {report.bottleneck_avg_ms:.0f}ms). İzlemeye devam."
            )
        if report.error_rate > 0.05:
            return (
                f"Yüksek hata oranı: {report.error_rate:.1%}. "
                f"Sorunlu modüller: {report.error_modules[:3]}."
            )
        return "Sistem sağlıklı. Darboğaz yok."


# ═══════════════════════════════════════════════
#  TELEMETRY TRACER (Ana Sınıf)
# ═══════════════════════════════════════════════
class TelemetryTracer:
    """Dağıtık izleme ve gözlemlenebilirlik motoru.

    Kullanım:
        tracer = TelemetryTracer(service_name="quant-bot")

        # Context manager ile span
        with tracer.span("neo4j_query", module="graph") as s:
            result = neo4j.query(...)
            s.attributes["rows"] = len(result)

        # Dekoratör ile otomatik izleme
        @tracer.trace("vision")
        async def analyze_frame(frame):
            ...

        # Rapor
        report = tracer.get_bottleneck_report()
    """

    def __init__(self, service_name: str = "quant-betting-bot",
                 export_console: bool = False):
        self._service_name = service_name
        self._profiler = SimpleProfiler()
        self._otel_tracer = None

        if OTEL_OK:
            try:
                resource = Resource.create({"service.name": service_name})
                provider = TracerProvider(resource=resource)

                if export_console:
                    provider.add_span_processor(
                        SimpleSpanProcessor(ConsoleSpanExporter())
                    )

                trace.set_tracer_provider(provider)
                self._otel_tracer = trace.get_tracer(service_name)
                logger.debug(f"[Telemetry] OpenTelemetry başlatıldı: {service_name}")
            except Exception as e:
                logger.debug(f"[Telemetry] OTel başlatma hatası: {e}")
        else:
            logger.debug(f"[Telemetry] SimpleProfiler modu: {service_name}")

    @contextmanager
    def span(self, name: str, module: str = "",
             attributes: dict | None = None) -> Generator:
        """Span oluştur (OTel veya fallback)."""
        attrs = attributes or {}

        if self._otel_tracer:
            with self._otel_tracer.start_as_current_span(name) as otel_span:
                for k, v in attrs.items():
                    otel_span.set_attribute(k, str(v))
                otel_span.set_attribute("module", module)

                t0 = time.perf_counter()
                try:
                    yield otel_span
                except Exception as e:
                    otel_span.set_attribute("error", True)
                    otel_span.set_attribute("error.type", type(e).__name__)
                    raise
                finally:
                    dur = (time.perf_counter() - t0) * 1000
                    self._profiler.record_span(
                        name, module, dur,
                        status="ok", attributes=attrs,
                    )
        else:
            with self._profiler.span(name, module, attrs) as record:
                yield record

    def trace(self, module: str = "") -> Callable:
        """Fonksiyon dekoratörü – otomatik span."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.span(func.__name__, module=module):
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.span(func.__name__, module=module):
                    return func(*args, **kwargs)

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator

    def get_bottleneck_report(self, top_n: int = 10) -> BottleneckReport:
        """Darboğaz raporu."""
        return self._profiler.get_report(top_n)

    def flush(self) -> int:
        """Span'leri dosyaya yaz."""
        return self._profiler.flush_to_file()

    def reset(self) -> None:
        """İstatistikleri sıfırla."""
        self._profiler.reset()
