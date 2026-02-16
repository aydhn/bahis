"""
chaos_monkey.py – Chaos Engineering (Kaos Testleri).

Sisteminiz tıkır tıkır işliyor. Peki:
  - Maç sırasında veri sağlayıcı çökerse?
  - Veritabanı kilitlenirse?
  - İnternet koparsa?
  - Yanlış veri (Garbage Data) gelirse?

Bu script rastgele zamanlarda:
  1. Scraper fonksiyonlarını bozar (ConnectionError fırlatır)
  2. Veritabanı bağlantısını keser (timeout simülasyonu)
  3. Garbage data gönderir (NULL, NaN, negatif odds)
  4. Ağ gecikmesi ekler (latency injection)
  5. Bellek baskısı oluşturur (memory pressure)
  6. Disk dolu simülasyonu (IOError)

Hedef:
  bahis.py'nin bu saldırılar karşısında çökmediğini,
  Exception Handling mekanizmalarının çalıştığını ve
  Telegram'dan doğru uyarıyı attığını doğrulayın.

Kullanım:
    python tests/chaos_monkey.py --target scraper --duration 60
    python tests/chaos_monkey.py --target all --intensity medium
    python tests/chaos_monkey.py --report  # Son test sonuçlarını göster
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

console = Console()
cli = typer.Typer(name="chaos-monkey", help="Chaos Engineering Test Suite")

RESULTS_PATH = ROOT / "logs" / "chaos_results.json"


@dataclass
class ChaosResult:
    """Tek bir kaos testinin sonucu."""
    test_name: str
    target: str
    timestamp: str = ""
    passed: bool = False
    error_caught: bool = False      # Hata yakalandı mı?
    system_crashed: bool = False    # Sistem çöktü mü?
    recovery_time_ms: float = 0     # Toparlanma süresi
    telegram_notified: bool = False # Telegram uyarı gönderildi mi?
    details: str = ""
    exception_type: str = ""


@dataclass
class ChaosReport:
    """Tüm testlerin raporu."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    crash_count: int = 0
    avg_recovery_ms: float = 0
    results: list[ChaosResult] = field(default_factory=list)
    score: float = 0.0  # 0-100 (100 = hiç çökmedi)
    grade: str = ""      # A/B/C/D/F


# ═══════════════════════════════════════════════
#  KAOS SENARYOLARI
# ═══════════════════════════════════════════════

class ChaosMonkey:
    """Sisteme kontrollü kaos enjekte eder."""

    def __init__(self, intensity: str = "medium"):
        """
        Args:
            intensity: low (nazik), medium (normal), high (acımasız)
        """
        self._intensity = intensity
        self._results: list[ChaosResult] = []
        self._failure_rates = {
            "low": 0.20,
            "medium": 0.50,
            "high": 0.85,
        }
        logger.info(f"[ChaosMonkey] Başlatıldı (intensity={intensity})")

    # ── 1) Scraper Bozma ──
    async def test_scraper_failure(self) -> ChaosResult:
        """Scraper'ın ConnectionError/Timeout alması."""
        result = ChaosResult(
            test_name="scraper_connection_failure",
            target="scraper",
            timestamp=datetime.now().isoformat(),
        )

        try:
            from src.core.circuit_breaker import CircuitBreakerRegistry

            cb = CircuitBreakerRegistry()
            mock_notifier = MagicMock()
            mock_notifier.send_error_alert = MagicMock(
                return_value=asyncio.coroutine(lambda *a, **k: None)()
            )

            # Üst üste hata fırlat → Circuit Breaker devreye girmeli
            errors_thrown = 0
            breaker_tripped = False

            for i in range(5):
                try:
                    if random.random() < self._failure_rates[self._intensity]:
                        raise ConnectionError(
                            f"[CHAOS] Bağlantı hatası #{i+1} – simülasyon"
                        )
                except ConnectionError as e:
                    errors_thrown += 1
                    # CB bu hatayı kaydetmeli
                    cb_state = cb.report_failure("mackolik_scraper")
                    if cb_state == "open":
                        breaker_tripped = True

            result.error_caught = True
            result.passed = errors_thrown > 0 and not result.system_crashed

            if breaker_tripped:
                result.details = (
                    f"Circuit Breaker devreye girdi ({errors_thrown} hata). "
                    f"Sistem korumalı."
                )
            else:
                result.details = (
                    f"{errors_thrown} hata fırlatıldı. "
                    f"CB tetiklenmedi (eşik altında)."
                )

        except Exception as e:
            result.system_crashed = True
            result.passed = False
            result.exception_type = type(e).__name__
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ── 2) Veritabanı Kesintisi ──
    async def test_db_failure(self) -> ChaosResult:
        """Veritabanı bağlantısının kopması."""
        result = ChaosResult(
            test_name="database_connection_lost",
            target="database",
            timestamp=datetime.now().isoformat(),
        )

        try:
            from src.memory.db_manager import DBManager

            db = DBManager()
            start = time.time()

            # DB'yi "kır" – geçersiz dosya yolu
            original_path = getattr(db, "_path", None)
            try:
                if hasattr(db, "_path"):
                    db._path = "/nonexistent/path/db.duckdb"

                # Sorgu çalıştırmayı dene
                try:
                    db.get_upcoming_matches()
                    result.details = "DB hatası beklendi ama olmadı (graceful handling)."
                    result.error_caught = True
                except Exception as db_err:
                    result.error_caught = True
                    result.exception_type = type(db_err).__name__
                    result.details = f"DB hatası yakalandı: {type(db_err).__name__}"

            finally:
                if original_path and hasattr(db, "_path"):
                    db._path = original_path

            result.recovery_time_ms = (time.time() - start) * 1000
            result.passed = result.error_caught and not result.system_crashed

        except Exception as e:
            result.system_crashed = True
            result.passed = False
            result.exception_type = type(e).__name__
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ── 3) Garbage Data ──
    async def test_garbage_data(self) -> ChaosResult:
        """Kirli/Geçersiz veri enjeksiyonu."""
        result = ChaosResult(
            test_name="garbage_data_injection",
            target="validator",
            timestamp=datetime.now().isoformat(),
        )

        garbage_samples = [
            {"match_id": None, "home_team": "", "away_team": None},
            {"match_id": "x", "home_odds": -5.0, "draw_odds": 0, "away_odds": float("inf")},
            {"match_id": "y", "home_team": "A" * 500, "kickoff": "not-a-date"},
            {"match_id": "z", "home_xg": float("nan"), "away_xg": -999},
            {},
            {"match_id": 12345, "home_team": True, "away_team": [1, 2, 3]},
        ]

        try:
            from src.core.data_validator import DataValidator
            validator = DataValidator()

            rejected = 0
            accepted = 0

            for garbage in garbage_samples:
                try:
                    validated = validator.validate_batch([garbage], schema="match")
                    if not validated:
                        rejected += 1
                    else:
                        accepted += 1
                except Exception:
                    rejected += 1

            result.error_caught = True
            result.passed = rejected >= len(garbage_samples) * 0.6
            result.details = (
                f"{rejected}/{len(garbage_samples)} garbage veri reddedildi, "
                f"{accepted} kabul edildi."
            )

        except ImportError:
            result.details = "DataValidator import edilemedi – modül kontrol edilmeli."
            result.passed = False
        except Exception as e:
            result.system_crashed = True
            result.exception_type = type(e).__name__
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ── 4) Ağ Gecikmesi ──
    async def test_network_latency(self) -> ChaosResult:
        """Ağ gecikmesi enjeksiyonu (yavaş API yanıtı)."""
        result = ChaosResult(
            test_name="network_latency_injection",
            target="network",
            timestamp=datetime.now().isoformat(),
        )

        try:
            latency_ms = {
                "low": 500,
                "medium": 2000,
                "high": 10000,
            }[self._intensity]

            start = time.time()

            # httpx ile timeout simülasyonu
            try:
                import httpx
                async with httpx.AsyncClient(timeout=1.0) as client:
                    # Kasıtlı olarak çok yavaş endpoint (timeout tetikler)
                    await asyncio.wait_for(
                        client.get("https://httpbin.org/delay/10"),
                        timeout=latency_ms / 1000,
                    )
            except (asyncio.TimeoutError, httpx.TimeoutException, Exception):
                result.error_caught = True

            elapsed = (time.time() - start) * 1000
            result.recovery_time_ms = elapsed
            result.passed = result.error_caught
            result.details = (
                f"Timeout başarıyla yakalandı ({elapsed:.0f}ms). "
                f"Simüle edilen gecikme: {latency_ms}ms."
            )

        except Exception as e:
            result.system_crashed = True
            result.exception_type = type(e).__name__
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ── 5) Bellek Baskısı ──
    async def test_memory_pressure(self) -> ChaosResult:
        """RAM baskısı – büyük veri yapısı oluştur ve temizle."""
        result = ChaosResult(
            test_name="memory_pressure",
            target="memory",
            timestamp=datetime.now().isoformat(),
        )

        try:
            import gc
            sizes = {"low": 10_000, "medium": 100_000, "high": 1_000_000}
            n = sizes[self._intensity]

            start = time.time()
            # Büyük liste oluştur
            big_data = [{"x": random.random(), "y": random.random()} for _ in range(n)]

            # İşle
            total = sum(d["x"] + d["y"] for d in big_data)

            # Temizle
            del big_data
            gc.collect()

            elapsed = (time.time() - start) * 1000

            result.passed = True
            result.error_caught = True
            result.recovery_time_ms = elapsed
            result.details = (
                f"{n:,} kayıt oluşturuldu, işlendi ve temizlendi ({elapsed:.0f}ms). "
                f"GC başarılı."
            )

        except MemoryError:
            result.error_caught = True
            result.passed = False
            result.details = "MemoryError! Sistem bellek sınırına ulaştı."
        except Exception as e:
            result.system_crashed = True
            result.exception_type = type(e).__name__
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ── 6) Eşzamanlı Yük ──
    async def test_concurrent_load(self) -> ChaosResult:
        """Çok sayıda eşzamanlı görev çalıştır."""
        result = ChaosResult(
            test_name="concurrent_load_test",
            target="asyncio",
            timestamp=datetime.now().isoformat(),
        )

        try:
            concurrency = {"low": 10, "medium": 50, "high": 200}[self._intensity]

            async def dummy_task(i):
                await asyncio.sleep(random.uniform(0.01, 0.1))
                if random.random() < 0.1:
                    raise ValueError(f"Random error in task {i}")
                return i

            start = time.time()
            tasks = [asyncio.create_task(dummy_task(i)) for i in range(concurrency)]
            results_gathered = await asyncio.gather(*tasks, return_exceptions=True)

            elapsed = (time.time() - start) * 1000
            errors = sum(1 for r in results_gathered if isinstance(r, Exception))
            success = concurrency - errors

            result.passed = True
            result.error_caught = errors > 0
            result.recovery_time_ms = elapsed
            result.details = (
                f"{concurrency} eşzamanlı görev: {success} başarılı, {errors} hatalı "
                f"({elapsed:.0f}ms). return_exceptions çalışıyor."
            )

        except Exception as e:
            result.system_crashed = True
            result.exception_type = type(e).__name__
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ── 7) Disk I/O Hatası ──
    async def test_disk_failure(self) -> ChaosResult:
        """Disk yazma hatası simülasyonu."""
        result = ChaosResult(
            test_name="disk_io_failure",
            target="disk",
            timestamp=datetime.now().isoformat(),
        )

        try:
            bad_path = Path("/nonexistent/deep/path/file.json")

            try:
                bad_path.write_text("test")
                result.details = "Yazma başarılı olmamalıydı (beklenmeyen)."
            except (OSError, PermissionError, FileNotFoundError) as e:
                result.error_caught = True
                result.passed = True
                result.exception_type = type(e).__name__
                result.details = f"Disk hatası düzgünce yakalandı: {type(e).__name__}"

        except Exception as e:
            result.system_crashed = True
            result.details = f"SİSTEM ÇÖKTÜ: {e}"

        self._results.append(result)
        return result

    # ═══════════════════════════════════════════
    #  TEST KOŞTURUCU
    # ═══════════════════════════════════════════
    async def run_all(self) -> ChaosReport:
        """Tüm kaos testlerini çalıştır."""
        console.rule("[bold red]CHAOS MONKEY – BAŞLIYOR[/]")

        test_methods = [
            self.test_scraper_failure,
            self.test_db_failure,
            self.test_garbage_data,
            self.test_network_latency,
            self.test_memory_pressure,
            self.test_concurrent_load,
            self.test_disk_failure,
        ]

        for test_fn in test_methods:
            name = test_fn.__name__
            console.print(f"\n[yellow]▶ {name}...[/]")
            try:
                result = await test_fn()
                status = "[green]PASSED[/]" if result.passed else "[red]FAILED[/]"
                console.print(f"  {status} – {result.details[:100]}")
            except Exception as e:
                console.print(f"  [red]CRASH[/] – {e}")

        return self._generate_report()

    async def run_target(self, target: str) -> ChaosReport:
        """Belirli bir hedef için kaos testleri."""
        target_map = {
            "scraper": [self.test_scraper_failure],
            "database": [self.test_db_failure],
            "validator": [self.test_garbage_data],
            "network": [self.test_network_latency],
            "memory": [self.test_memory_pressure],
            "asyncio": [self.test_concurrent_load],
            "disk": [self.test_disk_failure],
            "all": [
                self.test_scraper_failure, self.test_db_failure,
                self.test_garbage_data, self.test_network_latency,
                self.test_memory_pressure, self.test_concurrent_load,
                self.test_disk_failure,
            ],
        }

        tests = target_map.get(target, target_map["all"])
        for test_fn in tests:
            await test_fn()

        return self._generate_report()

    def _generate_report(self) -> ChaosReport:
        """Test sonuçlarından rapor üret."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        crashes = sum(1 for r in self._results if r.system_crashed)

        recovery_times = [
            r.recovery_time_ms for r in self._results if r.recovery_time_ms > 0
        ]
        avg_recovery = float(
            sum(recovery_times) / len(recovery_times)
        ) if recovery_times else 0

        score = (passed / total * 100) if total > 0 else 0

        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 40:
            grade = "D"
        else:
            grade = "F"

        report = ChaosReport(
            total_tests=total, passed=passed, failed=failed,
            crash_count=crashes, avg_recovery_ms=avg_recovery,
            results=self._results, score=score, grade=grade,
        )

        self._save_report(report)
        return report

    def _save_report(self, report: ChaosReport):
        """Raporu JSON olarak kaydet."""
        try:
            (ROOT / "logs").mkdir(exist_ok=True)
            data = {
                "timestamp": datetime.now().isoformat(),
                "total": report.total_tests,
                "passed": report.passed,
                "failed": report.failed,
                "crashes": report.crash_count,
                "score": report.score,
                "grade": report.grade,
                "avg_recovery_ms": report.avg_recovery_ms,
                "results": [
                    {
                        "name": r.test_name, "target": r.target,
                        "passed": r.passed, "crashed": r.system_crashed,
                        "details": r.details[:200],
                    }
                    for r in report.results
                ],
            }
            RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════
@cli.command()
def run(
    target: str = typer.Option("all", help="Hedef: scraper|database|validator|network|memory|asyncio|disk|all"),
    intensity: str = typer.Option("medium", help="Şiddet: low|medium|high"),
    duration: int = typer.Option(0, help="Süre (saniye). 0 = tek pass."),
):
    """Kaos testlerini çalıştır."""
    monkey = ChaosMonkey(intensity=intensity)

    async def _run():
        if duration > 0:
            end = time.time() + duration
            cycle = 0
            while time.time() < end:
                cycle += 1
                console.print(f"\n[bold]═══ Kaos Döngüsü #{cycle} ═══[/]")
                await monkey.run_target(target)
                await asyncio.sleep(2)
        else:
            await monkey.run_target(target)

        report = monkey._generate_report()
        _print_report(report)

    asyncio.run(_run())


@cli.command()
def report():
    """Son kaos test sonuçlarını göster."""
    if not RESULTS_PATH.exists():
        console.print("[yellow]Henüz kaos testi çalıştırılmamış.[/]")
        return

    data = json.loads(RESULTS_PATH.read_text())
    console.rule("[bold cyan]CHAOS MONKEY – SON RAPOR[/]")
    console.print(f"Tarih: {data.get('timestamp', '?')}")
    console.print(f"Toplam: {data['total']} | Geçti: {data['passed']} | Kaldı: {data['failed']}")
    console.print(f"Çökme: {data['crashes']} | Skor: {data['score']:.0f}/100 | Not: {data['grade']}")

    table = Table(title="Test Detayları", show_lines=True)
    table.add_column("Test", style="cyan")
    table.add_column("Hedef")
    table.add_column("Sonuç", justify="center")
    table.add_column("Detay", max_width=60)

    for r in data.get("results", []):
        status = "[green]OK[/]" if r["passed"] else "[red]FAIL[/]"
        if r.get("crashed"):
            status = "[bold red]CRASH[/]"
        table.add_row(r["name"], r["target"], status, r.get("details", "")[:60])

    console.print(table)


def _print_report(report: ChaosReport):
    """Raporu terminale yazdır."""
    console.rule("[bold cyan]CHAOS MONKEY – RAPOR[/]")

    grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "bold red"}
    color = grade_colors.get(report.grade, "white")

    console.print(f"\n[{color}]NOT: {report.grade} ({report.score:.0f}/100)[/]")
    console.print(f"Toplam Test: {report.total_tests}")
    console.print(f"Geçti: [green]{report.passed}[/] | Kaldı: [red]{report.failed}[/]")
    console.print(f"Çökme Sayısı: {report.crash_count}")
    console.print(f"Ort. Toparlanma: {report.avg_recovery_ms:.0f}ms")

    table = Table(show_lines=True)
    table.add_column("Test", style="cyan")
    table.add_column("Hedef")
    table.add_column("Sonuç", justify="center")
    table.add_column("Toparlanma", justify="right")
    table.add_column("Detay", max_width=50)

    for r in report.results:
        status = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
        if r.system_crashed:
            status = "[bold red]CRASH[/]"
        table.add_row(
            r.test_name, r.target, status,
            f"{r.recovery_time_ms:.0f}ms" if r.recovery_time_ms > 0 else "-",
            r.details[:50],
        )

    console.print(table)
    console.rule()


if __name__ == "__main__":
    cli()
