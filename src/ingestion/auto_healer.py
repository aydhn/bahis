"""
auto_healer.py – Kendi kendini onaran (Self-healing) sistem modülü.
Tüm bileşenlerin sağlık kontrolünü yapar, hataları tespit eder ve düzeltir.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]


class AutoHealer:
    """Sistem bileşenlerini kontrol eden ve onaran modül."""

    REQUIRED_MODULES = {
        "numpy": "numpy",
        "polars": "polars",
        "duckdb": "duckdb",
        "diskcache": "diskcache",
        "httpx": "httpx",
        "loguru": "loguru",
        "rich": "rich",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
    }

    OPTIONAL_MODULES = {
        "torch": "torch",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "pymc": "pymc",
        "lancedb": "lancedb",
        "playwright": "playwright",
        "cv2": "opencv-python-headless",
        "whisper": "openai-whisper",
        "prometheus_client": "prometheus-client",
        "networkx": "networkx",
        "jax": "jax",
        "vectorbt": "vectorbt",
    }

    def __init__(self):
        self._results: list[dict] = []
        logger.debug("AutoHealer başlatıldı.")

    def diagnose(self) -> list[dict]:
        """Tüm bileşenleri kontrol eder."""
        console = Console()
        self._results = []

        # 1) Python sürümü
        py_ver = sys.version
        self._results.append({
            "component": "Python",
            "status": "OK" if sys.version_info >= (3, 10) else "WARN",
            "detail": py_ver,
        })

        # 2) Zorunlu modüller
        table = Table(title="Sistem Sağlık Kontrolü", show_lines=True)
        table.add_column("Bileşen", style="cyan")
        table.add_column("Durum", style="bold")
        table.add_column("Detay")

        for mod_name, pip_name in self.REQUIRED_MODULES.items():
            status, detail = self._check_module(mod_name)
            if status == "MISSING":
                self._try_install(pip_name)
                status, detail = self._check_module(mod_name)
                if status == "OK":
                    detail += " (otomatik yüklendi)"
            color = "green" if status == "OK" else "red"
            table.add_row(mod_name, f"[{color}]{status}[/]", detail)
            self._results.append({"component": mod_name, "status": status, "detail": detail})

        # 3) Opsiyonel modüller
        for mod_name, pip_name in self.OPTIONAL_MODULES.items():
            status, detail = self._check_module(mod_name)
            color = "green" if status == "OK" else "yellow"
            table.add_row(f"{mod_name} (opt)", f"[{color}]{status}[/]", detail)
            self._results.append({"component": mod_name, "status": status, "detail": detail})

        # 4) Dizin yapısı
        expected_dirs = [
            "src/ingestion", "src/memory", "src/quant",
            "src/core", "src/utils", "src/ui",
        ]
        for d in expected_dirs:
            full = ROOT / d
            exists = full.exists()
            status = "OK" if exists else "MISSING"
            color = "green" if exists else "red"
            table.add_row(d, f"[{color}]{status}[/]", str(full))
            self._results.append({"component": d, "status": status, "detail": str(full)})

        # 5) Data dizini
        data_dir = ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        table.add_row("data/", "[green]OK[/]", str(data_dir))

        console.print(table)
        return self._results

    def _check_module(self, module_name: str) -> tuple[str, str]:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "?")
            return "OK", f"v{version}"
        except ImportError:
            return "MISSING", "Yüklü değil"
        except Exception as e:
            return "ERROR", str(e)[:60]

    def _try_install(self, pip_name: str) -> bool:
        logger.info(f"Otomatik yükleniyor: {pip_name}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                timeout=120,
            )
            return True
        except Exception as e:
            logger.warning(f"Otomatik yükleme başarısız ({pip_name}): {e}")
            return False

    def heal(self) -> int:
        """Eksik zorunlu bileşenleri yükler. Düzeltilen sayısını döndürür."""
        fixed = 0
        for mod_name, pip_name in self.REQUIRED_MODULES.items():
            status, _ = self._check_module(mod_name)
            if status != "OK":
                if self._try_install(pip_name):
                    fixed += 1
        logger.info(f"AutoHealer: {fixed} bileşen düzeltildi.")
        return fixed
