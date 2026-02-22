#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════╗
║  QUANT BETTING BOT – ENTRY POINT (bahis.py)                        ║
║  Refactored v2.0                                                   ║
╚════════════════════════════════════════════════════════════════════╝
"""
import asyncio
import sys
from pathlib import Path
import typer
from loguru import logger

# Proje kök dizini
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# .env yükle
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Gürültücü kütüphaneleri sustur
import logging
for lib in ["jax", "neo4j", "numba", "matplotlib", "tensorflow", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Log yapılandırması
from src.utils.super_logger import configure_loguru
configure_loguru(ROOT / "logs")

app = typer.Typer(name="bahis", help="Quant Betting Bot v2", add_completion=False)

@app.command()
def run(
    mode: str = typer.Option("full", help="live, pre, full"),
    headless: bool = typer.Option(True, help="Tarayıcı arkaplanda çalışsın"),
    telegram: bool = typer.Option(True, help="Telegram botu aktif"),
    dashboard: bool = typer.Option(False, help="Terminal dashboard"),
):
    """Botu başlatır."""
    from src.core.bootstrap import SystemBootstrapper
    
    bootstrapper = SystemBootstrapper(
        mode=mode, 
        headless=headless, 
        telegram_enabled=telegram, 
        dashboard=dashboard
    )
    
    try:
        asyncio.run(bootstrapper.boot())
        asyncio.run(bootstrapper.run_forever())
    except KeyboardInterrupt:
        logger.warning("Kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.exception(f"Kritik Hata: {e}")
        raise

if __name__ == "__main__":
    app()
