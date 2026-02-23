#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  QUANT BETTING BOT – SENTINEL (v2.0)                        ║
║  Modular, High-Frequency, Enterprise Grade.                  ║
╚══════════════════════════════════════════════════════════════╝
"""
import sys
import asyncio
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.system.sentinel import Sentinel

if __name__ == "__main__":
    sentinel = Sentinel(daemon_mode=True)
    try:
        asyncio.run(sentinel.run())
    except KeyboardInterrupt:
        sentinel.shutdown()
