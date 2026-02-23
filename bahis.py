#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  QUANT BETTING BOT – ORCHESTRATOR (v2)                      ║
║  Modular, High-Frequency, Enterprise Grade.                  ║
╚══════════════════════════════════════════════════════════════╝
"""
import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.main import app

if __name__ == "__main__":
    app()
