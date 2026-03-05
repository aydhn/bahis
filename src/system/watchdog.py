"""
watchdog.py - System Health Monitor & Auto-Recovery.

This service runs independently of the main application.
It monitors the heartbeat file and restarts the system if it freezes.
"""
import time
import os
import sys
import subprocess
import requests
from pathlib import Path
from loguru import logger

# Configuration
HEARTBEAT_FILE = Path("data/heartbeat.txt")
TIMEOUT_SECONDS = 300  # 5 minutes
CHECK_INTERVAL = 60
MAIN_SCRIPT = "bahis.py"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_alert(message: str):
    """Send critical alert via Telegram (Direct API)."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(f"Telegram not configured. Alert: {message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🚨 *WATCHDOG ALERT*\n{message}",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

async def await restart_system():
    """Kill and restart the main process."""
    logger.warning("Attempting system restart...")

    # 1. Find and Kill
    # This is a bit aggressive. In production, we should store the PID.
    # For now, we rely on pkill.
    try:
        subprocess.run(["pkill", "-f", MAIN_SCRIPT], check=False)
        await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"Kill failed: {e}")

    # 2. Restart
    try:
        # Assuming we are in the root directory
        subprocess.Popen([sys.executable, MAIN_SCRIPT])
        send_alert("System restarted successfully after freeze detection.")
    except Exception as e:
        send_alert(f"Failed to restart system: {e}")

async def run_watchdog():
    logger.info("Watchdog started. Monitoring heartbeat...")

    while True:
        try:
            if not HEARTBEAT_FILE.exists():
                logger.warning("No heartbeat file found. Waiting...")
            else:
                content = HEARTBEAT_FILE.read_text().strip()
                if content:
                    last_beat = float(content)
                    delta = time.time() - last_beat

                    if delta > TIMEOUT_SECONDS:
                        logger.error(f"Heartbeat lost! Last beat was {delta:.1f}s ago.")
                        send_alert(f"System Freeze Detected! Last heartbeat: {delta:.0f}s ago.")
                        await restart_system()
                        # Reset heartbeat to avoid loop while starting
                        HEARTBEAT_FILE.write_text(str(time.time()))
                    else:
                        logger.debug(f"System healthy. Last beat: {delta:.1f}s ago.")

        except Exception as e:
            logger.error(f"Watchdog error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_watchdog())
