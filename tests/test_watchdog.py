import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import time

from src.system.watchdog import (
    send_alert,
    restart_system,
    run_watchdog,
    TIMEOUT_SECONDS
)
import src.system.watchdog as watchdog

# Ensure TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are available for testing send_alert
@patch('src.system.watchdog.TELEGRAM_TOKEN', 'fake_token')
@patch('src.system.watchdog.TELEGRAM_CHAT_ID', 'fake_chat_id')
@patch('src.system.watchdog.httpx.AsyncClient.post')
@patch('src.system.watchdog.logger')
@pytest.mark.asyncio
async def test_send_alert_success(mock_logger, mock_post):
    await send_alert("Test message")
    mock_post.assert_called_once()
    mock_logger.error.assert_not_called()

@patch('src.system.watchdog.TELEGRAM_TOKEN', 'fake_token')
@patch('src.system.watchdog.TELEGRAM_CHAT_ID', 'fake_chat_id')
@patch('src.system.watchdog.httpx.AsyncClient.post', side_effect=Exception("Network error"))
@patch('src.system.watchdog.logger')
@pytest.mark.asyncio
async def test_send_alert_failure(mock_logger, mock_post):
    await send_alert("Test message")
    mock_post.assert_called_once()
    mock_logger.error.assert_called_with("Failed to send alert: Network error")

@patch('src.system.watchdog.TELEGRAM_TOKEN', None)
@patch('src.system.watchdog.logger')
@pytest.mark.asyncio
async def test_send_alert_no_token(mock_logger):
    await send_alert("Test message")
    mock_logger.warning.assert_called_with("Telegram not configured. Alert: Test message")

@pytest.mark.asyncio
@patch('src.system.watchdog.subprocess.Popen')
@patch('src.system.watchdog.subprocess.run')
@patch('src.system.watchdog.asyncio.sleep', new_callable=AsyncMock)
@patch('src.system.watchdog.send_alert')
async def test_restart_system_success(mock_send_alert, mock_sleep, mock_run, mock_popen):
    await restart_system()
    mock_run.assert_called_once_with(["pkill", "-f", watchdog.MAIN_SCRIPT], check=False)
    mock_sleep.assert_called_once_with(5)
    mock_popen.assert_called_once()
    mock_send_alert.assert_called_with("System restarted successfully after freeze detection.")

@pytest.mark.asyncio
@patch('src.system.watchdog.subprocess.Popen', side_effect=Exception("Popen failed"))
@patch('src.system.watchdog.subprocess.run')
@patch('src.system.watchdog.asyncio.sleep', new_callable=AsyncMock)
@patch('src.system.watchdog.send_alert')
async def test_restart_system_popen_failure(mock_send_alert, mock_sleep, mock_run, mock_popen):
    await restart_system()
    mock_run.assert_called_once()
    mock_sleep.assert_called_once()
    mock_popen.assert_called_once()
    mock_send_alert.assert_called_with("Failed to restart system: Popen failed")

@pytest.mark.asyncio
@patch('src.system.watchdog.subprocess.run', side_effect=Exception("Kill failed"))
@patch('src.system.watchdog.subprocess.Popen')
@patch('src.system.watchdog.asyncio.sleep', new_callable=AsyncMock)
@patch('src.system.watchdog.send_alert')
@patch('src.system.watchdog.logger')
async def test_restart_system_kill_failure(mock_logger, mock_send_alert, mock_sleep, mock_popen, mock_run):
    await restart_system()
    mock_run.assert_called_once()
    mock_logger.error.assert_called_with("Kill failed: Kill failed")
    mock_sleep.assert_not_called()  # Due to exception in subprocess.run
    mock_popen.assert_called_once()
    mock_send_alert.assert_called_with("System restarted successfully after freeze detection.")


@pytest.mark.asyncio
@patch('src.system.watchdog.HEARTBEAT_FILE')
@patch('src.system.watchdog.logger')
@patch('src.system.watchdog.asyncio.sleep', side_effect=Exception("Break loop"))
async def test_run_watchdog_no_heartbeat(mock_sleep, mock_logger, mock_heartbeat_file):
    mock_heartbeat_file.exists.return_value = False

    with pytest.raises(Exception, match="Break loop"):
        await run_watchdog()

    mock_logger.warning.assert_called_with("No heartbeat file found. Waiting...")

@pytest.mark.asyncio
@patch('src.system.watchdog.HEARTBEAT_FILE')
@patch('src.system.watchdog.time.time')
@patch('src.system.watchdog.logger')
@patch('src.system.watchdog.asyncio.sleep', side_effect=Exception("Break loop"))
async def test_run_watchdog_healthy(mock_sleep, mock_logger, mock_time, mock_heartbeat_file):
    mock_heartbeat_file.exists.return_value = True
    current_time = 1000.0
    mock_time.return_value = current_time
    mock_heartbeat_file.read_text.return_value = str(current_time - 10.0) # 10s ago, healthy

    with pytest.raises(Exception, match="Break loop"):
        await run_watchdog()

    mock_logger.debug.assert_called_with(f"System healthy. Last beat: 10.0s ago.")

@pytest.mark.asyncio
@patch('src.system.watchdog.HEARTBEAT_FILE')
@patch('src.system.watchdog.time.time')
@patch('src.system.watchdog.logger')
@patch('src.system.watchdog.send_alert')
@patch('src.system.watchdog.restart_system', new_callable=AsyncMock)
@patch('src.system.watchdog.asyncio.sleep', side_effect=Exception("Break loop"))
async def test_run_watchdog_freeze_detected(mock_sleep, mock_restart, mock_send_alert, mock_logger, mock_time, mock_heartbeat_file):
    mock_heartbeat_file.exists.return_value = True
    current_time = 1000.0
    mock_time.return_value = current_time
    # Set heartbeat exactlyTIMEOUT_SECONDS + 10s ago to trigger restart
    mock_heartbeat_file.read_text.return_value = str(current_time - TIMEOUT_SECONDS - 10.0)

    with pytest.raises(Exception, match="Break loop"):
        await run_watchdog()

    mock_logger.error.assert_called_with(f"Heartbeat lost! Last beat was {TIMEOUT_SECONDS + 10.0:.1f}s ago.")
    mock_send_alert.assert_called_with(f"System Freeze Detected! Last heartbeat: {TIMEOUT_SECONDS + 10.0:.0f}s ago.")
    mock_restart.assert_called_once()
    mock_heartbeat_file.write_text.assert_called_with(str(current_time))
