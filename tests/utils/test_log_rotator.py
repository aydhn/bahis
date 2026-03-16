import pytest
import os
import gzip
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.log_rotator import LogRotator

@pytest.fixture
def rotator(tmp_path):
    """Provides a fresh LogRotator instance in a temporary directory."""
    log_dir = tmp_path / "logs"
    return LogRotator(log_dir=str(log_dir), archive_days=3, compress=False, max_archive_days=30)

@pytest.fixture
def compressed_rotator(tmp_path):
    """Provides a fresh LogRotator instance with compression enabled."""
    log_dir = tmp_path / "compressed_logs"
    return LogRotator(log_dir=str(log_dir), archive_days=3, compress=True, max_archive_days=30)

def set_mtime(path: Path, delta_days: int):
    """Set the modification time of a file to N days ago."""
    dt = datetime.now() - timedelta(days=delta_days)
    mtime = dt.timestamp()
    os.utime(str(path), (mtime, mtime))

def test_initialization(tmp_path):
    log_dir = tmp_path / "init_logs"
    rotator = LogRotator(log_dir=str(log_dir), archive_days=3, compress=True, max_archive_days=30)

    assert rotator._log_dir.exists()
    assert rotator._archive_dir.exists()
    assert rotator._archive_days == 3
    assert rotator._compress is True
    assert rotator._max_archive == 30

def test_rotate_uncompressed(rotator):
    # Create mock active logs
    active_log = rotator._log_dir / "active.log"
    active_log.write_text("active")

    # Create mock old log
    old_log = rotator._log_dir / "old.log"
    old_log.write_text("old")
    set_mtime(old_log, 5) # 5 days old, threshold is 3

    # Run rotation
    report = rotator.rotate()

    # Assert old log is moved
    assert not old_log.exists()
    assert (rotator._archive_dir / "old.log").exists()
    assert (rotator._archive_dir / "old.log").read_text() == "old"

    # Assert active log is kept
    assert active_log.exists()

    # Check report
    assert "old.log" in report["archived"]
    assert "active.log" in report["kept"]
    assert len(report["archived"]) == 1
    assert len(report["kept"]) == 1

def test_rotate_compressed(compressed_rotator):
    old_log = compressed_rotator._log_dir / "old.log"
    old_log.write_text("old log content")
    set_mtime(old_log, 5)

    report = compressed_rotator.rotate()

    assert not old_log.exists()
    gz_path = compressed_rotator._archive_dir / "old.log.gz"
    assert gz_path.exists()

    # Check gz content
    with gzip.open(gz_path, "rt") as f:
        assert f.read() == "old log content"

    assert "old.log.gz" in report["archived"]

def test_rotate_old_archives_deletion(rotator):
    # Create a mock old archive file
    very_old_archive = rotator._archive_dir / "very_old.log"
    very_old_archive.write_text("very old")
    set_mtime(very_old_archive, 35) # > max_archive_days (30)

    recent_archive = rotator._archive_dir / "recent.log"
    recent_archive.write_text("recent")
    set_mtime(recent_archive, 15) # < max_archive_days

    report = rotator.rotate()

    assert not very_old_archive.exists()
    assert recent_archive.exists()

    assert "very_old.log" in report["deleted"]
    assert "recent.log" not in report["deleted"]

def test_rotate_date_pattern(rotator):
    # bot_YYYY-MM-DD.log logic uses the date from the filename, not mtime

    # Old date
    old_date = datetime.now() - timedelta(days=5)
    old_log_name = f"bot_{old_date.strftime('%Y-%m-%d')}.log"
    old_log = rotator._log_dir / old_log_name
    old_log.write_text("old bot log")
    # Even if mtime is recent, filename should dictate archiving

    # Recent date
    recent_date = datetime.now() - timedelta(days=1)
    recent_log_name = f"bot_{recent_date.strftime('%Y-%m-%d')}.log"
    recent_log = rotator._log_dir / recent_log_name
    recent_log.write_text("recent bot log")

    report = rotator.rotate()

    assert not old_log.exists()
    assert (rotator._archive_dir / old_log_name).exists()
    assert old_log_name in report["archived"]

    assert recent_log.exists()
    assert not (rotator._archive_dir / recent_log_name).exists()
    # It might not be in 'kept' list specifically for the pattern matching part
    # depending on implementation, but the file should remain.

def test_get_active_logs(rotator):
    # Create files
    log1 = rotator._log_dir / "1.log"
    log1.write_text("1")
    set_mtime(log1, 1) # Active

    log2 = rotator._log_dir / "2.log"
    log2.write_text("2")
    set_mtime(log2, 5) # Old

    active_logs = rotator.get_active_logs()

    assert len(active_logs) == 1
    assert log1 in active_logs
    assert log2 not in active_logs

def test_get_stats(rotator):
    log1 = rotator._log_dir / "1.log"
    log1.write_text("1" * 1024) # 1KB

    log2 = rotator._log_dir / "2.log"
    log2.write_text("2" * 2048) # 2KB

    archive1 = rotator._archive_dir / "arch.log"
    archive1.write_text("a" * 4096) # 4KB

    stats = rotator.get_stats()

    assert stats["active_count"] == 2
    assert stats["archive_count"] == 1

    active_mb = round((1024 + 2048) / (1024 * 1024), 2)
    assert stats["active_size_mb"] == active_mb

    archive_mb = round(4096 / (1024 * 1024), 2)
    assert stats["archive_size_mb"] == archive_mb

    assert stats["total_size_mb"] == round((1024 + 2048 + 4096) / (1024 * 1024), 2)

def test_rotate_error_handling(rotator, monkeypatch):
    # Test error handling when moving fails
    old_log = rotator._log_dir / "error.log"
    old_log.write_text("error")
    set_mtime(old_log, 5)

    def mock_move(*args, **kwargs):
        raise Exception("Mocked move error")

    monkeypatch.setattr(shutil, "move", mock_move)

    report = rotator.rotate()

    assert "error.log: Mocked move error" in report["errors"]
    assert old_log.exists() # Should still exist
