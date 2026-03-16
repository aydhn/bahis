"""
log_rotator.py – Log Rotasyon & Arşivleme Sistemi.

Son 3 günlük logları aktif tutar, daha eski logları
logs/archive/ klasörüne taşır ve sıkıştırır.

Bu sayede:
  - AI context window'u sadece güncel logları okur
  - Eski loglar archive/'de saklanır (gitignore'da engelli)
  - API kota tüketimi düşer (daha az token)
  - Disk alanı korunur (gz sıkıştırma)

Kullanım:
  rotator = LogRotator(log_dir="logs", archive_days=3)
  rotator.rotate()  # Eski logları arşivle
"""
from __future__ import annotations

import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


class LogRotator:
    """Log dosyalarını otomatik arşivler.

    Kullanım:
        rotator = LogRotator(log_dir="logs", archive_days=3)

        # Manuel çalıştır
        rotator.rotate()

        # Scheduler ile (her gece 02:00)
        scheduler.add_cron("log_rotate", rotator.rotate, hour=2, minute=0)
    """

    def __init__(self, log_dir: str = "logs",
                 archive_days: int = 3,
                 compress: bool = True,
                 max_archive_days: int = 30):
        self._log_dir = Path(log_dir)
        self._archive_dir = self._log_dir / "archive"
        self._archive_days = archive_days
        self._compress = compress
        self._max_archive = max_archive_days

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"[LogRotator] Başlatıldı: log_dir={log_dir}, "
            f"archive_days={archive_days}, compress={compress}"
        )

    def _archive_file(self, log_file: Path, report: dict, check_exists: bool = False) -> None:
        """Helper to archive or compress a single log file."""
        dest = self._archive_dir / log_file.name

        if self._compress:
            gz_path = dest.with_suffix(dest.suffix + ".gz")
            if check_exists and gz_path.exists():
                return
            with open(log_file, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log_file.unlink()
            report["archived"].append(str(gz_path.name))
            logger.info(f"[LogRotator] Arşivlendi: {log_file.name} → archive/{gz_path.name}")
        else:
            if check_exists and dest.exists():
                return
            shutil.move(str(log_file), str(dest))
            report["archived"].append(str(dest.name))
            logger.info(f"[LogRotator] Taşındı: {log_file.name} → archive/{dest.name}")

    def _process_active_logs(self, cutoff: datetime, report: dict) -> None:
        """Process active logs checking mtime."""
        for log_file in self._log_dir.glob("*.log"):
            if log_file.is_dir() or log_file.name.startswith("bot_"):
                # Ignore directory and bot_ logs (handled separately)
                continue

            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff:
                    self._archive_file(log_file, report)
                else:
                    report["kept"].append(str(log_file.name))
            except Exception as e:
                report["errors"].append(f"{log_file.name}: {e}")
                logger.debug(f"[LogRotator] Hata: {log_file.name} — {e}")

    def _process_date_pattern_logs(self, cutoff: datetime, report: dict) -> None:
        """Process log files with date pattern like bot_YYYY-MM-DD.log."""
        for log_file in self._log_dir.glob("bot_*.log"):
            if log_file.is_dir():
                continue
            try:
                date_str = log_file.stem.replace("bot_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    self._archive_file(log_file, report, check_exists=True)
            except (ValueError, Exception) as e:
                report["errors"].append(f"{log_file.name}: {e}")
                logger.debug(f"[LogRotator] Hata: {log_file.name} — {e}")

    def _delete_old_archives(self, archive_cutoff: datetime, report: dict) -> None:
        """Delete archived logs older than max_archive_days."""
        for archive_file in self._archive_dir.iterdir():
            if archive_file.is_dir():
                continue
            try:
                file_mtime = datetime.fromtimestamp(archive_file.stat().st_mtime)
                if file_mtime < archive_cutoff:
                    archive_file.unlink()
                    report["deleted"].append(str(archive_file.name))
                    logger.info(
                        f"[LogRotator] Silindi (>{self._max_archive}gün): "
                        f"archive/{archive_file.name}"
                    )
            except Exception as e:
                report["errors"].append(f"archive/{archive_file.name}: {e}")
                logger.debug(f"[LogRotator] Hata: archive/{archive_file.name} — {e}")

    def rotate(self) -> dict:
        """Eski logları arşivle, çok eski arşivleri sil.

        Returns:
            dict: Rotasyon raporu
        """
        report = {
            "archived": [],
            "deleted": [],
            "kept": [],
            "errors": [],
        }

        cutoff = datetime.now() - timedelta(days=self._archive_days)
        archive_cutoff = datetime.now() - timedelta(days=self._max_archive)

        self._process_active_logs(cutoff, report)
        self._process_date_pattern_logs(cutoff, report)
        self._delete_old_archives(archive_cutoff, report)

        logger.info(
            f"[LogRotator] Rotasyon tamamlandı: "
            f"{len(report['archived'])} arşivlendi, "
            f"{len(report['deleted'])} silindi, "
            f"{len(report['kept'])} korundu"
        )

        return report

    def get_active_logs(self) -> list[Path]:
        """Güncel (son 3 gün) log dosyalarını listele."""
        cutoff = datetime.now() - timedelta(days=self._archive_days)
        active = []
        for f in sorted(self._log_dir.glob("*.log")):
            if f.is_file():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime >= cutoff:
                    active.append(f)
        return active

    def get_stats(self) -> dict:
        """Log istatistikleri."""
        active = list(self._log_dir.glob("*.log"))
        archived = list(self._archive_dir.iterdir()) if self._archive_dir.exists() else []

        active_size = sum(f.stat().st_size for f in active if f.is_file())
        archive_size = sum(f.stat().st_size for f in archived if f.is_file())

        return {
            "active_count": len(active),
            "active_size_mb": round(active_size / (1024 * 1024), 2),
            "archive_count": len(archived),
            "archive_size_mb": round(archive_size / (1024 * 1024), 2),
            "total_size_mb": round((active_size + archive_size) / (1024 * 1024), 2),
        }
