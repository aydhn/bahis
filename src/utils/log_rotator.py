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

        # 1) Aktif logları tara
        for log_file in self._log_dir.glob("*.log"):
            if log_file.is_dir():
                continue

            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

                if file_mtime < cutoff:
                    # Eski — arşivle
                    dest = self._archive_dir / log_file.name

                    if self._compress:
                        gz_path = dest.with_suffix(dest.suffix + ".gz")
                        with open(log_file, "rb") as f_in:
                            with gzip.open(gz_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        log_file.unlink()
                        report["archived"].append(str(gz_path.name))
                        logger.info(
                            f"[LogRotator] Arşivlendi: {log_file.name} → "
                            f"archive/{gz_path.name}"
                        )
                    else:
                        shutil.move(str(log_file), str(dest))
                        report["archived"].append(str(dest.name))
                        logger.info(
                            f"[LogRotator] Taşındı: {log_file.name} → "
                            f"archive/{dest.name}"
                        )
                else:
                    report["kept"].append(str(log_file.name))

            except Exception as e:
                report["errors"].append(f"{log_file.name}: {e}")
                logger.debug(f"[LogRotator] Hata: {log_file.name} — {e}")

        # Tarih kalıplı logları da tara (bot_2026-02-13.log gibi)
        for log_file in self._log_dir.glob("bot_*.log"):
            if log_file.is_dir():
                continue
            try:
                date_str = log_file.stem.replace("bot_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    dest = self._archive_dir / log_file.name
                    if self._compress:
                        gz_path = dest.with_suffix(dest.suffix + ".gz")
                        if not gz_path.exists():
                            with open(log_file, "rb") as f_in:
                                with gzip.open(gz_path, "wb") as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            log_file.unlink()
                            report["archived"].append(str(gz_path.name))
                    else:
                        if not dest.exists():
                            shutil.move(str(log_file), str(dest))
                            report["archived"].append(str(dest.name))
            except (ValueError, Exception) as e:
                report["errors"].append(f"{log_file.name}: {e}")

        # 2) Çok eski arşivleri sil
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
