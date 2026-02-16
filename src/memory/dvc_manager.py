"""
dvc_manager.py – DVC tabanlı veri ve model versiyonlama.
Dataset'leri ve .pth model dosyalarını versiyonlar.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]


class DVCManager:
    """Data Version Control (DVC) yöneticisi."""

    def __init__(self, data_dir: Path | str = ROOT / "data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._dvc_available = self._check_dvc()
        logger.debug(f"DVCManager başlatıldı (DVC: {'aktif' if self._dvc_available else 'yok'}).")

    def _check_dvc(self) -> bool:
        try:
            result = subprocess.run(
                ["dvc", "version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def init(self) -> bool:
        """DVC'yi başlatır (git repo gerektirir)."""
        if not self._dvc_available:
            logger.warning("DVC yüklü değil – versiyonlama devre dışı.")
            return False
        try:
            subprocess.run(["dvc", "init"], cwd=str(ROOT), capture_output=True)
            logger.info("DVC başlatıldı.")
            return True
        except Exception as e:
            logger.error(f"DVC init hatası: {e}")
            return False

    def track(self, filepath: str | Path) -> bool:
        """Dosyayı DVC ile takibe alır."""
        if not self._dvc_available:
            return False
        try:
            subprocess.run(["dvc", "add", str(filepath)], cwd=str(ROOT), capture_output=True, check=True)
            logger.info(f"DVC tracking: {filepath}")
            return True
        except Exception as e:
            logger.error(f"DVC add hatası: {e}")
            return False

    def push(self, remote: str = "origin") -> bool:
        """DVC verisini remote'a gönderir."""
        if not self._dvc_available:
            return False
        try:
            subprocess.run(["dvc", "push", "-r", remote], cwd=str(ROOT), capture_output=True, check=True)
            logger.info(f"DVC push tamamlandı → {remote}")
            return True
        except Exception as e:
            logger.error(f"DVC push hatası: {e}")
            return False

    def snapshot(self, tag: str | None = None) -> str:
        """Mevcut veri durumunun anlık görüntüsünü oluşturur."""
        tag = tag or f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        manifest = {
            "tag": tag,
            "timestamp": datetime.utcnow().isoformat(),
            "files": [],
        }
        for f in self._data_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                manifest["files"].append({
                    "path": str(f.relative_to(ROOT)),
                    "size_mb": f.stat().st_size / 1e6,
                })

        manifest_path = self._data_dir / f"{tag}.json"
        import json
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Snapshot oluşturuldu: {tag} ({len(manifest['files'])} dosya)")
        return str(manifest_path)
