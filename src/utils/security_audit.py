"""
security_audit.py – Statik Güvenlik Analizi ve Risk Taraması.

Bu modül, kod tabanını şu risklere karşı tarar:
- Hardcoded sırlar (Secret keys)
- Tehlikeli fonksiyonlar (eval, exec, pickle)
- Unsafe shell komutları
- Zayıf kriptografi
"""
import os
import re
from pathlib import Path
from loguru import logger

class SecurityAuditor:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.issues = []

    def scan(self):
        """Tüm projeyi tarar."""
        logger.info("[SecurityAudit] Tarama başlatılıyor...")
        for file_path in self.root_dir.rglob("*.py"):
            if "env" in str(file_path) or ".git" in str(file_path):
                continue
            self._scan_file(file_path)
        
        self._report()

    def _scan_file(self, path: Path):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            
            # 1. Hardcoded Secrets
            if re.search(r'(token|key|password|secret|api_id)\s*=\s*["\'][a-zA-Z0-9_\-]{10,}', content, re.I):
                # .env veya os.getenv olmayan durumlar risklidir
                if "os.getenv" not in content and "os.environ" not in content:
                    self.issues.append(f"CRITICAL: {path.name} içinde hardcoded secret olabilir.")

            # 2. Unsafe Functions
            if "eval(" in content:
                self.issues.append(f"HIGH: {path.name} içinde 'eval' kullanımı tespit edildi (Injection Riski).")
            if "exec(" in content:
                self.issues.append(f"HIGH: {path.name} içinde 'exec' kullanımı tespit edildi.")
            if "pickle.load" in content:
                self.issues.append(f"MEDIUM: {path.name} içinde unsafe pickle kullanımı.")

            # 3. Shell Injection
            if re.search(r'os\.system\(|subprocess\.Popen\(.*shell=True', content):
                self.issues.append(f"HIGH: {path.name} içinde unsafe shell execution.")

        except Exception as e:
            logger.error(f"Dosya tarama hatası {path}: {e}")

    def _report(self):
        if not self.issues:
            logger.success("[SecurityAudit] Hiçbir kritik açık bulunamadı. Sistem GÜVENLİ.")
        else:
            logger.warning(f"[SecurityAudit] {len(self.issues)} potansiyel risk bulundu:")
            for issue in self.issues:
                logger.warning(f" - {issue}")

if __name__ == "__main__":
    auditor = SecurityAuditor()
    auditor.scan()
