#!/usr/bin/env python3
"""
system_check.py – Sistem sağlık kontrolü ve otomatik düzeltme aracı.

Kullanım:
    python system_check.py [--fix]

Yapılan kontroller:
    - Gerekli dizinler (logs/, data/, models/)
    - DuckDB bağlantısı ve kilit durumu
    - Neo4j bağlantısı (isteğe bağlı)
    - Python bağımlılıkları
    - Telegram bot token
    - Eksik model dosyaları
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Proje kökünü bul
ROOT = Path(__file__).resolve().parent

def log(status: str, message: str):
    """Renkli log çıktısı."""
    colors = {
        "OK": "\033[92m",      # Yeşil
        "WARN": "\033[93m",    # Sarı
        "FAIL": "\033[91m",    # Kırmızı
        "INFO": "\033[94m",    # Mavi
        "RESET": "\033[0m",
    }
    color = colors.get(status, colors["INFO"])
    status_str = f"[{status}]".ljust(6)
    try:
        print(f"{color}{status_str}{colors['RESET']} {message}")
    except UnicodeEncodeError:
        # Windows console encoding sorunu durumunda ASCII karakterler kullan
        print(f"[{status}] {message}")

def check_directories(fix: bool = False) -> bool:
    """Gerekli dizinlerin varlığını kontrol et."""
    required_dirs = [
        ROOT / "logs",
        ROOT / "logs" / "archive",
        ROOT / "data",
        ROOT / "data" / "cache",
        ROOT / "data" / "logs",
        ROOT / "models",
        ROOT / "models" / "rl",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists():
            log("OK", f"Dizin mevcut: {dir_path.relative_to(ROOT)}")
        else:
            if fix:
                dir_path.mkdir(parents=True, exist_ok=True)
                log("INFO", f"Dizin oluşturuldu: {dir_path.relative_to(ROOT)}")
            else:
                log("WARN", f"Dizin eksik: {dir_path.relative_to(ROOT)}")
                all_ok = False
    
    return all_ok

def check_duckdb(fix: bool = False) -> bool:
    """DuckDB dosya kilidini kontrol et."""
    db_path = ROOT / "data" / "bahis.duckdb"
    
    # Önce eski süreçleri kontrol et
    try:
        import psutil
        current_pid = os.getpid()
        db_path_str = str(db_path).lower()
        
        stale_pids = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.pid == current_pid:
                    continue
                pname = (proc.info.get("name") or "").lower()
                if "python" not in pname:
                    continue
                try:
                    open_files = proc.open_files()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
                for f in open_files:
                    if db_path_str in f.path.lower():
                        stale_pids.append(proc.pid)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        if stale_pids:
            if fix:
                log("WARN", f"DuckDB kilitleyen süreçler bulundu: {stale_pids}")
                for pid in stale_pids:
                    try:
                        p = psutil.Process(pid)
                        p.terminate()
                        p.wait(timeout=3)
                        log("INFO", f"Süreç sonlandırıldı: PID={pid}")
                    except Exception as e:
                        log("FAIL", f"Süreç sonlandırılamadı PID={pid}: {e}")
            else:
                log("WARN", f"DuckDB kilitli – Süreçler: {stale_pids} (python system_check.py --fix)")
                return False
        else:
            log("OK", "DuckDB dosyası serbest")
            
    except ImportError:
        log("WARN", "psutil yüklü değil – süreç kontrolü atlanıyor")
    
    # .wal ve .tmp dosyalarını temizle
    for suffix in (".wal", ".tmp"):
        temp_file = db_path.with_suffix(db_path.suffix + suffix)
        if temp_file.exists():
            if fix:
                try:
                    temp_file.unlink()
                    log("INFO", f"Temizlendi: {temp_file.name}")
                except OSError as e:
                    log("FAIL", f"Temizlenemedi: {temp_file.name} – {e}")
            else:
                log("WARN", f"Geçici dosya mevcut: {temp_file.name}")
    
    return True

def check_neo4j() -> bool:
    """Neo4j bağlantısını kontrol et."""
    try:
        from neo4j import GraphDatabase
        uri = "bolt://localhost:7687"
        driver = GraphDatabase.driver(uri, auth=None)
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        log("OK", "Neo4j bağlantısı aktif (localhost:7687)")
        return True
    except Exception as e:
        log("WARN", f"Neo4j bağlantısı yok – fallback mod aktif olacak ({type(e).__name__})")
        return False

def check_dependencies() -> bool:
    """Kritik Python bağımlılıklarını kontrol et."""
    critical_deps = [
        "polars",
        "duckdb",
        "numpy",
        "loguru",
        "typer",
        "rich",
        "telegram",
        "pydantic",
    ]
    
    optional_deps = [
        "pymc",
        "jax",
        "torch",
        "lightgbm",
        "xgboost",
        "psutil",
        "neo4j",
    ]
    
    all_ok = True
    for dep in critical_deps:
        try:
            __import__(dep)
            log("OK", f"Bağımlılık: {dep}")
        except ImportError:
            log("FAIL", f"EKSİK KRİTİK: {dep} – pip install {dep}")
            all_ok = False
    
    for dep in optional_deps:
        try:
            __import__(dep)
            log("OK", f"Opsiyonel: {dep}")
        except ImportError:
            log("INFO", f"Opsiyonel eksik: {dep}")
    
    return all_ok

def check_telegram_config() -> bool:
    """Telegram yapılandırmasını kontrol et."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        log("OK", "Telegram yapılandırması tamam")
        return True
    else:
        log("WARN", "Telegram token/chat_id eksik – .env dosyasını kontrol edin")
        return False

def check_models() -> bool:
    """Model dosyalarının varlığını kontrol et."""
    models_dir = ROOT / "models"
    expected_models = [
        "lstm_trend.pt",
        "ppo_betting_agent.zip",
    ]
    
    for model in expected_models:
        model_path = models_dir / model
        if model_path.exists():
            log("OK", f"Model mevcut: {model}")
        else:
            log("INFO", f"Model oluşturulacak (ilk çalıştırmada): {model}")
    
    return True

def clean_logs():
    """Eski log dosyalarını temizle."""
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        return
    
    import time
    now = time.time()
    max_age_days = 7
    
    cleaned = 0
    for log_file in logs_dir.glob("*.log*"):
        if log_file.is_file():
            age_days = (now - log_file.stat().st_mtime) / (24 * 3600)
            if age_days > max_age_days:
                try:
                    log_file.unlink()
                    cleaned += 1
                except OSError:
                    pass
    
    if cleaned:
        log("INFO", f"{cleaned} eski log dosyası temizlendi")

def main():
    parser = argparse.ArgumentParser(description="Quant Betting Bot – Sistem Kontrolü")
    parser.add_argument("--fix", action="store_true", help="Bulunan sorunları otomatik düzelt")
    parser.add_argument("--clean", action="store_true", help="Eski logları temizle")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  QUANT BETTING BOT – SİSTEM KONTROLÜ")
    print("="*60 + "\n")
    
    if args.clean:
        clean_logs()
    
    checks = [
        ("Dizinler", check_directories(args.fix)),
        ("DuckDB", check_duckdb(args.fix)),
        ("Neo4j", check_neo4j()),
        ("Bağımlılıklar", check_dependencies()),
        ("Telegram", check_telegram_config()),
        ("Modeller", check_models()),
    ]
    
    print("\n" + "="*60)
    print("  KONTROL SONUÇLARI")
    print("="*60)
    
    all_passed = True
    for name, result in checks:
        status = "[OK]" if result else "[X]"
        print(f"  {status} {name}")
        if not result:
            all_passed = False
    
    print("="*60 + "\n")
    
    if all_passed:
        msg = "[OK] Tum kontroller basarili – Sistem calismaya hazir!"
        try:
            print(f"\033[92m{msg}\033[0m\n")
        except UnicodeEncodeError:
            print(msg + "\n")
        return 0
    else:
        if args.fix:
            msg = "[!] Bazi sorunlar otomatik duzeltildi – Tekrar kontrol edin."
            try:
                print(f"\033[93m{msg}\033[0m\n")
            except UnicodeEncodeError:
                print(msg + "\n")
        else:
            msg = "[X] Bazi sorunlar var – Duzeltmek icin: python system_check.py --fix"
            try:
                print(f"\033[91m{msg}\033[0m\n")
            except UnicodeEncodeError:
                print(msg + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
