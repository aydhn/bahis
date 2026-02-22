import sys
from pathlib import Path
from loguru import logger

def configure_loguru(log_dir: Path):
    """Loguru yapılandırması."""
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "archive").mkdir(exist_ok=True)
    
    logger.remove()
    
    # Konsol (Renkli, Info)
    try:
        if sys.platform == "win32":
            import os
            os.system('chcp 65001 > nul')
    except Exception:
        pass
        
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> – {message}",
        level="INFO",
        enqueue=True
    )
    
    # Dosya (Debug, Detaylı)
    logger.add(
        log_dir / "bot_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="3 days",
        compression="gz",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
        colorize=False,
    )
    
    # Hata Logu
    logger.add(
        log_dir / "error.log",
        level="ERROR",
        rotation="5 MB",
        retention="7 days",
    )

class SuperLogger:
    """Merkezi loglama yöneticisi."""
    def __init__(self, log_dir: str | Path):
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        configure_loguru(log_dir)
        logger.info(f"SuperLogger başlatıldı: {log_dir}")

