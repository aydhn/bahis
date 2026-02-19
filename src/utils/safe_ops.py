"""
safe_ops.py – Global Null-Safe Utility Fonksiyonları.

Tüm modüller tarafından kullanılan defansif veri dönüşüm fonksiyonları.
Polars null, NaN, None ve beklenmeyen tipleri güvenli şekilde dönüştürür.

Kullanım:
    from src.utils.safe_ops import safe_float, safe_str, safe_int, safe_dict_get

    val = safe_float(row.get("odds"))        # None → 0.0
    name = safe_str(row.get("team"))          # None → ""
    count = safe_int(row.get("goals"))        # None → 0
"""
from __future__ import annotations

import math
import time
from typing import Any

from loguru import logger


def safe_float(val: Any, default: float = 0.0) -> float:
    """Herhangi bir değeri güvenli float'a çevirir.

    Handles: None, NaN, Polars null, string, int, bool, inf.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("none", "null", "nan", "inf", "-inf", "n/a", ""):
            return default
        try:
            v = float(val.replace(",", "."))
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except (ValueError, TypeError):
            return default
    # Polars null, numpy types, etc.
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError, OverflowError):
        return default


def safe_str(val: Any, default: str = "") -> str:
    """Herhangi bir değeri güvenli string'e çevirir."""
    if val is None:
        return default
    s = str(val).strip()
    if s.lower() in ("none", "null"):
        return default
    return s


def safe_int(val: Any, default: int = 0) -> int:
    """Herhangi bir değeri güvenli int'e çevirir."""
    f = safe_float(val, float(default))
    return int(f)


def safe_dict_get(d: dict, key: str, default: Any = None,
                  cast: type | None = None) -> Any:
    """Dict'ten güvenli değer çeker, opsiyonel tip dönüşümü."""
    val = d.get(key, default)
    if val is None:
        return default
    if cast is float:
        return safe_float(val, default if isinstance(default, (int, float)) else 0.0)
    if cast is str:
        return safe_str(val, default if isinstance(default, str) else "")
    if cast is int:
        return safe_int(val, default if isinstance(default, int) else 0)
    return val


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Güvenli bölme - sıfıra bölme koruması."""
    n = safe_float(numerator, 0.0)
    d = safe_float(denominator, 0.0)
    if abs(d) < 1e-12:
        return default
    result = n / d
    if math.isnan(result) or math.isinf(result):
        return default
    return result


class PerfTimer:
    """Bağlam yöneticisi ile performans ölçümü.

    Kullanım:
        with PerfTimer("model_predict") as t:
            result = model.predict(data)
        print(f"Süre: {t.elapsed_ms:.1f}ms")
    """

    def __init__(self, label: str = "", log_level: str = "DEBUG",
                 threshold_ms: float = 0.0):
        self.label = label
        self.log_level = log_level
        self.threshold_ms = threshold_ms
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if self.label and self.elapsed_ms >= self.threshold_ms:
            lg = getattr(logger, self.log_level.lower(), logger.debug)
            lg(f"[Perf] {self.label}: {self.elapsed_ms:.1f}ms")


def clamp(value: float, low: float, high: float) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, safe_float(value, low)))


def normalize_probs(probs: list[float]) -> list[float]:
    """Olasılık vektörünü normalize eder (toplamı 1 yapar)."""
    safe_probs = [max(0.0, safe_float(p, 0.0)) for p in probs]
    total = sum(safe_probs)
    if total < 1e-12:
        n = len(safe_probs)
        return [1.0 / n] * n if n > 0 else []
    return [p / total for p in safe_probs]
