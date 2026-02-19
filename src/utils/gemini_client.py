"""
gemini_client.py – Google Gemini API Unified Client.

google-generativeai deprecated olduğu için google-genai SDK'ya geçildi.
Tüm modüller bu wrapper üzerinden Gemini'ye erişir.

Özellikler:
  - Singleton client (tek bağlantı, her yerde paylaşım)
  - Rate limiting (dakikada max 15 istek – ücretsiz tier)
  - Retry with exponential backoff (429/500 hatalarında)
  - Token sayacı (maliyet takibi)
  - Model seçimi: gemini-2.0-flash (hızlı), gemini-2.0-pro (kaliteli)

Kullanım:
    from src.utils.gemini_client import gemini_generate
    text = gemini_generate(prompt="Merhaba", system="Sen bir analistsin")
"""
from __future__ import annotations

import os
import time
from collections import deque
from loguru import logger

GEMINI_OK = False
_client = None
_call_timestamps: deque = deque(maxlen=100)
_total_calls = 0
_total_tokens_approx = 0
_rate_limit_per_minute = 15

try:
    from google import genai
    GEMINI_OK = True
except ImportError:
    pass


def _get_client():
    """Singleton Gemini client."""
    global _client
    if _client is None and GEMINI_OK:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            _client = genai.Client(api_key=api_key)
        else:
            try:
                _client = genai.Client()
            except Exception as e:
                logger.debug(f"[GeminiClient] Client oluşturulamadı: {e}")
    return _client


def _rate_limit_check() -> bool:
    """Dakikadaki çağrı sayısını kontrol eder."""
    now = time.time()
    while _call_timestamps and _call_timestamps[0] < now - 60:
        _call_timestamps.popleft()
    if len(_call_timestamps) >= _rate_limit_per_minute:
        return False
    return True


def gemini_generate(prompt: str, system: str = "",
                    model: str = "gemini-2.0-flash",
                    temperature: float = 0.3,
                    max_tokens: int = 500,
                    retries: int = 2) -> str:
    """Gemini API üzerinden metin üretir.

    Args:
        prompt: Ana prompt
        system: Sistem mesajı (prompt'un başına eklenir)
        model: Model adı (gemini-2.0-flash | gemini-2.0-pro)
        temperature: Yaratıcılık seviyesi (0.0=deterministik, 1.0=yaratıcı)
        max_tokens: Maks token sayısı
        retries: Hata durumunda tekrar deneme sayısı

    Returns:
        Üretilen metin veya hata durumunda boş string
    """
    global _total_calls, _total_tokens_approx

    if not GEMINI_OK:
        return ""

    client = _get_client()
    if client is None:
        return ""

    if not _rate_limit_check():
        logger.debug("[GeminiClient] Rate limit aşıldı, bekleniyor…")
        time.sleep(5)
        if not _rate_limit_check():
            return ""

    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    for attempt in range(retries + 1):
        try:
            _call_timestamps.append(time.time())
            _total_calls += 1

            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            if response and response.text:
                text = response.text.strip()
                _total_tokens_approx += len(text.split())
                return text
            return ""

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 2 ** (attempt + 1)
                logger.debug(f"[GeminiClient] Rate limit, {wait}s bekleniyor…")
                time.sleep(wait)
                continue
            if "500" in err_str or "503" in err_str:
                time.sleep(1)
                continue
            logger.debug(f"[GeminiClient] Hata: {e}")
            return ""

    return ""


def gemini_stats() -> dict:
    """Gemini kullanım istatistikleri."""
    return {
        "total_calls": _total_calls,
        "approx_tokens": _total_tokens_approx,
        "rate_limit": _rate_limit_per_minute,
        "sdk_ok": GEMINI_OK,
        "client_ready": _client is not None,
    }
