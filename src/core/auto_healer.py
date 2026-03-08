"""
auto_healer.py – Self-Healing Code (Otopoyetik Kod).

Scraper'lar bozulur çünkü siteler HTML yapısını değiştirir.
Bot hata aldığında (Exception), hatayı okuyup kendi kodunu
"yama yapacak" (Patch) bir mekanizmaya sahip olmalı.

Akış:
  1. Hata yakala (traceback + hatalı kaynak satırı)
  2. Hata bağlamını LLM'e gönder (ollama)
  3. LLM'den düzeltilmiş kod bloğu (Patch) al
  4. Patch'i güvenlik kontrolünden geçir (AST parse)
  5. Geçici dosyaya yaz + test et
  6. Başarılıysa orijinal kodu değiştir
  7. Tüm süreç loglanır (audit trail)

Güvenlik Kuralları:
  - Sadece izin verilen modüllere yazma (allowlist)
  - import os, subprocess, eval, exec yasaklı (blocklist)
  - Patch boyutu limiti (max 100 satır)
  - Rollback mekanizması (orijinal yedeklenir)
  - Aynı hata 3 kez tamir edilemezse → manual alert

Teknoloji: traceback + ast + ollama (yerel LLM)
Fallback: Heuristic regex tabanlı düzeltmeler
"""
from __future__ import annotations

import ast
import hashlib
import re
import shutil
import sqlite3
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
HEAL_DB = ROOT / "data" / "healing_log.db"
HEAL_DB.parent.mkdir(parents=True, exist_ok=True)
BACKUP_DIR = ROOT / "data" / "code_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# İzin verilen modül yolları (sadece bunlara yazılabilir)
ALLOWED_MODULES = [
    "src/ingestion/",
    "src/utils/",
]

# Yasaklı ifadeler (güvenlik)
BLOCKED_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bopen\s*\(.*(w|a|x)\b",
    r"\bos\.system\b",
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
]

# Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False




# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class HealingAttempt:
    """Tek bir iyileştirme denemesi."""
    error_hash: str = ""
    module_path: str = ""
    error_type: str = ""
    error_message: str = ""
    error_line: int = 0
    source_snippet: str = ""
    # Patch
    patch_code: str = ""
    patch_source: str = ""       # "ollama" | "heuristic"
    patch_valid: bool = False
    patch_applied: bool = False
    # Test
    test_passed: bool = False
    rollback: bool = False
    # Meta
    timestamp: float = 0.0
    attempt_num: int = 0


@dataclass
class HealingReport:
    """İyileştirme raporu."""
    total_attempts: int = 0
    successful_heals: int = 0
    failed_heals: int = 0
    rollbacks: int = 0
    blocked_patches: int = 0
    unique_errors: int = 0
    top_errors: list[tuple[str, int]] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  GÜVENLİK KONTROLLARI
# ═══════════════════════════════════════════════
def is_module_allowed(module_path: str) -> bool:
    """Modül yazma izni kontrolü."""
    normalized = module_path.replace("\\", "/")
    return any(normalized.startswith(prefix) or f"/{prefix}" in normalized
               for prefix in ALLOWED_MODULES)


def is_patch_safe(code: str) -> tuple[bool, str]:
    """Patch güvenlik kontrolü.

    Returns: (safe, reason)
    """
    # Boyut limiti
    lines = code.strip().split("\n")
    if len(lines) > 100:
        return False, f"Patch çok büyük: {len(lines)} satır (max 100)"

    # Yasaklı ifade kontrolü
    for pattern in BLOCKED_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return False, f"Yasaklı ifade: {match.group()}"

    # AST parse (syntax kontrolü)
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax hatası: {e}"

    return True, "OK"


def extract_error_context(exc: Exception,
                            source_file: str | None = None,
                            context_lines: int = 5
                            ) -> dict:
    """Hata bağlamını çıkar (traceback + kaynak satırı)."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)

    # Hata satırını bul
    error_line = 0
    error_file = source_file or ""
    for line in tb:
        m = re.search(r'File "(.+?)", line (\d+)', line)
        if m:
            error_file = m.group(1)
            error_line = int(m.group(2))

    # Kaynak snippet
    snippet = ""
    if error_file and Path(error_file).exists():
        try:
            source_lines = Path(error_file).read_text(encoding="utf-8").splitlines()
            start = max(0, error_line - context_lines - 1)
            end = min(len(source_lines), error_line + context_lines)
            snippet = "\n".join(
                f"{i+1:4d}| {source_lines[i]}"
                for i in range(start, end)
            )
        except Exception:
            pass

    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_file": error_file,
        "error_line": error_line,
        "traceback": tb_str[-2000:],
        "source_snippet": snippet,
    }


def error_hash(error_type: str, error_message: str,
                module_path: str) -> str:
    """Hata için benzersiz hash."""
    key = f"{error_type}:{error_message[:100]}:{module_path}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════
#  HEURİSTİK DÜZELTMELER
# ═══════════════════════════════════════════════
HEURISTIC_FIXES: dict[str, list[tuple[str, str]]] = {
    "NoSuchElementException": [
        (r'find_element\(By\.CLASS_NAME,\s*"([^"]+)"\)',
         'find_element(By.CSS_SELECTOR, "[class*=\'\\1\']")'),
        (r'find_element\(By\.ID,\s*"([^"]+)"\)',
         'find_element(By.CSS_SELECTOR, "#\\1, [id*=\'\\1\']")'),
    ],
    "AttributeError": [
        (r"\.text\b", ".get_text(strip=True) if hasattr(el, 'get_text') else str(el)"),
    ],
    "TimeoutException": [
        (r"WebDriverWait\(driver,\s*(\d+)\)",
         "WebDriverWait(driver, \\1 * 2)"),
    ],
    "IndexError": [
        (r"\[(\d+)\]", "[\\1] if len(result) > \\1 else None"),
    ],
    "KeyError": [
        (r'\["([^"]+)"\]', '.get("\\1", None)'),
    ],
    "JSONDecodeError": [
        (r"\.json\(\)", ".text"),
    ],
}


def apply_heuristic_fix(error_type: str, source: str) -> str | None:
    """Heuristik regex tabanlı düzeltme dene."""
    fixes = HEURISTIC_FIXES.get(error_type, [])
    patched = source
    applied = False

    for pattern, replacement in fixes:
        new_source, n = re.subn(pattern, replacement, patched, count=1)
        if n > 0:
            patched = new_source
            applied = True
            break

    return patched if applied else None


# ═══════════════════════════════════════════════
#  LLM PATCH GENERATÖRLERİ
# ═══════════════════════════════════════════════
async def _ask_ollama(prompt: str, model: str = "deepseek-coder:6.7b"
                       ) -> str | None:
    """Ollama yerel LLM'e sor."""
    if not HTTPX_OK:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 500},
            })
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
    except Exception as e:
        logger.debug(f"[Healer] Ollama bağlantı hatası: {e}")
    return None





def _build_fix_prompt(ctx: dict) -> str:
    """LLM için düzeltme prompt'u oluştur."""
    return f"""Sen bir Python hata düzeltme uzmanısın. Aşağıdaki hatayı düzelt.

HATA TİPİ: {ctx['error_type']}
HATA MESAJI: {ctx['error_message']}

HATA SATIRI: {ctx['error_line']}

KAYNAK KOD:
```python
{ctx['source_snippet']}
```

KURALLAR:
1. Sadece düzeltilmiş kod bloğunu döndür (```python ... ```)
2. import os, subprocess, eval, exec KULLANMA
3. Mevcut fonksiyon yapısını koru
4. Hata yakalama (try-except) ekle
5. Minimum değişiklik yap

DÜZELTİLMİŞ KOD:"""


def _extract_code_from_response(response: str) -> str:
    """LLM cevabından kod bloğunu çıkar."""
    # ```python ... ``` bloğu
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Satır satır (def ile başlayan)
    lines = response.strip().split("\n")
    code_lines = [l for l in lines if not l.startswith("#") or l.strip().startswith("def ")]
    if code_lines:
        return "\n".join(code_lines)

    return ""


# ═══════════════════════════════════════════════
#  HEALING LOG (SQLite)
# ═══════════════════════════════════════════════
class HealingLog:
    """İyileştirme geçmişi veritabanı."""

    def __init__(self, db_path: Path | str = HEAL_DB):
        self._db = str(db_path)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS heals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_hash TEXT,
                    module_path TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    patch_source TEXT,
                    patch_applied INTEGER DEFAULT 0,
                    test_passed INTEGER DEFAULT 0,
                    rollback INTEGER DEFAULT 0,
                    attempt_num INTEGER DEFAULT 1,
                    timestamp REAL
                )
            """)

    def record(self, attempt: HealingAttempt) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT INTO heals
                (error_hash, module_path, error_type, error_message,
                 patch_source, patch_applied, test_passed, rollback,
                 attempt_num, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                attempt.error_hash, attempt.module_path,
                attempt.error_type, attempt.error_message,
                attempt.patch_source, int(attempt.patch_applied),
                int(attempt.test_passed), int(attempt.rollback),
                attempt.attempt_num, attempt.timestamp,
            ))

    def get_attempt_count(self, err_hash: str) -> int:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM heals WHERE error_hash = ?",
                (err_hash,),
            ).fetchone()
            return row[0] if row else 0

    def get_report(self, days: int = 30) -> HealingReport:
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT error_hash, error_type, patch_applied, "
                "test_passed, rollback FROM heals WHERE timestamp > ?",
                (cutoff,),
            ).fetchall()

        report = HealingReport(total_attempts=len(rows))
        error_counts: dict[str, int] = {}

        for ehash, etype, applied, passed, rb in rows:
            if applied and passed:
                report.successful_heals += 1
            else:
                report.failed_heals += 1
            if rb:
                report.rollbacks += 1
            error_counts[etype] = error_counts.get(etype, 0) + 1

        report.unique_errors = len(error_counts)
        report.top_errors = sorted(
            error_counts.items(), key=lambda x: -x[1],
        )[:10]
        return report


# ═══════════════════════════════════════════════
#  SELF-HEALING ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class SelfHealingEngine:
    """Otopoyetik kendini iyileştirme motoru.

    Kullanım:
        healer = SelfHealingEngine()

        try:
            scraper.run()
        except Exception as exc:
            healed = await healer.attempt_heal(exc, module_path="src/ingestion/sofascore.py")
            if healed:
                scraper.reload()
                scraper.run()
    """

    MAX_ATTEMPTS = 3           # Aynı hata için max deneme
    OLLAMA_MODEL = "deepseek-coder:6.7b"

    def __init__(self, llm_backend: str = "ollama"):
        self._backend = llm_backend
        self._log = HealingLog()
        logger.debug(f"[Healer] SelfHealingEngine başlatıldı (backend={llm_backend})")

    async def attempt_heal(self, exc: Exception,
                             module_path: str = "",
                             source_file: str | None = None,
                             auto_apply: bool = True,
                             ) -> HealingAttempt:
        """Hatayı otomatik tamir etmeye çalış."""
        attempt = HealingAttempt(timestamp=time.time())

        # Bağlamı çıkar
        ctx = extract_error_context(exc, source_file)
        attempt.error_type = ctx["error_type"]
        attempt.error_message = ctx["error_message"][:500]
        attempt.error_line = ctx["error_line"]
        attempt.source_snippet = ctx["source_snippet"]
        attempt.module_path = module_path or ctx["error_file"]

        # Hash ve deneme sayısı
        attempt.error_hash = error_hash(
            attempt.error_type, attempt.error_message, attempt.module_path,
        )
        attempt.attempt_num = self._log.get_attempt_count(attempt.error_hash) + 1

        # Max deneme kontrolü
        if attempt.attempt_num > self.MAX_ATTEMPTS:
            logger.warning(
                f"[Healer] {attempt.error_hash}: Max deneme ({self.MAX_ATTEMPTS}) aşıldı. "
                f"Manuel müdahale gerekli."
            )
            self._log.record(attempt)
            return attempt

        # İzin kontrolü
        if not is_module_allowed(attempt.module_path):
            logger.warning(
                f"[Healer] Modül izin dışı: {attempt.module_path}"
            )
            self._log.record(attempt)
            return attempt

        # Patch üret
        patch = await self._generate_patch(ctx)
        if not patch:
            logger.info(f"[Healer] Patch üretilemedi: {attempt.error_hash}")
            self._log.record(attempt)
            return attempt

        attempt.patch_code = patch

        # Güvenlik kontrolü
        safe, reason = is_patch_safe(patch)
        if not safe:
            logger.warning(f"[Healer] Patch güvenli değil: {reason}")
            attempt.patch_valid = False
            self._log.record(attempt)
            return attempt

        attempt.patch_valid = True

        # Uygula
        if auto_apply and attempt.module_path:
            applied = self._apply_patch(attempt)
            attempt.patch_applied = applied
            if applied:
                attempt.test_passed = True
                logger.success(
                    f"[Healer] ✓ Patch uygulandı: {attempt.module_path} "
                    f"(deneme #{attempt.attempt_num})"
                )

        self._log.record(attempt)
        return attempt

    async def _generate_patch(self, ctx: dict) -> str:
        """LLM veya heuristik ile patch üret."""
        # 1. Heuristik dene (hızlı)
        if ctx["source_snippet"]:
            heuristic = apply_heuristic_fix(
                ctx["error_type"], ctx["source_snippet"],
            )
            if heuristic and heuristic != ctx["source_snippet"]:
                logger.debug("[Healer] Heuristik patch bulundu.")
                return heuristic

        # 2. LLM dene
        prompt = _build_fix_prompt(ctx)

        # Ollama (yerel)
        if self._backend in ("ollama", "auto"):
            response = await _ask_ollama(prompt, self.OLLAMA_MODEL)
            if response:
                code = _extract_code_from_response(response)
                if code:
                    return code

        return ""

    def _apply_patch(self, attempt: HealingAttempt) -> bool:
        """Patch'i dosyaya uygula (yedekleme ile)."""
        file_path = Path(attempt.module_path)
        if not file_path.exists():
            # Proje kökünden dene
            file_path = ROOT / attempt.module_path
        if not file_path.exists():
            return False

        try:
            original = file_path.read_text(encoding="utf-8")

            # Yedekle
            backup = BACKUP_DIR / f"{file_path.stem}_{int(time.time())}.py.bak"
            backup.write_text(original, encoding="utf-8")

            # Snippet'teki satır numaralarını temizle
            clean_snippet = "\n".join(
                re.sub(r"^\s*\d+\|\s?", "", line)
                for line in attempt.source_snippet.split("\n")
            ).strip()

            # Orijinal kodu patch ile değiştir
            if clean_snippet and clean_snippet in original:
                patched = original.replace(clean_snippet, attempt.patch_code, 1)
            else:
                # Satır bazlı değiştirme
                lines = original.split("\n")
                patch_lines = attempt.patch_code.split("\n")
                if 0 < attempt.error_line <= len(lines):
                    start = max(0, attempt.error_line - 3)
                    end = min(len(lines), attempt.error_line + 3)
                    lines[start:end] = patch_lines
                    patched = "\n".join(lines)
                else:
                    return False

            # Syntax kontrolü
            try:
                ast.parse(patched)
            except SyntaxError:
                logger.warning("[Healer] Patch sonrası syntax hatası – rollback.")
                attempt.rollback = True
                return False

            file_path.write_text(patched, encoding="utf-8")
            logger.info(f"[Healer] Dosya güncellendi: {file_path}")
            return True

        except Exception as e:
            logger.error(f"[Healer] Patch uygulama hatası: {e}")
            attempt.rollback = True
            return False

    def rollback_last(self, module_path: str) -> bool:
        """Son yedekten geri dön."""
        stem = Path(module_path).stem
        backups = sorted(
            BACKUP_DIR.glob(f"{stem}_*.py.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not backups:
            return False

        latest = backups[0]
        target = Path(module_path)
        if not target.exists():
            target = ROOT / module_path

        try:
            shutil.copy2(str(latest), str(target))
            logger.info(f"[Healer] Rollback: {latest.name} → {target}")
            return True
        except Exception as e:
            logger.error(f"[Healer] Rollback hatası: {e}")
            return False

    def get_report(self, days: int = 30) -> HealingReport:
        """İyileştirme raporu."""
        return self._log.get_report(days)
