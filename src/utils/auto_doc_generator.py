"""
auto_doc_generator.py – Otomatik dokümantasyon üretici.
Tüm sistemin teknik "Yönetici El Kitabı"nı otomatik oluşturur.
"""
from __future__ import annotations

import ast
from pathlib import Path
from datetime import datetime

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"


class AutoDocGenerator:
    """Kaynak koddan otomatik dokümantasyon üreten modül."""

    SCAN_DIRS = [
        "src/ingestion",
        "src/memory",
        "src/quant",
        "src/core",
        "src/utils",
        "src/ui",
    ]

    def __init__(self):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("AutoDocGenerator başlatıldı.")

    def generate(self) -> str:
        """Tüm modülleri tarayarak dokümantasyon oluşturur."""
        sections = []
        sections.append(self._header())

        for scan_dir in self.SCAN_DIRS:
            full_path = ROOT / scan_dir
            if not full_path.exists():
                continue
            section = self._scan_directory(full_path, scan_dir)
            if section:
                sections.append(section)

        content = "\n\n".join(sections)
        doc_path = DOCS_DIR / "API_REFERENCE.md"
        doc_path.write_text(content, encoding="utf-8")
        logger.info(f"Dokümantasyon oluşturuldu: {doc_path}")
        return str(doc_path)

    def _header(self) -> str:
        return (
            f"# Quant Betting Bot – API Reference\n\n"
            f"*Otomatik oluşturuldu: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            f"---\n"
        )

    def _scan_directory(self, path: Path, label: str) -> str:
        lines = [f"## {label}\n"]
        py_files = sorted(path.glob("*.py"))

        for py_file in py_files:
            if py_file.name.startswith("__"):
                continue
            doc = self._analyze_file(py_file)
            if doc:
                lines.append(doc)

        return "\n".join(lines)

    def _analyze_file(self, filepath: Path) -> str:
        """Tek bir Python dosyasını analiz eder."""
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            return f"### {filepath.name}\n\n*Parse hatası: {e}*\n"

        lines = [f"### {filepath.name}\n"]

        # Modül docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            lines.append(f"> {module_doc.split(chr(10))[0]}\n")

        # Sınıflar ve fonksiyonlar
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node)
                lines.append(f"**class `{node.name}`**")
                if class_doc:
                    lines.append(f"  {class_doc.split(chr(10))[0]}")
                lines.append("")

                # Methodlar
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("_") and item.name != "__init__":
                            continue
                        args = [a.arg for a in item.args.args if a.arg != "self"]
                        sig = f"({', '.join(args)})"
                        method_doc = ast.get_docstring(item) or ""
                        first_line = method_doc.split("\n")[0] if method_doc else ""
                        lines.append(f"  - `{item.name}{sig}` – {first_line}")

                lines.append("")

            elif isinstance(node, ast.FunctionDef) and not any(
                isinstance(p, ast.ClassDef) for p in ast.walk(tree)
            ):
                func_doc = ast.get_docstring(node)
                first_line = func_doc.split("\n")[0] if func_doc else ""
                lines.append(f"**`{node.name}()`** – {first_line}\n")

        return "\n".join(lines)
