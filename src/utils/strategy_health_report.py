"""
strategy_health_report.py – Strateji sağlık raporu.
Sharpe Ratio, Drawdown, ROI ve modül performanslarını PDF'e dönüştürür.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table


REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


class StrategyHealthReport:
    """Strateji performans raporu üretici."""

    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []
        self._signals_history: list[dict] = []
        logger.debug("StrategyHealthReport başlatıldı.")

    def update(self, bets: list[dict], ensemble: list[dict]):
        """Her döngüde güncelleme alır."""
        self._signals_history.extend(bets)
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_bets": len([b for b in bets if b.get("selection") != "skip"]),
            "avg_ev": np.mean([b.get("ev", 0) for b in bets]) if bets else 0,
            "avg_confidence": np.mean([b.get("confidence", 0) for b in bets]) if bets else 0,
            "total_stake": sum(b.get("stake_pct", 0) for b in bets),
        }
        self._history.append(record)

    def generate_pdf(self) -> str:
        """PDF rapor üretir."""
        try:
            from fpdf import FPDF
        except ImportError:
            logger.warning("fpdf2 yüklü değil – PDF oluşturulamıyor.")
            return self._generate_text_report()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Quant Betting Bot - Strateji Saglik Raporu", ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)

        # Özet metrikler
        metrics = self._compute_metrics()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Ozet Metrikler", ln=True)
        pdf.set_font("Helvetica", "", 10)

        for key, value in metrics.items():
            label = key.replace("_", " ").title()
            pdf.cell(60, 7, f"{label}:", border=0)
            pdf.cell(0, 7, str(value), border=0, ln=True)

        pdf.ln(5)

        # Sinyal geçmişi
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Son 20 Sinyal", ln=True)
        pdf.set_font("Helvetica", "", 8)

        recent = self._signals_history[-20:]
        for sig in recent:
            line = (
                f"  {sig.get('match_id', '')[:20]:20s} | "
                f"{sig.get('selection', '-'):6s} | "
                f"EV={sig.get('ev', 0):.3f} | "
                f"Stake={sig.get('stake_pct', 0):.3%}"
            )
            pdf.cell(0, 5, line, ln=True)

        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path = REPORTS_DIR / filename
        pdf.output(str(path))
        logger.info(f"PDF rapor kaydedildi: {path}")
        return str(path)

    def _generate_text_report(self) -> str:
        """PDF yokken metin raporu."""
        metrics = self._compute_metrics()
        lines = ["=" * 50, "QUANT BETTING BOT – STRATEJİ SAĞLIK RAPORU", "=" * 50, ""]
        for k, v in metrics.items():
            lines.append(f"  {k.replace('_', ' ').title():30s}: {v}")
        lines.append("")

        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = REPORTS_DIR / filename
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Metin rapor kaydedildi: {path}")
        return str(path)

    def _compute_metrics(self) -> dict:
        if not self._history:
            return {"status": "Henüz veri yok"}

        total_bets = sum(h["n_bets"] for h in self._history)
        avg_ev = np.mean([h["avg_ev"] for h in self._history])
        avg_conf = np.mean([h["avg_confidence"] for h in self._history])
        total_stake = sum(h["total_stake"] for h in self._history)

        # Sharpe basit tahmini
        evs = [h["avg_ev"] for h in self._history if h["avg_ev"] != 0]
        sharpe = float(np.mean(evs) / (np.std(evs) + 1e-10) * np.sqrt(252)) if evs else 0

        return {
            "toplam_dongu": len(self._history),
            "toplam_bahis": total_bets,
            "ortalama_ev": f"{avg_ev:.4f}",
            "ortalama_guven": f"{avg_conf:.2%}",
            "toplam_stake": f"{total_stake:.4f}",
            "sharpe_ratio": f"{sharpe:.2f}",
        }

    def console_summary(self):
        """Konsola özet yazdırır."""
        console = Console()
        metrics = self._compute_metrics()
        table = Table(title="Strateji Sağlık Özeti", show_lines=True)
        table.add_column("Metrik", style="cyan")
        table.add_column("Değer", style="bold")
        for k, v in metrics.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)
