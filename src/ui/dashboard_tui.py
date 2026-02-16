"""
dashboard_tui.py – Rich/Textual tabanlı terminal dashboard.
Bloomberg stili canlı akan profesyonel ekran.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from loguru import logger


class DashboardTUI:
    """Terminalde canlı akan Bloomberg stili dashboard."""

    def __init__(self):
        self._console = Console()
        self._data: dict = {
            "signals": [],
            "bankroll": 10000.0,
            "cycle": 0,
            "status": "Bekleniyor",
            "risk_level": "low",
            "active_bets": 0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
        }
        logger.debug("DashboardTUI başlatıldı.")

    def update(self, **kwargs):
        self._data.update(kwargs)

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="signals", ratio=3),
            Layout(name="metrics"),
        )
        return layout

    def _render_header(self) -> Panel:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = Text()
        title.append(" QUANT BETTING BOT ", style="bold white on blue")
        title.append(f"  {now}  ", style="dim")
        title.append(f"  Döngü: #{self._data['cycle']}  ", style="cyan")
        title.append(f"  Durum: {self._data['status']}  ",
                      style="green" if self._data["status"] == "Çalışıyor" else "yellow")
        return Panel(title, style="blue")

    def _render_signals(self) -> Panel:
        table = Table(title="Son Sinyaller", show_lines=True, expand=True)
        table.add_column("Maç", style="cyan", ratio=3)
        table.add_column("Pazar", ratio=1)
        table.add_column("Seçim", ratio=1)
        table.add_column("Oran", justify="right", ratio=1)
        table.add_column("Stake", justify="right", ratio=1)
        table.add_column("EV", justify="right", ratio=1)
        table.add_column("Güven", justify="right", ratio=1)

        for sig in self._data.get("signals", [])[:15]:
            ev = sig.get("ev", 0)
            ev_style = "green" if ev > 0 else "red"
            conf = sig.get("confidence", 0)
            conf_style = "green" if conf > 0.6 else "yellow" if conf > 0.4 else "red"

            table.add_row(
                sig.get("match_id", "")[:25],
                sig.get("market", "1X2"),
                sig.get("selection", "-"),
                f"{sig.get('odds', 0):.2f}",
                f"{sig.get('stake_pct', 0):.3%}",
                f"[{ev_style}]{ev:.3f}[/]",
                f"[{conf_style}]{conf:.1%}[/]",
            )

        return Panel(table, title="Sinyaller", border_style="cyan")

    def _render_metrics(self) -> Panel:
        table = Table(show_header=False, expand=True, padding=(0, 1))
        table.add_column("Metrik", style="bold cyan")
        table.add_column("Değer", justify="right")

        bankroll = self._data["bankroll"]
        dd = self._data["drawdown"]
        risk = self._data["risk_level"]

        risk_color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(risk, "white")

        table.add_row("Kasa", f"₺{bankroll:,.2f}")
        table.add_row("Aktif Bahis", str(self._data["active_bets"]))
        table.add_row("Kazanma Oranı", f"{self._data['win_rate']:.1%}")
        table.add_row("Sharpe Ratio", f"{self._data['sharpe']:.2f}")
        table.add_row("Drawdown", f"[{'red' if dd < -0.05 else 'yellow'}]{dd:.2%}[/]")
        table.add_row("Risk Seviyesi", f"[{risk_color}]{risk.upper()}[/]")

        return Panel(table, title="Portföy Metrikleri", border_style="green")

    def _render_risk_panel(self) -> Panel:
        lines = []
        risk = self._data["risk_level"]

        bar_map = {"low": 2, "medium": 5, "high": 8, "critical": 10}
        bar_len = bar_map.get(risk, 3)
        bar = "█" * bar_len + "░" * (10 - bar_len)

        lines.append(f"Risk: [{bar}]")
        lines.append(f"DD Limit: 10%")
        lines.append(f"Max Tekli: 5%")
        lines.append(f"Max Toplam: 20%")
        lines.append("")
        lines.append(f"Döngü: {self._data['cycle']}")

        return Panel("\n".join(lines), title="Risk Göstergesi", border_style="red")

    def _render_footer(self) -> Panel:
        text = Text()
        text.append(" [Q] Çık ", style="bold white on red")
        text.append("  [R] Rapor ", style="bold white on blue")
        text.append("  [B] Backtest ", style="bold white on green")
        text.append("  [D] Doktor ", style="bold white on magenta")
        return Panel(text, style="dim")

    async def run(self, shutdown: asyncio.Event):
        """Canlı dashboard döngüsü."""
        logger.info("TUI Dashboard başlatılıyor…")
        layout = self._build_layout()

        with Live(layout, console=self._console, refresh_per_second=2, screen=True) as live:
            while not shutdown.is_set():
                layout["header"].update(self._render_header())
                layout["signals"].update(self._render_signals())
                layout["metrics"].update(self._render_metrics())
                layout["right"].update(self._render_risk_panel())
                layout["footer"].update(self._render_footer())
                await asyncio.sleep(0.5)
