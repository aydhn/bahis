"""
telegram_chart_sender.py – Telegram'a grafik/görsel gönderme motoru.

Metin yerine analiz grafiklerini resim olarak gönderir:
- Gol beklentisi dağılımı (Poisson/Dixon-Coles heatmap)
- Son 10 maç trendi (form çizgisi)
- Kasa eğrisi (bankroll curve)
- CLV trend grafiği
- Korelasyon matrisi heatmap

matplotlib → BytesIO buffer → Telegram sendPhoto
"""
from __future__ import annotations

import io
from pathlib import Path

from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("matplotlib yüklü değil – grafik gönderimi devre dışı.")

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


# ── Tema ayarları ──
DARK_BG = "#1a1a2e"
ACCENT = "#e94560"
TEXT_COLOR = "#eaeaea"
GRID_COLOR = "#333355"


def _apply_dark_theme():
    """Profesyonel koyu tema uygula."""
    if not MPL_AVAILABLE:
        return
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.size": 10,
        "figure.dpi": 150,
    })


class TelegramChartSender:
    """Telegram'a matplotlib grafikleri gönderen modül."""

    def __init__(self, notifier=None):
        self._notifier = notifier
        _apply_dark_theme()
        logger.debug("TelegramChartSender başlatıldı.")

    async def send_chart(self, fig, caption: str = "") -> bool:
        """matplotlib Figure'ü Telegram'a gönderir."""
        if not MPL_AVAILABLE or not self._notifier:
            return False

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=DARK_BG, edgecolor="none")
        buf.seek(0)
        plt.close(fig)

        try:
            if not self._notifier._ready or not self._notifier._bot:
                logger.debug(f"Chart DEMO: {caption[:50]}")
                return False

            await self._notifier._bot.send_photo(
                chat_id=self._notifier._chat_id,
                photo=buf,
                caption=caption,
                parse_mode="HTML",
            )
            return True
        except Exception as e:
            logger.error(f"Grafik gönderim hatası: {e}")
            return False

    # ═══════════════════════════════════════════
    #  GOL BEKLENTİSİ HEATMAP
    # ═══════════════════════════════════════════
    async def send_score_heatmap(self, score_matrix: "np.ndarray",
                                 home: str, away: str) -> bool:
        """Dixon-Coles / Poisson skor olasılık matrisi heatmap."""
        if not MPL_AVAILABLE or not NP_AVAILABLE:
            return False

        fig, ax = plt.subplots(figsize=(8, 6))
        n = min(score_matrix.shape[0], 7)
        data = score_matrix[:n, :n] * 100

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

        # Etiketler
        for i in range(n):
            for j in range(n):
                val = data[i][j]
                color = "white" if val > data.max() * 0.6 else TEXT_COLOR
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n))
        ax.set_yticklabels(range(n))
        ax.set_xlabel(f"{away} Gol", fontsize=12)
        ax.set_ylabel(f"{home} Gol", fontsize=12)
        ax.set_title(f"Skor Olasılıkları: {home} vs {away}", fontsize=14,
                     fontweight="bold", pad=15)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Olasılık (%)", color=TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

        caption = f"📊 <b>{home} vs {away}</b> – Skor Olasılık Matrisi"
        return await self.send_chart(fig, caption)

    # ═══════════════════════════════════════════
    #  FORM TRENDİ (Son N Maç)
    # ═══════════════════════════════════════════
    async def send_form_trend(self, team: str,
                              results: list[dict]) -> bool:
        """Son N maç form trendi çizgi grafiği.

        results: [{goals_for, goals_against, opponent, date}, ...]
        """
        if not MPL_AVAILABLE or not results:
            return False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])

        n = len(results)
        x = range(n)

        gf = [r.get("goals_for", 0) for r in results]
        ga = [r.get("goals_against", 0) for r in results]

        # Gol grafiği
        ax1.bar([i - 0.15 for i in x], gf, width=0.3, color="#4CAF50",
                label="Atılan", alpha=0.8)
        ax1.bar([i + 0.15 for i in x], ga, width=0.3, color="#F44336",
                label="Yenilen", alpha=0.8)
        ax1.set_ylabel("Gol")
        ax1.set_title(f"{team} – Son {n} Maç Performansı", fontsize=13,
                      fontweight="bold")
        ax1.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_COLOR)
        ax1.grid(axis="y", alpha=0.3)

        # Kümülatif puan
        points = [3 if g > ga[i] else 1 if g == ga[i] else 0 for i, g in enumerate(gf)]
        cum_pts = np.cumsum(points)
        ax2.plot(x, cum_pts, color=ACCENT, linewidth=2.5, marker="o", markersize=5)
        ax2.fill_between(x, cum_pts, alpha=0.15, color=ACCENT)
        ax2.set_ylabel("Kümülatif Puan")
        ax2.set_xlabel("Maç #")
        ax2.grid(axis="both", alpha=0.3)

        labels = [r.get("opponent", f"M{i+1}")[:8] for i, r in enumerate(results)]
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        fig.tight_layout()
        caption = f"📈 <b>{team}</b> – Son {n} Maç Formu"
        return await self.send_chart(fig, caption)

    # ═══════════════════════════════════════════
    #  KASA EĞRİSİ (Bankroll Curve)
    # ═══════════════════════════════════════════
    async def send_bankroll_curve(self, pnl_history: list[float],
                                  initial_bankroll: float = 10000) -> bool:
        """Kasa eğrisi grafiği."""
        if not MPL_AVAILABLE or not pnl_history:
            return False

        fig, ax = plt.subplots(figsize=(10, 5))

        bankroll = [initial_bankroll + sum(pnl_history[:i+1]) for i in range(len(pnl_history))]
        x = range(len(bankroll))

        ax.plot(x, bankroll, color=ACCENT, linewidth=2)
        ax.axhline(y=initial_bankroll, color=GRID_COLOR, linestyle="--", alpha=0.5,
                   label=f"Başlangıç: ₺{initial_bankroll:,.0f}")
        ax.fill_between(x, initial_bankroll, bankroll, alpha=0.15, color=ACCENT)

        current = bankroll[-1]
        pnl_total = current - initial_bankroll
        roi = pnl_total / initial_bankroll * 100

        ax.set_title(f"Kasa Eğrisi | ₺{current:,.0f} | ROI: {roi:+.1f}%",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Bahis #")
        ax.set_ylabel("Kasa (₺)")
        ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₺{x:,.0f}"))

        fig.tight_layout()
        caption = f"🏦 <b>Kasa Eğrisi</b> | ROI: {roi:+.1f}%"
        return await self.send_chart(fig, caption)

    # ═══════════════════════════════════════════
    #  CLV TREND
    # ═══════════════════════════════════════════
    async def send_clv_trend(self, clv_series: list[float]) -> bool:
        """Kümülatif CLV trend grafiği."""
        if not MPL_AVAILABLE or not clv_series:
            return False

        fig, ax = plt.subplots(figsize=(10, 5))

        cumulative = np.cumsum(clv_series)
        x = range(len(cumulative))

        ax.plot(x, cumulative, color="#2196F3", linewidth=2, label="Kümülatif CLV")
        ax.axhline(y=0, color=GRID_COLOR, linestyle="--", alpha=0.5)
        ax.fill_between(x, 0, cumulative,
                        where=[c >= 0 for c in cumulative], alpha=0.15, color="#4CAF50")
        ax.fill_between(x, 0, cumulative,
                        where=[c < 0 for c in cumulative], alpha=0.15, color="#F44336")

        avg_clv = float(np.mean(clv_series))
        ax.set_title(f"CLV Trend | Ort: {avg_clv:+.3f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Bahis #")
        ax.set_ylabel("Kümülatif CLV")
        ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        caption = f"📊 <b>Closing Line Value Trend</b> | Ort CLV: {avg_clv:+.3f}"
        return await self.send_chart(fig, caption)

    # ═══════════════════════════════════════════
    #  KORELASYON MATRİSİ
    # ═══════════════════════════════════════════
    async def send_correlation_heatmap(self, matrix: "np.ndarray",
                                       labels: list[str]) -> bool:
        """Bahis korelasyon matrisi heatmap."""
        if not MPL_AVAILABLE or not NP_AVAILABLE:
            return False

        fig, ax = plt.subplots(figsize=(8, 6))
        n = min(len(labels), matrix.shape[0])

        im = ax.imshow(matrix[:n, :n], cmap="RdYlGn_r", vmin=-1, vmax=1)

        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                color = "white" if abs(val) > 0.5 else TEXT_COLOR
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_labels = [l[:12] for l in labels[:n]]
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title("Bahis Korelasyon Matrisi", fontsize=13, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Korelasyon", color=TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

        fig.tight_layout()
        return await self.send_chart(fig, "🔗 <b>Bahis Korelasyon Matrisi</b>")
