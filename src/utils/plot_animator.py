"""
plot_animator.py – Hareketli Isı Haritaları ve GIF Üretici.

Statik resimler maçı anlatmaz. Gol olduğunda veya maç bittiğinde,
takımın baskı kurduğu bölgeleri gösteren hareketli bir GIF gelir.

İşlevler:
  1. Pitch Heatmap GIF: Topla oynama verisinden ısı haritası animasyonu
  2. Odds Movement GIF: Oran hareketlerinin zaman animasyonu
  3. Pressure Map GIF: Baskı bölgeleri animasyonu
  4. Bankroll History GIF: Kasa değişim animasyonu

Teknoloji: matplotlib.animation + imageio
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.patches import Arc, Circle
    MPL_OK = True
except ImportError:
    MPL_OK = False
    logger.info("matplotlib yüklü değil – animasyon devre dışı.")

try:
    import imageio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False
    logger.info("imageio yüklü değil – GIF oluşturulamaz.")

ROOT = Path(__file__).resolve().parent.parent.parent
GIF_DIR = ROOT / "output" / "gifs"
GIF_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  SAHA ÇİZİCİ
# ═══════════════════════════════════════════════
def draw_pitch(ax: Any, color: str = "white", linewidth: float = 1.5):
    """FIFA standart futbol sahası çizimi (105x68)."""
    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.set_facecolor("#1a472a")
    ax.set_aspect("equal")
    ax.axis("off")

    lc = color
    lw = linewidth

    # Saha çizgileri
    ax.plot([0, 105], [0, 0], color=lc, lw=lw)       # alt
    ax.plot([0, 105], [68, 68], color=lc, lw=lw)      # üst
    ax.plot([0, 0], [0, 68], color=lc, lw=lw)         # sol
    ax.plot([105, 105], [0, 68], color=lc, lw=lw)     # sağ
    ax.plot([52.5, 52.5], [0, 68], color=lc, lw=lw)   # orta çizgi

    # Orta daire
    ax.add_patch(Circle((52.5, 34), 9.15, ec=lc, fc="none", lw=lw))
    ax.plot(52.5, 34, "o", color=lc, ms=3)

    # Sol ceza sahası
    ax.plot([0, 16.5], [13.84, 13.84], color=lc, lw=lw)
    ax.plot([16.5, 16.5], [13.84, 54.16], color=lc, lw=lw)
    ax.plot([0, 16.5], [54.16, 54.16], color=lc, lw=lw)

    # Sağ ceza sahası
    ax.plot([88.5, 105], [13.84, 13.84], color=lc, lw=lw)
    ax.plot([88.5, 88.5], [13.84, 54.16], color=lc, lw=lw)
    ax.plot([88.5, 105], [54.16, 54.16], color=lc, lw=lw)

    # Sol kale alanı
    ax.plot([0, 5.5], [24.84, 24.84], color=lc, lw=lw)
    ax.plot([5.5, 5.5], [24.84, 43.16], color=lc, lw=lw)
    ax.plot([0, 5.5], [43.16, 43.16], color=lc, lw=lw)

    # Sağ kale alanı
    ax.plot([99.5, 105], [24.84, 24.84], color=lc, lw=lw)
    ax.plot([99.5, 99.5], [24.84, 43.16], color=lc, lw=lw)
    ax.plot([99.5, 105], [43.16, 43.16], color=lc, lw=lw)

    # Penaltı noktaları
    ax.plot(11, 34, "o", color=lc, ms=2)
    ax.plot(94, 34, "o", color=lc, ms=2)

    # Korner bayrakları
    for cx, cy in [(0, 0), (0, 68), (105, 0), (105, 68)]:
        ax.add_patch(Arc((cx, cy), 2, 2, angle=0, theta1=0, theta2=90,
                         ec=lc, lw=lw * 0.8))


# ═══════════════════════════════════════════════
#  ISI HARİTASI ANİMASYON MOTORU
# ═══════════════════════════════════════════════
class PlotAnimator:
    """Hareketli grafik ve GIF üretici.

    Kullanım:
        animator = PlotAnimator()
        # Isı haritası GIF
        path = animator.create_heatmap_gif(
            "GS_FB", touch_data, duration=5
        )
        # Oran hareketi GIF
        path = animator.create_odds_gif("GS_FB", odds_timeline)
        # Telegram'a gönder
        await notifier.send_animation(path)
    """

    DEFAULT_FPS = 4
    DEFAULT_DURATION = 5  # saniye

    def __init__(self, output_dir: str | Path | None = None):
        self._output_dir = Path(output_dir) if output_dir else GIF_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("[PlotAnimator] Başlatıldı.")

    # ═══════════════════════════════════════════
    #  1. PITCH HEATMAP GIF
    # ═══════════════════════════════════════════
    def create_heatmap_gif(self, match_id: str,
                            touch_data: list[dict],
                            team: str = "home",
                            duration: float = 5.0,
                            fps: int = 4) -> Path | None:
        """Topla oynama verisinden ısı haritası animasyonu.

        touch_data: [
            {"x": 45.2, "y": 30.1, "minute": 5, "type": "pass"},
            {"x": 80.0, "y": 40.0, "minute": 10, "type": "shot"},
            ...
        ]
        """
        if not MPL_OK or not IMAGEIO_OK:
            logger.warning("[Animator] matplotlib veya imageio yüklü değil.")
            return None

        if not touch_data:
            return None

        n_frames = int(duration * fps)
        max_minute = max(t.get("minute", 0) for t in touch_data) or 90
        minutes_per_frame = max_minute / n_frames

        frames = []

        for frame_idx in range(n_frames):
            end_minute = (frame_idx + 1) * minutes_per_frame

            # Bu kareye kadar olan dokunuşlar
            touches = [
                t for t in touch_data
                if t.get("minute", 0) <= end_minute
            ]

            if not touches:
                continue

            x = [t.get("x", 0) for t in touches]
            y = [t.get("y", 0) for t in touches]

            fig, ax = plt.subplots(figsize=(10.5, 6.8))
            draw_pitch(ax)

            # Isı haritası
            if len(x) > 2:
                heatmap, xedges, yedges = np.histogram2d(
                    x, y, bins=[21, 14],
                    range=[[0, 105], [0, 68]],
                )
                ax.imshow(
                    heatmap.T, origin="lower",
                    extent=[0, 105, 0, 68],
                    cmap="YlOrRd", alpha=0.6,
                    interpolation="gaussian",
                )

            # Son 10 dokunuşu nokta olarak göster
            recent = touches[-10:]
            rx = [t.get("x", 0) for t in recent]
            ry = [t.get("y", 0) for t in recent]
            ax.scatter(rx, ry, c="cyan", s=30, zorder=5, alpha=0.8)

            ax.set_title(
                f"{match_id} | {team.upper()} | dk {int(end_minute)}'",
                color="white", fontsize=14, fontweight="bold",
                pad=10,
            )
            fig.patch.set_facecolor("#0d1117")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.v3.imread(buf))

        if not frames:
            return None

        output = self._output_dir / f"heatmap_{match_id}_{team}.gif"
        imageio.v3.imwrite(output, frames, duration=1.0 / fps, loop=0)
        logger.info(f"[Animator] Heatmap GIF: {output} ({len(frames)} kare)")
        return output

    # ═══════════════════════════════════════════
    #  2. ODDS MOVEMENT GIF
    # ═══════════════════════════════════════════
    def create_odds_gif(self, match_id: str,
                         odds_timeline: list[dict],
                         duration: float = 5.0,
                         fps: int = 4) -> Path | None:
        """Oran hareket animasyonu.

        odds_timeline: [
            {"timestamp": 1700000000, "home": 1.80, "draw": 3.50, "away": 4.20},
            ...
        ]
        """
        if not MPL_OK or not IMAGEIO_OK:
            return None

        if len(odds_timeline) < 3:
            return None

        n_frames = int(duration * fps)
        step = max(1, len(odds_timeline) // n_frames)
        frames = []

        for end_idx in range(2, len(odds_timeline), step):
            subset = odds_timeline[:end_idx + 1]
            times = list(range(len(subset)))
            home_odds = [d.get("home", 0) for d in subset]
            draw_odds = [d.get("draw", 0) for d in subset]
            away_odds = [d.get("away", 0) for d in subset]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")

            ax.plot(times, home_odds, "g-o", label="Ev", ms=3, lw=2)
            ax.plot(times, draw_odds, "y-s", label="Ber", ms=3, lw=2)
            ax.plot(times, away_odds, "r-^", label="Dep", ms=3, lw=2)

            ax.set_xlabel("Zaman", color="white")
            ax.set_ylabel("Oran", color="white")
            ax.set_title(
                f"{match_id} | Oran Hareketi",
                color="white", fontsize=14, fontweight="bold",
            )
            ax.legend(facecolor="#1a1a2e", edgecolor="white",
                      labelcolor="white")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, color="white")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.v3.imread(buf))

        if not frames:
            return None

        output = self._output_dir / f"odds_{match_id}.gif"
        imageio.v3.imwrite(output, frames, duration=1.0 / fps, loop=0)
        logger.info(f"[Animator] Odds GIF: {output} ({len(frames)} kare)")
        return output

    # ═══════════════════════════════════════════
    #  3. BASKI HARİTASI GIF
    # ═══════════════════════════════════════════
    def create_pressure_gif(self, match_id: str,
                              home_team: str, away_team: str,
                              events: list[dict],
                              duration: float = 5.0,
                              fps: int = 3) -> Path | None:
        """Baskı bölgeleri animasyonu.

        events: [
            {"minute": 5, "type": "shot", "team": "home", "x": 85, "y": 35},
            {"minute": 8, "type": "cross", "team": "away", "x": 20, "y": 55},
            ...
        ]
        """
        if not MPL_OK or not IMAGEIO_OK:
            return None

        if not events:
            return None

        n_frames = int(duration * fps)
        max_min = max(e.get("minute", 0) for e in events) or 90
        frames = []

        for frame_idx in range(n_frames):
            end_min = (frame_idx + 1) * (max_min / n_frames)
            window_start = max(0, end_min - 15)

            # Son 15 dakikadaki olaylar
            window = [
                e for e in events
                if window_start <= e.get("minute", 0) <= end_min
            ]

            home_events = [e for e in window if e.get("team") == "home"]
            away_events = [e for e in window if e.get("team") == "away"]

            fig, ax = plt.subplots(figsize=(10.5, 6.8))
            draw_pitch(ax)

            # Ev sahibi baskı noktaları (yeşil)
            if home_events:
                hx = [e.get("x", 50) for e in home_events]
                hy = [e.get("y", 34) for e in home_events]
                ax.scatter(hx, hy, c="lime", s=80, alpha=0.6,
                           zorder=5, label=home_team)

                if len(hx) > 2:
                    heatmap, _, _ = np.histogram2d(
                        hx, hy, bins=[10, 7],
                        range=[[0, 105], [0, 68]],
                    )
                    ax.imshow(
                        heatmap.T, origin="lower",
                        extent=[0, 105, 0, 68],
                        cmap="Greens", alpha=0.3,
                        interpolation="gaussian",
                    )

            # Deplasman baskı noktaları (kırmızı)
            if away_events:
                ax_pts = [e.get("x", 50) for e in away_events]
                ay_pts = [e.get("y", 34) for e in away_events]
                ax.scatter(ax_pts, ay_pts, c="red", s=80, alpha=0.6,
                           zorder=5, label=away_team)

            ax.legend(loc="upper center", ncol=2,
                      facecolor="#1a1a2e", edgecolor="white",
                      labelcolor="white", fontsize=10)
            ax.set_title(
                f"{home_team} vs {away_team} | Baskı | dk {int(end_min)}'",
                color="white", fontsize=14, fontweight="bold",
                pad=10,
            )
            fig.patch.set_facecolor("#0d1117")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.v3.imread(buf))

        if not frames:
            return None

        output = self._output_dir / f"pressure_{match_id}.gif"
        imageio.v3.imwrite(output, frames, duration=1.0 / fps, loop=0)
        logger.info(f"[Animator] Pressure GIF: {output} ({len(frames)} kare)")
        return output

    # ═══════════════════════════════════════════
    #  4. BANKROLL GEÇMİŞ GIF
    # ═══════════════════════════════════════════
    def create_bankroll_gif(self, history: list[dict],
                              duration: float = 4.0,
                              fps: int = 4) -> Path | None:
        """Kasa değişim animasyonu.

        history: [
            {"date": "2026-01-01", "bankroll": 10000, "pnl": 0},
            {"date": "2026-01-02", "bankroll": 10150, "pnl": 150},
            ...
        ]
        """
        if not MPL_OK or not IMAGEIO_OK:
            return None

        if len(history) < 3:
            return None

        n_frames = int(duration * fps)
        step = max(1, len(history) // n_frames)
        frames = []

        for end_idx in range(2, len(history), step):
            subset = history[:end_idx + 1]
            dates = list(range(len(subset)))
            values = [d.get("bankroll", 10000) for d in subset]
            initial = values[0]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")

            # Renk: kâr → yeşil, zarar → kırmızı
            ax.fill_between(dates, initial, values, alpha=0.3,
                            color="#00ff88" if values[-1] >= initial else "#ff4444")
            ax.plot(dates, values, color="cyan", lw=2.5)
            ax.axhline(y=initial, color="white", lw=0.8, ls="--", alpha=0.5)

            # Son değer etiketi
            ax.annotate(
                f"  {values[-1]:,.0f}",
                xy=(dates[-1], values[-1]),
                fontsize=14, color="cyan", fontweight="bold",
            )

            roi = (values[-1] - initial) / initial
            ax.set_title(
                f"Kasa Geçmişi | ROI: {roi:+.1%}",
                color="white", fontsize=14, fontweight="bold",
            )
            ax.set_ylabel("TL", color="white")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.15, color="white")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.v3.imread(buf))

        if not frames:
            return None

        output = self._output_dir / "bankroll_history.gif"
        imageio.v3.imwrite(output, frames, duration=1.0 / fps, loop=0)
        logger.info(f"[Animator] Bankroll GIF: {output} ({len(frames)} kare)")
        return output

    # ═══════════════════════════════════════════
    #  TELEGRAM GÖNDERİM
    # ═══════════════════════════════════════════
    async def send_heatmap(self, match_id: str, touch_data: list[dict],
                            team: str, notifier: Any) -> bool:
        """Isı haritası GIF'ini Telegram'a gönder."""
        path = self.create_heatmap_gif(match_id, touch_data, team)
        if path and path.exists():
            try:
                await notifier.send_animation(str(path), caption=f"Isı Haritası: {match_id}")
                return True
            except Exception as e:
                logger.debug(f"[Animator] Telegram gönderim hatası: {e}")
        return False

    async def send_odds_animation(self, match_id: str,
                                    odds_timeline: list[dict],
                                    notifier: Any) -> bool:
        """Oran hareketi GIF'ini Telegram'a gönder."""
        path = self.create_odds_gif(match_id, odds_timeline)
        if path and path.exists():
            try:
                await notifier.send_animation(str(path), caption=f"Oran Hareketi: {match_id}")
                return True
            except Exception as e:
                logger.debug(f"[Animator] Telegram gönderim hatası: {e}")
        return False
