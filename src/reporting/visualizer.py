import io
from typing import Optional, List, Dict, Any
from loguru import logger
import polars as pl
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg') # Server-side rendering
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not installed. Visualization disabled.")

class Visualizer:
    """
    Generates charts for Telegram reporting.
    (CEO Dashboard Vision)
    """

    @staticmethod
    def generate_pnl_chart(pnl_history: List[float], labels: List[str] = None) -> Optional[io.BytesIO]:
        """Generates a cumulative PnL chart."""
        if not HAS_MATPLOTLIB or not pnl_history:
            return None

        try:
            plt.figure(figsize=(10, 5))
            cumulative = np.cumsum(pnl_history)

            # Style
            plt.style.use('bmh') # Clean style
            plt.plot(cumulative, label='Cumulative PnL', color='#2ecc71', linewidth=2)
            plt.fill_between(range(len(cumulative)), cumulative, alpha=0.1, color='#2ecc71')

            plt.title("Bankroll Growth (PnL)", fontsize=14, fontweight='bold')
            plt.xlabel("Trades")
            plt.ylabel("PnL (TL)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()
            return buf
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

    @staticmethod
    def generate_dummy_chart() -> Optional[io.BytesIO]:
        """Generates a dummy chart for testing."""
        data = np.random.normal(0, 1, 100).cumsum()
        return Visualizer.generate_pnl_chart(list(data))
