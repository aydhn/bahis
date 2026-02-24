import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from typing import Optional, List
from loguru import logger

class Visualizer:
    """
    Generates Visual Intelligence for the 'War Room'.
    Creates PNG charts from data distributions.
    """

    @staticmethod
    def generate_value_chart(
        home_team: str,
        away_team: str,
        model_probs: List[float],
        market_probs: List[float]
    ) -> Optional[io.BytesIO]:
        """
        Generates a bar chart comparing Model vs Market probabilities.
        Returns a BytesIO buffer containing the PNG image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")

            labels = ['Home', 'Draw', 'Away']
            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, model_probs, width, label='Model (AI)', color='#2ecc71')
            rects2 = ax.bar(x + width/2, market_probs, width, label='Market (Bookie)', color='#e74c3c')

            ax.set_ylabel('Probability')
            ax.set_title(f'Value Analysis: {home_team} vs {away_team}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            # Add value tags
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close('all')
            return buf

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

    @staticmethod
    def generate_dummy_chart() -> Optional[io.BytesIO]:
        """Generates a dummy chart for testing integration."""
        try:
            # Create dummy data
            x = np.linspace(0, 10, 100)
            y = np.sin(x)

            plt.figure(figsize=(8, 4))
            plt.plot(x, y, label='PnL Simulation', color='blue')
            plt.title('Bankroll Simulation (Dummy)')
            plt.xlabel('Time')
            plt.ylabel('PnL')
            plt.legend()
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close('all')
            return buf
        except Exception as e:
            logger.error(f"Dummy chart generation failed: {e}")
            return None
