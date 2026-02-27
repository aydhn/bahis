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
    def generate_gods_eye_radar(
        match_id: str,
        value_score: float,
        physics_score: float,
        narrative_score: float,
        market_score: float,
        risk_score: float
    ) -> Optional[io.BytesIO]:
        """
        Generates the 'God's Eye' Radar Chart covering 5 system pillars.
        Scores should be 0.0 to 1.0.
        """
        try:
            # Labels and values
            labels = np.array(['Value (Kelly)', 'Physics (Chaos)', 'Narrative (Teleology)', 'Market (Smart Money)', 'Risk (Safety)'])
            values = np.array([value_score, physics_score, narrative_score, market_score, risk_score])

            # Close the loop
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            # Draw plot
            ax.plot(angles, values, color='#3498db', linewidth=2, linestyle='solid')
            ax.fill(angles, values, color='#3498db', alpha=0.4)

            # Customization
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)

            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

            plt.title(f"God's Eye Analysis: {match_id}", size=14, y=1.1)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close('all')
            return buf

        except Exception as e:
            logger.error(f"Radar chart generation failed: {e}")
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
