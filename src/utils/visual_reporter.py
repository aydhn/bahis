"""
visual_reporter.py – Görsel Raporlama Aracı.

Amacı: Telegram'a sadece metin değil, "Hawkeye" tarzı görsel analizler atmak.
Teknoloji: Matplotlib (Headless backend).

Çıktılar:
- PnL Grafiği (Kasa değişimi)
- Maç Olasılık Pastası
- Momentum Grafiği (Hibrit modelden gelen)
"""
import matplotlib
matplotlib.use('Agg') # GUI yok, arka plan modu
import matplotlib.pyplot as plt
import io
import numpy as np

class VisualReporter:
    def __init__(self):
        plt.style.use('dark_background') # Cyberpunk havası
        
    def generate_pnl_chart(self, pnl_history: list[float], dates: list[str]) -> bytes:
        """Kasa gelişim grafiği oluşturur ve bytes döndürür."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Veri yoksa dummy
        if not pnl_history:
            pnl_history = [1000, 1050, 1020, 1100]
            dates = ["Mon", "Tue", "Wed", "Thu"]
            
        x = np.arange(len(pnl_history))
        y = np.array(pnl_history)
        
        # Renk: Kazançsa yeşil, kayıpsa kırmızı
        color = '#00ff00' if y[-1] >= y[0] else '#ff0000'
        
        ax.plot(x, y, color=color, linewidth=2, marker='o')
        ax.fill_between(x, y, y.min(), color=color, alpha=0.1)
        
        ax.set_title("💰 Bankroll Evolution", fontsize=14, color='white', fontweight='bold')
        ax.set_xlabel("Bets", color='gray')
        ax.set_ylabel("Currency", color='gray')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Buffer'a kaydet
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#121212')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def generate_momentum_chart(self, home: str, away: str, 
                              home_probs: list[float], away_probs: list[float]) -> bytes:
        """Maç içi gol ihtimali (Momentum) grafiği."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        x = np.arange(10, 100, 10) # 10, 20, ..., 90
        
        ax.plot(x, home_probs, label=home, color='#00ffff', linewidth=2)
        ax.plot(x, away_probs, label=away, color='#ff00ff', linewidth=2)
        
        ax.set_title(f"🔥 Goal Momentum: {home} vs {away}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Minute")
        ax.set_ylabel("Goal Probability (10m)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#121212')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
