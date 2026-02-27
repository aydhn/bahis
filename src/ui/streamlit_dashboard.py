"""
streamlit_dashboard.py – Streamlit tabanlı web dashboard.
Canlı Radar, Value Finder ve Kasa Eğrisi görselleştirmesi.
Çalıştırma: streamlit run src/ui/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Proje kökünü PYTHONPATH'e ekle
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    # ── Sayfa Ayarları ──
    st.set_page_config(
        page_title="Quant Betting Bot",
        page_icon="🏟️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Bağlantılar ──
    @st.cache_resource
    def get_db():
        from src.memory.db_manager import DBManager

        return DBManager()

    @st.cache_resource
    def get_poisson():
        from src.quant.models.poisson_model import PoissonModel

        return PoissonModel()

    @st.cache_resource
    def get_monte_carlo():
        from src.quant.analysis.monte_carlo_engine import MonteCarloEngine

        return MonteCarloEngine()

    @st.cache_resource
    def get_elo():
        from src.quant.models.elo_glicko_rating import EloGlickoSystem

        return EloGlickoSystem()

    @st.cache_resource
    def get_anomaly():
        from src.quant.analysis.anomaly_detector import AnomalyDetector

        return AnomalyDetector()

    # ── CSS ──
    st.markdown(
        """
    <style>
    .main {background-color: #0e1117;}
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #00d4ff;
    }
    .value-positive {color: #00ff88; font-weight: bold; font-size: 1.2em;}
    .value-negative {color: #ff4444; font-weight: bold; font-size: 1.2em;}
    .stMetric {background: #1a1a2e; border-radius: 8px; padding: 10px;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    st.sidebar.title("🏟️ Quant Betting Bot")
    page = st.sidebar.radio(
        "Modül Seçin",
        [
            "Dashboard",
            "Value Finder",
            "Maç Analizi",
            "Kasa Eğrisi",
            "Canlı Radar",
            "Anomali Tespiti",
        ],
    )

    # 🎨 Palette: Refresh control for data freshness
    if st.sidebar.button("🔄 Verileri Yenile", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.caption(f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")

    # ═══════════════════════════════════════════════════
    # DASHBOARD
    # ═══════════════════════════════════════════════════
    if page == "Dashboard":
        st.title("📊 Ana Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Kasa", "₺10,000", "+2.3%", help="Toplam bakiye ve günlük değişim."
            )
        with col2:
            st.metric(
                "Aktif Bahis",
                "3",
                "+1",
                help="Şu anda açık olan (sonuçlanmamış) bahis sayısı.",
            )
        with col3:
            st.metric(
                "Günlük ROI",
                "+1.8%",
                "0.5%",
                help="Günlük Yatırım Getirisi (Return on Investment).",
            )
        with col4:
            st.metric(
                "Sharpe Ratio",
                "1.45",
                "+0.12",
                help="Riske göre düzeltilmiş getiri performansı. >1 iyidir, >2 mükemmeldir.",
            )

        st.divider()

        # Son sinyaller
        st.subheader("🎯 Son Sinyaller")
        db = get_db()
        signals = db.get_signals()

        if not signals.is_empty():
            st.dataframe(
                signals.select(
                    [
                        "match_id",
                        "market",
                        "selection",
                        "odds",
                        "stake_pct",
                        "confidence",
                        "ev",
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Henüz sinyal üretilmedi. Botu çalıştırın: `python bahis.py run`")

    # ═══════════════════════════════════════════════════
    # VALUE FINDER
    # ═══════════════════════════════════════════════════
    elif page == "Value Finder":
        st.title("💎 Value Finder")
        st.caption("Modelimizin oranı ile İddaa oranı arasındaki farkı gösteren tablo")

        with st.expander("ℹ️ Renk Kodları ve EV Nedir?"):
            st.markdown("""
            **EV (Beklenen Değer):** Bahisin uzun vadede ne kadar kârlı olacağını gösterir.

            *   🟢 **Koyu Yeşil:** EV > %5 (Yüksek Değer - Önerilen)
            *   🟢 **Açık Yeşil:** EV > %0 (Pozitif Değer)
            *   🔴 **Kırmızı:** EV < %0 (Negatif Değer - Uzak Durun)

            **Formül:** `(Model Olasılığı × Bahis Oranı) - 1`
            """)

        # 🎨 Palette: Toggle to focus only on opportunities
        show_only_value = st.toggle("Sadece Fırsatları Göster (Value > 0)", value=False)

        db = get_db()
        matches = db.get_upcoming_matches()

        if not matches.is_empty():
            poisson = get_poisson()
            rows = []
            for match_row in matches.iter_rows(named=True):
                mid = match_row.get("match_id", "")
                home = match_row.get("home_team", "")
                away = match_row.get("away_team", "")
                ho = match_row.get("home_odds", 2.5) or 2.5
                do_ = match_row.get("draw_odds", 3.3) or 3.3
                ao = match_row.get("away_odds", 3.0) or 3.0

                home_xg = match_row.get("home_xg", 1.3) or 1.3
                away_xg = match_row.get("away_xg", 1.1) or 1.1

                probs = poisson.match_outcome_probs(home_xg, away_xg)

                # Model oranları (fair odds)
                fair_home = 1 / max(probs["prob_home"], 0.01)
                fair_draw = 1 / max(probs["prob_draw"], 0.01)
                fair_away = 1 / max(probs["prob_away"], 0.01)

                # EV hesaplama
                ev_home = probs["prob_home"] * ho - 1
                ev_draw = probs["prob_draw"] * do_ - 1
                ev_away = probs["prob_away"] * ao - 1

                rows.append(
                    {
                        "Maç": f"{home} vs {away}",
                        "İddaa 1": ho,
                        "Model 1": round(fair_home, 2),
                        "EV 1": round(ev_home, 3),
                        "İddaa X": do_,
                        "Model X": round(fair_draw, 2),
                        "EV X": round(ev_draw, 3),
                        "İddaa 2": ao,
                        "Model 2": round(fair_away, 2),
                        "EV 2": round(ev_away, 3),
                        "En İyi": max(
                            ["1", "X", "2"],
                            key=lambda x: {"1": ev_home, "X": ev_draw, "2": ev_away}[x],
                        ),
                        "Value?": "✅"
                        if max(ev_home, ev_draw, ev_away) > 0.02
                        else "❌",
                    }
                )

            import pandas as pd

            df = pd.DataFrame(rows)

            # 🎨 Palette: Apply filter if toggle is active
            if show_only_value:
                df = df[df["Value?"] == "✅"]

            if df.empty and show_only_value:
                st.info("Şu an için kriterlere uyan fırsat bulunamadı.")

            if not df.empty:
                # Renklendirme
                def color_ev(val):
                    if isinstance(val, (int, float)):
                        if val > 0.05:
                            return (
                                "background-color: #00ff88; color: black; font-weight: bold"
                            )
                        elif val > 0:
                            return "background-color: #88ff88; color: black"
                        else:
                            return "background-color: #ff8888; color: black"
                    return ""

                styled = df.style.applymap(color_ev, subset=["EV 1", "EV X", "EV 2"])
                styled = styled.format(
                    {
                        "EV 1": "{:.1%}",
                        "EV X": "{:.1%}",
                        "EV 2": "{:.1%}",
                        "İddaa 1": "{:.2f}",
                        "İddaa X": "{:.2f}",
                        "İddaa 2": "{:.2f}",
                        "Model 1": "{:.2f}",
                        "Model X": "{:.2f}",
                        "Model 2": "{:.2f}",
                    }
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

            # Value bahisleri filtrele (always calculate count from full rows for global context)
            value_bets = [r for r in rows if r["Value?"] == "✅"]
            if value_bets:
                st.success(f"🎯 {len(value_bets)} adet değer bahisi bulundu!")
            else:
                st.warning("Şu an değer bahisi eşiğini geçen maç yok.")
        else:
            st.info("Maç verisi yok. Scraper'ları çalıştırın.")

    # ═══════════════════════════════════════════════════
    # MAÇ ANALİZİ
    # ═══════════════════════════════════════════════════
    elif page == "Maç Analizi":
        st.title("🔬 Maç Analizi")

        col1, col2 = st.columns(2)
        with col1:
            home_xg = st.slider(
                "Ev Sahibi xG",
                0.5,
                3.5,
                1.4,
                0.1,
                help="Beklenen Gol (Expected Goals): Takımın girdiği gol pozisyonlarının kalitesi ve sayısı.",
            )
        with col2:
            away_xg = st.slider(
                "Deplasman xG",
                0.5,
                3.5,
                1.1,
                0.1,
                help="Beklenen Gol (Expected Goals): Takımın girdiği gol pozisyonlarının kalitesi ve sayısı.",
            )

        poisson = get_poisson()
        mc = get_monte_carlo()

        # Poisson skor matrisi
        st.subheader("📊 Skor Olasılık Matrisi (Poisson)")
        mat = poisson.score_matrix(home_xg, away_xg)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=mat[:6, :6] * 100,
                x=[f"Dep {i}" for i in range(6)],
                y=[f"Ev {i}" for i in range(6)],
                colorscale="YlOrRd",
                text=np.round(mat[:6, :6] * 100, 1),
                texttemplate="%{text}%",
                textfont={"size": 12},
            )
        )
        fig_heatmap.update_layout(title="Skor Olasılıkları (%)", height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Monte Carlo
        st.subheader("🎲 Monte Carlo Simülasyonu (10,000 maç)")
        sim = mc.simulate_match(home_xg, away_xg)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Ev Kazanır", f"{sim['prob_home']:.1%}", f"{sim['home_win_count']:,}"
            )
        with col2:
            st.metric("Beraberlik", f"{sim['prob_draw']:.1%}", f"{sim['draw_count']:,}")
        with col3:
            st.metric(
                "Dep Kazanır", f"{sim['prob_away']:.1%}", f"{sim['away_win_count']:,}"
            )

        # Top 10 skor
        st.subheader("🏆 En Olası Skorlar")
        top = sim["top_scores"]
        fig_bar = go.Figure(
            data=go.Bar(
                x=[s["score"] for s in top],
                y=[s["pct"] * 100 for s in top],
                marker_color=[
                    "#00d4ff" if s["pct"] == max(t["pct"] for t in top) else "#4a9eff"
                    for s in top
                ],
                text=[f"{s['pct']:.1%}" for s in top],
                textposition="auto",
            )
        )
        fig_bar.update_layout(title="En Olası Skorlar (%)", yaxis_title="%", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("📊 Bu simülasyon ne anlatıyor?"):
            st.markdown("""
            Bu simülasyon iki farklı yaklaşımı birleştirir:
            1.  **Poisson Dağılımı:** Takımların gol atma potansiyellerine göre saf olasılıkları hesaplar.
            2.  **Monte Carlo:** Maçı sanal ortamda 10,000 kez oynatarak şans faktörünü simüle eder.

            **Sonuç:** Poisson size "teorik" ihtimali, Monte Carlo ise "pratik" dağılımı gösterir. İkisi birbirini destekliyorsa güven artar.
            """)

    # ═══════════════════════════════════════════════════
    # KASA EĞRİSİ
    # ═══════════════════════════════════════════════════
    elif page == "Kasa Eğrisi":
        st.title("📈 Kasa Eğrisi (Kümülatif P&L)")

        # Demo veri (backtest'ten gelecek)
        np.random.seed(42)
        n_days = 180
        daily_returns = np.random.normal(0.002, 0.015, n_days)
        bankroll = 10000 * np.cumprod(1 + daily_returns)

        import pandas as pd

        dates = pd.date_range("2025-08-01", periods=n_days, freq="D")
        df_equity = pd.DataFrame({"Tarih": dates, "Kasa": bankroll})

        peak = np.maximum.accumulate(bankroll)
        drawdown = (bankroll - peak) / peak

        # Equity curve
        fig_equity = go.Figure()
        fig_equity.add_trace(
            go.Scatter(
                x=dates,
                y=bankroll,
                mode="lines",
                name="Kasa",
                line=dict(color="#00d4ff", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.1)",
            )
        )
        fig_equity.add_trace(
            go.Scatter(
                x=dates,
                y=peak,
                mode="lines",
                name="Peak",
                line=dict(color="#ffffff", width=1, dash="dot"),
            )
        )
        fig_equity.update_layout(
            title="Kasa Eğrisi",
            yaxis_title="₺",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Drawdown
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown * 100,
                mode="lines",
                name="Drawdown",
                line=dict(color="#ff4444", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,68,68,0.2)",
            )
        )
        fig_dd.update_layout(
            title="Drawdown (%)",
            yaxis_title="%",
            template="plotly_dark",
            height=250,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Metrikler
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roi = (bankroll[-1] - 10000) / 10000
            st.metric("Toplam ROI", f"{roi:.1%}")
        with col2:
            st.metric("Max Drawdown", f"{drawdown.min():.1%}")
        with col3:
            sharpe = (
                np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
            )
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with col4:
            st.metric("Final Kasa", f"₺{bankroll[-1]:,.0f}")

    # ═══════════════════════════════════════════════════
    # CANLI RADAR
    # ═══════════════════════════════════════════════════
    elif page == "Canlı Radar":
        st.title("📡 Canlı Maç Radar")
        st.caption("O an oynanan maçtaki baskı düzeyi (Momentum)")

        # Demo radar chart
        categories = [
            "Baskı",
            "Şut",
            "Korner",
            "Toplam Pas",
            "Pas İsabeti",
            "Hücum Etkinliği",
        ]

        home_vals = [72, 8, 5, 450, 85, 65]
        away_vals = [28, 4, 3, 310, 78, 42]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=home_vals + [home_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Ev Sahibi",
                fillcolor="rgba(0, 212, 255, 0.3)",
                line=dict(color="#00d4ff", width=2),
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=away_vals + [away_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Deplasman",
                fillcolor="rgba(255, 99, 71, 0.3)",
                line=dict(color="#ff6347", width=2),
            )
        )
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)"),
            template="plotly_dark",
            height=500,
            title="Maç Momentum Radarı",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Momentum çizgisi
        st.subheader("⚡ Momentum Zaman Serisi")
        minutes = list(range(1, 91))
        np.random.seed(123)
        momentum = np.cumsum(np.random.normal(0.5, 2, 90))
        momentum = np.clip(momentum, -30, 30)

        fig_mom = go.Figure()
        colors = ["#00d4ff" if m > 0 else "#ff6347" for m in momentum]
        fig_mom.add_trace(
            go.Bar(x=minutes, y=momentum, marker_color=colors, name="Momentum")
        )
        fig_mom.add_hline(y=0, line_dash="dash", line_color="white")
        fig_mom.update_layout(
            xaxis_title="Dakika",
            yaxis_title="Momentum (Ev+/Dep-)",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_mom, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # ANOMALİ TESPİTİ
    # ═══════════════════════════════════════════════════
    elif page == "Anomali Tespiti":
        st.title("🔍 Oran Anomalisi Tespiti")
        st.caption("Z-Score ile Dropping Odds ve Smart Money takibi")

        # Demo dropping odds
        st.subheader("📉 Dropping Odds Uyarıları")

        demo_alerts = [
            {
                "Maç": "GS vs FB",
                "Seçim": "1",
                "Oran": "1.85 → 1.62",
                "Z-Score": -2.8,
                "Düşüş": "-12.4%",
                "Seviye": "🔴 HIGH",
            },
            {
                "Maç": "BJK vs TS",
                "Seçim": "X",
                "Oran": "3.40 → 3.10",
                "Z-Score": -2.2,
                "Düşüş": "-8.8%",
                "Seviye": "🟡 MEDIUM",
            },
            {
                "Maç": "Ank vs Kny",
                "Seçim": "2",
                "Oran": "2.60 → 2.25",
                "Z-Score": -3.1,
                "Düşüş": "-13.5%",
                "Seviye": "🔴 HIGH",
            },
        ]
        import pandas as pd

        st.dataframe(
            pd.DataFrame(demo_alerts), use_container_width=True, hide_index=True
        )

        # Z-Score dağılımı
        st.subheader("📊 Oran Hareketi Z-Score Dağılımı")
        np.random.seed(77)
        z_scores = np.random.normal(0, 1.2, 200)
        anomalies = z_scores[np.abs(z_scores) > 2]

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(x=z_scores, nbinsx=40, name="Normal", marker_color="#4a9eff")
        )
        fig_hist.add_trace(
            go.Histogram(x=anomalies, nbinsx=20, name="Anomali", marker_color="#ff4444")
        )
        fig_hist.add_vline(
            x=-2, line_dash="dash", line_color="red", annotation_text="Z=-2"
        )
        fig_hist.add_vline(
            x=2, line_dash="dash", line_color="red", annotation_text="Z=+2"
        )
        fig_hist.update_layout(
            barmode="overlay",
            template="plotly_dark",
            height=350,
            title="Z-Score Dağılımı (Kırmızı = Anomali)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    # Streamlit yüklü değilse
    from loguru import logger

    logger.warning("Streamlit yüklü değil – dashboard devre dışı.")

    def launch():
        print("streamlit yüklü değil. Yüklemek için: pip install streamlit plotly")
