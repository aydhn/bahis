"""
dashboard_mini_app.py – Streamlit-Tabanlı Gerçek Zamanlı Dashboard.

Bu dosya, Telegram Mini App içinde veya harici bir browser'da 
çalışabilecek görsel bir kontrol paneli sunar.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
from datetime import datetime

st.set_page_config(page_title="Quant Betting Dashboard", page_icon="📈", layout="wide")

def load_data():
    con = duckdb.connect("data/bahis.duckdb")
    signals = con.execute("SELECT * FROM signals").df()
    matches = con.execute("SELECT * FROM matches").df()
    con.close()
    return signals, matches

st.title("🚀 Quant Betting Bot - Real-Time Dashboard")

try:
    signals, matches = load_data()
    
    # Sidebar Metrics
    st.sidebar.header("Portföy Özet")
    total_signals = len(signals)
    avg_ev = signals['ev'].mean() if not signals.empty else 0
    st.sidebar.metric("Toplam Sinyal", total_signals)
    st.sidebar.metric("Ortalama EV", f"%{avg_ev*100:.1f}")
    
    # PnL Chart
    st.subheader("Profit & Loss (PnL) Serisi")
    if not signals.empty:
        # PnL verisi olmadığı için mock bir seri oluşturuyoruz
        chart_data = pd.DataFrame({
            "Bahis No": range(len(signals)),
            "Kümülatif PnL": signals['ev'].cumsum() * 100
        })
        fig = px.line(chart_data, x="Bahis No", y="Kümülatif PnL", title="PnL Trendi")
        st.plotly_chart(fig, use_container_width=True)
    
    # Active Signals Table
    st.subheader("Aktif Value Sinyalleri")
    st.dataframe(signals.tail(10), use_container_width=True)
    
    # Match Volatility Analysis
    st.subheader("Maç Volatilite Analizi")
    if not matches.empty:
        fig_vol = px.histogram(matches, x="home_score", title="Gol Dağılımı")
        st.plotly_chart(fig_vol, use_container_width=True)

except Exception as e:
    st.error(f"Veri yükleme hatası: {e}")
    st.info("Sistem henüz veri üretmemiş olabilir.")

if st.button("🔄 Veriyi Yenile"):
    st.rerun()
