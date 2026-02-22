
import asyncio
import numpy as np
from src.quant.portfolio_optimizer import PortfolioOptimizer
from src.ingestion.news_rag import NewsRAGAnalyzer
from src.quant.anomaly_detector import AnomalyDetector
import polars as pl
import pandas as pd # Legacy compatibility

async def test_portfolio():
    print("\n[TEST] Portfolio Optimizer...")
    optimizer = PortfolioOptimizer()
    
    # 3 Bahis Senaryosu:
    # 1. Yüksek EV, Düşük Olasılık (Riskli)
    # 2. Orta EV, Orta Olasılık
    # 3. Düşük EV, Yüksek Olasılık (Güvenli)
    bets = [
        {"id": "match_risky", "ev": 0.20, "prob": 0.3},
        {"id": "match_mid",   "ev": 0.08, "prob": 0.5},
        {"id": "match_safe",  "ev": 0.04, "prob": 0.8},
    ]
    
    res = optimizer.optimize(bets)
    print(f"Weights: {res.weights}")
    print(f"Sharpe: {res.sharpe_ratio:.2f}")
    
    if res.status == "success" and len(res.weights) > 0:
        print("✅ Portfolio Optimization SUCCESS")
        return True
    else:
        print("❌ Portfolio Optimization FAILED")
        return False

async def test_news_rag():
    print("\n[TEST] News RAG (Free Mode)...")
    rag = NewsRAGAnalyzer() # No API keys, should default to rule-based
    
    # Mock fetching news (to avoid network call waiting or failure)
    # But let's try a real fetch for one team if possible, or just test the fallback logic
    # We will test the ANALYSIS logic with mock news items
    from src.ingestion.news_rag import NewsItem
    
    mock_news = [
        NewsItem(title="Galatasaray derbiyi muhteşem bir oyunla kazandı", snippet="Harika performans."),
        NewsItem(title="Icardi sakatlandı, sezonu kapattı", snippet="Büyük şok."),
        NewsItem(title="Yönetim istifa sesleri", snippet="Kriz büyüyor.")
    ]
    
    # "Galatasaray" için analiz
    # 1 pozitif, 2 negatif -> Skor < 0.5 olmalı
    res = await rag._analyze_with_llm("Galatasaray", mock_news)
    print(f"Sentiment: {res.sentiment_score:.2f} ({res.method})")
    
    if res.sentiment_score < 0.5 and res.method == "rule_based":
        print("✅ News RAG (Rule Based) SUCCESS")
        return True
    else:
        print(f"❌ News RAG FAILED or Logic Error (Score: {res.sentiment_score})")
        return False

def test_anomaly():
    print("\n[TEST] Anomaly Detector...")
    detector = AnomalyDetector(z_threshold=2.0)
    
    # Create synthetic odds history: Stable then Drop
    odds = [2.0, 2.0, 2.05, 1.95, 2.0, 2.0] * 5 # Stable
    odds.extend([1.8, 1.6, 1.5]) # Drop
    
    df = pl.DataFrame({
        "match_id": ["TEST"] * len(odds),
        "selection": ["1"] * len(odds),
        "odds": odds
    })
    
    alerts = detector.detect_dropping_odds(df)
    
    if len(alerts) > 0 and alerts[-1]["type"] == "DROPPING_ODDS":
        print(f"✅ Anomaly Detected: {alerts[-1]['pct_drop']:.1f}% drop")
        return True
    else:
        print("❌ Anomaly Detection FAILED")
        return False

async def main():
    p = await test_portfolio()
    n = await test_news_rag()
    a = test_anomaly()
    
    if p and n and a:
        print("\n🚀 ALL ADVANCED CAPABILITIES VERIFIED")
    else:
        print("\n⚠️ SOME TESTS FAILED")

if __name__ == "__main__":
    asyncio.run(main())
