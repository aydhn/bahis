"""
sentiment_analyzer.py – NLP ile duygu analizi.
VADER + TextBlob kullanarak taraftar/medya duygusunu
sayısal veriye (-1 ile +1) dönüştürür.
Kaynak: Twitter (X) anahtar kelimeler, haber siteleri.
"""
from __future__ import annotations

import asyncio
import re

import numpy as np
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment yüklü değil.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("textblob yüklü değil.")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Türkçe futbol duygu sözlüğü (VADER'ı desteklemek için)
TR_SENTIMENT_LEXICON = {
    # Pozitif
    "galibiyet": 2.0, "zafer": 2.5, "şampiyon": 3.0, "harika": 2.0,
    "mükemmel": 2.5, "kazandı": 1.5, "gol": 1.0, "form": 1.0,
    "yıldız": 1.0, "başarı": 2.0, "güçlü": 1.5, "muhteşem": 2.0,
    "iyi": 1.0, "süper": 1.5, "kadro": 0.5, "hazır": 0.5,
    # Negatif
    "mağlubiyet": -2.0, "yenilgi": -2.0, "sakatlık": -2.5, "sakatlandı": -2.5,
    "ceza": -1.5, "kırmızı kart": -2.0, "cezalı": -2.0, "kadro dışı": -2.5,
    "kötü": -1.5, "kriz": -2.0, "kavga": -1.5, "istifa": -2.0,
    "transfer yasağı": -2.5, "hoca krizi": -2.0, "tribün kapama": -1.5,
    "sakat": -2.0, "revir": -1.5, "tehlike": -1.0, "düşüş": -1.5,
    "yenildi": -1.5, "kaybetti": -1.5,
}


class SentimentAnalyzer:
    """NLP tabanlı duygu analizi motoru."""

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        # Türkçe sözlüğü VADER'a ekle
        if self._vader:
            self._vader.lexicon.update(TR_SENTIMENT_LEXICON)
        self._client: httpx.AsyncClient | None = None
        logger.debug("SentimentAnalyzer başlatıldı.")

    async def _ensure_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15, follow_redirects=True)

    def analyze_text(self, text: str) -> dict:
        """Tek bir metni analiz eder. Çıktı: -1 (negatif) ile +1 (pozitif)."""
        scores = []

        if self._vader:
            vader_scores = self._vader.polarity_scores(text)
            scores.append(vader_scores["compound"])

        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                scores.append(blob.sentiment.polarity)
            except Exception as e:
                logger.debug(f"Exception caught: {e}")

        # Türkçe keyword tabanlı analiz (her zaman çalışır)
        tr_score = self._turkish_sentiment(text)
        scores.append(tr_score)

        if not scores:
            return {"sentiment": 0.0, "confidence": 0.0}

        avg = float(np.mean(scores))
        std = float(np.std(scores)) if len(scores) > 1 else 0.5
        confidence = max(0.1, 1 - std)

        return {
            "sentiment": float(np.clip(avg, -1, 1)),
            "confidence": float(np.clip(confidence, 0, 1)),
            "vader": scores[0] if self._vader else None,
            "textblob": scores[1] if TEXTBLOB_AVAILABLE and len(scores) > 1 else None,
            "turkish": tr_score,
        }

    def _turkish_sentiment(self, text: str) -> float:
        """Türkçe futbol sözlüğü ile duygu analizi."""
        text_lower = text.lower()
        total_score = 0.0
        matches = 0

        for keyword, score in TR_SENTIMENT_LEXICON.items():
            if keyword in text_lower:
                total_score += score
                matches += 1

        if matches == 0:
            return 0.0
        return float(np.clip(total_score / (matches * 2), -1, 1))

    def analyze_batch(self, texts: list[str]) -> dict:
        """Birden fazla metni analiz edip özet çıkarır."""
        if not texts:
            return {"sentiment": 0.0, "confidence": 0.0, "count": 0}

        sentiments = [self.analyze_text(t)["sentiment"] for t in texts]
        return {
            "sentiment": float(np.mean(sentiments)),
            "std": float(np.std(sentiments)),
            "confidence": float(1 - np.std(sentiments)),
            "count": len(texts),
            "positive_pct": float(sum(1 for s in sentiments if s > 0.1) / len(sentiments)),
            "negative_pct": float(sum(1 for s in sentiments if s < -0.1) / len(sentiments)),
        }

    async def scrape_news_sentiment(self, team_name: str) -> dict:
        """Haber sitelerinden takım haberlerini çekip analiz eder."""
        await self._ensure_client()
        headlines = []

        # Google News (Türkçe) üzerinden basit arama
        try:
            search_url = f"https://news.google.com/rss/search?q={team_name}+futbol&hl=tr&gl=TR&ceid=TR:tr"
            resp = await self._client.get(search_url)
            if resp.status_code == 200:
                # XML parse
                items = re.findall(r"<title>(.*?)</title>", resp.text)
                headlines.extend(items[:20])
        except Exception as e:
            logger.debug(f"Google News hatası: {e}")

        if not headlines:
            return {"team": team_name, "sentiment": 0.0, "confidence": 0.0, "headlines": 0}

        analysis = self.analyze_batch(headlines)
        analysis["team"] = team_name
        analysis["headlines"] = len(headlines)
        logger.info(f"[Sentiment] {team_name}: {analysis['sentiment']:.2f} ({len(headlines)} haber)")
        return analysis

    def analyze_for_match(self, home_team: str, away_team: str) -> dict:
        """Maç bazında duygu analizi özeti."""
        # Senkron versiyon (async dışı kullanım için)
        return {
            "home_sentiment": 0.0,
            "away_sentiment": 0.0,
            "sentiment_edge": 0.0,
            "note": "Async scrape_news_sentiment() ile güncellenecek",
        }

    async def analyze_for_match_async(self, home_team: str, away_team: str) -> dict:
        """Maç bazında asenkron duygu analizi."""
        home_sent, away_sent = await asyncio.gather(
            self.scrape_news_sentiment(home_team),
            self.scrape_news_sentiment(away_team),
        )

        edge = home_sent.get("sentiment", 0) - away_sent.get("sentiment", 0)
        return {
            "home_sentiment": home_sent.get("sentiment", 0),
            "away_sentiment": away_sent.get("sentiment", 0),
            "sentiment_edge": float(np.clip(edge, -1, 1)),
            "home_headlines": home_sent.get("headlines", 0),
            "away_headlines": away_sent.get("headlines", 0),
        }
