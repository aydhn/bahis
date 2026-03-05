"""
news_rag.py – RAG Destekli Haber Analizi (LLM Entegrasyonu).

Sayısal veriler tamam, peki ya metinler?
Teknik direktörün endişeli tonu, sakatlık haberleri, transfer dedikoduları…

Teknoloji:
  - HuggingFace Inference API
  - RAG: Retrieval-Augmented Generation
  - Son 24 saat haber başlıkları + tweetler → LLM özeti
  - Çıktı: 0-1 arası Sentiment Skoru → ensemble_stacking.py'ye sinyal

İşleyiş:
  1. Haberleri / tweetleri çek (RSS, haber siteleri)
  2. LLM ile özetle ve sentiment skorla
  3. Skoru ensemble modeline "news_sentiment" olarak gönder
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False


@dataclass
class NewsItem:
    """Tek bir haber/tweet."""
    title: str
    source: str = ""
    url: str = ""
    published: str = ""
    snippet: str = ""
    team_mentioned: str = ""
    raw_sentiment: float = 0.5  # Ön NLP skoru


@dataclass
class RAGResult:
    """RAG analiz sonucu."""
    team: str
    sentiment_score: float = 0.5   # 0=çok negatif, 1=çok pozitif
    summary: str = ""              # LLM özeti
    key_topics: list[str] = field(default_factory=list)
    confidence: float = 0.5
    n_sources: int = 0
    method: str = ""               # huggingface / rule_based


class NewsRAGAnalyzer:
    """RAG destekli haber analizi motoru.

    Kullanım:
        rag = NewsRAGAnalyzer()
        result = await rag.analyze_team("Galatasaray")
        print(result.sentiment_score)   # 0.72
        print(result.summary)           # "Icardi'nin sakatlığı endişe yaratıyor..."
    """
    HF_URL = "https://api-inference.huggingface.co/models/"

    # Haber kaynakları
    NEWS_SOURCES = {
        "sporx": "https://www.sporx.com/rss/futbol.xml",
        "fanatik": "https://www.fanatik.com.tr/rss/futbol",
        "mackolik": "https://www.mackolik.com/rss",
    }

    def __init__(self, hf_token: str = "",
                 hf_model: str = "facebook/bart-large-mnli",
                 ollama_url: str = "",
                 ollama_model: str = "llama3"):
        self._hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self._hf_model = hf_model
        self._ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self._ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3")
        self._cache: dict[str, tuple[float, RAGResult]] = {}
        self._cache_ttl = 3600.0  # 1 saat
        logger.debug(
            f"NewsRAG başlatıldı (Ollama='✓', "
            f"HF={'✓' if self._hf_token else '✗'})."
        )

    # ═══════════════════════════════════════════
    #  ANA ANALİZ
    # ═══════════════════════════════════════════
    async def analyze_team(self, team: str) -> RAGResult:
        """Takım hakkındaki haberleri topla ve LLM ile analiz et."""
        # Cache kontrolü
        cached = self._get_cache(team)
        if cached:
            return cached

        # 1. Haberleri topla
        news = await self._fetch_news(team)

        if not news:
            result = RAGResult(
                team=team, sentiment_score=0.5,
                summary="Haber bulunamadı.", confidence=0.1,
                method="no_data",
            )
            self._set_cache(team, result)
            return result

        # 2. Haberleri özetle ve sentiment üret
        result = await self._analyze_with_llm(team, news)
        self._set_cache(team, result)
        return result

    async def analyze_match(self, home: str, away: str) -> dict:
        """İki takım için karşılaştırmalı sentiment analizi."""
        home_result, away_result = await asyncio.gather(
            self.analyze_team(home),
            self.analyze_team(away),
        )

        diff = home_result.sentiment_score - away_result.sentiment_score

        return {
            "home_sentiment": home_result.sentiment_score,
            "away_sentiment": away_result.sentiment_score,
            "sentiment_diff": diff,
            "home_summary": home_result.summary,
            "away_summary": away_result.summary,
            "home_topics": home_result.key_topics,
            "away_topics": away_result.key_topics,
            "confidence": (home_result.confidence + away_result.confidence) / 2,
            "edge_direction": (
                "home" if diff > 0.1 else
                "away" if diff < -0.1 else
                "neutral"
            ),
        }

    # ═══════════════════════════════════════════
    #  HABER TOPLAMA
    # ═══════════════════════════════════════════
    async def _fetch_news(self, team: str) -> list[NewsItem]:
        """RSS + web'den haberleri topla."""
        all_news = []

        tasks = [
            self._fetch_rss(team, name, url)
            for name, url in self.NEWS_SOURCES.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_news.extend(result)

        # Google News search (yedek)
        if len(all_news) < 3:
            google_news = await self._fetch_google_news(team)
            all_news.extend(google_news)

        logger.debug(f"[RAG] {team}: {len(all_news)} haber toplandı.")
        return all_news[:20]  # En fazla 20 haber

    async def _fetch_rss(self, team: str, source: str,
                          url: str) -> list[NewsItem]:
        """RSS feed'den haberleri çek."""
        if not HTTPX_OK:
            return []

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return []

            if not BS4_OK:
                return []

            soup = BeautifulSoup(resp.text, "lxml-xml")
            items = []

            for item in soup.find_all("item"):
                title = item.find("title")
                if not title:
                    continue
                title_text = title.text.strip()

                # Takım adı geçiyor mu?
                team_lower = team.lower()
                if team_lower not in title_text.lower():
                    desc = item.find("description")
                    if not desc or team_lower not in desc.text.lower():
                        continue

                pub_date = item.find("pubDate")
                link = item.find("link")

                items.append(NewsItem(
                    title=title_text,
                    source=source,
                    url=link.text.strip() if link else "",
                    published=pub_date.text.strip() if pub_date else "",
                    snippet=(desc.text.strip()[:200] if desc else ""),
                    team_mentioned=team,
                ))

            return items[:5]  # Kaynak başına max 5
        except Exception as e:
            logger.debug(f"[RAG] RSS hatası ({source}): {e}")
            return []

    async def _fetch_google_news(self, team: str) -> list[NewsItem]:
        """Google News'ten haber çek (yedek kaynak)."""
        if not HTTPX_OK:
            return []

        try:
            query = f"{team} futbol"
            url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return []

            if not BS4_OK:
                return []

            soup = BeautifulSoup(resp.text, "lxml-xml")
            items = []
            for item in soup.find_all("item")[:5]:
                title = item.find("title")
                if title:
                    items.append(NewsItem(
                        title=title.text.strip(),
                        source="google_news",
                        team_mentioned=team,
                    ))
            return items
        except Exception:
            return []

    # ═══════════════════════════════════════════
    #  LLM ANALİZİ
    # ═══════════════════════════════════════════
    async def _analyze_with_llm(self, team: str,
                                 news: list[NewsItem]) -> RAGResult:
        """Haberleri LLM ile analiz et."""
        # 1. Ollama (Yerel & Ücretsiz) öncelikli
        result = await self._analyze_ollama(team, news)
        if result:
            return result

        # 3. HuggingFace yedek
        if self._hf_token:
            result = await self._analyze_huggingface(team, news)
            if result:
                return result

        # 4. Kural tabanlı fallback
        return self._analyze_rule_based(team, news)

    async def _analyze_ollama(self, team: str,
                               news: list[NewsItem]) -> RAGResult | None:
        """Ollama ile yerel sentiment analizi (Tamamen Ücretsiz ve Özel)."""
        if not HTTPX_OK:
            return None

        headlines = "\n".join(f"- {n.title}" for n in news[:10])
        prompt = (
            f"Aşağıdaki haber başlıkları {team} futbol takımı hakkındadır.\n\n"
            f"{headlines}\n\n"
            f"Lütfen şu soruları yanıtla:\n"
            f"1. Genel atmosfer (0=çok negatif, 1=çok pozitif) -> sadece sayı yaz.\n"
            f"2. Anahtar konular (virgülle ayır, max 5 kelime/konu)\n"
            f"3. 2 cümlelik özet\n\n"
            f"Lütfen SADECE şu formatta yanıt ver ve format dışına çıkma:\n"
            f"SKOR|KONULAR|ÖZET"
        )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    self._ollama_url,
                    json={
                        "model": self._ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    },
                )
                if resp.status_code != 200:
                    logger.debug(f"[RAG] Ollama HTTP hatası: {resp.status_code}")
                    return None

                data = resp.json()
                text = data.get("response", "")

                if text:
                    logger.info(f"Ollama yanıt verdi: {text[:50]}...")
                    return self._parse_llm_response(team, text, news, f"ollama_{self._ollama_model}")
                return None
        except Exception as e:
            logger.debug(f"[RAG] Ollama hatası: {e}. Ollama kapalı olabilir.")
            return None



    async def _analyze_huggingface(self, team: str,
                                    news: list[NewsItem]) -> RAGResult | None:
        """HuggingFace Inference API ile sentiment."""
        if not HTTPX_OK:
            return None

        headlines = ". ".join(n.title for n in news[:5])

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.HF_URL}cardiffnlp/twitter-roberta-base-sentiment-latest",
                    headers={"Authorization": f"Bearer {self._hf_token}"},
                    json={"inputs": headlines[:500]},
                )
                if resp.status_code != 200:
                    return None

                data = resp.json()
                if not data or not isinstance(data, list):
                    return None

                scores = data[0] if isinstance(data[0], list) else data
                sentiment_map = {}
                for item in scores:
                    label = item.get("label", "").lower()
                    score = item.get("score", 0)
                    sentiment_map[label] = score

                # positive, neutral, negative → 0-1 skor
                pos = sentiment_map.get("positive", 0)
                neg = sentiment_map.get("negative", 0)
                sentiment = 0.5 + (pos - neg) * 0.5

                return RAGResult(
                    team=team,
                    sentiment_score=float(max(0, min(1, sentiment))),
                    summary=f"HF sentiment: pos={pos:.2f}, neg={neg:.2f}",
                    confidence=0.6,
                    n_sources=len(news),
                    method="huggingface",
                )
        except Exception as e:
            logger.debug(f"[RAG] HF hatası: {e}")
            return None

    def _analyze_rule_based(self, team: str,
                             news: list[NewsItem]) -> RAGResult:
        """Kural tabanlı basit sentiment (LLM yokken)."""
        positive_words = {
            "galibiyet", "kazandı", "zafer", "muhteşem", "form", "motivasyon",
            "şampiyonluk", "transfer", "güçlendi", "başarı", "gol", "rekor",
            "imza", "anlaşma", "takviye", "dönüş", "iyileşti",
        }
        negative_words = {
            "yenilgi", "kaybetti", "sakatlık", "ceza", "kriz", "kovuldu",
            "kırmızı", "ban", "sakatlandı", "kadro dışı", "kavga", "istifa",
            "mağlubiyet", "düşüş", "hayal kırıklığı", "başarısız",
        }

        pos_count = neg_count = 0
        topics = set()

        for n in news:
            text_lower = (n.title + " " + n.snippet).lower()
            for w in positive_words:
                if w in text_lower:
                    pos_count += 1
                    topics.add(w)
            for w in negative_words:
                if w in text_lower:
                    neg_count += 1
                    topics.add(w)

        total = pos_count + neg_count
        if total > 0:
            sentiment = 0.5 + (pos_count - neg_count) / (2 * total)
        else:
            sentiment = 0.5

        return RAGResult(
            team=team,
            sentiment_score=float(max(0, min(1, sentiment))),
            summary=f"{len(news)} haber analiz edildi (kural tabanlı).",
            key_topics=list(topics)[:5],
            confidence=0.3,
            n_sources=len(news),
            method="rule_based",
        )

    def _parse_llm_response(self, team: str, text: str,
                             news: list[NewsItem],
                             method: str) -> RAGResult:
        """LLM yanıtını parse et: SKOR|KONULAR|ÖZET formatı."""
        try:
            parts = text.strip().split("|")
            score_str = parts[0].strip() if len(parts) > 0 else "0.5"
            topics_str = parts[1].strip() if len(parts) > 1 else ""
            summary = parts[2].strip() if len(parts) > 2 else text[:200]

            score = float(score_str)
            if score > 1:
                score /= 10  # 0-10 yerine 0-1 dönüşümü
            score = max(0, min(1, score))

            topics = [t.strip() for t in topics_str.split(",") if t.strip()]

            return RAGResult(
                team=team,
                sentiment_score=score,
                summary=summary,
                key_topics=topics[:5],
                confidence=0.7,
                n_sources=len(news),
                method=method,
            )
        except Exception:
            return RAGResult(
                team=team,
                sentiment_score=0.5,
                summary=text[:200] if text else "Parse hatası.",
                confidence=0.4,
                n_sources=len(news),
                method=method,
            )

    # ═══════════════════════════════════════════
    #  CACHE
    # ═══════════════════════════════════════════
    def _get_cache(self, key: str) -> RAGResult | None:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, result: RAGResult):
        self._cache[key] = (time.time(), result)
