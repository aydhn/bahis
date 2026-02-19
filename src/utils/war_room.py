"""
war_room.py – Multi-Agent Debate (The War Room).

Tek bir botun karar vermesi riskli. 3 farklı yapay zeka
kişiliği tartışır, siz jüri olursunuz.

Ajanlar:
  - The Gambler: Risk seven, sürpriz arayan kişilik
  - The Risk Manager: Tutucu, kasayı koruyan kişilik
  - The Quant: Sadece matematik ve veri konuşan kişilik

Akış:
  1. Maç verisi ve model çıktıları toplanır
  2. Her ajan kendi perspektifinden analiz yapar
  3. 3 turda birbiriyle tartışır
  4. Final kararı çoğunluk oylaması ile belirlenir
  5. Tartışma Telegram'a gönderilir

Teknoloji: LangChain + Ollama (yerel LLM) veya Gemini
Fallback: Template-based persona responses
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    from src.utils.gemini_client import gemini_generate, GEMINI_OK as _GEMINI_OK
    GEMINI_OK = _GEMINI_OK
except ImportError:
    GEMINI_OK = False

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

OLLAMA_URL = "http://localhost:11434/api/generate"


# ═══════════════════════════════════════════════
#  AJAN PERSONALARl
# ═══════════════════════════════════════════════
AGENT_PROFILES = {
    "gambler": {
        "name": "🎰 The Gambler",
        "emoji": "🎰",
        "style": "aggressive",
        "system_prompt": (
            "Sen agresif bir bahisçisin. Risk alırsın çünkü büyük kazançlar "
            "büyük risklerle gelir. Yüksek oranları seversin, sürprizlere "
            "inanırsın. 'Value' gördüğün yerde asla pas geçmezsin. "
            "Konuşman enerjik ve cesaret vericidir. Türkçe konuş."
        ),
    },
    "risk_manager": {
        "name": "🛡️ The Risk Manager",
        "emoji": "🛡️",
        "style": "conservative",
        "system_prompt": (
            "Sen tutucu bir risk yöneticisisin. Kasayı korumak her şeyden "
            "önemli. Kelly kriteri %2'den fazla diyorsa bile şüphelenirsin. "
            "Drawdown'dan nefret edersin. 'Kaybetmemek kazanmaktan önemlidir' "
            "felsefesiyle yaşarsın. Konuşman temkinli ve uyarıcıdır. Türkçe konuş."
        ),
    },
    "quant": {
        "name": "📊 The Quant",
        "emoji": "📊",
        "style": "analytical",
        "system_prompt": (
            "Sen soğukkanlı bir kantitatif analistsin. Sadece sayılar konuşur. "
            "Duygulara yer yok, her şey olasılık ve beklenen değer (EV). "
            "Model çıktılarını analiz edersin, ne fazlasını ne eksiğini söylersin. "
            "Konuşman kısa, net ve veri odaklıdır. Türkçe konuş."
        ),
    },
}


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class AgentOpinion:
    """Bir ajanın görüşü."""
    agent_name: str = ""
    agent_style: str = ""
    opinion: str = ""
    verdict: str = "HOLD"     # "BET" | "HOLD" | "SKIP"
    confidence: float = 0.5
    key_argument: str = ""


@dataclass
class DebateResult:
    """Tartışma sonucu."""
    match_id: str = ""
    match_info: str = ""
    # Ajan görüşleri
    opinions: list[AgentOpinion] = field(default_factory=list)
    # Tartışma metni
    dialogue: str = ""
    # Final karar
    majority_verdict: str = "HOLD"
    consensus: bool = False
    bet_count: int = 0
    skip_count: int = 0
    hold_count: int = 0
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  LLM BACKEND
# ═══════════════════════════════════════════════
def _ask_ollama_sync(prompt: str, system: str,
                       model: str = "llama3:8b") -> str:
    """Ollama yerel LLM (senkron)."""
    if not HTTPX_OK:
        return ""
    try:
        resp = httpx.post(OLLAMA_URL, json={
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300},
        }, timeout=30.0)
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception:
        pass
    return ""


def _ask_gemini_sync(prompt: str, system: str) -> str:
    """Google Gemini API."""
    if not GEMINI_OK:
        return ""
    try:
        result = gemini_generate(prompt=prompt, system=system)
        return result
    except Exception:
        pass
    return ""


def _template_response(agent_key: str, match_info: dict) -> str:
    """Template-based persona cevabı (LLM yoksa)."""
    odds = match_info.get("odds", 2.0)
    ev = match_info.get("ev", 0.0)
    prob = match_info.get("prob", 0.5)
    home = match_info.get("home", "Ev Sahibi")
    away = match_info.get("away", "Deplasman")
    kelly = match_info.get("kelly", 0.0)

    if agent_key == "gambler":
        if odds > 3.0:
            return (
                f"Bu oran {odds:.2f} harika! {home} son maçlarda "
                f"zorlanıyor ama tam da sürpriz zamanı. "
                f"EV {ev:+.1%} pozitif, bu fırsatı kaçırmayalım! "
                f"BET diyorum, cesaret zamanı!"
            )
        return (
            f"Oran {odds:.2f}, olasılık {prob:.0%}. "
            f"Kötü değil ama beni heyecanlandırmıyor. "
            f"Yine de EV pozitifse ({ev:+.1%}) oynamaya değer."
        )

    if agent_key == "risk_manager":
        if kelly > 0.05:
            return (
                f"Dur bir dakika. Kelly {kelly:.1%} diyor ama "
                f"bu lig çok kaotik. Entropi yüksekse pas geçmeliyiz. "
                f"Drawdown limitimize yaklaşıyorsak riske girmeyelim. "
                f"SKIP veya en fazla yarım stake."
            )
        return (
            f"Kelly {kelly:.1%} düşük, oran {odds:.2f}. "
            f"Bu maçta edge yeterli değil. Kasayı koruyalım. "
            f"SKIP diyorum, daha iyi fırsatlar olacak."
        )

    # quant
    return (
        f"Verilere bakıyorum: P(home)={prob:.1%}, odds={odds:.2f}, "
        f"EV={ev:+.2%}, Kelly={kelly:.2%}. "
        f"{'Pozitif EV var, matematiksel olarak BET mantıklı.' if ev > 0.02 else 'Edge marjinal, HOLD.'} "
        f"Hawkes BR ve Hurst değerlerine de bakılmalı."
    )


# ═══════════════════════════════════════════════
#  WAR ROOM (Ana Sınıf)
# ═══════════════════════════════════════════════
class WarRoom:
    """Multi-Agent Debate sistemi.

    Kullanım:
        wr = WarRoom(llm_backend="auto")

        result = wr.debate(
            match_info={
                "home": "Galatasaray",
                "away": "Fenerbahçe",
                "odds": 2.10,
                "prob": 0.55,
                "ev": 0.08,
                "kelly": 0.04,
            },
            match_id="gs_fb",
        )

        # Telegram'a gönder
        telegram_text = wr.format_telegram(result)
    """

    def __init__(self, llm_backend: str = "auto",
                 ollama_model: str = "llama3:8b"):
        self._backend = llm_backend
        self._ollama_model = ollama_model
        logger.debug(f"[WarRoom] Debate sistemi başlatıldı (backend={llm_backend})")

    def debate(self, match_info: dict,
                 match_id: str = "",
                 n_rounds: int = 1) -> DebateResult:
        """3 ajan tartışması başlat."""
        result = DebateResult(
            match_id=match_id,
            match_info=self._format_match_context(match_info),
        )

        context = self._format_match_context(match_info)

        # Her ajan görüş verir
        for agent_key, profile in AGENT_PROFILES.items():
            opinion = self._get_agent_opinion(
                agent_key, profile, match_info, context,
            )
            result.opinions.append(opinion)

        # Tartışma diyalogu
        result.dialogue = self._build_dialogue(result.opinions, match_info)

        # Oylama
        for op in result.opinions:
            if op.verdict == "BET":
                result.bet_count += 1
            elif op.verdict == "SKIP":
                result.skip_count += 1
            else:
                result.hold_count += 1

        if result.bet_count > result.skip_count:
            result.majority_verdict = "BET"
        elif result.skip_count > result.bet_count:
            result.majority_verdict = "SKIP"
        else:
            result.majority_verdict = "HOLD"

        result.consensus = (
            result.bet_count == 3 or result.skip_count == 3 or result.hold_count == 3
        )

        result.recommendation = (
            f"Çoğunluk: {result.majority_verdict} "
            f"({'OYBIRLIGI' if result.consensus else f'{result.bet_count}B/{result.skip_count}S/{result.hold_count}H'})"
        )

        return result

    def _get_agent_opinion(self, agent_key: str, profile: dict,
                             match_info: dict, context: str) -> AgentOpinion:
        """Bir ajanın görüşünü al."""
        opinion = AgentOpinion(
            agent_name=profile["name"],
            agent_style=profile["style"],
        )

        prompt = (
            f"Maç analizi:\n{context}\n\n"
            f"Bu maç hakkında 2-3 cümlelik görüşünü ver. "
            f"Sonunda kararını belirt: BET, HOLD veya SKIP."
        )

        # LLM dene
        response = ""
        if self._backend in ("ollama", "auto"):
            response = _ask_ollama_sync(
                prompt, profile["system_prompt"], self._ollama_model,
            )
        if not response and self._backend in ("gemini", "auto"):
            response = _ask_gemini_sync(prompt, profile["system_prompt"])

        if not response:
            response = _template_response(agent_key, match_info)

        opinion.opinion = response.strip()

        # Karar çıkar
        upper = response.upper()
        if "BET" in upper or "OYNA" in upper or "AL" in upper:
            opinion.verdict = "BET"
        elif "SKIP" in upper or "PAS" in upper or "KAÇIN" in upper:
            opinion.verdict = "SKIP"
        else:
            opinion.verdict = "HOLD"

        # Güven (basit heuristik)
        if "kesinlikle" in response.lower() or "mutlaka" in response.lower():
            opinion.confidence = 0.9
        elif "belki" in response.lower() or "düşünülebilir" in response.lower():
            opinion.confidence = 0.5
        else:
            opinion.confidence = 0.7

        return opinion

    def _format_match_context(self, info: dict) -> str:
        """Maç bilgisini metin formatına çevir."""
        return (
            f"Maç: {info.get('home', '?')} vs {info.get('away', '?')}\n"
            f"Oran: {info.get('odds', '?')}\n"
            f"Model Olasılığı: {info.get('prob', 0):.0%}\n"
            f"EV (Beklenen Değer): {info.get('ev', 0):+.1%}\n"
            f"Kelly Stake: {info.get('kelly', 0):.1%}\n"
            f"Güven Skoru: {info.get('confidence', 0):.0%}\n"
            f"Entropi: {info.get('entropy', '?')}\n"
            f"Hurst: {info.get('hurst', '?')}"
        )

    def _build_dialogue(self, opinions: list[AgentOpinion],
                          match_info: dict) -> str:
        """Tartışma diyalogu oluştur."""
        home = match_info.get("home", "Ev Sahibi")
        away = match_info.get("away", "Deplasman")

        lines = [
            f"🏟️ <b>WAR ROOM: {home} vs {away}</b>\n",
        ]

        for op in opinions:
            lines.append(f"{op.agent_name}:")
            lines.append(f"<i>\"{op.opinion}\"</i>")
            lines.append(f"Karar: <b>{op.verdict}</b>\n")

        return "\n".join(lines)

    def format_telegram(self, result: DebateResult) -> str:
        """Telegram formatında tartışma çıktısı."""
        avg_conf = sum(op.confidence for op in result.opinions) / max(len(result.opinions), 1)
        lines = [
            result.dialogue,
            "─" * 30,
            f"🗳️ <b>OYLAMA:</b>",
            f"  ✅ BET: {result.bet_count}",
            f"  ⏸️ HOLD: {result.hold_count}",
            f"  ❌ SKIP: {result.skip_count}",
            f"  📊 Güven: {avg_conf:.0%}",
            "",
            f"📋 <b>Karar: {result.majority_verdict}</b>"
            + (" (OYBİRLİĞİ)" if result.consensus else ""),
        ]
        return "\n".join(lines)

    def quick_verdict(self, match_info: dict) -> str:
        """Hızlı oylama – tam tartışma yapmadan karar üretir."""
        ev = match_info.get("ev", 0.0)
        kelly = match_info.get("kelly", 0.0)
        odds = match_info.get("odds", 2.0)
        prob = match_info.get("prob", 0.5)

        score = 0
        if ev > 0.05:
            score += 2
        elif ev > 0.02:
            score += 1
        if kelly > 0.03:
            score += 1
        if prob > 0.55:
            score += 1
        if odds > 1.5 and odds < 5.0:
            score += 1

        if score >= 4:
            return "BET"
        elif score >= 2:
            return "HOLD"
        return "SKIP"
