"""
telegram_live.py – Canlı Skor Panosu (Live Dashboarding via editMessage).

Telegram'ı kirletmeden, maç başladığında TEK bir mesaj sabitlener.
Dakika, skor veya kart değiştikçe mesaj düzenlenir (editMessage).
Maç bittiğinde "Analiz Başarısı: ✅/❌" damgası vurulur.

→ Yeni mesaj spam'i yok, temiz bir canlı skorboard.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class LiveMatchState:
    """Canlı maçın anlık durumu."""
    match_id: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    minute: int = 0
    period: str = "PRE"       # PRE, 1H, HT, 2H, FT, ET
    events: list[str] = field(default_factory=list)   # ⚽ GOL, 🟨 KART vb.
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_possession: float = 50.0
    away_possession: float = 50.0
    # Bot tahmini
    prediction: str = ""     # home / draw / away
    predicted_odds: float = 0.0
    bet_placed: bool = False
    # Telegram mesaj ID'si
    message_id: int = 0
    last_updated: float = 0.0


class TelegramLiveDashboard:
    """Canlı skor panosu – editMessage ile tek mesaj güncelleme.

    Kullanım:
        live = TelegramLiveDashboard(notifier=notifier)
        await live.start_tracking(match_state)
        # Veri güncelleme döngüsünde:
        await live.update(match_state)
        # Maç bittiğinde:
        await live.finalize(match_state, actual_result="home")
    """

    UPDATE_INTERVAL = 30  # Minimum güncelleme aralığı (saniye)

    def __init__(self, notifier=None):
        self._notifier = notifier
        self._active: dict[str, LiveMatchState] = {}
        logger.debug("TelegramLiveDashboard başlatıldı.")

    async def start_tracking(self, state: LiveMatchState) -> bool:
        """Maç takibini başlat – ilk mesajı gönder ve sabitle."""
        text = self._render(state)

        msg_id = await self._send_and_pin(text)
        if msg_id:
            state.message_id = msg_id
            state.last_updated = time.time()
            self._active[state.match_id] = state
            logger.info(
                f"[Live] Takip başladı: {state.home_team} vs {state.away_team} "
                f"(msg_id={msg_id})"
            )
            return True
        return False

    async def update(self, state: LiveMatchState) -> bool:
        """Maç durumunu güncelle – mesajı düzenle."""
        if state.match_id not in self._active:
            return await self.start_tracking(state)

        # Minimum güncelleme aralığı
        elapsed = time.time() - state.last_updated
        if elapsed < self.UPDATE_INTERVAL:
            return False

        text = self._render(state)
        success = await self._edit_message(state.message_id, text)

        if success:
            state.last_updated = time.time()
            self._active[state.match_id] = state

        return success

    async def finalize(self, state: LiveMatchState,
                       actual_result: str = "") -> bool:
        """Maç bitti – son güncelleme + analiz başarısı."""
        state.period = "FT"

        # Tahmin doğruluğu
        if actual_result and state.prediction:
            success = actual_result == state.prediction
            result_emoji = "✅" if success else "❌"
            verdict = "BAŞARILI" if success else "BAŞARISIZ"
        else:
            result_emoji = "➖"
            verdict = "DEĞERLENDİRİLEMEDİ"

        text = self._render(state)
        text += (
            f"\n\n{'━' * 28}\n"
            f"{result_emoji} <b>Analiz Sonucu: {verdict}</b>\n"
        )

        if state.bet_placed:
            text += (
                f"💰 <b>Bahis:</b> {state.prediction.upper()} @ {state.predicted_odds:.2f}\n"
                f"📊 <b>Sonuç:</b> {state.home_score}-{state.away_score}\n"
            )

        await self._edit_message(state.message_id, text)

        # Aktif listeden çıkar
        self._active.pop(state.match_id, None)
        logger.info(
            f"[Live] Maç tamamlandı: {state.home_team} {state.home_score}-"
            f"{state.away_score} {state.away_team} | {result_emoji} {verdict}"
        )
        return True

    # ═══════════════════════════════════════════
    #  RENDER: Mesaj şablonu
    # ═══════════════════════════════════════════
    def _render(self, state: LiveMatchState) -> str:
        """Canlı skor mesajını HTML olarak oluştur."""
        period_emoji = {
            "PRE": "🕐", "1H": "⚽", "HT": "☕",
            "2H": "⚽", "FT": "🏁", "ET": "⏰",
        }.get(state.period, "📊")

        minute_str = f"'{state.minute}" if state.minute > 0 else "Başlamadı"
        if state.period == "HT":
            minute_str = "Devre Arası"
        elif state.period == "FT":
            minute_str = "Maç Bitti"

        # Skor çubuğu
        score_line = (
            f"<b>{state.home_team}</b>  "
            f"<code>[ {state.home_score} - {state.away_score} ]</code>  "
            f"<b>{state.away_team}</b>"
        )

        # xG karşılaştırma
        xg_bar = self._xg_bar(state.home_xg, state.away_xg)

        # Possession
        poss_bar = self._possession_bar(state.home_possession)

        # Olay akışı (son 5)
        events_str = ""
        if state.events:
            recent = state.events[-5:]
            events_str = "\n".join(f"  {e}" for e in recent)
            events_str = f"\n📋 <b>Son Olaylar:</b>\n{events_str}"

        # Bot tahmini
        pred_str = ""
        if state.prediction:
            pred_str = (
                f"\n🤖 <b>Bot Tahmini:</b> {state.prediction.upper()} "
                f"@ {state.predicted_odds:.2f}"
            )
            if state.bet_placed:
                pred_str += " 💰"

        text = (
            f"{period_emoji} <b>CANLI SKOR</b>  ⏱ {minute_str}\n"
            f"{'━' * 28}\n\n"
            f"{score_line}\n\n"
            f"📊 <b>xG:</b> {xg_bar}\n"
            f"🎮 <b>Pas:</b> {poss_bar}\n"
            f"{events_str}"
            f"{pred_str}\n\n"
            f"<i>Son güncelleme: {datetime.now().strftime('%H:%M:%S')}</i>"
        )
        return text

    @staticmethod
    def _xg_bar(home_xg: float, away_xg: float) -> str:
        """xG karşılaştırma çubuğu."""
        total = max(home_xg + away_xg, 0.01)
        home_pct = home_xg / total
        n_blocks = 10
        home_blocks = round(home_pct * n_blocks)
        away_blocks = n_blocks - home_blocks
        return f"{home_xg:.2f} {'▓' * home_blocks}{'░' * away_blocks} {away_xg:.2f}"

    @staticmethod
    def _possession_bar(home_poss: float) -> str:
        """Topla oynama çubuğu."""
        n = 10
        home_n = round(home_poss / 10)
        away_n = n - home_n
        return f"{home_poss:.0f}% {'▓' * home_n}{'░' * away_n} {100-home_poss:.0f}%"

    # ═══════════════════════════════════════════
    #  TELEGRAM API
    # ═══════════════════════════════════════════
    async def _send_and_pin(self, text: str) -> int:
        """Mesaj gönder ve sabitle."""
        if not self._notifier:
            return 0

        try:
            msg_id = await self._notifier.send(text, return_message_id=True)
            if msg_id and hasattr(self._notifier, "pin_message"):
                await self._notifier.pin_message(msg_id)
            return msg_id or 0
        except Exception as e:
            logger.error(f"[Live] Mesaj gönderme hatası: {e}")
            return 0

    async def _edit_message(self, message_id: int, text: str) -> bool:
        """Mevcut mesajı düzenle."""
        if not self._notifier or not message_id:
            return False

        try:
            if hasattr(self._notifier, "edit_message"):
                return await self._notifier.edit_message(message_id, text)
            return False
        except Exception as e:
            logger.debug(f"[Live] Mesaj düzenleme hatası: {e}")
            return False

    # ═══════════════════════════════════════════
    #  OLAY GÜNCELLEMELERİ
    # ═══════════════════════════════════════════
    def add_event(self, match_id: str, event_text: str):
        """Maça olay ekle (gol, kart, değişiklik)."""
        state = self._active.get(match_id)
        if state:
            state.events.append(event_text)

    def update_score(self, match_id: str, home: int, away: int, minute: int):
        """Skor güncelle."""
        state = self._active.get(match_id)
        if state:
            state.home_score = home
            state.away_score = away
            state.minute = minute

    @property
    def active_matches(self) -> list[str]:
        return list(self._active.keys())

    @property
    def active_count(self) -> int:
        return len(self._active)
