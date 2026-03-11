"""
agent_poll_system.py – Multi-Agent Collaboration Polls (Ajan Oylaması).

war_room.py modülündeki 3 ajan (Gambler, Risk Manager, Quant)
artık sadece tartışmaz — bir oylama yapar ve sonucu interaktif
bir Telegram anketi olarak sunar. Son karar kullanıcınındır.

Kavramlar:
  - Agent Council: 3 ajan bir "Karar Konseyi" oluşturur
  - Vote: Her ajanın EVET/HAYIR oyu + gerekçesi
  - Consensus: 3/3 oy birliği — güçlü sinyal
  - Majority: 2/3 çoğunluk — orta sinyal
  - Split: Her biri farklı — kaos, pas geç
  - Telegram Poll: Kullanıcıya interaktif anket gönderimi
  - Callback: Kullanıcının oyu kaydedilir (HITL)

Akış:
  1. war_room.debate() çağrılır → 3 ajan görüş verir
  2. AgentPollSystem ajanların oylarını ve gerekçelerini toplar
  3. Oylama özeti HTML formatında hazırlanır
  4. Telegram Poll (anket) oluşturulur: "Bu maça bahis oynamalı mıyız?"
  5. Ajan gerekçeleri mesaj olarak gönderilir
  6. Kullanıcı ankette oy verir → sonuç veritabanına kaydedilir

Teknoloji: python-telegram-bot (Polls / Quiz Mode)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    logger.debug("python-telegram-bot yüklü değil – poll sistemi offline modda.")

# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class AgentVote:
    """Bir ajanın oyu."""
    agent_name: str = ""
    agent_emoji: str = ""
    vote: str = "KARASIZ"       # "EVET" | "HAYIR" | "KARASIZ"
    confidence: float = 0.5     # [0, 1]
    reasoning: str = ""         # Gerekçe (kısa)
    key_metric: str = ""        # En önemli metrik (ör: "EV=+8.5%")
    risk_flag: str = ""         # Risk uyarısı (varsa)

@dataclass
class CouncilDecision:
    """Konsey kararı."""
    match_id: str = ""
    home: str = ""
    away: str = ""
    # Oylar
    votes: list[AgentVote] = field(default_factory=list)
    yes_count: int = 0
    no_count: int = 0
    undecided_count: int = 0
    # Konsensüs
    consensus_type: str = ""     # "unanimous_yes" | "unanimous_no" | "majority_yes" | "majority_no" | "split"
    consensus_emoji: str = ""
    council_verdict: str = ""    # "OYNA" | "PAS GEÇ" | "KARARSIZ"
    # Model verileri
    odds: float = 0.0
    ev_pct: float = 0.0
    kelly_pct: float = 0.0
    model_prob: float = 0.0
    # Zaman
    timestamp: str = ""
    # Telegram
    poll_message_id: int = 0
    detail_message_id: int = 0
    user_vote: str = ""          # Kullanıcının oyu (callback ile gelir)

@dataclass
class PollStats:
    """Anket istatistikleri."""
    total_polls: int = 0
    user_agreed_with_council: int = 0
    user_disagreed: int = 0
    council_accuracy: float = 0.0  # Konsey kararının tutma oranı
    user_accuracy: float = 0.0     # Kullanıcı kararının tutma oranı

# ═══════════════════════════════════════════════
#  AGENT POLL SYSTEM (Ana Sınıf)
# ═══════════════════════════════════════════════
class AgentPollSystem:
    """Multi-Agent Collaboration Polls.

    war_room ajanlarını Telegram anketine dönüştürür.

    Kullanım:
        from src.utils.war_room import WarRoom

        wr = WarRoom(llm_backend="auto")
        poll_sys = AgentPollSystem(notifier=telegram_notifier)

        # Debate + Poll
        debate_result = wr.debate(match_info, match_id="gs_fb")
        council = poll_sys.create_council_decision(debate_result, match_info)

        # Telegram'a gönder
        await poll_sys.send_poll(council)

        # Kullanıcı oyu geldiğinde
        poll_sys.record_user_vote(council.match_id, "EVET")
    """

    # Konsensüs tiplerine göre emoji ve açıklama
    CONSENSUS_MAP = {
        "unanimous_yes": ("🟢🟢🟢", "OYBİRLİĞİ: OYNA"),
        "unanimous_no": ("🔴🔴🔴", "OYBİRLİĞİ: PAS GEÇ"),
        "majority_yes": ("🟢🟢🔴", "ÇOĞUNLUK: OYNA (2/3)"),
        "majority_no": ("🟢🔴🔴", "ÇOĞUNLUK: PAS GEÇ (2/3)"),
        "split": ("🟡🟡🟡", "BÖLÜNME: KARARSIZ"),
    }

    def __init__(self, notifier: Any = None,
                 auto_send: bool = True):
        self._notifier = notifier
        self._auto_send = auto_send
        self._history: list[CouncilDecision] = []
        self._user_votes: dict[str, str] = {}

        logger.debug(
            f"[AgentPoll] Sistem başlatıldı: "
            f"telegram={'OK' if notifier else 'offline'}"
        )

    def create_council_decision(self, debate_result,
                                  match_info: dict) -> CouncilDecision:
        """WarRoom debate sonucunu konsey kararına çevir."""
        council = CouncilDecision(
            match_id=debate_result.match_id,
            home=match_info.get("home", "?"),
            away=match_info.get("away", "?"),
            odds=match_info.get("odds", 0.0),
            ev_pct=match_info.get("ev", 0.0),
            kelly_pct=match_info.get("kelly", 0.0),
            model_prob=match_info.get("prob", 0.0),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Ajan oylarını çıkar
        for opinion in debate_result.opinions:
            vote = self._opinion_to_vote(opinion, match_info)
            council.votes.append(vote)

            if vote.vote == "EVET":
                council.yes_count += 1
            elif vote.vote == "HAYIR":
                council.no_count += 1
            else:
                council.undecided_count += 1

        # Konsensüs belirle
        council = self._determine_consensus(council)

        self._history.append(council)
        return council

    def _opinion_to_vote(self, opinion, match_info: dict) -> AgentVote:
        """AgentOpinion → AgentVote dönüşümü."""
        vote = AgentVote(
            agent_name=opinion.agent_name,
            confidence=opinion.confidence,
            reasoning=opinion.opinion[:200],
        )

        # Emoji ata
        if "Gambler" in opinion.agent_name:
            vote.agent_emoji = "🎰"
        elif "Risk" in opinion.agent_name:
            vote.agent_emoji = "🛡️"
        else:
            vote.agent_emoji = "📊"

        # Verdict → Vote
        if opinion.verdict == "BET":
            vote.vote = "EVET"
        elif opinion.verdict == "SKIP":
            vote.vote = "HAYIR"
        else:
            vote.vote = "KARASIZ"

        # Anahtar metrik
        ev = match_info.get("ev", 0)
        kelly = match_info.get("kelly", 0)
        if "Gambler" in opinion.agent_name:
            vote.key_metric = f"Oran: {match_info.get('odds', '?')}"
        elif "Risk" in opinion.agent_name:
            vote.key_metric = f"Kelly: {kelly:.1%}"
            if kelly > 0.05:
                vote.risk_flag = "Yüksek stake uyarısı"
        else:
            vote.key_metric = f"EV: {ev:+.1%}"

        return vote

    def _determine_consensus(self, council: CouncilDecision) -> CouncilDecision:
        """Konsensüs tipini belirle."""
        y, n = council.yes_count, council.no_count

        if y == 3:
            council.consensus_type = "unanimous_yes"
            council.council_verdict = "OYNA"
        elif n == 3:
            council.consensus_type = "unanimous_no"
            council.council_verdict = "PAS GEÇ"
        elif y >= 2:
            council.consensus_type = "majority_yes"
            council.council_verdict = "OYNA"
        elif n >= 2:
            council.consensus_type = "majority_no"
            council.council_verdict = "PAS GEÇ"
        else:
            council.consensus_type = "split"
            council.council_verdict = "KARARSIZ"

        emoji, _ = self.CONSENSUS_MAP.get(
            council.consensus_type, ("❓", "BİLİNMİYOR"),
        )
        council.consensus_emoji = emoji

        return council

    # ─────────────────────────────────────────────
    #  TELEGRAM FORMATLAMA
    # ─────────────────────────────────────────────
    def format_council_message(self, council: CouncilDecision) -> str:
        """Konsey kararını HTML formatında hazırla."""
        _, desc = self.CONSENSUS_MAP.get(
            council.consensus_type, ("❓", "BİLİNMİYOR"),
        )

        lines = [
            "🏛️ <b>KARAR KONSEYİ</b>",
            f"⚽ <b>{council.home} vs {council.away}</b>",
            f"📊 Oran: {council.odds:.2f} | EV: {council.ev_pct:+.1%} | "
            f"Kelly: {council.kelly_pct:.1%}",
            "",
            "─" * 28,
            "",
        ]

        # Her ajanın oyu
        for v in council.votes:
            vote_emoji = "✅" if v.vote == "EVET" else "❌" if v.vote == "HAYIR" else "🤔"
            conf_bar = "█" * int(v.confidence * 5) + "░" * (5 - int(v.confidence * 5))

            lines.append(
                f"{v.agent_emoji} <b>{v.agent_name}</b>"
            )
            lines.append(
                f"   {vote_emoji} <b>{v.vote}</b> | "
                f"Güven: [{conf_bar}] {v.confidence:.0%}"
            )
            lines.append(f"   📌 {v.key_metric}")
            if v.risk_flag:
                lines.append(f"   ⚠️ {v.risk_flag}")
            lines.append(f"   💬 <i>\"{v.reasoning[:100]}\"</i>")
            lines.append("")

        lines.extend([
            "─" * 28,
            "",
            f"{council.consensus_emoji} <b>{desc}</b>",
            "",
            f"🗳️ Konsey Kararı: <b>{council.council_verdict}</b>",
            f"   EVET: {council.yes_count} | HAYIR: {council.no_count} | "
            f"KARASIZ: {council.undecided_count}",
            "",
            "👇 <b>Son karar sizin. Aşağıdaki ankete oy verin.</b>",
        ])

        return "\n".join(lines)

    def format_poll_question(self, council: CouncilDecision) -> str:
        """Anket sorusu."""
        return (
            f"{council.home} vs {council.away}\n"
            f"Oran: {council.odds:.2f} | EV: {council.ev_pct:+.1%}\n"
            f"Konsey: {council.council_verdict} "
            f"({council.yes_count}E/{council.no_count}H)"
        )

    def get_poll_options(self) -> list[str]:
        """Anket seçenekleri."""
        return [
            "✅ OYNA – Konseye katılıyorum",
            "⚡ OYNA – Ama yarı stake ile",
            "⏸️ BEKLE – Canlı bahiste değerlendireceğim",
            "❌ PAS GEÇ – Oynamıyorum",
        ]

    # ─────────────────────────────────────────────
    #  TELEGRAM GÖNDERİM
    # ─────────────────────────────────────────────
    async def send_poll(self, council: CouncilDecision,
                          chat_id: str | None = None) -> CouncilDecision:
        """Konsey kararını Telegram anketi olarak gönder."""
        if not self._notifier:
            logger.debug("[AgentPoll] Notifier yok – offline mod.")
            return council

        # 1) Detay mesajı gönder
        detail_text = self.format_council_message(council)
        try:
            detail_msg_id = await self._notifier.send(
                detail_text, chat_id=chat_id,
                parse_mode="HTML", return_message_id=True,
            )
            if isinstance(detail_msg_id, int):
                council.detail_message_id = detail_msg_id
        except Exception as e:
            logger.debug(f"[AgentPoll] Detay mesajı hatası: {e}")

        # 2) Anket gönder
        try:
            poll_msg_id = await self._send_telegram_poll(
                council, chat_id=chat_id,
            )
            if isinstance(poll_msg_id, int):
                council.poll_message_id = poll_msg_id
        except Exception as e:
            logger.debug(f"[AgentPoll] Anket gönderim hatası: {e}")
            # Fallback: inline butonlar
            await self._send_inline_buttons(council, chat_id)

        return council

    async def _send_telegram_poll(self, council: CouncilDecision,
                                    chat_id: str | None = None) -> int | None:
        """Telegram Poll API ile anket gönder."""
        if not self._notifier or not hasattr(self._notifier, '_bot'):
            return None

        bot = self._notifier._bot
        target = chat_id or self._notifier._chat_id
        if not bot or not target:
            return None

        try:
            question = self.format_poll_question(council)
            options = self.get_poll_options()

            msg = await bot.send_poll(
                chat_id=target,
                question=question,
                options=options,
                is_anonymous=False,
                allows_multiple_answers=False,
            )
            logger.info(
                f"[AgentPoll] Anket gönderildi: "
                f"{council.home} vs {council.away} "
                f"(msg_id={msg.message_id})"
            )
            return msg.message_id
        except Exception as e:
            logger.debug(f"[AgentPoll] Poll API hatası: {e}")
            return None

    async def _send_inline_buttons(self, council: CouncilDecision,
                                     chat_id: str | None = None) -> None:
        """Fallback: Inline butonlar ile oylama."""
        if not self._notifier or not TELEGRAM_OK:
            return

        try:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        "✅ OYNA", callback_data=f"poll_yes_{council.match_id}",
                    ),
                    InlineKeyboardButton(
                        "⚡ YARI", callback_data=f"poll_half_{council.match_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "⏸️ BEKLE", callback_data=f"poll_wait_{council.match_id}",
                    ),
                    InlineKeyboardButton(
                        "❌ PAS", callback_data=f"poll_no_{council.match_id}",
                    ),
                ],
            ])

            text = (
                f"🗳️ <b>OYUNUZU VERİN</b>\n"
                f"⚽ {council.home} vs {council.away}\n"
                f"Konsey: {council.consensus_emoji} {council.council_verdict}"
            )

            await self._notifier.send(
                text, chat_id=chat_id,
                parse_mode="HTML", reply_markup=keyboard,
            )
        except Exception as e:
            logger.debug(f"[AgentPoll] Inline button hatası: {e}")

    # ─────────────────────────────────────────────
    #  KULLANICI OYU KAYIT
    # ─────────────────────────────────────────────
    def record_user_vote(self, match_id: str, user_choice: str) -> None:
        """Kullanıcının oyunu kaydet."""
        self._user_votes[match_id] = user_choice

        # İlgili konsey kararını bul
        for c in self._history:
            if c.match_id == match_id:
                c.user_vote = user_choice
                break

        logger.info(
            f"[AgentPoll] Kullanıcı oyu kaydedildi: "
            f"{match_id} → {user_choice}"
        )

    def record_match_result(self, match_id: str,
                              result: str) -> dict | None:
        """Maç sonucunu kaydet ve doğruluk hesapla.

        result: "won" | "lost" | "void"
        """
        for c in self._history:
            if c.match_id == match_id:
                council_correct = (
                    (c.council_verdict == "OYNA" and result == "won")
                    or (c.council_verdict == "PAS GEÇ" and result == "lost")
                )
                user_vote = c.user_vote
                user_played = user_vote in ("EVET", "OYNA", "✅ OYNA – Konseye katılıyorum")
                user_correct = (
                    (user_played and result == "won")
                    or (not user_played and result == "lost")
                )

                return {
                    "match_id": match_id,
                    "council_verdict": c.council_verdict,
                    "user_vote": user_vote,
                    "result": result,
                    "council_correct": council_correct,
                    "user_correct": user_correct,
                }
        return None

    # ─────────────────────────────────────────────
    #  İSTATİSTİKLER
    # ─────────────────────────────────────────────
    def get_stats(self) -> PollStats:
        """Anket istatistikleri."""
        stats = PollStats(total_polls=len(self._history))

        agreed = 0
        disagreed = 0
        for c in self._history:
            if not c.user_vote:
                continue
            user_yes = c.user_vote in ("EVET", "OYNA",
                                        "✅ OYNA – Konseye katılıyorum",
                                        "⚡ OYNA – Ama yarı stake ile")
            council_yes = c.council_verdict == "OYNA"
            if user_yes == council_yes:
                agreed += 1
            else:
                disagreed += 1

        stats.user_agreed_with_council = agreed
        stats.user_disagreed = disagreed

        return stats

    def get_history(self) -> list[CouncilDecision]:
        """Tüm konsey kararları."""
        return list(self._history)

    def get_last_decision(self) -> CouncilDecision | None:
        """Son konsey kararı."""
        return self._history[-1] if self._history else None
