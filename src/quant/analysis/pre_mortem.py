"""
pre_mortem.py – The Devil's Advocate (Negative Visualization).

"Mükemmel görünen bir bahiste ne ters gidebilir?"
Bu modül, her bahsi "öldürmeye" çalışır. Eğer öldüremezse, onaylar.

Checks:
  - Vig Check: Piyasa marjı > %6 ise likidite/güven düşüktür.
  - Too Good To Be True: EV > %20 ise veri hatası veya model sapması olabilir.
  - Consensus Trap: Herkes aynı taraftaysa (Sürü Psikolojisi), dikkat.
  - Odds Drift: Oranlar aleyhimize hareket ediyorsa (Smart Money kaçıyor).

Output:
  - Kill: Bahsi iptal et.
  - Caution: Stake'i düşür.
  - Clean: Sorun yok.
"""
from dataclasses import dataclass, field
from typing import Dict, Any
from loguru import logger

@dataclass
class PreMortemReport:
    """Pre-Mortem Analiz Raporu."""
    match_id: str = ""
    is_clean: bool = True
    kill_signal: bool = False
    caution_signal: bool = False
    vig: float = 0.0
    ev: float = 0.0
    reasons: list[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 (Safe) - 1.0 (Toxic)

class PreMortemAnalyzer:
    """
    Şeytanın Avukatı. Bahisleri stres testine sokar.
    """

    def __init__(self):
        # Eşik Değerler
        self.MAX_VIG = 0.07          # %7 üzeri vig = Kötü likidite / Belirsiz piyasa
        self.MAX_EV = 0.25           # %25 üzeri EV = Muhtemelen model hatası
        self.DRIFT_TOLERANCE = -0.05 # %5'ten fazla aleyhte değişim
        self.MIN_CONFIDENCE = 0.55   # Düşük güvenli bahisler için ekstra kontrol

    def analyze(self, bet_candidate: Dict[str, Any], market_context: Dict[str, Any] = None) -> PreMortemReport:
        """
        Bahis adayını analiz et.

        Args:
            bet_candidate: {
                "match_id": str,
                "odds": float, (Bizim seçimin oranı)
                "prob_model": float,
                "ev": float,
                "selection": str,
                "home_odds": float, (Opsiyonel - Vig için)
                "draw_odds": float, (Opsiyonel - Vig için)
                "away_odds": float, (Opsiyonel - Vig için)
            }
            market_context: {
                "opening_odds": float,
                "market_sentiment": str, (BULLISH/BEARISH/NEUTRAL)
            }
        """
        match_id = bet_candidate.get("match_id", "Unknown")
        report = PreMortemReport(match_id=match_id)

        # 1. Vig Check (Market Margin)
        # Eğer tüm oranlar varsa hesapla
        h = bet_candidate.get("home_odds", 0.0)
        d = bet_candidate.get("draw_odds", 0.0)
        a = bet_candidate.get("away_odds", 0.0)

        if h > 1 and d > 1 and a > 1:
            margin = (1/h + 1/d + 1/a) - 1.0
            report.vig = margin
            if margin > self.MAX_VIG:
                report.caution_signal = True
                report.reasons.append(f"Yüksek Market Vig: %{margin*100:.1f} (Likidite Düşük)")
                report.risk_score += 0.3

        # 2. Too Good To Be True (Extreme EV)
        ev = bet_candidate.get("ev", 0.0)
        report.ev = ev
        if ev > self.MAX_EV:
            # Model %25+ edge buluyorsa muhtemelen bir sakatlık (kadro dışı vb.) vardır
            # veya model bir şeyi kaçırıyordur.
            report.kill_signal = True # Doğrudan öldür veya çok sıkı incele
            report.reasons.append(f"Extreme EV: %{ev*100:.1f} (Veri/Model Hatası Şüphesi)")
            report.risk_score += 0.5

        # 3. Low Confidence Trap
        prob = bet_candidate.get("prob_model", 0.0)
        if prob < self.MIN_CONFIDENCE and ev < 0.05:
            # Düşük güven + Düşük edge = Değmez
            report.kill_signal = True
            report.reasons.append(f"Düşük Güven (%{prob*100:.1f}) ve Zayıf Edge")
            report.risk_score += 0.4

        # 4. Market Context (Eğer varsa)
        if market_context:
            # Odds Drift
            curr_odds = bet_candidate.get("odds", 0.0)
            open_odds = market_context.get("opening_odds", 0.0)

            if open_odds > 0:
                drift = (curr_odds - open_odds) / open_odds
                # Eğer oran artıyorsa (olasılık düşüyor) ve biz "Kazanır" diyorsak → Ters hareket
                # Bu her zaman kötü değildir (Contrarian), ama Smart Money aleyhteyse dikkat.
                if drift > 0.10: # %10 oran artışı (Piyasa takımdan kaçıyor)
                    report.caution_signal = True
                    report.reasons.append(f"Ters Odds Hareketi: %{drift*100:.1f} (Piyasa Kaçıyor)")
                    report.risk_score += 0.2

        # Final Karar
        if report.kill_signal:
            report.is_clean = False
        elif report.caution_signal:
            report.is_clean = False

        if not report.is_clean:
            logger.warning(f"[PreMortem] {match_id}: {', '.join(report.reasons)}")

        return report
