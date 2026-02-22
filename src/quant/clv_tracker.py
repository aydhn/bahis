"""
clv_tracker.py – Closing Line Value (CLV) Takibi.

Botun başarısını sadece "Kazandı/Kaybetti" ile ölçme.
Asıl soru: "Ben bahsi aldığımda oran 2.10'du, maç başladığında
1.80'e düştü mü?"

Oran düştüyse → Piyasa seninle aynı fikre gelmiş demektir.
Kaybetsen bile matematiksel olarak DOĞRU yoldasın.

CLV = (Alınan Oran / Kapanış Oranı) - 1
CLV > 0 → Doğru yolda (uzun vadede kazanırsın)
CLV < 0 → Piyasanın arkasındasın (uzun vadede kaybedersin)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from loguru import logger


@dataclass
class CLVRecord:
    """Tek bir bahisin CLV kaydı."""
    match_id: str
    selection: str           # home / draw / away / over / under
    entry_odds: float        # Bahsi aldığındaki oran
    closing_odds: float = 0.0  # Maç başlangıcındaki oran (kapanış)
    result: str = ""         # won / lost / pending
    timestamp: str = ""
    clv: float = 0.0         # Hesaplanmış CLV
    clv_pct: float = 0.0     # CLV yüzde
    edge_at_entry: float = 0.0  # Giriş anındaki EV


class CLVTracker:
    """Closing Line Value takip ve analiz modülü.

    Neden önemli?
    - Win rate yanıltıcıdır (yüksek oranlarla kaybedebilirsin ama karda olursun)
    - CLV, bahisçinin matematiksel doğruluğunu ölçer
    - +EV bahis yapıp yapmadığını kanıtlar
    - Sharpe Ratio benzeri bir "yetenek metriği"dir
    """

    def __init__(self):
        self._records: list[CLVRecord] = []
        logger.debug("CLVTracker başlatıldı.")

    def record_entry(self, match_id: str, selection: str,
                     entry_odds: float, ev: float = 0.0) -> CLVRecord:
        """Bahis giriş kaydı oluştur."""
        record = CLVRecord(
            match_id=match_id,
            selection=selection,
            entry_odds=entry_odds,
            edge_at_entry=ev,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._records.append(record)
        logger.debug(f"CLV kayıt: {match_id} {selection} @{entry_odds:.2f}")
        return record

    def update_closing_odds(self, match_id: str, selection: str,
                            closing_odds: float) -> CLVRecord | None:
        """Maç başlangıcında kapanış oranını güncelle ve CLV hesapla."""
        for r in reversed(self._records):
            if r.match_id == match_id and r.selection == selection:
                r.closing_odds = closing_odds
                r.clv = self.calculate_clv(r.entry_odds, closing_odds)
                r.clv_pct = r.clv * 100
                logger.info(
                    f"CLV güncellendi: {match_id} {selection} "
                    f"| Giriş: {r.entry_odds:.2f} → Kapanış: {closing_odds:.2f} "
                    f"| CLV: {r.clv:+.3f} ({r.clv_pct:+.1f}%)"
                )
                return r
        return None

    def update_result(self, match_id: str, selection: str,
                      result: str) -> None:
        """Maç sonucunu güncelle (won/lost)."""
        for r in reversed(self._records):
            if r.match_id == match_id and r.selection == selection:
                r.result = result
                return

    @staticmethod
    def calculate_clv(entry_odds: float, closing_odds: float) -> float:
        """CLV = (Giriş Oranı / Kapanış Oranı) - 1

        Örnekler:
        - 2.10 giriş, 1.80 kapanış → CLV = +0.167 (+%16.7) ✓
        - 2.10 giriş, 2.30 kapanış → CLV = -0.087 (-%8.7) ✗
        """
        if closing_odds <= 1.0:
            return 0.0
        return (entry_odds / closing_odds) - 1.0

    def aggregate_stats(self) -> dict:
        """Tüm CLV verilerinin özet istatistikleri."""
        records_with_clv = [r for r in self._records if r.closing_odds > 1.0]
        if not records_with_clv:
            return {"status": "yeterli veri yok", "n": 0}

        clvs = np.array([r.clv for r in records_with_clv])
        n = len(clvs)
        positive = sum(1 for c in clvs if c > 0)
        won = [r for r in records_with_clv if r.result == "won"]
        lost = [r for r in records_with_clv if r.result == "lost"]

        return {
            "n_bets": n,
            "avg_clv": float(np.mean(clvs)),
            "median_clv": float(np.median(clvs)),
            "clv_std": float(np.std(clvs)),
            "positive_clv_rate": positive / n,
            "total_clv": float(np.sum(clvs)),
            "win_rate": len(won) / max(len(won) + len(lost), 1),
            "sharpe_like": float(np.mean(clvs) / max(np.std(clvs), 0.001)),
            "interpretation": self._interpret(float(np.mean(clvs))),
        }

    @staticmethod
    def _interpret(avg_clv: float) -> str:
        if avg_clv > 0.05:
            return "Mükemmel! Piyasanın çok önündesin – bu sürdürülebilir alpha."
        elif avg_clv > 0.02:
            return "İyi! Tutarlı pozitif CLV – doğru yoldasın."
        elif avg_clv > 0:
            return "Marjinal pozitif CLV – biraz daha hassasiyet gerekli."
        elif avg_clv > -0.02:
            return "Nötr – piyasayla aynı seviyedesin, edge yok."
        else:
            return "Negatif CLV – piyasanın arkasındasın, strateji revizyonu gerekli."

    def check_retraining_trigger(self, window: int = 50, threshold: float = -0.02) -> bool:
        """
        CLV drift kontrolü: Son N bahisteki ortalama CLV, threshold'un altına düştüyse
        strateji/model bozulmuş demektir (Negatif Drift).
        Bu durumda 'True' döner ve retraining tetiklenebilir.
        """
        records_with_clv = [r for r in self._records if r.closing_odds > 1.0]
        if len(records_with_clv) < window:
            return False
            
        recent_clvs = [r.clv for r in records_with_clv[-window:]]
        avg_recent = np.mean(recent_clvs)
        
        if avg_recent < threshold:
            logger.warning(
                f"🚨 ADAPTIVE RETRAINING TETİKLENDİ! "
                f"Son {window} bahis CLV: {avg_recent:+.3f} < {threshold:+.3f}"
            )
            return True
        return False


class CorrelationMatrix:
    """Bahis korelasyon matrisi – portföy varyansını düşürür.

    "X Takımı Gol Yemez" + "X Takımı Kazanır" = YÜKSEK korelasyon.
    Bu tür ilişkili bahisleri aynı kupona koymak riski artırır.

    Korelasyonu düşük bahisleri seçerek çeşitlendirme (diversification)
    yapar, toplam portföy varyansını düşürür.
    """

    # Bilinen bahis korelasyonları (domain knowledge)
    _KNOWN_CORRELATIONS: dict[tuple[str, str], float] = {
        ("home_win", "clean_sheet_home"): 0.65,
        ("home_win", "over_25"): 0.20,
        ("away_win", "clean_sheet_away"): 0.65,
        ("draw", "under_25"): 0.55,
        ("over_25", "btts_yes"): 0.70,
        ("under_25", "btts_no"): 0.65,
        ("home_win", "btts_yes"): 0.15,
        ("home_win", "under_25"): -0.10,
        ("over_25", "under_25"): -1.0,  # Tam ters
        ("btts_yes", "btts_no"): -1.0,
    }

    def __init__(self, max_portfolio_correlation: float = 0.40):
        self._max_corr = max_portfolio_correlation
        self._history: list[dict] = []
        logger.debug(f"CorrelationMatrix başlatıldı (max_corr={max_portfolio_correlation})")

    def get_correlation(self, market_a: str, market_b: str) -> float:
        """İki bahis pazarı arasındaki korelasyonu döndür."""
        if market_a == market_b:
            return 1.0

        key = (market_a, market_b)
        rev_key = (market_b, market_a)

        if key in self._KNOWN_CORRELATIONS:
            return self._KNOWN_CORRELATIONS[key]
        if rev_key in self._KNOWN_CORRELATIONS:
            return self._KNOWN_CORRELATIONS[rev_key]

        # Aynı maçın farklı pazarları → orta korelasyon
        return 0.30

    def portfolio_correlation(self, bets: list[dict]) -> float:
        """Portföydeki ortalama ikili korelasyon."""
        if len(bets) < 2:
            return 0.0

        correlations = []
        for i, a in enumerate(bets):
            for j, b in enumerate(bets):
                if i >= j:
                    continue
                m_a = a.get("market", "1X2") + "_" + a.get("selection", "home")
                m_b = b.get("market", "1X2") + "_" + b.get("selection", "home")

                # Aynı maç mı?
                same_match = a.get("match_id") == b.get("match_id")
                corr = self.get_correlation(m_a, m_b) if not same_match else 0.7

                correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def filter_diversified(self, candidates: list[dict],
                           max_bets: int = 5) -> list[dict]:
        """Korelasyonu düşük bahisleri seçerek portföy oluştur.

        Greedy algoritma:
        1. En yüksek EV'li bahsi al
        2. Kalan bahislerden, mevcut portföyle korelasyonu düşük olanı ekle
        3. Max korelasyon eşiğini aşanları reddet
        """
        if not candidates:
            return []

        # EV'ye göre sırala
        sorted_bets = sorted(candidates, key=lambda x: x.get("ev", 0), reverse=True)

        portfolio: list[dict] = [sorted_bets[0]]

        for bet in sorted_bets[1:]:
            if len(portfolio) >= max_bets:
                break

            # Mevcut portföyle korelasyonu kontrol et
            max_corr_with_portfolio = 0.0
            for existing in portfolio:
                m_new = bet.get("market", "1X2") + "_" + bet.get("selection", "home")
                m_exist = existing.get("market", "1X2") + "_" + existing.get("selection", "home")
                same_match = bet.get("match_id") == existing.get("match_id")

                if same_match:
                    corr = 0.8
                elif m_new == m_exist:
                    # Farklı maçlar, aynı market tipi → bağımsız sinyaller
                    corr = 0.20
                else:
                    corr = self.get_correlation(m_new, m_exist)
                max_corr_with_portfolio = max(max_corr_with_portfolio, abs(corr))

            if max_corr_with_portfolio <= self._max_corr:
                portfolio.append(bet)
            else:
                logger.debug(
                    f"Bahis reddedildi (yüksek korelasyon {max_corr_with_portfolio:.2f}): "
                    f"{bet.get('match_id', '')}"
                )

        avg_corr = self.portfolio_correlation(portfolio)
        logger.info(
            f"Portföy oluşturuldu: {len(portfolio)} bahis, "
            f"ort. korelasyon: {avg_corr:.2f}"
        )

        return portfolio

    def build_matrix(self, bets: list[dict]) -> np.ndarray:
        """Tam korelasyon matrisini oluştur (grafik için)."""
        n = len(bets)
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                m_i = bets[i].get("market", "1X2") + "_" + bets[i].get("selection", "home")
                m_j = bets[j].get("market", "1X2") + "_" + bets[j].get("selection", "home")
                corr = self.get_correlation(m_i, m_j)
                matrix[i][j] = corr
                matrix[j][i] = corr

        return matrix
