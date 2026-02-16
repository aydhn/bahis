"""
entropy_meter.py – Shannon Entropisi ile Bilgi Fiziği.

Bir maçın ne kadar "belirsiz" olduğunu standart sapma ile
ölçemezsiniz. Bunu Bilgi Fiziği (Information Theory) ile
ölçmelisiniz.

Entropi: Bir sistemdeki düzensizlik veya sürpriz miktarı.
  - 0 bit → Kesin sonuç (bilgi yok)
  - 1 bit → İkili seçim (yazı-tura)
  - 1.58 bit → 3 eşit olasılıklı sonuç (1X2)
  - 2.5+ bit → Kaotik sistem → TÜM BAHİSLERİ İPTAL ET

Kavramlar:
  - Shannon Entropy: H(X) = -Σ p(x) log₂ p(x)
  - KL-Divergence: Model ile piyasa arasındaki bilgi farkı
  - Mutual Information: İki değişken arasındaki bilgi paylaşımı
  - Cross Entropy: Modelin ne kadar "doğru" tahmin ettiği

Kill Switch:
  Entropi > 2.5 bit → "Kaotik" → Tüm bahisler iptal!
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from scipy.stats import entropy as scipy_entropy
    from scipy.special import rel_entr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


@dataclass
class EntropyReport:
    """Entropi analiz raporu."""
    match_id: str = ""
    home_team: str = ""
    away_team: str = ""
    # Shannon Entropy
    match_entropy: float = 0.0         # Maç entropi (bit)
    home_entropy: float = 0.0          # Ev sahibi geçmiş entropi
    away_entropy: float = 0.0          # Deplasman geçmiş entropi
    league_entropy: float = 0.0        # Lig genel entropi
    # Karşılaştırma
    kl_divergence: float = 0.0         # Model vs Piyasa KL-diverjansı
    cross_entropy: float = 0.0         # Model çapraz entropi
    mutual_info: float = 0.0           # Özellikler arası karşılıklı bilgi
    # Karar
    is_chaotic: bool = False           # Entropi çok yüksek
    chaos_level: str = "stable"        # stable | uncertain | volatile | chaotic
    kill_switch: bool = False          # TÜM BAHİSLERİ İPTAL ET
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  ENTROPİ HESAPLAYICILARI
# ═══════════════════════════════════════════════
def shannon_entropy(probs: np.ndarray, base: int = 2) -> float:
    """Shannon Entropy hesapla (bit cinsinden).

    H(X) = -Σ p(x) log₂ p(x)

    Örnekler:
        [1.0, 0.0, 0.0] → 0.0 bit (kesin sonuç)
        [0.5, 0.5, 0.0] → 1.0 bit (yazı-tura)
        [0.33, 0.33, 0.34] → 1.58 bit (max belirsizlik)
    """
    probs = np.array(probs, dtype=np.float64)
    probs = probs[probs > 1e-12]

    if len(probs) == 0:
        return 0.0

    # Normalize
    probs = probs / probs.sum()

    if SCIPY_OK:
        return float(scipy_entropy(probs, base=base))

    return float(-np.sum(probs * np.log2(probs)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL-Divergence: D_KL(P || Q).

    Model (P) ile Piyasa (Q) arasındaki bilgi farkı.
    D_KL = 0 → İkisi aynı şeyi söylüyor
    D_KL > 0.5 → Büyük ayrışma (değer fırsatı veya model hatası)
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)

    # Sıfır koruma
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    if SCIPY_OK:
        return float(np.sum(rel_entr(p, q)))

    return float(np.sum(p * np.log(p / q)))


def cross_entropy(true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
    """Cross Entropy: H(P, Q) = -Σ p(x) log q(x).

    Modelin tahmin kalitesini ölçer.
    Düşük = iyi tahmin, Yüksek = kötü tahmin.
    """
    p = np.array(true_probs, dtype=np.float64)
    q = np.array(pred_probs, dtype=np.float64)
    q = np.maximum(q, 1e-10)
    p = np.maximum(p, 1e-10)
    p = p / p.sum()
    q = q / q.sum()
    return float(-np.sum(p * np.log2(q)))


def mutual_information(joint_probs: np.ndarray) -> float:
    """Mutual Information: I(X; Y).

    İki değişken arasındaki bilgi paylaşımı.
    0 → Bağımsız, Yüksek → Güçlü bağımlılık.
    """
    joint = np.array(joint_probs, dtype=np.float64)
    joint = joint / joint.sum()

    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 1e-12:
                mi += joint[i, j] * np.log2(
                    joint[i, j] / (marginal_x[i] * marginal_y[j] + 1e-12)
                )
    return max(0.0, float(mi))


# ═══════════════════════════════════════════════
#  ENTROPİ ÖLÇER (Ana Sınıf)
# ═══════════════════════════════════════════════
class EntropyMeter:
    """Shannon Entropy bazlı maç/lig belirsizlik ölçer.

    Kullanım:
        meter = EntropyMeter()
        # Maç entropi analizi
        report = meter.analyze_match(
            match_id="GS_FB",
            model_probs={"prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20},
            market_odds={"home": 1.80, "draw": 3.50, "away": 4.50},
        )
        # Kill switch kontrolü
        if report.kill_switch:
            cancel_all_bets()
    """

    # Kaos eşikleri (bit cinsinden)
    STABLE_MAX = 1.20              # < 1.20 bit → stabil
    UNCERTAIN_MAX = 1.45           # < 1.45 bit → belirsiz
    VOLATILE_MAX = 2.00            # < 2.00 bit → volatil
    CHAOTIC_THRESHOLD = 2.50       # ≥ 2.50 bit → KAOTİK → KILL SWITCH

    # KL-Divergence eşikleri
    KL_VALUE_THRESHOLD = 0.10      # Model-piyasa ayrışması → değer
    KL_DANGER_THRESHOLD = 0.50     # Çok büyük ayrışma → model hatası?

    def __init__(self, kill_threshold: float = 2.50):
        self._kill_threshold = kill_threshold
        self._league_cache: dict[str, float] = {}
        self._team_cache: dict[str, list[float]] = {}
        logger.debug(f"[Entropy] Meter başlatıldı (kill={kill_threshold} bit).")

    # ═══════════════════════════════════════════
    #  MAÇ ANALİZİ
    # ═══════════════════════════════════════════
    def analyze_match(self, match_id: str = "",
                       model_probs: dict | None = None,
                       market_odds: dict | None = None,
                       home_team: str = "",
                       away_team: str = "",
                       home_history: list[str] | None = None,
                       away_history: list[str] | None = None,
                       ) -> EntropyReport:
        """Maç entropi analizi.

        Args:
            model_probs: {"prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20}
            market_odds: {"home": 1.80, "draw": 3.50, "away": 4.50}
            home_history: Son maç sonuçları ["W", "W", "D", "L", "W"]
            away_history: Son maç sonuçları ["L", "D", "W", "L", "D"]
        """
        report = EntropyReport(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
        )

        # ── Maç entropi ──
        if model_probs:
            probs = np.array([
                model_probs.get("prob_home", 0.33),
                model_probs.get("prob_draw", 0.33),
                model_probs.get("prob_away", 0.34),
            ])
            probs = np.maximum(probs, 1e-10)
            probs = probs / probs.sum()
            report.match_entropy = round(shannon_entropy(probs), 4)

        # ── Takım geçmiş entropileri ──
        if home_history:
            report.home_entropy = round(
                self._history_entropy(home_history), 4,
            )
        if away_history:
            report.away_entropy = round(
                self._history_entropy(away_history), 4,
            )

        # ── KL-Divergence: Model vs Piyasa ──
        if model_probs and market_odds:
            oh = market_odds.get("home", 2)
            od = market_odds.get("draw", 3.5)
            oa = market_odds.get("away", 4)

            if oh > 1 and od > 1 and oa > 1:
                market_probs = np.array([1 / oh, 1 / od, 1 / oa])
                market_probs = market_probs / market_probs.sum()

                model_arr = np.array([
                    model_probs.get("prob_home", 0.33),
                    model_probs.get("prob_draw", 0.33),
                    model_probs.get("prob_away", 0.34),
                ])
                model_arr = model_arr / model_arr.sum()

                report.kl_divergence = round(
                    kl_divergence(model_arr, market_probs), 4,
                )
                report.cross_entropy = round(
                    cross_entropy(model_arr, market_probs), 4,
                )

        # ── Kaos seviyesi ──
        h = report.match_entropy
        if h < self.STABLE_MAX:
            report.chaos_level = "stable"
        elif h < self.UNCERTAIN_MAX:
            report.chaos_level = "uncertain"
        elif h < self.VOLATILE_MAX:
            report.chaos_level = "volatile"
        else:
            report.chaos_level = "chaotic"
            report.is_chaotic = True

        # ── Kill Switch ──
        report.kill_switch = h >= self._kill_threshold
        if report.kill_switch:
            report.recommendation = (
                f"KILL SWITCH: Entropi = {h:.2f} bit (eşik: {self._kill_threshold}). "
                f"Bu maç KAOTİK – tüm bahisler İPTAL!"
            )
        elif report.is_chaotic:
            report.recommendation = (
                f"UYARI: Yüksek entropi ({h:.2f} bit). "
                f"Düşük stake veya pas geç."
            )
        elif report.kl_divergence > self.KL_VALUE_THRESHOLD:
            report.recommendation = (
                f"DEĞER: Model-piyasa ayrışması yüksek "
                f"(KL={report.kl_divergence:.3f}). Fırsat olabilir."
            )
        else:
            report.recommendation = (
                f"Stabil ({h:.2f} bit). Normal bahis."
            )

        return report

    def _history_entropy(self, history: list[str]) -> float:
        """Takım geçmiş sonuçlarının entropisini hesapla."""
        if not history:
            return 0.0

        counts = {"W": 0, "D": 0, "L": 0}
        for r in history:
            r_upper = r.upper()
            if r_upper in counts:
                counts[r_upper] += 1

        total = sum(counts.values())
        if total == 0:
            return 0.0

        probs = np.array([c / total for c in counts.values()])
        return shannon_entropy(probs)

    # ═══════════════════════════════════════════
    #  TOPLU ANALİZ
    # ═══════════════════════════════════════════
    def analyze_batch(self, matches: list[dict]) -> list[EntropyReport]:
        """Birden fazla maçı analiz et."""
        reports = []
        for m in matches:
            probs = {}
            if "prob_home" in m:
                probs = {
                    "prob_home": m["prob_home"],
                    "prob_draw": m.get("prob_draw", 0.33),
                    "prob_away": m.get("prob_away", 0.34),
                }
            odds = {}
            if "home_odds" in m:
                odds = {
                    "home": m["home_odds"],
                    "draw": m.get("draw_odds", 3.5),
                    "away": m.get("away_odds", 4.0),
                }
            report = self.analyze_match(
                match_id=m.get("match_id", ""),
                model_probs=probs or None,
                market_odds=odds or None,
                home_team=m.get("home_team", ""),
                away_team=m.get("away_team", ""),
            )
            reports.append(report)
        return reports

    def filter_non_chaotic(self, bets: list[dict],
                            prob_key: str = "confidence") -> list[dict]:
        """Kaotik maçları bahis listesinden çıkar (Kill Switch)."""
        safe = []
        killed = 0

        for bet in bets:
            probs = {
                "prob_home": bet.get("prob_home", 0.33),
                "prob_draw": bet.get("prob_draw", 0.33),
                "prob_away": bet.get("prob_away", 0.34),
            }
            report = self.analyze_match(
                match_id=bet.get("match_id", ""),
                model_probs=probs,
            )
            if report.kill_switch:
                killed += 1
                bet["entropy_killed"] = True
                bet["entropy"] = report.match_entropy
                logger.warning(
                    f"[Entropy] KILL: {bet.get('match_id','')} → "
                    f"{report.match_entropy:.2f} bit (KAOTİK)"
                )
                continue

            bet["entropy"] = report.match_entropy
            bet["chaos_level"] = report.chaos_level
            safe.append(bet)

        if killed:
            logger.info(
                f"[Entropy] {killed} maç KILL SWITCH → iptal. "
                f"Güvenli: {len(safe)}"
            )
        return safe

    # ═══════════════════════════════════════════
    #  LİG ENTROPİSİ
    # ═══════════════════════════════════════════
    def league_entropy(self, results: list[dict],
                        league: str = "") -> float:
        """Bir ligin genel entropisini hesapla.

        Yüksek entropi → "Bu ligde analiz zor, dikkatli ol."
        """
        if not results:
            return 0.0

        outcomes = []
        for r in results:
            hg = r.get("home_goals", 0)
            ag = r.get("away_goals", 0)
            if hg > ag:
                outcomes.append("H")
            elif hg == ag:
                outcomes.append("D")
            else:
                outcomes.append("A")

        total = len(outcomes)
        counts = {
            "H": outcomes.count("H"),
            "D": outcomes.count("D"),
            "A": outcomes.count("A"),
        }
        probs = np.array([c / total for c in counts.values()])
        h = shannon_entropy(probs)

        if league:
            self._league_cache[league] = h

        return round(h, 4)
