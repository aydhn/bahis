"""
philosophical_engine.py – Felsefi Analiz Motoru (Epistemic Reasoning).

Bahis kararlarını salt sayısal modellerin ötesinde, epistemik ve
felsefi çerçevelerle sorgular. "Bir model yüksek olasılık verdi"
yetmez — "neden biliyoruz?" sorusunu sorar.

Kavramlar:
  - Epistemic Uncertainty: "Bilmediğimizi bilmek" — yetersiz veri
    kaynaklı belirsizlik (aleatorik ≠ epistemik)
  - Black Swan Awareness: Modelin hiç görmediği olaylar için hazırlık
  - Dunning-Kruger Filter: Model aşırı özgüvenli mi? Kalibrasyon.
  - Antifragility Score: Kayıplardan öğreniyor muyuz?
  - Falsifiability: Model yanlışlanabilir mi? Test edilebilir tahmin.
  - Meta-Uncertainty: Belirsizliğin belirsizliği — 2. derece kuşku
  - Wisdom of Crowds vs Herd: Piyasa bilge mi, sürü mü?
  - Regret Minimization: Maksimum pişmanlığı minimize et
  - Lindy Effect: Daha uzun süredir çalışan stratejiler daha güvenilir
  - Skin in the Game: Modelin kendi tahminlerine güveni

Akış:
  1. Model çıktılarını al (olasılık, güven, rejim)
  2. Epistemik filtreler uygula
  3. Felsefi sorgulamayı logla
  4. Anti-fragility ve calibration skorlarını hesapla
  5. Final "epistemik onay" ver veya reddet
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class EpistemicReport:
    """Felsefi analiz raporu."""
    match_id: str = ""
    # Girdiler
    model_probability: float = 0.0
    model_confidence: float = 0.0
    sample_size: int = 0
    # Epistemik filtreler
    dunning_kruger_score: float = 0.0  # 0=overfit, 1=calibrated
    black_swan_risk: float = 0.0       # 0=güvenli, 1=bilinmeyen yüksek
    antifragility: float = 0.0         # <0=fragile, 0=robust, >0=antifragile
    lindy_score: float = 0.0           # Uzun süredir başarılı mı?
    falsifiability: float = 0.0        # Tahmin test edilebilir mi?
    meta_uncertainty: float = 0.0      # 2. derece belirsizlik
    # Piyasa bilgeliği
    crowd_vs_herd: str = "unknown"     # "crowd_wisdom" | "herd_behavior" | "unknown"
    # Karar
    epistemic_approved: bool = False
    epistemic_score: float = 0.0       # Bileşik felsefi skor
    rejection_reasons: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  PHİLOSOPHİCAL ENGİNE (Ana Sınıf)
# ═══════════════════════════════════════════════
class PhilosophicalEngine:
    """Epistemic Reasoning motoru.

    Kullanım:
        phi = PhilosophicalEngine()

        report = phi.evaluate(
            probability=0.72, confidence=0.85,
            sample_size=200, strategy_age_days=45,
            recent_results=[1,1,0,1,0,0,1,1,1,0],
            market_odds_spread=0.03,
            match_id="gs_fb",
        )

        if not report.epistemic_approved:
            skip_bet()
    """

    def __init__(self, calibration_window: int = 200,
                 min_epistemic_score: float = 0.45):
        self._cal_window = calibration_window
        self._min_score = min_epistemic_score
        self._predictions: deque = deque(maxlen=500)
        self._results: deque = deque(maxlen=500)
        self._loss_learning: deque = deque(maxlen=100)

        logger.debug(
            f"[PhiloEngine] Başlatıldı: cal_win={calibration_window}, "
            f"min_score={min_epistemic_score}"
        )

    def evaluate(self, probability: float, confidence: float = 0.7,
                   sample_size: int = 100,
                   strategy_age_days: int = 30,
                   recent_results: list[int] | None = None,
                   market_odds_spread: float = 0.05,
                   model_count: int = 1,
                   match_id: str = "") -> EpistemicReport:
        """Felsefi değerlendirme uygula."""
        report = EpistemicReport(
            match_id=match_id,
            model_probability=probability,
            model_confidence=confidence,
            sample_size=sample_size,
        )

        # 1) Dunning-Kruger Filter (Overconfidence Detection)
        dk_score = self._dunning_kruger(probability, confidence, sample_size)
        report.dunning_kruger_score = round(dk_score, 4)
        if dk_score < 0.30:
            report.rejection_reasons.append(
                f"Dunning-Kruger: Model aşırı özgüvenli "
                f"(skor={dk_score:.2f}, güven={confidence:.0%} vs "
                f"N={sample_size})"
            )
            report.reflections.append(
                "Soru: Bu özgüven veri miktarıyla destekleniyor mu? "
                "Küçük örneklemde yüksek güven, bilgi değil yanılsamadır."
            )

        # 2) Black Swan Risk
        bs_risk = self._black_swan_risk(sample_size, probability, model_count)
        report.black_swan_risk = round(bs_risk, 4)
        if bs_risk > 0.70:
            report.rejection_reasons.append(
                f"Black Swan: Görülmemiş olay riski yüksek ({bs_risk:.0%})"
            )
            report.reflections.append(
                "Taleb'in uyarısı: 'Tarih sürüyor sırf görülmemiş olay "
                "henüz olmadığı için.' Model dışı senaryolara hazır mısın?"
            )

        # 3) Antifragility
        af_score = self._antifragility(recent_results)
        report.antifragility = round(af_score, 4)
        if af_score < -0.3:
            report.reflections.append(
                "Fragile sistem: Kayıplardan öğrenmiyor, aksine kırılıyor. "
                "Antifragil strateji kayıptan güçlenerek çıkar."
            )

        # 4) Lindy Effect
        lindy = self._lindy_score(strategy_age_days)
        report.lindy_score = round(lindy, 4)
        if lindy < 0.3:
            report.reflections.append(
                "Lindy Etkisi: Genç strateji. 'Ayakta kalma süresi arttıkça "
                "beklenen kalan ömür de artar.' Henüz kendini kanıtlamamış."
            )

        # 5) Falsifiability
        fals = self._falsifiability(probability)
        report.falsifiability = round(fals, 4)
        if fals < 0.3:
            report.reflections.append(
                "Popper: Yanlışlanamayan tahmin bilimsel değildir. "
                "p=0.50'ye yakın tahminler test edilemez."
            )

        # 6) Meta-Uncertainty
        meta = self._meta_uncertainty(confidence, sample_size, model_count)
        report.meta_uncertainty = round(meta, 4)
        if meta > 0.7:
            report.rejection_reasons.append(
                f"Meta-belirsizlik çok yüksek ({meta:.0%}): "
                f"Belirsizliğin kendisi belirsiz."
            )

        # 7) Crowd vs Herd
        crowd = self._crowd_vs_herd(probability, market_odds_spread)
        report.crowd_vs_herd = crowd
        if crowd == "herd_behavior":
            report.reflections.append(
                "Sürü davranışı tespit: Piyasa çok dar spread ile "
                "aynı yöne yığılmış. Contrarian fırsat olabilir ama "
                "sürüye karşı gitmek de risk."
            )

        # 8) Bileşik Epistemik Skor
        score = self._composite_score(
            dk_score, bs_risk, af_score, lindy, fals, meta,
        )
        report.epistemic_score = round(score, 4)

        # 9) Karar
        if report.rejection_reasons:
            report.epistemic_approved = False
        elif score < self._min_score:
            report.epistemic_approved = False
            report.rejection_reasons.append(
                f"Bileşik epistemik skor yetersiz: "
                f"{score:.2f} < {self._min_score:.2f}"
            )
        else:
            report.epistemic_approved = True
            report.reflections.append(
                f"Epistemik onay: Skor={score:.2f}. "
                f"Tüm felsefi filtreler geçildi."
            )

        self._log_report(report)
        return report

    def record_prediction(self, predicted_prob: float, actual: int) -> None:
        """Tahmin-sonuç kaydı (kalibrasyon için)."""
        self._predictions.append(predicted_prob)
        self._results.append(actual)

    # ═══════════════════════════════════════════
    #  FELSEFİ FİLTRELER
    # ═══════════════════════════════════════════
    def _dunning_kruger(self, prob: float, conf: float, n: int) -> float:
        """Dunning-Kruger filtresi.

        Yüksek güven + küçük örneklem = overconfidence.
        Kalibrasyon: tahmin edilen olasılık gerçek win rate'e ne kadar yakın?
        """
        # Örneklem penaltisi: N<50 → güven anlamsız
        n_factor = min(n / 200, 1.0)

        # Güven-belirsizlik uyumu
        # p=0.5'e yakınsa güven düşük olmalı, p=0.9'sa yüksek olabilir
        expected_confidence = abs(2 * prob - 1)  # p=0.5→0, p=1→1
        conf_gap = max(conf - expected_confidence - 0.1, 0)

        # Kalibrasyon (geçmiş verilerle)
        cal_score = 1.0
        if len(self._predictions) >= 30:
            preds = np.array(list(self._predictions)[-200:])
            acts = np.array(list(self._results)[-200:])
            # Brier score benzeri kalibrasyon
            brier = np.mean((preds - acts) ** 2)
            cal_score = max(1 - brier * 4, 0)  # brier 0=perfect, 0.25=random

        dk = n_factor * (1 - conf_gap) * cal_score
        return float(np.clip(dk, 0, 1))

    def _black_swan_risk(self, n: int, prob: float,
                           model_count: int) -> float:
        """Black Swan riski.

        Az veri + tek model + extreme probability = yüksek risk.
        """
        data_risk = max(1 - n / 500, 0)
        model_risk = max(1 - model_count / 5, 0)  # Tek model tehlikeli
        extreme = 2 * abs(prob - 0.5)  # p=0.5→0, p=0.9→0.8

        risk = 0.4 * data_risk + 0.3 * model_risk + 0.3 * extreme
        return float(np.clip(risk, 0, 1))

    def _antifragility(self, results: list[int] | None) -> float:
        """Antifragility skoru.

        Kayıp sonrası performans iyileşiyorsa → antifragile (>0)
        Kayıp sonrası kötüleşiyorsa → fragile (<0)
        """
        if not results or len(results) < 10:
            return 0.0

        r = np.array(results)
        # Kayıp sonrası win oranı vs kayıp öncesi
        loss_indices = np.where(r == 0)[0]
        if len(loss_indices) < 3:
            return 0.0

        post_loss_wins = 0
        post_loss_total = 0
        for idx in loss_indices:
            if idx + 1 < len(r):
                post_loss_total += 1
                if r[idx + 1] == 1:
                    post_loss_wins += 1

        if post_loss_total == 0:
            return 0.0

        post_loss_wr = post_loss_wins / post_loss_total
        overall_wr = np.mean(r)

        # Pozitif → kayıptan sonra daha iyi = antifragile
        af = (post_loss_wr - overall_wr) * 2
        return float(np.clip(af, -1, 1))

    def _lindy_score(self, age_days: int) -> float:
        """Lindy Effect skoru.

        Daha uzun süredir çalışan strateji → daha güvenilir.
        Logaritmik: ilk günler kritik, sonra yavaş artış.
        """
        if age_days <= 0:
            return 0.0
        return float(np.clip(math.log(1 + age_days) / math.log(365), 0, 1))

    def _falsifiability(self, prob: float) -> float:
        """Yanlışlanabilirlik skoru.

        p=0.50 → test edilemez (yanlışlanamaz)
        p=0.90 → güçlü tahmin, kolayca yanlışlanabilir
        """
        return float(abs(2 * prob - 1))

    def _meta_uncertainty(self, conf: float, n: int,
                            model_count: int) -> float:
        """2. derece belirsizlik.

        "Belirsizliğin kendisinden ne kadar eminiz?"
        """
        # Az model → meta-belirsizlik yüksek
        model_factor = max(1 - model_count / 8, 0)
        # Az veri → parametre belirsizliği
        data_factor = max(1 - n / 300, 0)
        # Güven düşükse zaten kabul ediliyor
        conf_factor = max(1 - conf, 0)

        meta = 0.4 * model_factor + 0.3 * data_factor + 0.3 * conf_factor
        return float(np.clip(meta, 0, 1))

    def _crowd_vs_herd(self, prob: float,
                         spread: float) -> str:
        """Piyasa bilgeliği mi sürü davranışı mı?

        Dar spread (< %2) + extreme odds = sürü
        Geniş spread + moderate odds = bilgelik
        """
        if spread < 0.02 and abs(prob - 0.5) > 0.2:
            return "herd_behavior"
        if spread > 0.05:
            return "crowd_wisdom"
        return "unknown"

    def _composite_score(self, dk: float, bs: float, af: float,
                           lindy: float, fals: float,
                           meta: float) -> float:
        """Bileşik epistemik skor."""
        score = (
            dk * 0.25           # Kalibrasyon
            + (1 - bs) * 0.20   # Black Swan güvenliği
            + (af + 1) / 2 * 0.10  # Antifragility [0,1]
            + lindy * 0.15      # Uzun ömür
            + fals * 0.15       # Yanlışlanabilirlik
            + (1 - meta) * 0.15  # Meta-kesinlik
        )
        return float(np.clip(score, 0, 1))

    def _log_report(self, r: EpistemicReport) -> None:
        level = "info" if r.epistemic_approved else "warning"
        msg = (
            f"[PhiloEngine] {r.match_id}: "
            f"score={r.epistemic_score:.2f}, "
            f"DK={r.dunning_kruger_score:.2f}, "
            f"BS={r.black_swan_risk:.2f}, "
            f"AF={r.antifragility:.2f}, "
            f"Lindy={r.lindy_score:.2f}, "
            f"{'✅' if r.epistemic_approved else '❌'}"
        )
        getattr(logger, level)(msg)
        for ref in r.reflections:
            logger.debug(f"[PhiloEngine] 💭 {ref}")
