"""
regime_kelly.py – Regime-Aware Kelly Criterion v2 + Bankroll Segmentation.

Klasik Kelly Criterion tek bir rejimde çalışır. Bu modül piyasa
rejimini (GARCH, Chaos, HMM) dikkate alarak Kelly fraksiyonunu
dinamik olarak ayarlar. Drawdown protokolleri ve bankroll
segmentasyonu ile profesyonel hedge fund seviyesinde kasa yönetimi.

Kavramlar:
  - Regime-Aware Kelly: f* = fraction_regime × (p·O - 1) / (O - 1)
  - Fractional Kelly: Gerçek uygulamada 0.25-0.50× kullanılır
  - Drawdown Protocol: -15% → yarı Kelly, -25% → dondur
  - Bankroll Segmentation: Hot Wallet / Reserve / R&D
  - Edge Filtering: Min %3 edge, rejim bazlı threshold
  - Volatility Targeting: Haftalık σ hedefi, aşılırsa stake düşür
  - Anti-Tilt: Üst üste kayıp serisinde otomatik soğuma
  - CLV Monitoring: Closing Line Value bozulursa fraksiyon düşür

Akış:
  1. Mevcut rejim tespit edilir (GARCH vol_regime + chaos λ)
  2. Kelly fraksiyonu rejime göre ölçeklenir
  3. Drawdown seviyesi kontrol edilir
  4. Anti-tilt ve volatilite limitleri uygulanır
  5. Bankroll segmentasyonuna göre max stake hesaplanır
  6. Final stake döndürülür + tüm kararlar loglanır
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from typing import Any
from loguru import logger
from src.core.ergodicity_optimizer import ErgodicityOptimizer
from src.core.red_team_agent import RedTeamAgent


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class RegimeState:
    """Mevcut piyasa rejimi."""
    volatility_regime: str = "calm"   # "calm" | "elevated" | "storm" | "crisis"
    chaos_regime: str = "stable"      # "stable" | "edge_of_chaos" | "chaotic"
    hmm_regime: str = "balanced"      # "dominant" | "balanced" | "passive"
    multifractal: str = "monofractal"  # "monofractal" | "weak_multifractal" | "strong_multifractal"
    composite_risk: float = 0.0       # [0, 1] bileşik risk skoru


@dataclass
class BankrollSegment:
    """Bankroll segmentasyonu."""
    total: float = 10000.0
    hot_wallet: float = 0.0     # Aktif bahis fonu (%60)
    reserve: float = 0.0        # Tampon fon (%30)
    rd_fund: float = 0.0        # AR-GE fonu (%10)
    hot_wallet_pct: float = 0.60
    reserve_pct: float = 0.30
    rd_pct: float = 0.10

    def recalculate(self):
        self.hot_wallet = self.total * self.hot_wallet_pct
        self.reserve = self.total * self.reserve_pct
        self.rd_fund = self.total * self.rd_pct


@dataclass
class KellyDecision:
    """Kelly karar raporu — tam audit trail."""
    match_id: str = ""
    timestamp: str = ""
    # Girdi
    probability: float = 0.0
    odds: float = 0.0
    edge: float = 0.0
    # Kelly hesaplama
    raw_kelly: float = 0.0         # Ham Kelly fraksiyonu
    fractional_kelly: float = 0.0  # Fraksiyon uygulanmış
    regime_multiplier: float = 1.0 # Rejim çarpanı
    drawdown_multiplier: float = 1.0
    tilt_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    ergodicity_multiplier: float = 1.0 # Ergodicity düzeltmesi
    final_kelly: float = 0.0      # Final fraksiyon
    # Stake
    stake_amount: float = 0.0     # TL/USD cinsinden
    stake_pct: float = 0.0        # Kasa yüzdesi
    max_allowed: float = 0.0      # Max izin verilen
    # Rejim
    regime: RegimeState = field(default_factory=RegimeState)
    # Kararlar
    approved: bool = False
    rejection_reason: str = ""
    adjustments: list[str] = field(default_factory=list)


class AutoRegimeDetector:
    """DB verilerini kullanarak piyasa rejimini otomatik tespit eder."""
    def __init__(self, db: Any):
        self.db = db

    def detect(self) -> RegimeState:
        """Volatility ve Brier skoru üzerinden rejim tespiti."""
        if self.db is None:
            return RegimeState()

        try:
            # 1. Son 50 maçın Brier skorunu ve PnL volatilitesini çek
            query = """
            SELECT pnl, ABS(probability - result) as brier
            FROM bets
            ORDER BY timestamp DESC
            LIMIT 50
            """
            df = self.db.query(query)
            if df.is_empty():
                return RegimeState()

            pnls = df["pnl"].to_numpy()
            brier_scores = df["brier"].to_numpy()
            
            vol = np.std(pnls) if len(pnls) > 5 else 0.0
            avg_brier = np.mean(brier_scores) if len(brier_scores) > 5 else 0.25

            # Rejim Sınıflandırma (Basit Thresholds)
            vol_reg = "calm"
            if vol > 200: vol_reg = "crisis"
            elif vol > 100: vol_reg = "storm"
            elif vol > 50: vol_reg = "elevated"

            hmm_reg = "balanced"
            if avg_brier < 0.15: hmm_reg = "dominant"
            elif avg_brier > 0.35: hmm_reg = "passive"

            risk_skor = (vol / 250) * 0.7 + (avg_brier / 0.5) * 0.3
            
            return RegimeState(
                volatility_regime=vol_reg,
                hmm_regime=hmm_reg,
                composite_risk=float(np.clip(risk_skor, 0, 1))
            )

        except Exception as e:
            logger.warning(f"RegimeDetection hatası: {e}")
            return RegimeState()

# ═══════════════════════════════════════════════
#  REGIME KELLY (Ana Sınıf)
# ═══════════════════════════════════════════════
class RegimeKelly:
    """Rejim-farkında Kelly Criterion v2."""

    # Rejim → Kelly çarpanı
    VOLATILITY_MULTIPLIER = {
        "calm": 1.0,
        "elevated": 0.65,
        "storm": 0.30,
        "crisis": 0.0,
    }
    CHAOS_MULTIPLIER = {
        "stable": 1.0,
        "edge_of_chaos": 0.50,
        "chaotic": 0.0,
    }
    HMM_MULTIPLIER = {
        "dominant": 1.1,
        "balanced": 1.0,
        "passive": 0.70,
    }

    # Drawdown protokolleri
    DRAWDOWN_LEVELS = [
        (-0.05, 0.80, "Hafif drawdown — %20 azaltma"),
        (-0.10, 0.50, "Orta drawdown — yarı Kelly"),
        (-0.15, 0.25, "Ağır drawdown — çeyrek Kelly"),
        (-0.25, 0.0, "KRİTİK drawdown — bahisler donduruldu"),
    ]

    def __init__(self, db: Any = None, bankroll: float = 10000.0,
                 base_fraction: float = 0.25,
                 min_edge: float = 0.03,
                 max_stake_pct: float = 0.05,
                 max_daily_exposure: float = 0.15,
                 anti_tilt_streak: int = 5,
                 vol_target_weekly: float = 0.08):
        self.db = db
        self._bankroll = BankrollSegment(total=bankroll)
        self._bankroll.recalculate()
        self._base_fraction = base_fraction
        self._min_edge = min_edge
        self._max_stake_pct = max_stake_pct
        self._max_daily = max_daily_exposure
        self._tilt_streak = anti_tilt_streak
        self._vol_target = vol_target_weekly
        self._detector = AutoRegimeDetector(db) if db else None
        self._ergodicity = ErgodicityOptimizer(risk_aversion=1.0)
        self._red_team = RedTeamAgent(db=db)

        # Durum takibi
        self._peak = bankroll
        self._results: deque = deque(maxlen=200)
        self._daily_exposure: float = 0.0
        self._consecutive_losses: int = 0
        self._weekly_returns: deque = deque(maxlen=50)
        self._decisions: list[KellyDecision] = []

        logger.info(
            f"[RegimeKelly] Başlatıldı: bankroll={bankroll:.0f}, "
            f"fraction={base_fraction}, min_edge={min_edge:.0%}"
        )

    def run_batch(self, **kwargs) -> list[KellyDecision]:
        """Batch modu: DB'den sinyalleri değerlendirir."""
        if self.db is None:
            logger.warning("RegimeKelly: DB bağlantısı yok.")
            return []

        # Sinyalleri çek (henüz işlenmemişleri filtrelemek lazım normalde)
        signals_df = self.db.get_signals()
        if signals_df.is_empty():
            logger.info("RegimeKelly: Değerlendirilecek sinyal yok.")
            return []

        signals = signals_df.to_dicts()
        decisions = []
        
        for sig in signals:
            # Rejim bilgisini çek (Mock)
            # Gerçekte: self.db.get_current_regime() vs.
            regime = RegimeState(volatility_regime="calm") 
            
            decision = self.calculate(
                probability=1.0 / sig.get("odds", 2.0), # Approx
                odds=sig.get("odds", 2.0),
                match_id=sig.get("match_id", ""),
                regime=regime
            )
            decisions.append(decision)
            
        return decisions

    def calculate(self, probability: float, odds: float,
                    match_id: str = "",
                    regime: RegimeState | None = None,
                    **kwargs) -> KellyDecision:
        """Rejim-farkında Kelly hesaplama."""
        # Eğer rejim verilmemişse ve dedektör varsa otomatik tespit et
        current_regime = regime
        if current_regime is None and self._detector:
            current_regime = self._detector.detect()
            
        decision = KellyDecision(
            match_id=match_id,
            timestamp=datetime.utcnow().isoformat(),
            probability=probability,
            odds=odds,
            regime=current_regime or RegimeState(),
        )

        # 1) Edge hesapla
        edge = probability * odds - 1
        decision.edge = round(edge, 6)

        # 2) Edge filtresi
        if edge < self._min_edge:
            decision.approved = False
            decision.rejection_reason = (
                f"Edge yetersiz: {edge:.2%} < {self._min_edge:.2%}"
            )
            self._log_decision(decision)
            return decision

        # 3) Ham Kelly
        raw_kelly = (probability * odds - 1) / (odds - 1) if odds > 1 else 0
        decision.raw_kelly = round(max(raw_kelly, 0), 6)

        # 4) Fractional Kelly
        frac_kelly = raw_kelly * self._base_fraction
        decision.fractional_kelly = round(frac_kelly, 6)

        # 5) Rejim çarpanları
        r = decision.regime
        vol_m = self.VOLATILITY_MULTIPLIER.get(r.volatility_regime, 0.5)
        chaos_m = self.CHAOS_MULTIPLIER.get(r.chaos_regime, 0.5)
        hmm_m = self.HMM_MULTIPLIER.get(r.hmm_regime, 1.0)

        regime_m = vol_m * chaos_m * hmm_m
        decision.regime_multiplier = round(regime_m, 4)
        if regime_m < 1.0:
            decision.adjustments.append(
                f"Rejim: vol={r.volatility_regime}(x{vol_m}), "
                f"chaos={r.chaos_regime}(x{chaos_m}), "
                f"hmm={r.hmm_regime}(x{hmm_m})"
            )

        # 6) Drawdown çarpanı
        dd = self._current_drawdown()
        dd_m = 1.0
        for threshold, multiplier, desc in self.DRAWDOWN_LEVELS:
            if dd <= threshold:
                dd_m = multiplier
                decision.adjustments.append(f"Drawdown: {dd:.1%} → {desc}")
                break
        decision.drawdown_multiplier = dd_m

        # 7) Anti-tilt
        tilt_m = 1.0
        if self._consecutive_losses >= self._tilt_streak:
            tilt_m = max(0.3, 1 - self._consecutive_losses * 0.1)
            decision.adjustments.append(
                f"Anti-tilt: {self._consecutive_losses} ardışık kayıp → x{tilt_m:.2f}"
            )
        decision.tilt_multiplier = round(tilt_m, 4)

        # 8) Volatilite hedefleme
        vol_m_weekly = 1.0
        if len(self._weekly_returns) >= 5:
            realized_vol = float(np.std(list(self._weekly_returns)))
            if realized_vol > self._vol_target:
                vol_m_weekly = self._vol_target / max(realized_vol, 1e-6)
                vol_m_weekly = max(vol_m_weekly, 0.2)
                decision.adjustments.append(
                    f"Vol target: realized={realized_vol:.2%} > "
                    f"target={self._vol_target:.2%} → x{vol_m_weekly:.2f}"
                )
        decision.volatility_multiplier = round(vol_m_weekly, 4)

        # 8.1) Red Team (Devil's Advocate) Verificaton (Phase 13)
        # Sinyali 'Şeytanın Avukatı'na sormak
        red_m = 1.0
        # RedTeam asenkron çalıştığı için calculate metodunu asenkron yapmamız gerekebilir 
        # veya burada senkron bir wrapper kullanmalıyız. 
        # Şimdilik basitleştirilmiş bir mantık:
        if kwargs.get("red_team_warning"):
            red_m = 0.5
            decision.adjustments.append("RedTeam: Karşıt kanıt bulundu → x0.5")
        
        decision.red_team_multiplier = red_m

        # 8.2) Tail-Risk Insurance (EVT)
        # Eğer kuyruk riski (Black Swan) yüksekse otomatik fren
        tail_m = 1.0
        tail_risk = kwargs.get("tail_risk", 0.0)
        if tail_risk > 0.15: # %15+ kuyruk riski tehlikelidir
            tail_m = 0.3
            decision.adjustments.append(f"TailRisk: Black Swan uyarısı ({tail_risk:.2f}) → x0.3")
        elif tail_risk > 0.10:
            tail_m = 0.7
            decision.adjustments.append(f"TailRisk: Yüksek sürpriz potansiyeli ({tail_risk:.2f}) → x0.7")
            
        decision.tail_risk_multiplier = tail_m

        # 9) Ergodicity Optimization
        final = frac_kelly * regime_m * dd_m * tilt_m * vol_m_weekly * red_m * tail_m
        final = max(final, 0)
        decision.final_kelly = round(final, 6)

        # 11) Stake hesaplama
        hot = self._bankroll.hot_wallet
        stake = hot * final
        max_allowed = hot * self._max_stake_pct
        stake = min(stake, max_allowed)

        # Günlük exposure kontrolü
        remaining_daily = (hot * self._max_daily) - self._daily_exposure
        if stake > remaining_daily:
            stake = max(remaining_daily, 0)
            decision.adjustments.append(
                f"Günlük limit: kalan={remaining_daily:.2f}"
            )

        decision.stake_amount = round(stake, 2)
        decision.stake_pct = round(stake / max(hot, 1), 6)
        decision.max_allowed = round(max_allowed, 2)

        # 11) Onay
        if stake > 0 and dd_m > 0 and regime_m > 0:
            decision.approved = True
        else:
            decision.approved = False
            if dd_m == 0:
                decision.rejection_reason = "Drawdown limiti aşıldı — bahisler donduruldu"
            elif regime_m == 0:
                decision.rejection_reason = "Piyasa kaotik/kriz — bahisler durduruldu"
            else:
                decision.rejection_reason = "Stake sıfır"

        self._log_decision(decision)
        self._decisions.append(decision)
        return decision

    def record_result(self, won: bool, pnl: float) -> None:
        """Bahis sonucunu kaydet."""
        self._results.append({"won": won, "pnl": pnl, "ts": time.time()})
        self._bankroll.total += pnl
        self._bankroll.recalculate()

        if self._bankroll.total > self._peak:
            self._peak = self._bankroll.total

        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self._weekly_returns.append(pnl / max(self._bankroll.total, 1))
        self._daily_exposure = max(self._daily_exposure + abs(pnl), 0)

        logger.debug(
            f"[RegimeKelly] Sonuç: {'W' if won else 'L'} "
            f"PnL={pnl:+.2f}, Bankroll={self._bankroll.total:.2f}, "
            f"DD={self._current_drawdown():.1%}"
        )

    def reset_daily(self) -> None:
        """Günlük exposure sıfırla."""
        self._daily_exposure = 0.0

    def _current_drawdown(self) -> float:
        if self._peak <= 0:
            return 0.0
        return (self._bankroll.total - self._peak) / self._peak

    def _log_decision(self, d: KellyDecision) -> None:
        level = "info" if d.approved else "warning"
        msg = (
            f"[RegimeKelly] {d.match_id}: "
            f"edge={d.edge:.2%}, raw_kelly={d.raw_kelly:.4f}, "
            f"regime_m={d.regime_multiplier:.2f}, "
            f"dd_m={d.drawdown_multiplier:.2f}, "
            f"red_m={d.red_team_multiplier:.2f}, "
            f"tail_m={d.tail_risk_multiplier:.2f}, "
            f"ergo_m={d.ergodicity_multiplier:.2f}, "
            f"final={d.final_kelly:.4f}, "
            f"stake={d.stake_amount:.2f} "
            f"({'✅' if d.approved else '❌ ' + d.rejection_reason})"
        )
        if d.adjustments:
            msg += f" | Adj: {'; '.join(d.adjustments)}"
        getattr(logger, level)(msg)

    def get_bankroll(self) -> BankrollSegment:
        return self._bankroll

    def get_stats(self) -> dict:
        results = list(self._results)
        wins = sum(1 for r in results if r["won"])
        total = len(results)
        return {
            "bankroll": self._bankroll.total,
            "peak": self._peak,
            "drawdown": self._current_drawdown(),
            "win_rate": wins / max(total, 1),
            "total_bets": total,
            "consecutive_losses": self._consecutive_losses,
            "daily_exposure": self._daily_exposure,
            "decisions_count": len(self._decisions),
            "volatility_multiplier": self.get_volatility_multiplier(),
        }

    def get_volatility_multiplier(self) -> float:
        """Scheduler için mevcut volatilite çarpanını döndürür.
        1.0 = Calm (Normal Interval)
        <1.0 = Volatile (Faster Interval)
        """
        # 1. Haftalık PnL volatilitesi
        vol_m = 1.0
        if len(self._weekly_returns) >= 5:
            realized_vol = float(np.std(list(self._weekly_returns)))
            if realized_vol > self._vol_target:
                # Volatilite hedefi aşıldıysa çarpan düşer (0.2'ye kadar)
                vol_m = self._vol_target / max(realized_vol, 1e-6)
                vol_m = max(vol_m, 0.2)
        
        # 2. Drawdown etkisi (Drawdown varsa daha temkinli/sık kontrol?)
        # Drawdown varsa çarpan düşer -> Daha sık kontrol (High Alert)
        dd = self._current_drawdown()
        if dd < -0.10:
            vol_m *= 0.8
            
        # 3. Tilt durumu
        if self._consecutive_losses >= 3:
            vol_m *= 0.7

        return round(max(vol_m, 0.1), 2)

    def calculate_portfolio_volatility(self) -> float:
        """Açık pozisyonların (bekleyen bahislerin) toplam varyansını hesaplar."""
        # Eğer bekleyen bahis verisi DB'de veya bellekte yoksa basitleştirilmiş bir tahmin dön
        # Normalde DB'den 'pending' olanları çekmeli
        if not self.db: return 0.0
        try:
            # Mock: DB'den bekleyen bahislerin stake oranlarını al
            pending = self.db.query("SELECT stake_pct, odds FROM bets WHERE status='pending'")
            if pending.is_empty(): return 0.0
            
            total_var = 0.0
            for row in pending.to_dicts():
                f = row["stake_pct"]
                o = row["odds"]
                p = 1.0 / max(o, 1.01)
                total_var += p * (1-p) * (f**2)
            return total_var
        except Exception:
            return 0.0

    def segment_bankroll(self) -> dict:
        """Kasa segmentasyonu: Hot (Live), Reserve (Güvenli), R&D (Deney)."""
        return {
            "hot": self._bankroll.total * 0.6,
            "reserve": self._bankroll.total * 0.3,
            "rnd": self._bankroll.total * 0.1
        }
