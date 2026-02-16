"""
shadow_manager.py – Shadow Testing (Gölge Modu / Paper Trading).

Yeni strateji geliştirdiniz (Korner bahsi, yeni model...).
Bunu hemen parayla denemek kumardır.

Shadow Mode:
  - Sistem aynı anda iki modda çalışır: Live (Canlı) + Shadow (Gölge)
  - Gölge modundaki modeller sanal para ile bahis alır
  - Hafta sonunda rapor: "Eğer Korner Modelini kullansaydık %20 daha fazla kazanacaktık"

Feature Flags:
  - Her strateji bir "flag" ile kontrol edilir
  - flag=True → Live (gerçek kasa)
  - flag=False → Shadow (sanal kasa)

A/B Testing:
  - İki farklı strateji aynı anda çalışır
  - Performansları karşılaştırılır
  - Kazanan strateji Live'a geçer
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
SHADOW_DIR = ROOT / "data" / "shadow"
SHADOW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ShadowBet:
    """Gölge modda alınan sanal bahis."""
    bet_id: str = ""
    strategy: str = ""            # Strateji adı
    match_id: str = ""
    selection: str = ""           # home / draw / away / over25 ...
    odds: float = 0.0
    stake: float = 0.0            # Sanal stake
    prob: float = 0.0
    value_edge: float = 0.0
    timestamp: float = 0.0
    settled: bool = False
    result: str = ""              # win / loss / push
    pnl: float = 0.0             # Kar / zarar
    metadata: dict = field(default_factory=dict)


@dataclass
class ShadowStrategy:
    """Gölge strateji profili."""
    name: str
    description: str = ""
    is_active: bool = True
    initial_bankroll: float = 10000.0
    current_bankroll: float = 10000.0
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    peak_bankroll: float = 10000.0
    max_drawdown: float = 0.0
    bets: list[ShadowBet] = field(default_factory=list)
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_bets, 1)

    @property
    def roi(self) -> float:
        return self.total_pnl / max(self.initial_bankroll, 1)

    @property
    def sharpe(self) -> float:
        """Basitleştirilmiş Sharpe oranı."""
        if not self.bets:
            return 0.0
        returns = [b.pnl / max(b.stake, 1) for b in self.bets if b.settled]
        if not returns:
            return 0.0
        import numpy as np
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        return mean_r / (std_r + 1e-8)


class ShadowManager:
    """Gölge test yöneticisi.

    Kullanım:
        shadow = ShadowManager()
        # Deneysel strateji ekle
        shadow.register_strategy("korner_modeli", bankroll=10000)
        # Gölge bahis al
        shadow.place_bet("korner_modeli", "GS_FB", "over_corners", 1.85, 200, prob=0.55)
        # Sonuçlandır
        shadow.settle("GS_FB", "over_corners", won=True)
        # Rapor
        report = shadow.compare_all()
    """

    def __init__(self):
        self._strategies: dict[str, ShadowStrategy] = {}
        self._feature_flags: dict[str, bool] = {}
        self._load_state()
        logger.debug("[Shadow] Manager başlatıldı.")

    # ═══════════════════════════════════════════
    #  STRATEJİ YÖNETİMİ
    # ═══════════════════════════════════════════
    def register_strategy(self, name: str, description: str = "",
                           bankroll: float = 10000.0,
                           live: bool = False) -> ShadowStrategy:
        """Yeni strateji kaydet."""
        if name in self._strategies:
            return self._strategies[name]

        strategy = ShadowStrategy(
            name=name,
            description=description,
            initial_bankroll=bankroll,
            current_bankroll=bankroll,
            peak_bankroll=bankroll,
        )
        self._strategies[name] = strategy
        self._feature_flags[name] = live

        logger.info(
            f"[Shadow] Strateji kayıt: '{name}' "
            f"({'LIVE' if live else 'SHADOW'}, kasa={bankroll:.0f})"
        )
        return strategy

    def set_live(self, name: str, live: bool = True):
        """Stratejiyi live/shadow moduna al."""
        self._feature_flags[name] = live
        mode = "LIVE" if live else "SHADOW"
        logger.info(f"[Shadow] '{name}' → {mode} moduna geçirildi.")

    def is_live(self, name: str) -> bool:
        """Strateji canlı mı?"""
        return self._feature_flags.get(name, False)

    def is_shadow(self, name: str) -> bool:
        return not self.is_live(name)

    # ═══════════════════════════════════════════
    #  GÖLGE BAHİS
    # ═══════════════════════════════════════════
    def place_bet(self, strategy: str, match_id: str,
                  selection: str, odds: float, stake: float,
                  prob: float = 0.0, value_edge: float = 0.0,
                  **metadata) -> ShadowBet | None:
        """Gölge bahis al."""
        strat = self._strategies.get(strategy)
        if not strat or not strat.is_active:
            return None

        if stake > strat.current_bankroll:
            stake = strat.current_bankroll * 0.05

        bet = ShadowBet(
            bet_id=f"shadow_{strategy}_{match_id}_{int(time.time())}",
            strategy=strategy,
            match_id=match_id,
            selection=selection,
            odds=odds,
            stake=stake,
            prob=prob,
            value_edge=value_edge,
            timestamp=time.time(),
            metadata=metadata,
        )

        strat.bets.append(bet)
        strat.total_bets += 1
        strat.last_updated = datetime.now().isoformat()

        logger.debug(
            f"[Shadow] '{strategy}' bahis: {match_id} {selection} "
            f"@{odds:.2f} stake={stake:.0f}"
        )
        return bet

    def settle(self, match_id: str, selection: str,
               won: bool, strategy: str | None = None):
        """Gölge bahisleri sonuçlandır."""
        settled_count = 0

        for strat_name, strat in self._strategies.items():
            if strategy and strat_name != strategy:
                continue

            for bet in strat.bets:
                if (bet.match_id == match_id and
                    bet.selection == selection and
                    not bet.settled):

                    bet.settled = True
                    if won:
                        bet.result = "win"
                        bet.pnl = bet.stake * (bet.odds - 1)
                        strat.wins += 1
                        strat.current_bankroll += bet.pnl
                    else:
                        bet.result = "loss"
                        bet.pnl = -bet.stake
                        strat.losses += 1
                        strat.current_bankroll -= bet.stake

                    strat.total_pnl += bet.pnl

                    # Peak / drawdown güncelle
                    if strat.current_bankroll > strat.peak_bankroll:
                        strat.peak_bankroll = strat.current_bankroll
                    dd = (strat.peak_bankroll - strat.current_bankroll) / max(strat.peak_bankroll, 1)
                    if dd > strat.max_drawdown:
                        strat.max_drawdown = dd

                    strat.last_updated = datetime.now().isoformat()
                    settled_count += 1

        if settled_count:
            self._save_state()
            logger.debug(
                f"[Shadow] {settled_count} bahis sonuçlandı: "
                f"{match_id} {selection} ({'W' if won else 'L'})"
            )

    def settle_all_for_match(self, match_id: str,
                              results: dict[str, bool]):
        """Maçın tüm bahislerini sonuçlandır.

        results: {"home": True, "draw": False, "away": False, "over25": True}
        """
        for selection, won in results.items():
            self.settle(match_id, selection, won)

    # ═══════════════════════════════════════════
    #  RAPORLAMA
    # ═══════════════════════════════════════════
    def get_strategy_report(self, name: str) -> dict:
        """Tek strateji raporu."""
        strat = self._strategies.get(name)
        if not strat:
            return {"error": f"Strateji bulunamadı: {name}"}

        return {
            "name": strat.name,
            "description": strat.description,
            "mode": "LIVE" if self.is_live(name) else "SHADOW",
            "initial_bankroll": strat.initial_bankroll,
            "current_bankroll": round(strat.current_bankroll, 2),
            "total_pnl": round(strat.total_pnl, 2),
            "roi": f"{strat.roi:.2%}",
            "total_bets": strat.total_bets,
            "wins": strat.wins,
            "losses": strat.losses,
            "win_rate": f"{strat.win_rate:.0%}",
            "max_drawdown": f"{strat.max_drawdown:.1%}",
            "sharpe": round(strat.sharpe, 3),
            "is_active": strat.is_active,
            "pending_bets": sum(1 for b in strat.bets if not b.settled),
        }

    def compare_all(self) -> dict:
        """Tüm stratejileri karşılaştır."""
        reports = {}
        for name in self._strategies:
            reports[name] = self.get_strategy_report(name)

        # Sıralama: ROI'ye göre
        ranked = sorted(
            reports.items(),
            key=lambda x: x[1].get("total_pnl", 0),
            reverse=True,
        )

        best = ranked[0] if ranked else None
        worst = ranked[-1] if ranked else None

        return {
            "strategies": dict(ranked),
            "total_strategies": len(ranked),
            "best_strategy": best[0] if best else None,
            "best_pnl": best[1]["total_pnl"] if best else 0,
            "worst_strategy": worst[0] if worst else None,
            "worst_pnl": worst[1]["total_pnl"] if worst else 0,
            "recommendation": self._recommend(ranked),
        }

    def _recommend(self, ranked: list[tuple]) -> str:
        """Strateji geçiş tavsiyesi."""
        if not ranked:
            return "Henüz veri yok."

        best_name, best_data = ranked[0]
        total_bets = best_data.get("total_bets", 0)

        if total_bets < 30:
            return (
                f"'{best_name}' en iyi performans gösteriyor ama "
                f"henüz {total_bets} bahis – min 30 bahis sonrası değerlendirin."
            )

        pnl = best_data.get("total_pnl", 0)
        if pnl > 0 and self.is_shadow(best_name):
            return (
                f"'{best_name}' gölge modda kârlı "
                f"(PnL: {pnl:.0f}, ROI: {best_data['roi']}). "
                f"LIVE moda geçiş önerilir!"
            )

        return f"Mevcut en iyi: '{best_name}' (ROI: {best_data['roi']})"

    # ═══════════════════════════════════════════
    #  FEATURE FLAG DECORATOR
    # ═══════════════════════════════════════════
    def shadow_mode(self, strategy_name: str):
        """Decorator: fonksiyonu shadow/live moduna göre yönlendirir.

        @shadow.shadow_mode("korner_modeli")
        def analyze_corners(match):
            ...

        Shadow modda → sonuç sanal kasaya kaydedilir
        Live modda → sonuç gerçek kasaya gider
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)

                if self.is_shadow(strategy_name):
                    # Sonucu gölge kasaya kaydet
                    if isinstance(result, dict) and result.get("bet"):
                        bet_info = result["bet"]
                        self.place_bet(
                            strategy_name,
                            bet_info.get("match_id", ""),
                            bet_info.get("selection", ""),
                            bet_info.get("odds", 0),
                            bet_info.get("stake", 0),
                            prob=bet_info.get("prob", 0),
                        )
                    result["mode"] = "SHADOW"
                else:
                    result["mode"] = "LIVE"

                return result
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    # ═══════════════════════════════════════════
    #  KALICILIK
    # ═══════════════════════════════════════════
    def _save_state(self):
        """Durumu diske kaydet."""
        state = {
            "strategies": {},
            "feature_flags": self._feature_flags,
            "saved_at": datetime.now().isoformat(),
        }
        for name, strat in self._strategies.items():
            state["strategies"][name] = {
                "name": strat.name,
                "description": strat.description,
                "is_active": strat.is_active,
                "initial_bankroll": strat.initial_bankroll,
                "current_bankroll": strat.current_bankroll,
                "total_bets": strat.total_bets,
                "wins": strat.wins,
                "losses": strat.losses,
                "total_pnl": strat.total_pnl,
                "peak_bankroll": strat.peak_bankroll,
                "max_drawdown": strat.max_drawdown,
                "created_at": strat.created_at,
                "last_updated": strat.last_updated,
                "bets_count": len(strat.bets),
                "recent_bets": [
                    asdict(b) for b in strat.bets[-50:]
                ],
            }

        path = SHADOW_DIR / "shadow_state.json"
        try:
            path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, default=str)
            )
        except Exception as e:
            logger.debug(f"[Shadow] Kaydetme hatası: {e}")

    def _load_state(self):
        """Durumu diskten yükle."""
        path = SHADOW_DIR / "shadow_state.json"
        if not path.exists():
            return

        try:
            state = json.loads(path.read_text())
            self._feature_flags = state.get("feature_flags", {})

            for name, data in state.get("strategies", {}).items():
                strat = ShadowStrategy(
                    name=data["name"],
                    description=data.get("description", ""),
                    is_active=data.get("is_active", True),
                    initial_bankroll=data.get("initial_bankroll", 10000),
                    current_bankroll=data.get("current_bankroll", 10000),
                    total_bets=data.get("total_bets", 0),
                    wins=data.get("wins", 0),
                    losses=data.get("losses", 0),
                    total_pnl=data.get("total_pnl", 0),
                    peak_bankroll=data.get("peak_bankroll", 10000),
                    max_drawdown=data.get("max_drawdown", 0),
                    created_at=data.get("created_at", ""),
                    last_updated=data.get("last_updated", ""),
                )

                # Son bahisleri yükle
                for bd in data.get("recent_bets", []):
                    strat.bets.append(ShadowBet(**{
                        k: v for k, v in bd.items()
                        if k in ShadowBet.__dataclass_fields__
                    }))

                self._strategies[name] = strat

            logger.info(
                f"[Shadow] Durum yüklendi: {len(self._strategies)} strateji."
            )
        except Exception as e:
            logger.debug(f"[Shadow] Yükleme hatası: {e}")

    @property
    def strategy_names(self) -> list[str]:
        return list(self._strategies.keys())
