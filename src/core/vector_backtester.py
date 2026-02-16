"""
vector_backtester.py – Vektörize geri-test motoru.
Stratejileri matris hızıyla (loop-free) test eder.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table


class VectorBacktester:
    """Vektörize backtest motoru – hızlı ve kapsamlı."""

    def __init__(self, initial_bankroll: float = 10000.0, commission: float = 0.0):
        self._bankroll = initial_bankroll
        self._commission = commission
        logger.debug("VectorBacktester başlatıldı.")

    def run(self, db, start: str = "2024-01-01", end: str = "2026-01-01") -> dict:
        """Tarihsel veri üzerinde strateji testi çalıştırır."""
        console = Console()

        # Tarihsel maçları al
        matches = db.query(
            "SELECT * FROM matches WHERE kickoff BETWEEN ? AND ? ORDER BY kickoff",
            [start, end],
        )

        if matches.is_empty():
            logger.warning("Backtest için yeterli veri yok – simülasyon çalıştırılıyor.")
            matches = self._generate_synthetic_data(500)

        logger.info(f"Backtest: {matches.height} maç üzerinde çalışıyor…")

        # Feature matrisi oluştur
        results = self._vectorized_backtest(matches)

        # Sonuçları göster
        self._display_results(console, results)
        return results

    def _vectorized_backtest(self, matches: pl.DataFrame) -> dict:
        """Vektörize backtest – tüm maçlar bir kerede."""
        n = matches.height

        # Oranlar
        home_odds = matches["home_odds"].to_numpy() if "home_odds" in matches.columns else np.full(n, 2.5)
        draw_odds = matches["draw_odds"].to_numpy() if "draw_odds" in matches.columns else np.full(n, 3.3)
        away_odds = matches["away_odds"].to_numpy() if "away_odds" in matches.columns else np.full(n, 3.0)

        # NaN temizle
        home_odds = np.nan_to_num(home_odds, nan=2.5)
        draw_odds = np.nan_to_num(draw_odds, nan=3.3)
        away_odds = np.nan_to_num(away_odds, nan=3.0)

        # İmplied probabilities
        imp_h = 1.0 / np.maximum(home_odds, 1.01)
        imp_d = 1.0 / np.maximum(draw_odds, 1.01)
        imp_a = 1.0 / np.maximum(away_odds, 1.01)
        total_imp = imp_h + imp_d + imp_a
        fair_h = imp_h / total_imp
        fair_d = imp_d / total_imp
        fair_a = imp_a / total_imp

        # Sonuçları simüle et (gerçek sonuçlar yoksa)
        if "home_score" in matches.columns and "away_score" in matches.columns:
            hs = matches["home_score"].to_numpy()
            as_ = matches["away_score"].to_numpy()
            hs = np.nan_to_num(hs, nan=-1)
            as_ = np.nan_to_num(as_, nan=-1)
            outcomes = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))
            valid = (hs >= 0) & (as_ >= 0)
        else:
            outcomes = np.array([np.random.choice([0, 1, 2], p=[fair_h[i], fair_d[i], fair_a[i]]) for i in range(n)])
            valid = np.ones(n, dtype=bool)

        # Value betting stratejisi
        ev_h = fair_h * home_odds - 1
        ev_d = fair_d * draw_odds - 1
        ev_a = fair_a * away_odds - 1

        # En iyi EV seçimi (vektörize)
        all_evs = np.stack([ev_h, ev_d, ev_a], axis=1)
        best_bet = np.argmax(all_evs, axis=1)
        best_ev = np.max(all_evs, axis=1)

        # Filtre: sadece EV > threshold
        ev_threshold = 0.02
        bet_mask = (best_ev > ev_threshold) & valid

        # Kelly stake
        all_probs = np.stack([fair_h, fair_d, fair_a], axis=1)
        all_odds = np.stack([home_odds, draw_odds, away_odds], axis=1)

        chosen_prob = all_probs[np.arange(n), best_bet]
        chosen_odds = all_odds[np.arange(n), best_bet]

        b = chosen_odds - 1
        kelly = (b * chosen_prob - (1 - chosen_prob)) / np.maximum(b, 0.01)
        kelly = np.clip(kelly, 0, 0.05) * 0.25  # Quarter Kelly

        # PnL hesaplama
        won = (best_bet == outcomes) & bet_mask
        pnl = np.where(
            bet_mask,
            np.where(won, kelly * (chosen_odds - 1), -kelly),
            0.0,
        )

        # Kümülatif bankroll
        bankroll = np.cumprod(1 + pnl) * self._bankroll

        # Metrikler
        total_bets = bet_mask.sum()
        wins = won.sum()
        total_pnl = bankroll[-1] - self._bankroll if n > 0 else 0

        # Drawdown
        peak = np.maximum.accumulate(bankroll)
        drawdown = (bankroll - peak) / peak
        max_dd = float(drawdown.min())

        # Sharpe Ratio (günlük)
        returns = pnl[bet_mask] if bet_mask.sum() > 0 else pnl
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252))

        return {
            "total_matches": int(n),
            "total_bets": int(total_bets),
            "wins": int(wins),
            "win_rate": float(wins / max(total_bets, 1)),
            "total_pnl": float(total_pnl),
            "final_bankroll": float(bankroll[-1]) if n > 0 else self._bankroll,
            "roi": float(total_pnl / self._bankroll),
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "avg_ev": float(best_ev[bet_mask].mean()) if bet_mask.sum() > 0 else 0.0,
            "avg_kelly": float(kelly[bet_mask].mean()) if bet_mask.sum() > 0 else 0.0,
            "bankroll_curve": bankroll.tolist(),
        }

    def _generate_synthetic_data(self, n: int = 500) -> pl.DataFrame:
        """Backtest için sentetik veri üretir."""
        np.random.seed(42)
        return pl.DataFrame({
            "match_id": [f"sim_{i:04d}" for i in range(n)],
            "home_odds": np.random.uniform(1.3, 5.0, n).round(2),
            "draw_odds": np.random.uniform(2.5, 5.0, n).round(2),
            "away_odds": np.random.uniform(1.5, 8.0, n).round(2),
            "home_score": np.random.poisson(1.3, n),
            "away_score": np.random.poisson(1.1, n),
            "kickoff": [f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}" for i in range(n)],
        })

    def _display_results(self, console: Console, results: dict):
        table = Table(title="Backtest Sonuçları", show_lines=True)
        table.add_column("Metrik", style="cyan")
        table.add_column("Değer", style="bold")

        rows = [
            ("Toplam Maç", str(results["total_matches"])),
            ("Toplam Bahis", str(results["total_bets"])),
            ("Kazanma", f"{results['wins']} ({results['win_rate']:.1%})"),
            ("Toplam PnL", f"₺{results['total_pnl']:,.2f}"),
            ("Final Kasa", f"₺{results['final_bankroll']:,.2f}"),
            ("ROI", f"{results['roi']:.2%}"),
            ("Max Drawdown", f"{results['max_drawdown']:.2%}"),
            ("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}"),
            ("Ort. EV", f"{results['avg_ev']:.4f}"),
            ("Ort. Kelly Stake", f"{results['avg_kelly']:.4f}"),
        ]
        for metric, value in rows:
            table.add_row(metric, value)
        console.print(table)
