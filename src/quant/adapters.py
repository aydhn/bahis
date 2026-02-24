"""
adapters.py – Mevcut quant modellerini sisteme uyarlayan adapter katmanı.
"""
from typing import Any, Dict, Optional
from loguru import logger
from src.core.interfaces import QuantModel
from src.quant.models.benter_model import BenterModel
from src.quant.models.lstm_trend import LSTMTrendAnalyzer
from src.quant.models.dixon_coles_model import DixonColesModel

class BenterAdapter(QuantModel):
    """
    Bill Benter modelini QuantModel arayüzüne uyarlar.
    """
    def __init__(self):
        self.model = BenterModel()

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Context'ten verileri çek
            h_xg = context.get("home_xg", 1.35)
            a_xg = context.get("away_xg", 1.10)

            # Contextual adjustments için gerekenler
            ctx = {
                "rain": context.get("rain_forecast", False),
                "missing_key_player_home": context.get("home_missing_players", 0) > 0,
                "missing_key_player_away": context.get("away_missing_players", 0) > 0,
                "motivation_high_home": context.get("derby_match", False)
            }

            res = self.model.calculate_benter_probabilities(h_xg, a_xg, ctx)

            # Arayüze uygun dönüş
            return {
                "model": "benter",
                "prob_home": res["prob_home"],
                "prob_draw": res["prob_draw"],
                "prob_away": res["prob_away"],
                "confidence": res["prob_home"] if res["prob_home"] > res["prob_away"] else res["prob_away"],
                "details": res
            }
        except Exception as e:
            logger.error(f"BenterAdapter hatası: {e}")
            return {"model": "benter", "error": str(e)}

class LSTMAdapter(QuantModel):
    """
    LSTM Trend modelini QuantModel arayüzüne uyarlar.
    """
    def __init__(self):
        self.model = LSTMTrendAnalyzer()

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            home = context.get("home_team", "Home")
            away = context.get("away_team", "Away")

            # Eğer gerçek history yoksa, mock history oluştur
            # context içinde 'home_history' bekleniyor normalde
            home_hist = context.get("home_history", [])
            away_hist = context.get("away_history", [])

            if not home_hist:
                # Basit bir mock history (son 5 maç puanına göre)
                points = context.get("home_last_5_points", 7)
                home_hist = self._mock_history(points)

            if not away_hist:
                points = context.get("away_last_5_points", 7)
                away_hist = self._mock_history(points)

            res = self.model.predict_for_match(home, away, home_hist, away_hist)

            return {
                "model": "lstm",
                "prob_home": res["prob_home"],
                "prob_draw": res["prob_draw"],
                "prob_away": res["prob_away"],
                "confidence": res["confidence"],
                "details": res
            }
        except Exception as e:
            logger.error(f"LSTMAdapter hatası: {e}")
            return {"model": "lstm", "error": str(e)}

    def _mock_history(self, points: int) -> list:
        """Puan durumuna göre sahte maç geçmişi üret."""
        # 5 maçta X puan nasıl alınır? Basit bir simülasyon.
        history = []
        wins = points // 3
        draws = (points % 3)
        losses = 5 - wins - draws

        for _ in range(wins): history.append({"result": "W", "xg": 1.8})
        for _ in range(draws): history.append({"result": "D", "xg": 1.0})
        for _ in range(losses): history.append({"result": "L", "xg": 0.5})

        return history

class DixonColesAdapter(QuantModel):
    """
    Dixon-Coles Modelini QuantModel arayüzüne uyarlar.
    """
    def __init__(self):
        self.model = DixonColesModel()

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            home = context.get("home_team", "Home")
            away = context.get("away_team", "Away")
            h_xg = context.get("home_xg", 1.35)
            a_xg = context.get("away_xg", 1.10)

            # Dixon-Coles Predict
            res = self.model.predict(home, away, h_xg, a_xg)

            return {
                "model": "dixon_coles",
                "prob_home": res["prob_home"],
                "prob_draw": res["prob_draw"],
                "prob_away": res["prob_away"],
                "confidence": res["prob_home"] if res["prob_home"] > res["prob_away"] else res["prob_away"],
                "details": res
            }
        except Exception as e:
            logger.error(f"DixonColesAdapter hatası: {e}")
            return {"model": "dixon_coles", "error": str(e)}
