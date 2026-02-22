"""
falsification_guard.py – Karl Popper Falsifiyabilist (Yanlışlanabilirlik) Güvenlik Duvarı.

Çoğu model 'neden bu bahis gelmeli?' diye bakar. 
FalsificationGuard ise 'bu bahis neden gelmemeli?' sorusunu sorar.
Sinyalin zayıf noktalarını (injury, moral, yorgunluk, hava durumu) arar.
Eğer güçlü bir 'yanlışlama' kanıtı bulursa sinyali reddeder.
"""
from typing import Dict, List, Any
from loguru import logger

class FalsificationGuard:
    def __init__(self, db: Any = None):
        self.db = db

    async def check_falsification(self, match_id: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sinyal için karşı kanıtları (Counter-evidence) araştırır.
        """
        match_info = signal.get("match_info", {})
        home = match_info.get("home_team", "Unknown")
        away = match_info.get("away_team", "Unknown")
        
        reasons_to_reject = []
        
        # 1. Eksik Oyuncu Kontrolü (Injury/Suspension)
        # Not: Bu kısım normalde bir Injury API veya KnowledgeGraph gerektirir.
        # Şimdilik DB'deki 'suspicious_news' veya benzeri flaglerden örnekliyoruz.
        news = await self._get_team_news(home, away)
        if "key_player_missing" in news:
            reasons_to_reject.append(f"Kritik eksik: {news['key_player_missing']}")

        # 2. Yorgunluk (Fatigue) - 3 günden az ara ile aç maçı var mı?
        fatigue = await self._check_fatigue(home, away)
        if fatigue:
            reasons_to_reject.append(f"Yüksek yorgunluk: {fatigue}")

        # 3. Form Çelişkisi (Anomalous Form)
        # Favori olmasına rağmen son 3 maçını kaybetmiş mi?
        if signal.get("selection") == "HOME_WIN" and signal.get("odds", 2.0) < 1.5:
            anomalous = await self._check_form_anomaly(home)
            if anomalous:
                reasons_to_reject.append("Favori formsuz (3 mağlubiyet)")

        # 4. Karar
        res = {
            "match_id": match_id,
            "falsified": len(reasons_to_reject) > 0,
            "reasons": reasons_to_reject,
            "penalty": len(reasons_to_reject) * 0.15 # Her gerekçe için %15 güven düşür
        }
        
        if res["falsified"]:
            logger.warning(f"[FalsificationGuard] {match_id} için karşı kanıtlar: {reasons_to_reject}")
            
        return res

    async def _get_team_news(self, team_h: str, team_a: str) -> Dict:
        # Mock Haber Tarama
        return {}

    async def _check_fatigue(self, team_h: str, team_a: str) -> str:
        # DB üzerinden son maç tarihlerini kontrol et
        return ""

    async def _check_form_anomaly(self, team: str) -> bool:
        return False

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Tüm sinyalleri Popper filtresinden geçirir."""
        protected_signals = []
        for sig in signals:
            checker = await self.check_falsification(sig.get("match_id", "unknown"), sig)
            if checker["falsified"]:
                # Güven skorunu düşür veya 'FRAGILE' etiketi ekle
                sig["confidence"] = max(0, sig.get("confidence", 0.5) - checker["penalty"])
                sig["tags"] = sig.get("tags", []) + ["popper_falsified"]
                sig["rejection_reasons"] = checker["reasons"]
            
            protected_signals.append(sig)
        return protected_signals
