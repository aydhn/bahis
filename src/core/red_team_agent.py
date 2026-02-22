"""
red_team_agent.py – Devils's Advocate / Falsification Agent.

Görevi: Bir sinyal üretildiğinde "Bu sinyal neden yanlıştır?" sorusuna
cevap aramak. Eğer yeterli karşıt kanıt bulursa sinyali veto eder.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Any
from loguru import logger

class RedTeamAgent:
    """Şeytanın Avukatı – Karar Doğrulama Katmanı."""
    
    def __init__(self, db: Any = None, llm_backend: str = "ollama"):
        self.db = db
        self._llm = llm_backend
        logger.debug(f"[RedTeam] Devil's Advocate Agent başlatıldı (backend={llm_backend})")

    async def challenge_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Sinyali çürütmeye çalış (Falsification)."""
        match_id = signal.get("match_id", "")
        home = signal.get("home_team", "Ev")
        away = signal.get("away_team", "Deplasman")
        selection = signal.get("selection", "?")
        
        logger.info(f"[RedTeam] CHALLENGE: {home} vs {away} ({selection})")
        
        # 1. İstatistiksel Check (Aşırı Güven Kontrolü)
        # Eğer model %80+ güven veriyorsa ama oran 2.0+ ise bir bit yeniği vardır.
        if signal.get("confidence", 0) > 0.8 and signal.get("odds", 0) > 2.0:
            signal["veto"] = True
            signal["veto_reason"] = "Aşırı güven anomalisi (Trap signal?)"
            return signal

        # 2. Karşıt Kanıt Arama (Counter-Evidence)
        # LLM'e "Bu bahsin yatma nedenleri neler olabilir?" diye sorulabilir.
        # Burada basitleştirilmiş bir Red-Team mantığı:
        reasons_to_fail = await self._get_negative_catalysts(signal)
        
        if len(reasons_to_fail) >= 2:
            signal["protection_multiplier"] = 0.5 # Bahis miktarını yarıya indir
            signal["tags"] = signal.get("tags", []) + ["red_team_warning"]
            logger.warning(f"[RedTeam] Uyarı: {len(reasons_to_fail)} negatif katalizör bulundu. Stake azaltıldı.")
            
        return signal

    async def _get_negative_catalysts(self, signal: Dict) -> List[str]:
        """Bahsin başarısız olma nedenlerini simüle eder."""
        negatives = []
        # Örnek negatif katalizörler (DB'den veya internetten çekilebilir)
        # 1. Takım yorgunluğu (Son 7 günde 3. maç)
        # 2. Hava durumu (Aşırı yağış - underdog avantajı)
        # 3. Hakem etkisi (Sert oyun - kırmızı kart riski)
        
        # Mock logic
        match_id = signal.get("match_id", "")
        if "liverpool" in match_id.lower(): # Örnek
            negatives.append("Yoğun maç trafiği (Fikstür yorgunluğu)")
            
        return negatives

    async def run_batch(self, signals: List[Dict], **kwargs) -> List[Dict]:
        """Batch halinde sinyalleri kırmızı ekip elemesinden geçirir."""
        if not signals: return []
        
        verified_signals = []
        for sig in signals:
            verified_sig = await self.challenge_signal(sig)
            # Eğer veto edilmediyse listeye ekle
            if not verified_sig.get("veto", False):
                verified_signals.append(verified_sig)
            else:
                logger.error(f"[RedTeam] VETO: {sig.get('match_id')} - {sig.get('veto_reason')}")
                
        return verified_signals
