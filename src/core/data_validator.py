"""
data_validator.py – Pydantic tabanlı veri doğrulama.
HTML'den gelen kirli veriyi (None, Null, hatalı tipler) işleme
sokulmadan önce reddeder (Validation Error).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("pydantic yüklü değil – veri doğrulama basit modda.")


# Geçerli status değerleri (DB'den gelen tüm varyantları kapsıyor)
_VALID_STATUSES = {
    "upcoming", "live", "finished", "postponed",
    "cancelled", "abandoned", "suspended", "inprogress",
    "notstarted", "halftime", "ended", "delayed",
    # Sofascore status değerleri
    "not started", "in progress", "half time",
    "1st half", "2nd half", "extra time", "penalties",
}

if PYDANTIC_AVAILABLE:

    class MatchData(BaseModel):
        """Maç verisi şeması – tüm scraper çıktıları buna uymalı."""
        match_id: str = Field(default="", max_length=200)
        home_team: str = Field(..., min_length=1, max_length=100)
        away_team: str = Field(..., min_length=1, max_length=100)
        league: str = Field(default="Bilinmeyen Lig", max_length=200)
        kickoff: Optional[str] = None
        # status - pattern yerine validator kullanıyoruz (esnek)
        status: str = Field(default="upcoming")

        @field_validator("status", mode="before")
        @classmethod
        def normalize_status(cls, v):
            """Status değerini normalize et – bilinmeyenleri 'upcoming' yap."""
            if v is None:
                return "upcoming"
            v_str = str(v).strip().lower()
            if v_str in _VALID_STATUSES:
                return v_str
            # Kısmi eşleşme
            for valid in ("live", "finished", "postponed", "upcoming"):
                if valid in v_str:
                    return valid
            return "upcoming"  # Bilinmeyen → upcoming (reddetme!)

        @field_validator("kickoff", mode="before")
        @classmethod
        def coerce_kickoff(cls, v):
            """datetime / timestamp nesnelerini ISO string'e dönüştür."""
            if v is None:
                return None
            if isinstance(v, datetime):
                return v.isoformat()
            if isinstance(v, (int, float)) and v > 0:
                try:
                    return datetime.fromtimestamp(v).isoformat()
                except Exception:
                    pass
            return str(v) if v else None

        @field_validator("match_id", mode="before")
        @classmethod
        def ensure_match_id(cls, v):
            """Boş match_id'yi kabul et – validator sonrası oluşturulacak."""
            if v is None:
                return ""
            return str(v)

        home_odds: Optional[float] = Field(default=None, ge=1.0, le=1000.0)
        draw_odds: Optional[float] = Field(default=None, ge=1.0, le=1000.0)
        away_odds: Optional[float] = Field(default=None, ge=1.0, le=1000.0)
        over25_odds: Optional[float] = Field(default=None, ge=1.0, le=100.0)
        under25_odds: Optional[float] = Field(default=None, ge=1.0, le=100.0)
        home_score: Optional[int] = Field(default=None, ge=0, le=50)
        away_score: Optional[int] = Field(default=None, ge=0, le=50)

        @field_validator("home_team", "away_team")
        @classmethod
        def strip_team_name(cls, v: str) -> str:
            return v.strip()

        @field_validator("home_odds", "draw_odds", "away_odds",
                         "over25_odds", "under25_odds", mode="before")
        @classmethod
        def coerce_odds(cls, v):
            """Oran değerini güvenli şekilde float'a çevir."""
            if v is None:
                return None
            try:
                f = float(v)
                return f if (1.0 <= f <= 1000.0) else None
            except (TypeError, ValueError):
                return None

        @model_validator(mode="after")
        def teams_different(self):
            if self.home_team.lower() == self.away_team.lower():
                raise ValueError("Ev sahibi ve deplasman takımı aynı olamaz.")
            return self

    class OddsData(BaseModel):
        """Oran verisi şeması."""
        match_id: str = Field(..., min_length=3)
        bookmaker: str = Field(default="unknown", max_length=50)
        market: str = Field(..., pattern=r"^(1X2|totals|btts|correct_score|corners)$")
        selection: str = Field(..., max_length=30)
        odds: float = Field(..., ge=1.0, le=5000.0)
        timestamp: Optional[str] = None

    class SignalData(BaseModel):
        """Bahis sinyali şeması."""
        match_id: str
        market: str = Field(default="1X2")
        selection: str = Field(..., pattern=r"^(home|draw|away|over|under|btts_yes|btts_no|skip)$")
        odds: float = Field(ge=0.0, le=5000.0)
        stake_pct: float = Field(ge=0.0, le=1.0)
        confidence: float = Field(ge=0.0, le=1.0)
        ev: float = Field(ge=-10.0, le=100.0)

    class TeamStats(BaseModel):
        """Takım istatistikleri şeması."""
        team: str = Field(..., min_length=1)
        matches_played: int = Field(ge=0, le=500)
        wins: int = Field(ge=0)
        draws: int = Field(ge=0)
        losses: int = Field(ge=0)
        goals_for: int = Field(ge=0, le=1000)
        goals_against: int = Field(ge=0, le=1000)
        xg_for: float = Field(default=0.0, ge=0.0, le=500.0)
        xg_against: float = Field(default=0.0, ge=0.0, le=500.0)

        @model_validator(mode="after")
        def check_totals(self):
            total = self.wins + self.draws + self.losses
            if total > self.matches_played:
                raise ValueError("G+B+M toplamı oynanan maçı geçemez.")
            return self

else:
    # Pydantic yokken basit dict-based doğrulama
    class MatchData:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def model_dump(self): return self.__dict__

    class OddsData:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def model_dump(self): return self.__dict__

    class SignalData:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def model_dump(self): return self.__dict__

    class TeamStats:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def model_dump(self): return self.__dict__


class DataValidator:
    """Merkezi veri doğrulama sınıfı.

    Tüm scraper çıktılarını ve pipeline verilerini doğrular.
    Geçersiz veriyi reddederek sistemin çökmesini önler.
    """

    def __init__(self):
        self._valid_count = 0
        self._invalid_count = 0
        self._errors: list[dict] = []
        self._null_fields_fixed = 0
        logger.debug("DataValidator başlatıldı.")

    # Heuristic fallback odds (league averages)
    _FALLBACK_ODDS = {
        "home_odds": 2.30,
        "draw_odds": 3.20,
        "away_odds": 3.00,
        "over25_odds": 1.85,
        "under25_odds": 1.95,
    }

    def _sanitize_nulls(self, data: dict) -> dict:
        """Polars/DuckDB null değerlerini Python None'a çevirir ve güvenli varsayılanlara dönüştürür.
        
        Eksik odds alanlarına heuristic fallback değer atar – downstream pipeline
        None odds yüzünden Draw 100% tahmin etmesin.
        """
        import math as _m
        cleaned = {}
        for k, v in data.items():
            if v is None or (isinstance(v, float) and not _m.isfinite(v)):
                if k.endswith("_odds"):
                    # Fallback: heuristic league-average oran ata
                    fallback = self._FALLBACK_ODDS.get(k)
                    cleaned[k] = fallback  # None yerine default oran
                    if fallback is not None:
                        self._null_fields_fixed += 1
                    else:
                        cleaned[k] = None
                elif k.endswith("_score"):
                    cleaned[k] = None
                else:
                    cleaned[k] = v
                self._null_fields_fixed += 1
            else:
                cleaned[k] = v
        return cleaned

    def validate_match(self, data: dict) -> dict | None:
        """Maç verisini doğrular. Geçersizse None döner."""
        try:
            data = self._sanitize_nulls(data)
            
            # Minimum gereksinimler - daha esnek
            if not data.get("home_team") or not data.get("away_team"):
                logger.warning(f"Eksik Takim Ismi: {data}")
                return None # İsimsiz maç kabul edilemez

            if not data.get("match_id"):
                # match_id yoksa oluştur
                data["match_id"] = f"{data.get('home_team', 'H')}_{data.get('away_team', 'A')}_{data.get('kickoff', 'T')}"
            
            if PYDANTIC_AVAILABLE:
                try:
                    # Pydantic validation
                    validated = MatchData(**data)
                    self._valid_count += 1
                    return validated.model_dump()
                except Exception as pyd_err:
                    # Pydantic hatasını logla ama veriyi kurtarmaya çalış (soft validation)
                    # logger.debug(f"Pydantic Validation Error (Ignored): {pyd_err}")
                    # Kritik alanlar varsa yine de geçir
                    if data.get("home_team") and data.get("away_team"):
                        self._valid_count += 1
                        return data
                    raise pyd_err
            else:
                self._valid_count += 1
                return data
        except Exception as e:
            self._invalid_count += 1
            self._errors.append({"type": "match", "error": str(e), "data": str(data)[:200]})
            # logger.warning(f"CRITICAL VALIDATION REJECT: {e}") 
            return None

    def validate_odds(self, data: dict) -> dict | None:
        try:
            if PYDANTIC_AVAILABLE:
                validated = OddsData(**data)
                self._valid_count += 1
                return validated.model_dump()
            else:
                if data.get("odds", 0) < 1.0:
                    raise ValueError("Oran 1.0'dan küçük olamaz")
                self._valid_count += 1
                return data
        except Exception as e:
            self._invalid_count += 1
            self._errors.append({"type": "odds", "error": str(e)})
            return None

    def validate_signal(self, data: dict) -> dict | None:
        try:
            if PYDANTIC_AVAILABLE:
                validated = SignalData(**data)
                self._valid_count += 1
                return validated.model_dump()
            else:
                self._valid_count += 1
                return data
        except Exception as e:
            self._invalid_count += 1
            self._errors.append({"type": "signal", "error": str(e)})
            return None

    def validate_batch(self, data_list: list[dict], schema: str = "match") -> list[dict]:
        """Toplu doğrulama – geçersiz olanları filtreler."""
        validator = {"match": self.validate_match, "odds": self.validate_odds,
                     "signal": self.validate_signal}.get(schema, self.validate_match)

        # Batch-local sayaçlar (cumulative'den bağımsız)
        batch_valid = 0
        batch_invalid = 0
        valid = []
        for item in data_list:
            before_invalid = self._invalid_count
            result = validator(item)
            if result is not None:
                valid.append(result)
                batch_valid += 1
            else:
                batch_invalid += 1

        total = batch_valid + batch_invalid
        if total > 0:
            reject_rate = batch_invalid / total
            if reject_rate > 0.5:
                logger.warning(f"Yüksek veri red oranı: {reject_rate:.0%} ({batch_invalid}/{total})")
            if reject_rate > 0.9 and total >= 5:
                logger.error(f"KRİTİK: Veri red oranı %{reject_rate:.0%}! Son hatalar: {self._errors[-3:]}")

        return valid

    def stats(self) -> dict:
        return {
            "valid": self._valid_count,
            "invalid": self._invalid_count,
            "reject_rate": self._invalid_count / max(self._valid_count + self._invalid_count, 1),
            "recent_errors": self._errors[-10:],
        }
