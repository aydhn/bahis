"""
data_validator.py – Pydantic tabanlı veri doğrulama.
HTML'den gelen kirli veriyi (None, Null, hatalı tipler) işleme
sokulmadan önce reddeder (Validation Error).
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("pydantic yüklü değil – veri doğrulama basit modda.")


if PYDANTIC_AVAILABLE:

    class MatchData(BaseModel):
        """Maç verisi şeması – tüm scraper çıktıları buna uymalı."""
        match_id: str = Field(..., min_length=3, max_length=50)
        home_team: str = Field(..., min_length=1, max_length=100)
        away_team: str = Field(..., min_length=1, max_length=100)
        league: str = Field(default="Bilinmeyen Lig", max_length=100)
        kickoff: Optional[str] = None
        status: str = Field(default="upcoming", pattern=r"^(upcoming|live|finished|postponed)$")
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
    """Merkezi veri doğrulama sınıfı."""

    def __init__(self):
        self._valid_count = 0
        self._invalid_count = 0
        self._errors: list[dict] = []
        logger.debug("DataValidator başlatıldı.")

    def validate_match(self, data: dict) -> dict | None:
        """Maç verisini doğrular. Geçersizse None döner."""
        try:
            if PYDANTIC_AVAILABLE:
                validated = MatchData(**data)
                self._valid_count += 1
                return validated.model_dump()
            else:
                if not data.get("home_team") or not data.get("away_team"):
                    raise ValueError("Takım adı eksik")
                if not data.get("match_id"):
                    raise ValueError("match_id eksik")
                self._valid_count += 1
                return data
        except Exception as e:
            self._invalid_count += 1
            self._errors.append({"type": "match", "data": str(data)[:200], "error": str(e)})
            logger.debug(f"Veri reddedildi: {e}")
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

        valid = []
        for item in data_list:
            result = validator(item)
            if result is not None:
                valid.append(result)

        reject_rate = self._invalid_count / max(self._valid_count + self._invalid_count, 1)
        if reject_rate > 0.5:
            logger.warning(f"Yüksek veri red oranı: {reject_rate:.0%}")

        return valid

    def stats(self) -> dict:
        return {
            "valid": self._valid_count,
            "invalid": self._invalid_count,
            "reject_rate": self._invalid_count / max(self._valid_count + self._invalid_count, 1),
            "recent_errors": self._errors[-10:],
        }
