"""
context.py – Pipeline State (BettingContext).

Bu modül, pipeline boyunca akan veri yapısını tanımlar.
Loose dictionary yerine Pydantic model kullanarak type safety sağlar.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import field

from pydantic import BaseModel, Field, ConfigDict
import polars as pl
from loguru import logger

# Lazy imports to avoid circular dependencies (Using TYPE_CHECKING pattern)
# Ancak Pydantic runtime validation yaptığı için forward ref string kullanabiliriz
# veya basitçe Any/Dict kullanıp type hint ekleyebiliriz.

class BettingContext(BaseModel):
    """
    Pipeline boyunca akan merkezi veri yapısı.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 1. Ingestion Stage Data
    matches: Any = Field(default=None, description="Polars DataFrame of matches")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw API responses")

    # 2. Features Stage Data
    features: Any = Field(default=None, description="Polars DataFrame with features")
    feature_metadata: Dict[str, Any] = Field(default_factory=dict)

    # 3. Inference Stage Data
    predictions: Any = Field(default=None, description="Polars DataFrame with probabilities")
    ensemble_results: List[Dict[str, Any]] = Field(default_factory=list, description="Ensemble decisions")

    # 4. Risk Stage Data
    # Match ID -> Report Map
    volatility_reports: Dict[str, Any] = Field(default_factory=dict, description="Map of VolatilityReport")
    philosophical_reports: Dict[str, Any] = Field(default_factory=dict, description="Map of EpistemicReport")
    narratives: Dict[str, str] = Field(default_factory=dict, description="Generated investment memos")

    # Final Bets
    final_bets: List[Dict[str, Any]] = Field(default_factory=list, description="Approved bets ready for execution")

    # 5. Metadata
    cycle_id: int = 0
    timestamp: float = 0.0
    errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Legacy dictionary formatına çevir (geriye uyumluluk için)."""
        return {
            "matches": self.matches,
            "features": self.features,
            "predictions": self.predictions,
            "ensemble_results": self.ensemble_results,
            "final_bets": self.final_bets,
            "ctx": self,  # Self-reference for stages that know about Context
            "cycle": self.cycle_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BettingContext":
        """Dictionary'den context oluştur veya var olanı al."""
        if "ctx" in data and isinstance(data["ctx"], cls):
            return data["ctx"]

        # Create new with best effort mapping
        return cls(
            matches=data.get("matches"),
            features=data.get("features"),
            predictions=data.get("predictions"),
            ensemble_results=data.get("ensemble_results", []),
            final_bets=data.get("final_bets", []),
            cycle_id=data.get("cycle", 0)
        )

    def add_error(self, msg: str):
        logger.error(f"[Context] {msg}")
        self.errors.append(msg)
