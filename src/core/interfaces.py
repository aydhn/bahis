"""
interfaces.py – Sistemin temel arayüz tanımları (Protocols).
"""
from typing import Any, Dict, Protocol, runtime_checkable

@runtime_checkable
class QuantModel(Protocol):
    """
    Tüm quant modelleri için standart arayüz.
    """
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model tahmini yapar.

        Args:
            context: Maç verileri, xG, news sentiment vb.

        Returns:
            Dict: Tahmin sonuçları (prob_home, confidence, vb.)
        """
        ...
