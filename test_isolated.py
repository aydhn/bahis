from src.extensions.regime_hmm import MarketRegimeHMM
from src.extensions.smart_money import SmartMoneyDetector
from src.ingestion.async_data_factory import DataFactory
from unittest.mock import MagicMock
import numpy as np

def test_imports_and_init():
    hmm = MarketRegimeHMM()
    assert hmm.n_states == 3

    # Test HMM EMA update
    obs = np.array([0.02, 0.05])
    old_means = hmm.means.copy()
    hmm.train(obs)
    # Means should have updated
    assert not np.allclose(hmm.means, old_means)

    sm = SmartMoneyDetector()
    assert sm.history == {}

    db_mock = MagicMock()
    cache_mock = MagicMock()
    df = DataFactory(db=db_mock, cache=cache_mock)
    assert hasattr(df, "smart_money")
