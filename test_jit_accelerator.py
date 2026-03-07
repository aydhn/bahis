import pytest
import numpy as np
from src.core.jit_accelerator import JITAccelerator, NUMBA_OK

@pytest.mark.skipif(not NUMBA_OK, reason="Numba is not available")
def test_jit_accelerator_methods():
    acc = JITAccelerator()
    acc.warmup()

    kelly_val = acc.kelly(0.55, 2.10)
    assert isinstance(kelly_val, float)

    poisson_val = acc.poisson_1x2(1.4, 1.1)
    assert isinstance(poisson_val, dict)
    assert "prob_home" in poisson_val
    assert "prob_draw" in poisson_val
    assert "prob_away" in poisson_val

    monte_carlo_val = acc.monte_carlo(1.4, 1.1, 1000)
    assert isinstance(monte_carlo_val, dict)
    assert "prob_home" in monte_carlo_val

    probs = np.array([0.5, 0.3, 0.2])
    entropy_val = acc.entropy(probs)
    assert isinstance(entropy_val, float)
