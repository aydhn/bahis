from src.core.pnl_stabilizer import PnLStabilizer

def test_metrics_empty_history():
    """Test metrics when history is empty."""
    stabilizer = PnLStabilizer()
    metrics = stabilizer.metrics()
    assert metrics == {"drawdown": 0, "peak": 0, "current": 0, "history_len": 0}

def test_metrics_populated_history():
    """Test metrics calculation with a normal PnL progression."""
    stabilizer = PnLStabilizer()

    # Record bankroll: 100 -> 110 -> 99
    stabilizer.record_pnl(100.0)
    stabilizer.record_pnl(110.0)
    stabilizer.record_pnl(99.0)

    metrics = stabilizer.metrics()

    assert metrics["peak"] == 110.0
    assert metrics["current"] == 99.0
    assert metrics["history_len"] == 3
    # Drawdown = (current - peak) / peak = (99 - 110) / 110 = -0.1
    assert metrics["drawdown"] == -0.1

def test_metrics_zero_peak():
    """Test metrics when peak is zero to ensure no division by zero error in drawdown."""
    stabilizer = PnLStabilizer()

    # Initial peak is 0.0. Recording 0.0 and -10.0 keeps peak at 0.0.
    stabilizer.record_pnl(0.0)
    stabilizer.record_pnl(-10.0)

    metrics = stabilizer.metrics()

    assert metrics["peak"] == 0.0
    assert metrics["current"] == -10.0
    assert metrics["history_len"] == 2
    # Code specifically returns 0.0 if _peak == 0
    assert metrics["drawdown"] == 0.0

def test_record_pnl_updates_history_and_peak():
    """Test that record_pnl correctly updates _history and _peak."""
    stabilizer = PnLStabilizer()

    assert stabilizer._history == []
    assert stabilizer._peak == 0.0

    stabilizer.record_pnl(100.0)
    assert stabilizer._history == [100.0]
    assert stabilizer._peak == 100.0

    stabilizer.record_pnl(110.0)
    assert stabilizer._history == [100.0, 110.0]
    assert stabilizer._peak == 110.0

def test_record_pnl_no_peak_update_on_drawdown():
    """Test that record_pnl does not update _peak on a drawdown."""
    stabilizer = PnLStabilizer()

    stabilizer.record_pnl(100.0)
    assert stabilizer._peak == 100.0

    stabilizer.record_pnl(90.0)
    assert stabilizer._history == [100.0, 90.0]
    assert stabilizer._peak == 100.0  # Peak should remain 100.0

def test_stabilize_empty_bets():
    """Test that stabilize returns empty list when no bets provided."""
    stabilizer = PnLStabilizer()
    assert stabilizer.stabilize([]) == []

def test_stabilize_skips_selection():
    """Test that bets marked with selection 'skip' are ignored."""
    stabilizer = PnLStabilizer()
    bets = [{"selection": "skip", "stake_pct": 5.0}]
    stabilized = stabilizer.stabilize(bets)

    assert len(stabilized) == 1
    assert stabilized[0]["stake_pct"] == 5.0  # Unchanged
    assert "pid_multiplier" not in stabilized[0]

def test_stabilize_adjusts_stake():
    """Test that valid bets have their stakes adjusted by the multiplier."""
    stabilizer = PnLStabilizer()
    # Force a known multiplier state (no drawdown, should be 1.0 or pid output)
    stabilizer._pid = None

    bets = [{"selection": "home", "stake_pct": 5.0}]
    stabilized = stabilizer.stabilize(bets)

    assert len(stabilized) == 1
    # Multiplier should be 1.0 based on simple rule for 0.0 drawdown
    assert stabilized[0]["stake_pct"] == 5.0
    assert stabilized[0]["pid_multiplier"] == 1.0

def test_stabilize_cuts_stake_on_drawdown():
    """Test aggressive stake cut when max drawdown limit is exceeded."""
    stabilizer = PnLStabilizer(max_drawdown_limit=0.10)
    stabilizer._pid = None

    # Create a > 10% drawdown
    stabilizer.record_pnl(100.0)
    stabilizer.record_pnl(80.0) # 20% drawdown

    bets = [{"selection": "home", "stake_pct": 10.0}]
    stabilized = stabilizer.stabilize(bets)

    assert len(stabilized) == 1

    # 20% drawdown -> _fallback_multiplier < -0.08 returns 0.2
    # So base adjusted = 10.0 * 0.2 = 2.0
    # Then dd (-0.20) < limit (-0.10), so adjusted *= 0.3
    # Result: 2.0 * 0.3 = 0.6

    expected_stake = float(round(max(0.6, 0), 5))
    assert stabilized[0]["stake_pct"] == expected_stake
    assert stabilized[0]["pid_multiplier"] == 0.2


def test_compute_multiplier_pid_active():
    """Test that _compute_multiplier delegates to PID when available."""
    stabilizer = PnLStabilizer()

    # Fake PID to return a constant multiplier for easier testing
    class DummyPID:
        def __call__(self, value):
            return 1.5

    stabilizer._pid = DummyPID()
    stabilizer.record_pnl(100.0)
    stabilizer.record_pnl(105.0)

    # Multiplier should be clipped to [0.1, 2.0]
    assert stabilizer._compute_multiplier(0.0) == 1.5

def test_compute_multiplier_pid_clipped():
    """Test that PID output is clipped properly."""
    stabilizer = PnLStabilizer()

    class DummyPID:
        def __call__(self, value):
            return 3.0 # Over the 2.0 limit

    stabilizer._pid = DummyPID()
    stabilizer.record_pnl(100.0)
    stabilizer.record_pnl(105.0)

    assert stabilizer._compute_multiplier(0.0) == 2.0

    class DummyPIDLow:
        def __call__(self, value):
            return -1.0 # Under the 0.1 limit

    stabilizer._pid = DummyPIDLow()
    assert stabilizer._compute_multiplier(0.0) == 0.1

def test_compute_multiplier_no_pid():
    """Test that _compute_multiplier falls back when PID is None."""
    stabilizer = PnLStabilizer()
    stabilizer._pid = None

    # -0.09 < -0.08 fallback is 0.2
    assert stabilizer._compute_multiplier(-0.09) == 0.2

def test_fallback_multiplier():
    """Test simple rule-based fallback multipliers."""
    stabilizer = PnLStabilizer()
    assert stabilizer._fallback_multiplier(-0.09) == 0.2
    assert stabilizer._fallback_multiplier(-0.06) == 0.5
    assert stabilizer._fallback_multiplier(-0.03) == 0.8
    assert stabilizer._fallback_multiplier(-0.01) == 1.0


def test_recent_return_empty_or_single():
    """Test _recent_return when history has 0 or 1 items."""
    stabilizer = PnLStabilizer()

    assert len(stabilizer._history) == 0
    assert stabilizer._recent_return() == 0.0

    stabilizer.record_pnl(100.0)
    assert len(stabilizer._history) == 1
    assert stabilizer._recent_return() == 0.0

def test_recent_return_calculation():
    """Test _recent_return with valid history."""
    stabilizer = PnLStabilizer()

    stabilizer.record_pnl(100.0)
    stabilizer.record_pnl(105.0)

    # calculation: (105.0 - 100.0) / max(abs(100.0), 1e-8) = 5.0 / 100.0 = 0.05
    assert stabilizer._recent_return() == 0.05

    # Drop test
    stabilizer.record_pnl(94.5)
    # (94.5 - 105.0) / max(abs(105.0), 1e-8) = -10.5 / 105.0 = -0.1
    assert stabilizer._recent_return() == -0.1

def test_recent_return_near_zero_previous():
    """Test _recent_return with near zero previous to avoid zero division."""
    stabilizer = PnLStabilizer()

    stabilizer.record_pnl(0.0)
    stabilizer.record_pnl(10.0)

    # (10.0 - 0.0) / max(0.0, 1e-8) = 10.0 / 1e-8 = 1000000000.0
    assert stabilizer._recent_return() == 10.0 / 1e-8

import sys
import importlib
from unittest.mock import patch

def test_init_without_pid():
    """Test initialization when PID is disabled."""
    with patch("src.core.pnl_stabilizer.PID_AVAILABLE", False):
        stabilizer = PnLStabilizer()
        assert stabilizer._pid is None

def test_import_error_handling():
    """Test import error handling gracefully."""
    with patch.dict('sys.modules', {'simple_pid': None}):
        import src.core.pnl_stabilizer
        importlib.reload(src.core.pnl_stabilizer)
        assert src.core.pnl_stabilizer.PID_AVAILABLE is False

    # Reload after patch to restore original state
    import src.core.pnl_stabilizer
    importlib.reload(src.core.pnl_stabilizer)
    assert src.core.pnl_stabilizer.PID_AVAILABLE is True

def test_current_drawdown_peak_update():
    """Test _current_drawdown updates peak when current > peak."""
    stabilizer = PnLStabilizer()
    stabilizer._history.append(150.0)
    stabilizer._peak = 100.0  # Force peak to be lower than current

    assert stabilizer._current_drawdown() == 0.0
    assert stabilizer._peak == 150.0

def test_stabilize_with_pid_active():
    """Test stabilize when PID logic is active."""
    stabilizer = PnLStabilizer()

    class DummyPID:
        def __call__(self, value):
            return 1.8

    stabilizer._pid = DummyPID()
    stabilizer.record_pnl(100.0)
    stabilizer.record_pnl(105.0)

    bets = [{"selection": "home", "stake_pct": 5.0}]
    stabilized = stabilizer.stabilize(bets)

    assert len(stabilized) == 1
    assert stabilized[0]["pid_multiplier"] == 1.8
    assert stabilized[0]["stake_pct"] == 9.0  # 5.0 * 1.8
