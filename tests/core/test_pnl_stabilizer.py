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
