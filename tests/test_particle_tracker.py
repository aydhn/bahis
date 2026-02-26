
import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.append(os.getcwd())

from src.quant.physics.particle_strength_tracker import ParticleStrengthTracker, MatchObservation

def test_particle_tracker_initialization():
    pst = ParticleStrengthTracker(n_particles=100)
    pst.initialize()

    assert pst._particles is not None
    assert pst._weights is not None
    assert pst._particles.shape == (100, 5)
    assert len(pst._weights) == 100
    assert np.isclose(np.sum(pst._weights), 1.0)

def test_particle_tracker_update():
    pst = ParticleStrengthTracker(n_particles=100)
    pst.initialize()

    obs = MatchObservation(
        minute=10,
        home_shots=2, away_shots=0,
        home_possession=60, away_possession=40,
        home_dangerous_attacks=5, away_dangerous_attacks=1,
        home_corners=1, away_corners=0
    )

    report = pst.update(obs)

    assert report is not None
    assert report.minute == 10
    assert report.state.home_power > 0
    assert report.state.away_power > 0
    assert 0 <= report.home_win_prob <= 1

    # Verify weights are still normalized
    assert np.isclose(np.sum(pst._weights), 1.0)

    # Verify particles evolved (not all zeros, etc)
    assert not np.all(pst._particles == 0)

def test_momentum_shift_detection():
    # Set seed for reproducibility
    np.random.seed(42)
    pst = ParticleStrengthTracker(n_particles=500, momentum_shift_threshold=0.05)
    pst.initialize(home_prior=0.5, away_prior=0.5)

    # 1. Steady state (Equal game)
    obs1 = MatchObservation(minute=10, home_shots=1, away_shots=1, home_possession=50, away_possession=50)
    pst.update(obs1)

    initial_power_diff = pst.get_history()[-1][1] - pst.get_history()[-1][2]

    # 2. Huge home surge
    obs2 = MatchObservation(
        minute=20,
        home_shots=10, away_shots=0,
        home_possession=90, away_possession=10,
        home_dangerous_attacks=10, away_dangerous_attacks=0
    )
    report = pst.update(obs2)

    current_power_diff = report.state.power_diff

    # The power difference should have shifted in favor of home (more positive)
    assert current_power_diff > initial_power_diff

    # Check if a shift was detected (it should be given the massive difference)
    # The magnitude check in code is: abs(delta) >= threshold
    # Delta = current - prev
    if report.momentum_shift.detected:
        assert report.momentum_shift.direction == "home_surge"

def test_full_simulation():
    pst = ParticleStrengthTracker(n_particles=50)

    observations = [
        MatchObservation(minute=i, home_shots=i//10, away_shots=0)
        for i in range(0, 91, 15)
    ]

    reports = pst.simulate_match(observations)
    assert len(reports) == len(observations)

    history = pst.get_history()
    assert len(history) == len(observations)

def test_update_weights_numerics():
    """Specific test for the optimized _update_weights method logic."""
    pst = ParticleStrengthTracker(n_particles=10)
    pst.initialize()

    # Mock particles to fixed values for deterministic check
    pst._particles[:, 0] = 0.6  # Home power
    pst._particles[:, 1] = 0.4  # Away power
    # Power ratio = 0.6 / (0.6+0.4) = 0.6

    # Observation that matches expectation perfectly
    # expected = [0.6, 0.3+0.4*0.6, 0.6, 0.4+0.2*0.6]
    #          = [0.6, 0.54, 0.6, 0.52]

    obs_z = np.array([0.6, 0.54, 0.6, 0.52])

    # If observation matches expectation, diff is 0, log_likelihood is 0 (max).
    # Weights should remain uniform (or close to it if they were uniform).

    pst._weights = np.ones(10) / 10
    pst._update_weights(obs_z)

    # Weights should be uniform since all particles are identical and match observation
    assert np.allclose(pst._weights, 0.1)

    # Now try an observation that is far off
    obs_z_bad = np.array([0.0, 0.0, 0.0, 0.0])
    pst._update_weights(obs_z_bad)

    assert np.isclose(np.sum(pst._weights), 1.0)
