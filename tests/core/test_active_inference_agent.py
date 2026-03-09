import pytest
import numpy as np

from src.core.active_inference_agent import (
    surprisal,
    free_energy,
    expected_information_gain,
    bayesian_update,
    ActiveInferenceAgent,
    ModuleState,
    ActiveInferenceReport,
    BeliefState
)

def test_surprisal():
    # p = [0.1, 0.8, 0.1], observed = 1 => -log(0.8) = 0.22314
    probs = np.array([0.1, 0.8, 0.1])
    s = surprisal(probs, 1)
    assert np.isclose(s, -np.log(0.8))

    # observed index out of bounds
    s_oob = surprisal(probs, 5)
    assert np.isclose(s_oob, -np.log(1e-10))

    # 0 probability (clipped to 1e-10)
    probs_zero = np.array([0.0, 1.0, 0.0])
    s_zero = surprisal(probs_zero, 0)
    assert np.isclose(s_zero, -np.log(1e-10))


def test_free_energy():
    q_beliefs = np.array([0.2, 0.5, 0.3])
    log_joint = np.log(np.array([0.1, 0.6, 0.3]))

    # F = sum(q * (log_q - log_joint))
    expected_f = np.sum(q_beliefs * (np.log(q_beliefs) - log_joint))
    f = free_energy(q_beliefs, log_joint)
    assert np.isclose(f, expected_f)

    # test clipping with 0
    q_beliefs_zero = np.array([0.0, 1.0, 0.0])
    f_zero = free_energy(q_beliefs_zero, log_joint)

    q_clipped = np.clip(q_beliefs_zero, 1e-10, 1.0)
    q_clipped /= q_clipped.sum()
    expected_f_zero = np.sum(q_clipped * (np.log(q_clipped) - log_joint))
    assert np.isclose(f_zero, expected_f_zero)


def test_expected_information_gain():
    beliefs = np.array([0.5, 0.5])
    # observation model: P(o|s) -> shape (n_obs, n_states)
    observation_model = np.array([
        [0.9, 0.1],  # obs 0 probabilities given states
        [0.1, 0.9],  # obs 1 probabilities given states
    ])

    gains = expected_information_gain(beliefs, observation_model)
    assert gains.shape == (2,)
    assert gains[0] > 0
    assert gains[1] > 0

    # perfect observation model should yield max gain
    observation_model_perfect = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    gains_perfect = expected_information_gain(beliefs, observation_model_perfect)
    assert np.all(gains_perfect > gains)

    # 1D observation model fallback
    obs_1d = np.array([0.6, 0.4])
    gains_1d = expected_information_gain(beliefs, obs_1d)
    assert gains_1d.shape == (1,)
    assert gains_1d[0] > 0


def test_bayesian_update():
    prior = np.array([0.3, 0.3, 0.4])
    likelihood = np.array([0.1, 0.8, 0.1])

    posterior = bayesian_update(prior, likelihood)
    assert np.isclose(posterior.sum(), 1.0)
    assert posterior[1] > prior[1]  # The likelihood strongly favored index 1

    # All zeros likelihood (edge case)
    zero_likelihood = np.zeros(3)
    posterior_zero = bayesian_update(prior, zero_likelihood)
    assert np.isclose(posterior_zero.sum(), 1.0)
    assert np.allclose(posterior_zero, np.ones(3)/3)


def test_agent_initialization():
    agent = ActiveInferenceAgent()
    assert len(agent._modules) == 7  # default modules
    assert "poisson" in agent._modules

    custom_agent = ActiveInferenceAgent(modules=["mod1", "mod2"])
    assert len(custom_agent._modules) == 2
    assert "mod1" in custom_agent._modules


def test_agent_observe():
    agent = ActiveInferenceAgent(["test_mod"])

    # Observe correct prediction
    s1 = agent.observe("test_mod", [0.8, 0.1, 0.1], 0)
    assert s1 < 1.0
    state = agent._modules["test_mod"]
    assert state.total_predictions == 1
    assert state.correct_predictions == 1
    assert state.accuracy == 1.0
    assert state.precision > 1.0  # Boosted

    # Observe incorrect prediction
    s2 = agent.observe("test_mod", [0.8, 0.1, 0.1], 1)
    assert s2 > 2.0  # High surprisal
    assert state.total_predictions == 2
    assert state.correct_predictions == 1
    assert state.accuracy == 0.5
    assert state.precision < 1.02  # Decayed

    # Test uninitialized module observe
    agent.observe("new_mod", [0.3, 0.4, 0.3], 1)
    assert "new_mod" in agent._modules


def test_agent_retrain_targets():
    agent = ActiveInferenceAgent(["good_mod", "bad_mod"])

    # Good mod predicts correctly
    agent.observe("good_mod", [0.9, 0.05, 0.05], 0)

    # Bad mod predicts incorrectly many times, triggers retrain (accuracy < 0.4 or surprisal > 2.0)
    for _ in range(3):
        agent.observe("bad_mod", [0.9, 0.05, 0.05], 1)

    targets = agent.get_retrain_targets()
    assert "bad_mod" in targets
    assert "good_mod" not in targets


def test_agent_resource_allocation_and_precision():
    agent = ActiveInferenceAgent(["mod1", "mod2"])
    agent.observe("mod1", [0.9, 0.05, 0.05], 0) # mod1 correct (high precision)
    agent.observe("mod2", [0.9, 0.05, 0.05], 1) # mod2 incorrect (low precision)

    alloc = agent.get_resource_allocation()
    # mod2 has lower precision, so resource weight is higher (1/precision)
    assert alloc["mod2"] > alloc["mod1"]

    weights = agent.get_precision_weights()
    # mod1 has higher precision
    assert weights["mod1"] > weights["mod2"]


def test_agent_active_sampling_targets():
    agent = ActiveInferenceAgent(["mod1"])
    # Cause high average surprisal (> 1.5)
    for _ in range(5):
        agent.observe("mod1", [0.9, 0.05, 0.05], 1)

    targets = agent.get_active_sampling_targets()
    assert "mod1" in targets


def test_agent_report_and_advice():
    agent = ActiveInferenceAgent(["mod1", "mod2"])
    agent.observe("mod1", [0.9, 0.05, 0.05], 0)

    # Generate report
    report = agent.get_report()
    assert isinstance(report, ActiveInferenceReport)
    assert report.method == "active_inference"
    assert "mod1" in report.module_states

    # Advice strings
    advice = agent._advice(report)
    assert "STABIL" in advice

    # Trigger retrain string
    for _ in range(5):
        agent.observe("mod2", [0.9, 0.05, 0.05], 1)

    report2 = agent.get_report()
    advice2 = agent._advice(report2)
    assert "YENİDEN EĞİTİM GEREKLİ" in advice2

    # Manually trigger sampling only
    report2.retrain_targets = []
    report2.active_sampling_targets = ["mod1"]
    advice3 = agent._advice(report2)
    assert "AKTİF ÖRNEKLEME" in advice3
