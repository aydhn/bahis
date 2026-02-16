"""
rl_trader.py – Reinforcement Learning ile otonom bahis ajanı.
Stable-Baselines3 (PPO/DQN) ile ödül maksimizasyonu yapar.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 yüklü değil – RL basit modda.")


class BettingEnv(gym.Env if GYM_AVAILABLE else object):
    """Bahis ortamı – RL ajanı için gymnasium uyumlu."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, initial_bankroll: float = 10000.0):
        super().__init__()
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.step_count = 0
        self.max_steps = 100

        if GYM_AVAILABLE:
            # Gözlem: [bankroll_ratio, prob_home, prob_draw, prob_away, ev_home, ev_draw, ev_away, confidence]
            self.observation_space = spaces.Box(low=-5, high=5, shape=(8,), dtype=np.float32)
            # Aksiyon: [bahis_yok=0, home=1, draw=2, away=3] + stake_level [0..4]
            self.action_space = spaces.MultiDiscrete([4, 5])

        self._current_match = None

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed) if GYM_AVAILABLE else None
        self.bankroll = self.initial_bankroll
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        bet_type, stake_level = action[0], action[1]
        stake_pct = [0.0, 0.01, 0.02, 0.03, 0.05][stake_level]
        stake = self.bankroll * stake_pct

        # Simüle maç sonucu
        reward = self._simulate_outcome(bet_type, stake)
        self.bankroll += reward
        self.step_count += 1

        terminated = self.bankroll <= 0 or self.step_count >= self.max_steps
        truncated = False

        info = {"bankroll": self.bankroll, "reward": reward, "step": self.step_count}
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._current_match is None:
            return np.zeros(8, dtype=np.float32)
        m = self._current_match
        return np.array([
            self.bankroll / self.initial_bankroll,
            m.get("prob_home", 0.4),
            m.get("prob_draw", 0.3),
            m.get("prob_away", 0.3),
            m.get("ev_home", 0.0),
            m.get("ev_draw", 0.0),
            m.get("ev_away", 0.0),
            m.get("confidence", 0.5),
        ], dtype=np.float32)

    def _simulate_outcome(self, bet_type: int, stake: float) -> float:
        if bet_type == 0 or stake <= 0:
            return 0.0

        if self._current_match is None:
            return -stake * 0.1

        probs = [
            self._current_match.get("prob_home", 0.4),
            self._current_match.get("prob_draw", 0.3),
            self._current_match.get("prob_away", 0.3),
        ]
        outcome = np.random.choice([1, 2, 3], p=probs)
        odds_map = {
            1: 1.0 / max(probs[0], 0.05),
            2: 1.0 / max(probs[1], 0.05),
            3: 1.0 / max(probs[2], 0.05),
        }

        if outcome == bet_type:
            return stake * (odds_map[bet_type] - 1)
        else:
            return -stake

    def set_match(self, match_data: dict):
        self._current_match = match_data


class RLTrader:
    """RL tabanlı otonom bahis karar verici."""

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._env = None

        if SB3_AVAILABLE and GYM_AVAILABLE:
            self._env = BettingEnv()
            if model_path:
                try:
                    self._model = PPO.load(model_path, env=self._env)
                    logger.info(f"RL model yüklendi: {model_path}")
                except Exception as e:
                    logger.warning(f"RL model yükleme hatası: {e}")

        logger.debug("RLTrader başlatıldı.")

    def decide(self, ensemble: list[dict]) -> list[dict]:
        """Ensemble sinyallerinden bahis kararları üretir."""
        decisions = []
        for match in ensemble:
            if SB3_AVAILABLE and self._model is not None and self._env is not None:
                decision = self._rl_decide(match)
            else:
                decision = self._heuristic_decide(match)
            decisions.append(decision)
        return decisions

    def _rl_decide(self, match: dict) -> dict:
        self._env.set_match(match)
        obs = self._env._get_obs()
        action, _ = self._model.predict(obs, deterministic=True)
        bet_type, stake_level = int(action[0]), int(action[1])
        stake_pct = [0.0, 0.01, 0.02, 0.03, 0.05][stake_level]

        bet_map = {0: "skip", 1: "home", 2: "draw", 3: "away"}
        return {
            "match_id": match.get("match_id", ""),
            "market": "1X2",
            "selection": bet_map[bet_type],
            "stake_pct": stake_pct,
            "confidence": match.get("confidence", 0.5),
            "ev": match.get(f"ev_{bet_map.get(bet_type, 'home')}", 0.0),
            "odds": 1.0 / max(match.get(f"prob_{bet_map.get(bet_type, 'home')}", 0.3), 0.05),
            "source": "rl",
        }

    def _heuristic_decide(self, match: dict) -> dict:
        """RL yokken Kelly Criterion tabanlı karar."""
        best_ev = max(
            match.get("ev_home", -1),
            match.get("ev_draw", -1),
            match.get("ev_away", -1),
        )
        confidence = match.get("confidence", 0.5)

        # Sadece pozitif EV ve yeterli güven varsa bahis yap
        if best_ev <= 0.02 or confidence < 0.4:
            return {
                "match_id": match.get("match_id", ""),
                "market": "1X2",
                "selection": "skip",
                "stake_pct": 0.0,
                "confidence": confidence,
                "ev": best_ev,
                "odds": 0.0,
                "source": "heuristic",
            }

        # En iyi EV'ye sahip seçim
        options = {"home": match.get("ev_home", -1), "draw": match.get("ev_draw", -1), "away": match.get("ev_away", -1)}
        best = max(options, key=options.get)
        prob = match.get(f"prob_{best}", 0.3)
        odds = 1.0 / max(prob, 0.05)

        # Kelly Criterion: f* = (bp - q) / b where b = odds - 1
        b = odds - 1
        p = prob
        q = 1 - p
        kelly = (b * p - q) / max(b, 0.01)
        kelly = max(0, min(kelly, 0.05))  # Fractional Kelly (%5 cap)

        # Güven ayarı
        stake_pct = kelly * confidence

        return {
            "match_id": match.get("match_id", ""),
            "market": "1X2",
            "selection": best,
            "stake_pct": float(round(stake_pct, 4)),
            "confidence": confidence,
            "ev": float(best_ev),
            "odds": float(round(odds, 2)),
            "source": "heuristic_kelly",
        }

    def train(self, total_timesteps: int = 50000):
        """RL ajanını eğitir."""
        if not SB3_AVAILABLE or not GYM_AVAILABLE:
            logger.warning("SB3/Gym yok – eğitim atlanıyor.")
            return
        self._env = BettingEnv()
        self._model = PPO("MlpPolicy", self._env, verbose=1, learning_rate=3e-4, n_steps=256)
        self._model.learn(total_timesteps=total_timesteps)
        logger.success(f"RL ajan eğitildi – {total_timesteps} adım.")
