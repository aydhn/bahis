"""
rl_betting_env.py – Reinforcement Learning: Bahis Ortamı + PPO Ajanı.

Gözetimli Öğrenme: "Geçmişte X oldu, şimdi ne olur?"
Pekiştirmeli Öğrenme: "Binlerce kez batıp, para yönetimini KENDİ öğren."

Gymnasium Custom Environment:
  - Ortam (Environment): Geçmiş maç verileri + bahis oranları
  - Ajan (Agent): Bot (PPO sinir ağı)
  - Aksiyonlar: {Pas Geç, %1 Bas, %3 Bas, %5 Bas, %10 Bas}
  - Gözlem: [model_prob, odds, bankroll_ratio, form, momentum, ...]
  - Ödül: Kasa artarsa (+), azalırsa (-), batarsa büyük (-)

Eğitim:
  stable-baselines3 PPO ile 3 yıllık veri üzerinde self-play.
  Ajan, nerede pas geçeceğini ve kasa yönetimini deneme-yanılma ile öğrenir.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_OK = True
except ImportError:
    GYM_OK = False
    logger.info("gymnasium yüklü değil – RL env devre dışı.")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_OK = True
except ImportError:
    SB3_OK = False
    logger.info("stable-baselines3 yüklü değil – heuristic RL aktif.")

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models" / "rl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BettingMatch:
    """Tek maç verisi (ortam state'i için)."""
    match_id: str = ""
    model_prob_home: float = 0.33
    model_prob_draw: float = 0.33
    model_prob_away: float = 0.34
    odds_home: float = 2.0
    odds_draw: float = 3.0
    odds_away: float = 3.5
    value_edge: float = 0.0
    confidence: float = 0.5
    home_form: float = 0.5      # Son 5 maç formu (0-1)
    away_form: float = 0.5
    home_momentum: float = 0.0  # Kalman momentum
    away_momentum: float = 0.0
    volatility: float = 0.5     # Maç volatilitesi
    result: int = -1            # 0=home, 1=draw, 2=away (-1=bilinmiyor)


# ═══════════════════════════════════════════════
#  GYMNASIUM CUSTOM ENVIRONMENT
# ═══════════════════════════════════════════════
if GYM_OK:
    class BettingEnv(gym.Env):
        """Bahis ortamı: ajan maç seçip stake belirler.

        Observation Space (12 boyut):
            [model_prob_home, model_prob_draw, model_prob_away,
             odds_home, odds_draw, odds_away,
             value_edge, confidence, bankroll_ratio,
             home_form, momentum_diff, volatility]

        Action Space (5 discrete):
            0: Pas Geç
            1: %1 Bas (home)
            2: %3 Bas (home)
            3: %5 Bas (home)
            4: %10 Bas (home)

        Reward:
            Kazanç → +profit / initial_bankroll
            Kayıp → -loss / initial_bankroll
            Pas   → küçük pozitif (sağduyulu)
            İflas → -10 (büyük ceza)
        """

        metadata = {"render_modes": ["human"]}

        # Aksiyon → stake yüzdesi mapping
        ACTION_STAKES = {0: 0.0, 1: 0.01, 2: 0.03, 3: 0.05, 4: 0.10}

        def __init__(self, matches: list[BettingMatch] | None = None,
                     initial_bankroll: float = 10000.0,
                     render_mode: str | None = None):
            super().__init__()

            self._matches = matches or []
            self._initial_bankroll = initial_bankroll
            self._bankroll = initial_bankroll
            self._step_idx = 0
            self._total_bets = 0
            self._total_won = 0
            self._history: list[dict] = []
            self.render_mode = render_mode

            # Observation: 12 boyutlu sürekli uzay
            self.observation_space = spaces.Box(
                low=0.0, high=10.0, shape=(12,), dtype=np.float32,
            )

            # Action: 5 discrete aksiyon
            self.action_space = spaces.Discrete(5)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._bankroll = self._initial_bankroll
            self._step_idx = 0
            self._total_bets = 0
            self._total_won = 0
            self._history = []

            if self.np_random is not None:
                self.np_random.shuffle(self._matches)

            obs = self._get_obs()
            return obs, {}

        def step(self, action: int):
            if self._step_idx >= len(self._matches):
                return self._get_obs(), 0.0, True, False, {}

            match = self._matches[self._step_idx]
            stake_pct = self.ACTION_STAKES.get(action, 0.0)
            stake = self._bankroll * stake_pct

            reward = 0.0
            bet_result = "pass"

            if stake_pct > 0 and match.result >= 0:
                self._total_bets += 1

                # En yüksek value edge'li seçimi oyna
                selections = [
                    ("home", match.model_prob_home, match.odds_home),
                    ("draw", match.model_prob_draw, match.odds_draw),
                    ("away", match.model_prob_away, match.odds_away),
                ]
                best = max(selections, key=lambda x: x[1] * x[2] - 1)
                sel_name, sel_prob, sel_odds = best
                sel_idx = {"home": 0, "draw": 1, "away": 2}[sel_name]

                if match.result == sel_idx:
                    profit = stake * (sel_odds - 1)
                    self._bankroll += profit
                    self._total_won += 1
                    reward = profit / self._initial_bankroll
                    bet_result = "win"
                else:
                    self._bankroll -= stake
                    reward = -stake / self._initial_bankroll
                    bet_result = "loss"

                self._history.append({
                    "step": self._step_idx,
                    "action": action,
                    "stake_pct": stake_pct,
                    "selection": sel_name,
                    "odds": sel_odds,
                    "result": bet_result,
                    "bankroll": self._bankroll,
                })

            elif stake_pct == 0:
                # Pas geçme ödülü: maçın value edge'i düşükse akıllıca
                if match.value_edge < 0.02:
                    reward = 0.001  # Doğru pas geçme bonus
                else:
                    reward = -0.001  # İyi fırsatı kaçırma ceza

            # İflas kontrolü
            if self._bankroll <= 0:
                reward = -10.0
                self._step_idx = len(self._matches)  # Oyun bitti
                return self._get_obs(), reward, True, False, {"bankrupt": True}

            self._step_idx += 1
            terminated = self._step_idx >= len(self._matches)
            truncated = False

            # Episode sonu bonus: kârdaysak
            if terminated and self._bankroll > self._initial_bankroll:
                roi = (self._bankroll - self._initial_bankroll) / self._initial_bankroll
                reward += roi * 2.0  # ROI bonusu

            return self._get_obs(), float(reward), terminated, truncated, {}

        def _get_obs(self) -> np.ndarray:
            """Güncel observation vektörü."""
            if self._step_idx >= len(self._matches):
                return np.zeros(12, dtype=np.float32)

            m = self._matches[self._step_idx]
            bankroll_ratio = self._bankroll / self._initial_bankroll

            return np.array([
                m.model_prob_home,
                m.model_prob_draw,
                m.model_prob_away,
                min(m.odds_home / 5.0, 2.0),   # Normalize (0-2)
                min(m.odds_draw / 5.0, 2.0),
                min(m.odds_away / 5.0, 2.0),
                max(min(m.value_edge, 0.5), -0.5) + 0.5,  # 0-1
                m.confidence,
                min(bankroll_ratio, 3.0),
                m.home_form,
                (m.home_momentum - m.away_momentum + 1) / 2,  # Normalize
                m.volatility,
            ], dtype=np.float32)

        def render(self):
            if self.render_mode == "human":
                br = self._bankroll
                roi = (br - self._initial_bankroll) / self._initial_bankroll
                print(
                    f"Step {self._step_idx}/{len(self._matches)} | "
                    f"Kasa: {br:.0f} | ROI: {roi:+.1%} | "
                    f"Bets: {self._total_bets} | Won: {self._total_won}"
                )


# ═══════════════════════════════════════════════
#  RL AJAN EĞİTİCİ
# ═══════════════════════════════════════════════
class RLBettingAgent:
    """PPO ile eğitilen bahis ajanı.

    Kullanım:
        agent = RLBettingAgent()
        # Eğitim verisi yükle
        agent.train(matches, total_timesteps=100000)
        # Tahmin
        action = agent.predict(observation)
        # Model kaydet/yükle
        agent.save()
        agent.load()
    """

    MODEL_NAME = "ppo_betting_agent"

    def __init__(self, initial_bankroll: float = 10000.0):
        self._initial_bankroll = initial_bankroll
        self._model = None
        self._env = None
        self._trained = False
        logger.debug("RLBettingAgent başlatıldı.")

    def train(self, matches: list[BettingMatch] | list[dict],
              total_timesteps: int = 100_000,
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 64) -> dict:
        """PPO ile ajanı eğit."""
        if not GYM_OK or not SB3_OK:
            logger.warning("[RL] gymnasium veya stable-baselines3 yüklü değil.")
            return {"status": "skipped", "reason": "missing dependencies"}

        # Dict'leri BettingMatch'e dönüştür
        match_objs = []
        for m in matches:
            if isinstance(m, dict):
                match_objs.append(BettingMatch(**{
                    k: v for k, v in m.items()
                    if k in BettingMatch.__dataclass_fields__
                }))
            else:
                match_objs.append(m)

        if len(match_objs) < 50:
            logger.warning(f"[RL] Eğitim verisi çok az ({len(match_objs)}). Min 50 maç.")
            return {"status": "skipped", "reason": "insufficient data"}

        logger.info(
            f"[RL] PPO eğitimi başlıyor: {len(match_objs)} maç, "
            f"{total_timesteps} timestep"
        )

        # Ortam oluştur
        def make_env():
            return BettingEnv(
                matches=match_objs,
                initial_bankroll=self._initial_bankroll,
            )

        vec_env = DummyVecEnv([make_env])

        # PPO model
        self._model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=min(n_steps, len(match_objs)),
            batch_size=min(batch_size, len(match_objs)),
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
        )

        self._model.learn(total_timesteps=total_timesteps)
        self._trained = True
        self._env = vec_env

        # Değerlendirme
        eval_result = self._evaluate(match_objs)
        logger.success(
            f"[RL] Eğitim tamamlandı. "
            f"ROI: {eval_result['roi']:.2%}, "
            f"Win Rate: {eval_result['win_rate']:.0%}"
        )

        return eval_result

    def _evaluate(self, matches: list[BettingMatch],
                  n_episodes: int = 5) -> dict:
        """Eğitilmiş ajanı değerlendir."""
        if not self._model or not GYM_OK:
            return {"status": "not_trained"}

        results = []
        for _ in range(n_episodes):
            env = BettingEnv(
                matches=matches,
                initial_bankroll=self._initial_bankroll,
            )
            obs, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                done = terminated or truncated

            roi = (env._bankroll - self._initial_bankroll) / self._initial_bankroll
            win_rate = env._total_won / max(env._total_bets, 1)
            results.append({
                "final_bankroll": env._bankroll,
                "roi": roi,
                "total_bets": env._total_bets,
                "win_rate": win_rate,
                "total_reward": total_reward,
            })

        avg_roi = float(np.mean([r["roi"] for r in results]))
        avg_wr = float(np.mean([r["win_rate"] for r in results]))
        avg_bets = float(np.mean([r["total_bets"] for r in results]))

        return {
            "status": "evaluated",
            "avg_roi": avg_roi,
            "roi": avg_roi,
            "avg_win_rate": avg_wr,
            "win_rate": avg_wr,
            "avg_bets_per_episode": avg_bets,
            "n_episodes": n_episodes,
        }

    def predict(self, observation: np.ndarray | dict) -> dict:
        """Verilen gözlem için aksiyon tavsiyesi."""
        if isinstance(observation, dict):
            obs = np.array([
                observation.get("model_prob_home", 0.33),
                observation.get("model_prob_draw", 0.33),
                observation.get("model_prob_away", 0.34),
                min(observation.get("odds_home", 2.0) / 5.0, 2.0),
                min(observation.get("odds_draw", 3.0) / 5.0, 2.0),
                min(observation.get("odds_away", 3.5) / 5.0, 2.0),
                observation.get("value_edge", 0) + 0.5,
                observation.get("confidence", 0.5),
                observation.get("bankroll_ratio", 1.0),
                observation.get("home_form", 0.5),
                observation.get("momentum_diff", 0.5),
                observation.get("volatility", 0.5),
            ], dtype=np.float32)
        else:
            obs = observation

        if self._model and self._trained:
            action, _ = self._model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = self._heuristic_action(obs)

        action_names = {
            0: "PAS GEÇ", 1: "%1 BAS", 2: "%3 BAS", 3: "%5 BAS", 4: "%10 BAS",
        }

        if GYM_OK:
            stake_pct = BettingEnv.ACTION_STAKES.get(action, 0)
        else:
            stake_pct = {0: 0, 1: 0.01, 2: 0.03, 3: 0.05, 4: 0.10}.get(action, 0)

        return {
            "action": action,
            "action_name": action_names.get(action, "?"),
            "stake_pct": stake_pct,
            "method": "ppo" if self._trained else "heuristic",
        }

    def _heuristic_action(self, obs: np.ndarray) -> int:
        """PPO yoksa kural tabanlı karar."""
        if len(obs) < 8:
            return 0

        value_edge = obs[6] - 0.5  # De-normalize
        confidence = obs[7]

        if value_edge < 0.02:
            return 0  # Pas
        elif value_edge < 0.05 or confidence < 0.4:
            return 1  # %1
        elif value_edge < 0.10:
            return 2  # %3
        elif value_edge < 0.15 and confidence > 0.6:
            return 3  # %5
        elif value_edge >= 0.15 and confidence > 0.7:
            return 4  # %10
        else:
            return 1

    def save(self, path: str | Path | None = None):
        """Eğitilmiş modeli kaydet."""
        if self._model:
            save_path = path or (MODELS_DIR / self.MODEL_NAME)
            self._model.save(str(save_path))
            logger.info(f"[RL] Model kaydedildi: {save_path}")

    def load(self, path: str | Path | None = None) -> bool:
        """Kaydedilmiş modeli yükle."""
        if not SB3_OK:
            return False
        load_path = path or (MODELS_DIR / f"{self.MODEL_NAME}.zip")
        load_path = Path(load_path)
        if not load_path.exists() and not Path(f"{load_path}.zip").exists():
            logger.debug(f"[RL] Model bulunamadı: {load_path}")
            return False
        try:
            self._model = PPO.load(str(load_path))
            self._trained = True
            logger.success(f"[RL] Model yüklendi: {load_path}")
            return True
        except Exception as e:
            logger.warning(f"[RL] Model yükleme hatası: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained
