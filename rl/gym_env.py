"""Wrapper Gymnasium single-agent."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - dépendance optionnelle
    gym = None
    spaces = None

from rl.session import SimulationConfig, SimulationSession


class PolytopiaEnv(gym.Env if gym else object):
    """Environnement Gym single-agent pilotant les IA internes."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[SimulationConfig] = None):
        if gym is None:
            raise ImportError("gymnasium est requis pour utiliser PolytopiaEnv")
        self.config = config or SimulationConfig()
        self.session = SimulationSession(self.config)
        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Discrete(1 << 30)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        self.session.reset(seed=seed)
        obs = self.session.observation()
        info = {"legal_actions_mask": self.session.legal_actions_mask()}
        return obs, info

    def step(self, action: int):
        self.session.apply_player_action(action)
        obs = self.session.observation()
        info = {"legal_actions_mask": self.session.legal_actions_mask()}
        terminated = self.session.is_done() and not self.session.reached_turn_limit()
        truncated = self.session.is_done() and self.session.reached_turn_limit()
        reward = self._compute_reward(terminated)
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - rendu non implémenté
        return None

    def close(self):
        return None

    def _compute_reward(self, terminated: bool) -> float:
        if not terminated:
            return 0.0
        state = self.session.state
        assert state is not None
        city_owner = np.array(state.city_owner, copy=True)
        player_alive = np.any(city_owner == 0)
        return 1.0 if player_alive else -1.0

    def _build_observation_space(self):
        obs = self.session.observation()
        height, width = self.config.height, self.config.width
        return spaces.Dict(
            {
                "terrain": spaces.Box(
                    low=0,
                    high=4,
                    shape=(height, width),
                    dtype=np.int32,
                ),
                "city_owner": spaces.Box(
                    low=-1,
                    high=9,
                    shape=(height, width),
                    dtype=np.int32,
                ),
                "city_level": spaces.Box(
                    low=0,
                    high=3,
                    shape=(height, width),
                    dtype=np.int32,
                ),
                "units": spaces.Box(
                    low=0,
                    high=16,
                    shape=(self.config.max_units,),
                    dtype=np.int32,
                ),
                "units_pos": spaces.Box(
                    low=0,
                    high=max(height, width),
                    shape=(self.config.max_units, 2),
                    dtype=np.int32,
                ),
                "player_stars": spaces.Box(
                    low=0,
                    high=999,
                    shape=(self.session.num_players,),
                    dtype=np.int32,
                ),
                "player_score": spaces.Box(
                    low=0,
                    high=9999,
                    shape=(self.session.num_players,),
                    dtype=np.int32,
                ),
                "current_player": spaces.Discrete(self.session.num_players),
                "turn": spaces.Discrete(self.config.max_turns + 2),
                "player_id": spaces.Discrete(self.session.num_players),
            }
        )
