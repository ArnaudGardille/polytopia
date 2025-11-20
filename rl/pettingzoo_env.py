"""Wrapper PettingZoo (AEC) pour le joueur humain."""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector
except ImportError:  # pragma: no cover - dépendance optionnelle
    AECEnv = object
    agent_selector = None

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover - dépendance optionnelle
    spaces = None

from rl.session import SimulationConfig, SimulationSession


class PolytopiaAECEnv(AECEnv if agent_selector else object):
    """Expose le joueur humain sous forme d'environnement AEC."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[SimulationConfig] = None):
        if agent_selector is None or spaces is None:
            raise ImportError("pettingzoo et gymnasium sont requis pour PolytopiaAECEnv")
        super().__init__()
        self.config = config or SimulationConfig()
        self.session = SimulationSession(self.config)
        self.possible_agents = ["player_0"]
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = {
            agent: spaces.Discrete(1 << 30) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "terrain": spaces.Box(
                        low=0,
                        high=4,
                        shape=(self.config.height, self.config.width),
                        dtype=np.int32,
                    ),
                    "city_owner": spaces.Box(
                        low=-1,
                        high=9,
                        shape=(self.config.height, self.config.width),
                        dtype=np.int32,
                    ),
                    "city_level": spaces.Box(
                        low=0,
                        high=3,
                        shape=(self.config.height, self.config.width),
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
                        high=max(self.config.height, self.config.width),
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
            for agent in self.possible_agents
        }
        self.has_reset = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.has_reset = True
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.session.reset(seed=seed)
        self.agent_selection = self._agent_selector.reset()
        self._clear_rewards()

    def observe(self, agent: str):
        if agent not in self.agents:
            raise ValueError(f"Agent inconnu: {agent}")
        return self.session.observation()

    def step(self, action: int):
        if self.agents is None or len(self.agents) == 0:
            return
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self.session.apply_player_action(action)
        reward = self._compute_reward()
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        done = self.session.is_done()
        self.terminations[agent] = done and not self.session.reached_turn_limit()
        self.truncations[agent] = done and self.session.reached_turn_limit()
        if done:
            self.agents = []
        self._accumulate_rewards()

    def _compute_reward(self) -> float:
        if not self.session.is_done():
            return 0.0
        state = self.session.state
        assert state is not None
        alive = np.any(np.array(state.city_owner, copy=True) == 0)
        return 1.0 if alive else -1.0
