"""Sessions de simulation partagées par les wrappers RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import random

import jax
import jax.numpy as jnp
import numpy as np

from polytopia_jax.core.init import GameConfig, init_random
from polytopia_jax.core.rules import step, legal_actions_mask
from polytopia_jax.core.actions import END_TURN_ACTION
from polytopia_jax.core.state import GameMode, GameState
from polytopia_jax.ai import HeuristicAI, DifficultyPreset, resolve_difficulty


@dataclass
class SimulationConfig:
    """Configuration haut-niveau pour un environnement RL."""

    height: int = 12
    width: int = 12
    opponents: int = 1
    max_units: int = 64
    max_turns: int = 30
    difficulty: str = "normal"
    game_mode: GameMode = GameMode.DOMINATION


class SimulationSession:
    """Pilote une simulation backend + IA adverses."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state: Optional[GameState] = None
        self.ai_agents: Dict[int, HeuristicAI] = {}
        self.difficulty: DifficultyPreset = resolve_difficulty(config.difficulty)
        self.seed: Optional[int] = None
        self.reset()

    @property
    def num_players(self) -> int:
        if self.state is None:
            return max(2, min(4, self.config.opponents + 1))
        return int(self.state.num_players)

    def reset(self, seed: Optional[int] = None) -> GameState:
        """Réinitialise la session entière."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.seed = seed
        num_players = max(2, min(4, self.config.opponents + 1))
        key = jax.random.PRNGKey(seed)
        engine_config = GameConfig(
            height=self.config.height,
            width=self.config.width,
            num_players=num_players,
            max_units=self.config.max_units,
            game_mode=self.config.game_mode,
            max_turns=self.config.max_turns,
        )
        state = init_random(key, engine_config)
        state = self._apply_difficulty_bonuses(state)
        self.ai_agents = {
            player_id: HeuristicAI(player_id, seed=seed + player_id)
            for player_id in range(1, num_players)
        }
        state = self._advance_ai_turns(state)
        self.state = state
        return state

    def apply_player_action(self, action_id: int) -> GameState:
        """Applique une action du joueur 0 et laisse les IA jouer."""
        if self.state is None:
            raise RuntimeError("Session non initialisée")
        self.state = step(self.state, action_id)
        self.state = self._advance_ai_turns(self.state)
        return self.state

    def legal_actions_mask(self) -> np.ndarray:
        """Masque d'actions valides pour le joueur courant."""
        if self.state is None:
            raise RuntimeError("Session non initialisée")
        mask = legal_actions_mask(self.state)
        return np.array(mask, dtype=np.bool_)

    def observation(self, player_id: int = 0) -> dict:
        """Vue numpy du plateau pour le joueur demandé."""
        if self.state is None:
            raise RuntimeError("Session non initialisée")
        state = self.state
        return {
            "terrain": np.array(state.terrain, copy=True),
            "city_owner": np.array(state.city_owner, copy=True),
            "city_level": np.array(state.city_level, copy=True),
            "units": np.array(state.units_type, copy=True),
            "units_pos": np.array(state.units_pos, copy=True),
            "current_player": int(np.asarray(state.current_player)),
            "turn": int(np.asarray(state.turn)),
            "player_stars": np.array(state.player_stars, copy=True),
            "player_score": np.array(state.player_score, copy=True),
            "player_id": player_id,
        }

    def is_done(self) -> bool:
        if self.state is None:
            return False
        return bool(np.asarray(self.state.done))

    def reached_turn_limit(self) -> bool:
        if self.state is None:
            return False
        turn = int(np.asarray(self.state.turn))
        return turn >= self.config.max_turns

    def _advance_ai_turns(self, state: GameState) -> GameState:
        """Laisse les IA jouer jusqu'à revenir au joueur 0."""
        loop_guard = 0
        while not self._is_done_state(state) and self._current_player(state) != 0:
            player_id = self._current_player(state)
            agent = self.ai_agents.get(player_id)
            if agent is None:
                state = step(state, END_TURN_ACTION)
            else:
                state = self._play_ai_turn(state, agent)
            loop_guard += 1
            if loop_guard > 32:
                break
        return state

    def _play_ai_turn(self, state: GameState, agent: HeuristicAI) -> GameState:
        """Laisse l'agent heuristique jouer son tour complet."""
        local_guard = 0
        while not self._is_done_state(state) and self._current_player(state) == agent.player_id:
            action = agent.choose_action(state)
            state = step(state, action)
            local_guard += 1
            if local_guard > state.max_units * 2:
                state = step(state, END_TURN_ACTION)
                break
        return state

    def _apply_difficulty_bonuses(self, state: GameState) -> GameState:
        bonus = jnp.zeros(state.num_players, dtype=jnp.int32)
        for player_id in range(1, state.num_players):
            bonus = bonus.at[player_id].set(self.difficulty.star_bonus)
        return state.replace(player_income_bonus=bonus)

    @staticmethod
    def _current_player(state: GameState) -> int:
        return int(np.asarray(state.current_player))

    @staticmethod
    def _is_done_state(state: GameState) -> bool:
        return bool(np.asarray(state.done))
