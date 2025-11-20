"""Stockage en mémoire des parties live (mode Perfection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
from typing import Dict, Optional
from uuid import uuid4

import jax
import jax.numpy as jnp

from polytopia_jax.core.init import init_random, GameConfig as EngineGameConfig
from polytopia_jax.core.rules import step
from polytopia_jax.core.actions import END_TURN_ACTION
from polytopia_jax.core.state import GameState, GameMode
from polytopia_jax.ai import DifficultyPreset, HeuristicAI, resolve_difficulty

from scripts.serialize import state_to_dict
from .view_options import ViewOptions, resolve_view_options


HUMAN_PLAYER_ID = 0
PERFECTION_MAX_TURNS = 30


@dataclass
class LiveGameSession:
    """Représente une partie live maintenue en mémoire."""

    id: str
    state: GameState
    max_turns: int = PERFECTION_MAX_TURNS
    opponents: int = 3
    difficulty: str = "crazy"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ai_agents: Dict[int, HeuristicAI] = field(default_factory=dict)
    view_options: ViewOptions = field(
        default_factory=lambda: resolve_view_options()
    )


_LIVE_GAMES: Dict[str, LiveGameSession] = {}


class LiveGameNotFound(Exception):
    """Levée quand une partie live est introuvable."""


def create_perfection_game(
    opponents: int = 3,
    difficulty: str = "crazy",
    seed: Optional[int] = None,
    view_options: Optional[ViewOptions] = None,
) -> LiveGameSession:
    """Crée une partie Perfection live et la stocke."""
    num_players = _clamp_players(opponents + 1)
    board_size = _compute_board_size(opponents)
    key_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    key = jax.random.PRNGKey(key_seed)

    engine_config = EngineGameConfig(
        height=board_size,
        width=board_size,
        num_players=num_players,
        max_units=max(64, board_size * 2),
        game_mode=GameMode.PERFECTION,
        max_turns=PERFECTION_MAX_TURNS,
    )

    state = init_random(key, engine_config)
    difficulty_preset = resolve_difficulty(difficulty)
    state = _apply_difficulty_bonuses(state, difficulty_preset)
    ai_agents = {
        player_id: HeuristicAI(player_id, seed=(key_seed + player_id))
        for player_id in range(1, num_players)
    }
    state = _enforce_turn_limit(state, PERFECTION_MAX_TURNS)

    game_id = uuid4().hex
    session = LiveGameSession(
        id=game_id,
        state=state,
        max_turns=PERFECTION_MAX_TURNS,
        opponents=opponents,
        difficulty=difficulty_preset.name,
        ai_agents=ai_agents,
        view_options=view_options or resolve_view_options(),
    )
    session.state = _advance_ai_turns(session, PERFECTION_MAX_TURNS)
    _LIVE_GAMES[game_id] = session
    return session


def get_game(game_id: str) -> LiveGameSession:
    """Retourne une partie live par ID."""
    try:
        return _LIVE_GAMES[game_id]
    except KeyError as exc:
        raise LiveGameNotFound(f"Partie live '{game_id}' introuvable") from exc


def remove_game(game_id: str) -> None:
    """Supprime une partie live (utilisé pour le nettoyage)."""
    _LIVE_GAMES.pop(game_id, None)


def apply_action(game_id: str, action_id: int) -> LiveGameSession:
    """Applique une action utilisateur sur la partie live."""
    session = get_game(game_id)
    state = step(session.state, action_id)
    state = _enforce_turn_limit(state, session.max_turns)
    session.state = state
    session.state = _advance_ai_turns(session, session.max_turns)
    return session


def end_turn(game_id: str) -> LiveGameSession:
    """Termine explicitement le tour du joueur humain."""
    session = get_game(game_id)
    state = step(session.state, END_TURN_ACTION)
    state = _enforce_turn_limit(state, session.max_turns)
    session.state = state
    session.state = _advance_ai_turns(session, session.max_turns)
    return session


def serialize_session(session: LiveGameSession) -> dict:
    """Retourne le payload sérialisé pour l'API."""
    return {
        "game_id": session.id,
        "state": state_to_dict(session.state),
        "max_turns": session.max_turns,
        "opponents": session.opponents,
        "difficulty": session.difficulty,
    }


def _clamp_players(num_players: int) -> int:
    return max(2, min(4, num_players))


def _compute_board_size(opponents: int) -> int:
    opponents = max(1, opponents)
    base = 10
    return base + min(5, opponents)


def _advance_ai_turns(session: LiveGameSession, max_turns: int) -> GameState:
    state = session.state
    loop_guard = 0
    while not _is_done(state) and _current_player(state) != HUMAN_PLAYER_ID:
        player_id = _current_player(state)
        agent = session.ai_agents.get(player_id)
        if agent is None:
            state = step(state, END_TURN_ACTION)
        else:
            state = _play_ai_turn(state, agent)
        state = _enforce_turn_limit(state, max_turns)
        loop_guard += 1
        if loop_guard > 32:
            break
    return state


def _enforce_turn_limit(state: GameState, max_turns: int) -> GameState:
    current_turn = int(jnp.asarray(state.turn))
    if current_turn >= max_turns:
        return state.replace(done=jnp.array(True, dtype=jnp.bool_))
    return state


def _is_done(state: GameState) -> bool:
    return bool(jnp.asarray(state.done))


def _current_player(state: GameState) -> int:
    return int(jnp.asarray(state.current_player))


def _play_ai_turn(state: GameState, agent: HeuristicAI) -> GameState:
    local_guard = 0
    while not _is_done(state) and _current_player(state) == agent.player_id:
        action = agent.choose_action(state)
        state = step(state, action)
        local_guard += 1
        if local_guard > state.max_units * 2:
            state = step(state, END_TURN_ACTION)
            break
    return state


def _apply_difficulty_bonuses(state: GameState, preset: DifficultyPreset) -> GameState:
    """Injecte les bonus de revenu dans l'état."""
    bonus = jnp.zeros(state.num_players, dtype=jnp.int32)
    for player_id in range(1, state.num_players):
        bonus = bonus.at[player_id].set(preset.star_bonus)
    return state.replace(player_income_bonus=bonus)
