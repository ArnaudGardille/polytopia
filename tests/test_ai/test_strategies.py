"""Tests basiques pour les stratégies IA."""

import jax

from polytopia_jax.ai.strategies import StrategyAI, resolve_strategy_name
from polytopia_jax.core.actions import END_TURN_ACTION
from polytopia_jax.core.init import GameConfig, init_random


def _make_state():
    key = jax.random.PRNGKey(1)
    config = GameConfig(height=6, width=6, num_players=2, max_units=16)
    return init_random(key, config)


def test_resolve_strategy_name_unknown_defaults_to_rush():
    assert resolve_strategy_name("unknown") == "rush"
    assert resolve_strategy_name(None) == "rush"


def test_idle_strategy_always_end_turn():
    state = _make_state()
    ai = StrategyAI(player_id=0, strategy_name="idle", seed=0)
    action = ai.choose_action(state)
    assert action == END_TURN_ACTION
