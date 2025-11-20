"""Tests pour rl.session.SimulationSession."""

import numpy as np

from polytopia_jax.core.actions import ActionType, encode_action
from rl.session import SimulationConfig, SimulationSession


def test_session_advances_ai_turns():
    """Le tour revient toujours au joueur 0 après une action."""
    config = SimulationConfig(height=6, width=6, opponents=2, max_units=16, max_turns=8, difficulty="hard")
    session = SimulationSession(config)
    assert session.state is not None
    assert int(np.asarray(session.state.current_player)) == 0

    session.apply_player_action(encode_action(ActionType.END_TURN))

    assert int(np.asarray(session.state.current_player)) == 0
    # Les bonus de difficulté doivent être non nuls pour les IA
    assert int(np.asarray(session.state.player_income_bonus[1])) == session.difficulty.star_bonus


def test_session_observation_and_mask():
    """Observation numpy et masque logique doivent être cohérents."""
    config = SimulationConfig(height=5, width=5, opponents=1, max_units=12)
    session = SimulationSession(config)
    obs = session.observation()
    assert obs["terrain"].shape == (config.height, config.width)
    mask = session.legal_actions_mask()
    assert mask.shape[0] == int(ActionType.NUM_ACTIONS)
    assert mask.dtype == np.bool_
