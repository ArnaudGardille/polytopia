"""Tests pour core/rules.py."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.rules import step, legal_actions_mask, _check_victory
from polytopia_jax.core.state import GameState, UnitType, NO_OWNER
from polytopia_jax.core.actions import ActionType, Direction, encode_action
from polytopia_jax.core.init import init_random, GameConfig


def test_step_no_op():
    """Test que NO_OP ne change pas l'état."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    action = encode_action(ActionType.NO_OP)
    new_state = step(state, action)
    
    # L'état ne devrait pas changer
    assert jnp.array_equal(new_state.terrain, state.terrain)
    assert new_state.current_player == state.current_player


def test_step_move_valid():
    """Test un mouvement valide."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Trouver la première unité du joueur 0
    player_0_units = jnp.where(
        (state.units_owner == 0) & state.units_active
    )[0]
    
    if len(player_0_units) > 0:
        unit_id = int(player_0_units[0])
        original_pos = state.units_pos[unit_id].copy()
        
        # Déplacer vers la droite
        action = encode_action(
            ActionType.MOVE,
            unit_id=unit_id,
            direction=Direction.RIGHT
        )
        
        new_state = step(state, action)
        new_pos = new_state.units_pos[unit_id]
        
        # La position devrait avoir changé
        assert new_pos[0] == original_pos[0] + 1
        assert new_pos[1] == original_pos[1]


def test_step_move_invalid_out_of_bounds():
    """Test qu'un mouvement hors limites est ignoré."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=5, width=5, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Placer une unité au bord
    new_units_pos = state.units_pos.at[0, 0].set(0)
    new_units_pos = new_units_pos.at[0, 1].set(0)
    state = state.replace(units_pos=new_units_pos)
    
    # Essayer de bouger vers le haut (hors limites)
    action = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.UP
    )
    
    new_state = step(state, action)
    
    # La position ne devrait pas avoir changé
    assert new_state.units_pos[0, 0] == 0
    assert new_state.units_pos[0, 1] == 0


def test_step_attack():
    """Test une attaque."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Placer deux unités adjacentes (joueur 0 et joueur 1)
    new_units_pos = state.units_pos.at[0, 0].set(5)
    new_units_pos = new_units_pos.at[0, 1].set(5)
    state = state.replace(
        units_pos=new_units_pos,
        units_owner=state.units_owner.at[0].set(0),
        units_active=state.units_active.at[0].set(True),
        units_hp=state.units_hp.at[0].set(10),
    )
    
    # Trouver une unité du joueur 1 ou en créer une
    player_1_units = jnp.where(
        (state.units_owner == 1) & state.units_active
    )[0]
    
    if len(player_1_units) > 0:
        target_id = int(player_1_units[0])
        # Placer la cible à côté de l'attaquant
        new_units_pos = state.units_pos.at[target_id, 0].set(6)
        new_units_pos = new_units_pos.at[target_id, 1].set(5)
        state = state.replace(units_pos=new_units_pos)
        
        original_hp_target = state.units_hp[target_id]
        
        # Attaquer
        action = encode_action(
            ActionType.ATTACK,
            unit_id=0,
            target_pos=(6, 5)
        )
        
        new_state = step(state, action)
        
        # La cible devrait avoir perdu des PV
        assert new_state.units_hp[target_id] < original_hp_target


def test_step_end_turn():
    """Test la fin de tour."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    original_player = state.current_player
    original_turn = state.turn
    
    # Fin de tour
    action = encode_action(ActionType.END_TURN)
    new_state = step(state, action)
    
    # Le joueur devrait avoir changé
    assert new_state.current_player != original_player
    assert new_state.current_player == (original_player + 1) % config.num_players


def test_step_end_turn_increments_turn():
    """Test que le tour s'incrémente quand on revient au joueur 0."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # S'assurer qu'on est au joueur 0
    state = state.replace(current_player=jnp.array(1))
    
    # Fin de tour (passe au joueur 0)
    action = encode_action(ActionType.END_TURN)
    new_state = step(state, action)
    
    # Le tour devrait s'être incrémenté
    assert new_state.turn == state.turn + 1
    assert new_state.current_player == 0


def test_legal_actions_mask():
    """Test le masque d'actions légales."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    mask = legal_actions_mask(state)
    
    # Le masque devrait être un array booléen
    assert mask.dtype == jnp.bool_
    assert len(mask) > 0


def test_legal_actions_mask_game_over():
    """Test que le masque est vide quand la partie est terminée."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Marquer la partie comme terminée
    state = state.replace(done=jnp.array(True))
    
    mask = legal_actions_mask(state)
    
    # Toutes les actions devraient être invalides
    assert jnp.all(~mask)


def test_check_victory():
    """Test la détection de victoire."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Initialement, la partie ne devrait pas être terminée
    state = _check_victory(state)
    assert not state.done
    
    # Éliminer le joueur 1 en supprimant sa capitale
    state = state.replace(
        city_owner=state.city_owner.at[:, :].set(0),
        city_level=state.city_level.at[:, :].set(1),
    )
    
    state = _check_victory(state)
    # Maintenant la partie devrait être terminée
    assert state.done


def test_step_jit_compatible():
    """Test que step est compatible avec jax.jit."""
    @jax.jit
    def step_jitted(state, action):
        return step(state, action)
    
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=8, width=8, num_players=2, max_units=20)
    state = init_random(key, config)
    
    action = encode_action(ActionType.END_TURN)
    new_state = step_jitted(state, action)
    
    # Vérifier que l'état est valide
    assert new_state.height == config.height
    assert new_state.width == config.width


def test_step_invalid_action_ignored():
    """Test qu'une action invalide est ignorée."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Action avec unit_id invalide
    action = encode_action(
        ActionType.MOVE,
        unit_id=999,  # ID invalide
        direction=Direction.RIGHT
    )
    
    new_state = step(state, action)
    
    # L'état ne devrait pas avoir changé
    assert jnp.array_equal(new_state.units_pos, state.units_pos)

