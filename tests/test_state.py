"""Tests pour polytopia_jax/core/state.py."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.state import (
    GameState,
    TerrainType,
    UnitType,
    NO_OWNER,
    GameMode,
    TechType,
    ResourceType,
)


def test_game_state_is_pytree():
    """Vérifie que GameState est un pytree JAX valide."""
    state = GameState.create_empty(height=10, width=10, max_units=20, num_players=2)
    
    # Test que l'on peut le transformer en pytree
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 0
    
    # Test que l'on peut le mapper
    def add_one(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.int32:
            return x + 1
        return x
    
    mapped = jax.tree_util.tree_map(add_one, state)
    assert mapped.turn == state.turn + 1


def test_create_empty_state():
    """Test la création d'un état vide."""
    height, width = 10, 10
    max_units = 20
    num_players = 2
    
    state = GameState.create_empty(height, width, max_units, num_players)
    
    # Vérifier les dimensions
    assert state.height == height
    assert state.width == width
    assert state.max_units == max_units
    assert state.num_players == num_players
    
    # Vérifier les shapes
    assert state.terrain.shape == (height, width)
    assert state.city_owner.shape == (height, width)
    assert state.city_level.shape == (height, width)
    assert state.city_population.shape == (height, width)
    assert state.units_type.shape == (max_units,)
    assert state.units_pos.shape == (max_units, 2)
    assert state.units_hp.shape == (max_units,)
    assert state.units_owner.shape == (max_units,)
    assert state.units_active.shape == (max_units,)
    assert state.resource_type.shape == (height, width)
    assert state.resource_available.shape == (height, width)
    
    # Vérifier les valeurs initiales
    assert jnp.all(state.terrain == TerrainType.PLAIN)
    assert jnp.all(state.city_owner == NO_OWNER)
    assert jnp.all(state.city_level == 0)
    assert jnp.all(state.units_type == UnitType.NONE)
    assert jnp.all(state.units_owner == NO_OWNER)
    assert jnp.all(state.units_active == False)
    assert state.current_player == 0
    assert state.turn == 0
    assert state.done == False
    assert state.player_stars.shape == (num_players,)
    assert jnp.all(state.player_stars == 0)
    assert jnp.all(state.city_population == 0)
    assert jnp.all(state.city_has_port == False)
    assert state.units_payload_type.shape == (max_units,)
    assert jnp.all(state.units_payload_type == 0)
    assert state.player_techs.shape == (num_players, TechType.NUM_TECHS)
    assert jnp.all(state.player_techs == 0)
    assert state.player_score.shape == (num_players,)
    assert jnp.all(state.player_score == 0)
    assert state.score_territory.shape == (num_players,)
    assert state.score_population.shape == (num_players,)
    assert state.score_military.shape == (num_players,)
    assert state.score_resources.shape == (num_players,)
    assert jnp.all(state.score_territory == 0)
    assert jnp.all(state.score_population == 0)
    assert jnp.all(state.score_military == 0)
    assert jnp.all(state.score_resources == 0)
    assert state.game_mode == GameMode.DOMINATION
    assert state.max_turns == 30
    assert jnp.all(state.resource_type == ResourceType.NONE)
    assert jnp.all(state.resource_available == False)


def test_state_jit_compatibility():
    """Test que GameState est compatible avec jax.jit."""
    @jax.jit
    def get_turn(state: GameState) -> jnp.ndarray:
        return state.turn
    
    state = GameState.create_empty(height=5, width=5, max_units=10, num_players=2)
    turn = get_turn(state)
    assert turn == 0


def test_state_vmap_compatibility():
    """Test que GameState est compatible avec jax.vmap."""
    @jax.vmap
    def get_turn(state: GameState) -> jnp.ndarray:
        return state.turn
    
    states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x, x]),
        GameState.create_empty(height=5, width=5, max_units=10, num_players=2)
    )
    
    turns = get_turn(states)
    assert turns.shape == (2,)
    assert jnp.all(turns == 0)


def test_state_types():
    """Vérifie les types des arrays."""
    state = GameState.create_empty(height=10, width=10, max_units=20, num_players=2)
    
    # Vérifier que tous les arrays sont des jnp.ndarray
    assert isinstance(state.terrain, jnp.ndarray)
    assert isinstance(state.city_owner, jnp.ndarray)
    assert isinstance(state.city_level, jnp.ndarray)
    assert isinstance(state.city_population, jnp.ndarray)
    assert isinstance(state.city_has_port, jnp.ndarray)
    assert isinstance(state.units_type, jnp.ndarray)
    assert isinstance(state.units_pos, jnp.ndarray)
    assert isinstance(state.units_hp, jnp.ndarray)
    assert isinstance(state.units_owner, jnp.ndarray)
    assert isinstance(state.units_active, jnp.ndarray)
    assert isinstance(state.current_player, jnp.ndarray)
    assert isinstance(state.turn, jnp.ndarray)
    assert isinstance(state.done, jnp.ndarray)
    assert isinstance(state.player_stars, jnp.ndarray)
    assert isinstance(state.player_techs, jnp.ndarray)
    assert isinstance(state.units_payload_type, jnp.ndarray)
    assert isinstance(state.player_score, jnp.ndarray)
    assert isinstance(state.score_territory, jnp.ndarray)
    assert isinstance(state.score_population, jnp.ndarray)
    assert isinstance(state.score_military, jnp.ndarray)
    assert isinstance(state.score_resources, jnp.ndarray)
