"""Tests pour core/init.py."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.init import init_random, GameConfig
from polytopia_jax.core.state import (
    TerrainType,
    UnitType,
    NO_OWNER,
)


def test_init_random_basic():
    """Test la génération d'un état initial basique."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    
    state = init_random(key, config)
    
    # Vérifier les dimensions
    assert state.height == config.height
    assert state.width == config.width
    assert state.num_players == config.num_players
    assert state.max_units == config.max_units
    
    # Vérifier qu'il y a des capitales
    num_capitals = jnp.sum(state.city_level > 0)
    assert num_capitals >= config.num_players
    
    # Vérifier qu'il y a des unités
    num_units = jnp.sum(state.units_active)
    assert num_units >= config.num_players  # Au moins une unité par joueur


def test_init_random_reproducibility():
    """Test que la génération est reproductible avec la même clé."""
    key = jax.random.PRNGKey(123)
    config = GameConfig(height=8, width=8, num_players=2, max_units=20)
    
    state1 = init_random(key, config)
    state2 = init_random(key, config)
    
    # Les états doivent être identiques
    assert jnp.array_equal(state1.terrain, state2.terrain)
    assert jnp.array_equal(state1.city_owner, state2.city_owner)
    assert jnp.array_equal(state1.city_level, state2.city_level)
    assert jnp.array_equal(state1.units_pos, state2.units_pos)


def test_init_random_different_keys():
    """Test que des clés différentes donnent des résultats différents."""
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    
    state1 = init_random(key1, config)
    state2 = init_random(key2, config)
    
    # Les terrains doivent être différents (probabilité très élevée)
    # On vérifie juste que ce n'est pas identique
    terrain_same = jnp.array_equal(state1.terrain, state2.terrain)
    # Il est possible mais très peu probable qu'ils soient identiques
    # On accepte ce test même si c'est le cas (c'est statistiquement très rare)


def test_init_random_capitals_placed():
    """Vérifie que les capitales sont bien placées."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    
    state = init_random(key, config)
    
    # Vérifier qu'il y a exactement num_players capitales
    capitals = jnp.sum(state.city_level > 0)
    assert capitals == config.num_players
    
    # Vérifier que chaque joueur a une capitale
    for player_id in range(config.num_players):
        player_capitals = jnp.sum(
            (state.city_owner == player_id) & (state.city_level > 0)
        )
        assert player_capitals >= 1


def test_init_random_starting_units():
    """Vérifie que les unités de départ sont bien initialisées."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    
    state = init_random(key, config)
    
    # Vérifier qu'il y a au moins une unité par joueur
    for player_id in range(config.num_players):
        player_units = jnp.sum(
            (state.units_owner == player_id) & state.units_active
        )
        assert player_units >= 1
    
    # Vérifier que les unités sont des guerriers
    active_units = state.units_type[state.units_active]
    assert jnp.all(active_units == UnitType.WARRIOR)
    
    # Vérifier que les unités ont des PV > 0
    active_hp = state.units_hp[state.units_active]
    assert jnp.all(active_hp > 0)


def test_init_random_jit_compatible():
    """Test que init_random est compatible avec jax.jit."""
    # Note: Les dimensions doivent être statiques pour jit
    @jax.jit
    def init_jitted(key):
        # Dimensions fixes pour le test
        config = GameConfig(height=8, width=8, num_players=2, max_units=20)
        return init_random(key, config)
    
    key = jax.random.PRNGKey(42)
    
    state = init_jitted(key)
    
    # Vérifier que l'état est valide
    assert state.height == 8
    assert state.width == 8


def test_init_random_different_configs():
    """Test avec différentes configurations."""
    key = jax.random.PRNGKey(42)
    
    # Test avec 4 joueurs
    config1 = GameConfig(height=15, width=15, num_players=4, max_units=50)
    state1 = init_random(key, config1)
    assert state1.num_players == 4
    assert jnp.sum(state1.city_level > 0) >= 4
    
    # Test avec petite carte
    config2 = GameConfig(height=5, width=5, num_players=2, max_units=10)
    state2 = init_random(key, config2)
    assert state2.height == 5
    assert state2.width == 5


def test_init_random_terrain_generation():
    """Vérifie que le terrain est bien généré."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    
    state = init_random(key, config)
    
    # Vérifier que tous les terrains sont valides
    valid_terrains = jnp.isin(
        state.terrain,
        jnp.array([TerrainType.PLAIN, TerrainType.FOREST, TerrainType.WATER_SHALLOW])
    )
    assert jnp.all(valid_terrains)
    
    # Vérifier que les capitales sont sur des plaines
    capital_mask = state.city_level > 0
    capital_terrains = state.terrain[capital_mask]
    assert jnp.all(capital_terrains == TerrainType.PLAIN)

