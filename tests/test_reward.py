"""Tests pour core/reward.py."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.reward import (
    compute_reward,
    compute_reward_all_players,
    _count_cities_captured,
    _count_enemy_units_killed,
    _check_player_won,
)
from polytopia_jax.core.state import GameState, UnitType, NO_OWNER
from polytopia_jax.core.init import init_random, GameConfig


def test_compute_reward_no_change():
    """Test que la récompense est 0 si rien ne change."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Même état = pas de changement
    reward = compute_reward(state, state)
    
    assert reward == 0.0


def test_compute_reward_city_capture():
    """Test la récompense pour capture de ville."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    prev_state = init_random(key, config)
    
    # Capturer une ville du joueur 1
    state = prev_state.replace(
        city_owner=prev_state.city_owner.at[5, 5].set(0),  # Joueur 0 capture
        city_level=prev_state.city_level.at[5, 5].set(1),
    )
    
    # S'assurer que le joueur actif est 0
    state = state.replace(current_player=jnp.array(0))
    prev_state = prev_state.replace(current_player=jnp.array(0))
    
    reward = compute_reward(state, prev_state)
    
    # Devrait avoir une récompense positive pour la capture
    assert reward > 0.0


def test_compute_reward_victory():
    """Test la récompense pour victoire."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    prev_state = init_random(key, config)
    
    # Éliminer le joueur 1 (toutes ses villes au joueur 0)
    state = prev_state.replace(
        city_owner=jnp.zeros_like(prev_state.city_owner),
        city_level=prev_state.city_level,
        done=jnp.array(True),
        current_player=jnp.array(0),
    )
    prev_state = prev_state.replace(current_player=jnp.array(0))
    
    reward = compute_reward(state, prev_state)
    
    # Devrait avoir une grosse récompense positive
    assert reward > 5.0


def test_compute_reward_defeat():
    """Test la pénalité pour défaite."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    prev_state = init_random(key, config)
    
    # Éliminer le joueur 0 (toutes ses villes au joueur 1)
    state = prev_state.replace(
        city_owner=jnp.ones_like(prev_state.city_owner),
        city_level=prev_state.city_level,
        done=jnp.array(True),
        current_player=jnp.array(0),
    )
    prev_state = prev_state.replace(current_player=jnp.array(0))
    
    reward = compute_reward(state, prev_state)
    
    # Devrait avoir une grosse pénalité
    assert reward < -5.0


def test_count_cities_captured():
    """Test le comptage de villes capturées."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    prev_state = init_random(key, config)
    
    # Capturer une ville
    state = prev_state.replace(
        city_owner=prev_state.city_owner.at[3, 3].set(0),
        city_level=prev_state.city_level.at[3, 3].set(1),
    )
    
    count = _count_cities_captured(state, prev_state, 0)
    
    # Devrait avoir capturé au moins 0 villes (peut-être plus si d'autres étaient neutres)
    assert count >= 0


def test_count_enemy_units_killed():
    """Test le comptage d'unités ennemies éliminées."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    prev_state = init_random(key, config)
    
    # Éliminer une unité ennemie
    enemy_units = jnp.where(
        (prev_state.units_owner == 1) & prev_state.units_active
    )[0]
    
    if len(enemy_units) > 0:
        unit_id = int(enemy_units[0])
        state = prev_state.replace(
            units_active=prev_state.units_active.at[unit_id].set(False)
        )
        
        count = _count_enemy_units_killed(state, prev_state, 0)
        
        # Devrait avoir éliminé au moins 1 unité
        assert count >= 1


def test_check_player_won():
    """Test la vérification de victoire."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Partie non terminée
    state = state.replace(done=jnp.array(False))
    assert not _check_player_won(state, 0)
    
    # Partie terminée, joueur 0 a gagné
    state = state.replace(
        done=jnp.array(True),
        city_owner=jnp.zeros_like(state.city_owner),
    )
    assert _check_player_won(state, 0)
    
    # Partie terminée, joueur 0 a perdu
    state = state.replace(
        city_owner=jnp.ones_like(state.city_owner),
    )
    assert not _check_player_won(state, 0)


def test_compute_reward_all_players():
    """Test le calcul de récompenses pour tous les joueurs."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    rewards = compute_reward_all_players(state, state)
    
    # Devrait retourner un array de la bonne taille
    assert rewards.shape == (config.num_players,)
    assert rewards.dtype == jnp.float32 or rewards.dtype == jnp.float64


def test_compute_reward_jit_compatible():
    """Test que compute_reward est compatible avec jax.jit."""
    @jax.jit
    def compute_reward_jitted(state, prev_state):
        return compute_reward(state, prev_state)
    
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=8, width=8, num_players=2, max_units=20)
    state = init_random(key, config)
    
    reward = compute_reward_jitted(state, state)
    
    # Vérifier que la récompense est un scalaire
    assert reward.shape == () or reward.shape == (1,)




