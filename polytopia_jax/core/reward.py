"""Fonctions de récompense pour l'apprentissage par renforcement."""

import jax
import jax.numpy as jnp
from .state import GameState, NO_OWNER, GameMode


def compute_reward(state: GameState, prev_state: GameState) -> jnp.ndarray:
    """Calcule la récompense pour le joueur actif entre deux états.
    
    Pour MVP, on utilise des récompenses simples :
    - +1.0 pour capturer une ville
    - +0.5 pour éliminer une unité ennemie
    - -0.5 pour perdre une unité
    - +10.0 pour gagner (éliminer tous les adversaires)
    - -10.0 pour perdre (être éliminé)
    
    Args:
        state: État actuel
        prev_state: État précédent
    
    Returns:
        Récompense pour le joueur actif (scalaire)
    """
    player = prev_state.current_player
    
    reward = jnp.array(0.0)
    
    # Récompense pour capture de ville
    cities_captured = _count_cities_captured(state, prev_state, player)
    reward = reward + cities_captured * 1.0
    
    # Récompense pour élimination d'unités ennemies
    enemy_units_killed = _count_enemy_units_killed(state, prev_state, player)
    reward = reward + enemy_units_killed * 0.5
    
    # Pénalité pour perte d'unités
    own_units_lost = _count_own_units_lost(state, prev_state, player)
    reward = reward - own_units_lost * 0.5
    
    # Récompense/pénalité pour victoire/défaite
    def add_victory_reward(reward):
        # Vérifier si le joueur a gagné
        player_won = _check_player_won(state, player)
        return reward + jnp.where(player_won, 10.0, -10.0)
    
    reward = jax.lax.cond(
        state.done,
        add_victory_reward,
        lambda r: r,
        reward
    )
    
    return reward


def _count_cities_captured(
    state: GameState,
    prev_state: GameState,
    player: int
) -> int:
    """Compte le nombre de villes capturées par le joueur."""
    # Villes appartenant au joueur maintenant
    current_cities = jnp.sum(
        (state.city_owner == player) & (state.city_level > 0)
    )
    
    # Villes appartenant au joueur avant
    prev_cities = jnp.sum(
        (prev_state.city_owner == player) & (prev_state.city_level > 0)
    )
    
    return current_cities - prev_cities


def _count_enemy_units_killed(
    state: GameState,
    prev_state: GameState,
    player: int
) -> int:
    """Compte le nombre d'unités ennemies éliminées."""
    # Compter les unités ennemies actives avant
    enemy_mask_prev = (prev_state.units_owner != player) & prev_state.units_active
    prev_enemy_units = jnp.sum(enemy_mask_prev)
    
    # Compter les unités ennemies actives maintenant
    enemy_mask_curr = (state.units_owner != player) & state.units_active
    current_enemy_units = jnp.sum(enemy_mask_curr)
    
    return prev_enemy_units - current_enemy_units


def _count_own_units_lost(
    state: GameState,
    prev_state: GameState,
    player: int
) -> int:
    """Compte le nombre d'unités perdues par le joueur."""
    # Unités du joueur avant
    prev_own_units = jnp.sum(
        (prev_state.units_owner == player) & prev_state.units_active
    )
    
    # Unités du joueur maintenant
    current_own_units = jnp.sum(
        (state.units_owner == player) & state.units_active
    )
    
    return prev_own_units - current_own_units


def _check_player_won(state: GameState, player: int) -> jnp.ndarray:
    """Vérifie si le joueur a gagné."""
    # Vérifier si le joueur a encore une capitale
    has_capital = jnp.any(
        (state.city_owner == player) & (state.city_level > 0)
    )
    
    # En mode Perfection, la victoire se base sur le score final
    top_score = jnp.max(state.player_score)
    is_top_scorer = state.player_score[player] == top_score
    perfection_win = state.done & (state.game_mode == GameMode.PERFECTION) & is_top_scorer
    
    domination_win = state.done & (state.game_mode != GameMode.PERFECTION) & has_capital
    
    return perfection_win | domination_win


def compute_reward_all_players(
    state: GameState,
    prev_state: GameState
) -> jnp.ndarray:
    """Calcule les récompenses pour tous les joueurs.
    
    Args:
        state: État actuel
        prev_state: État précédent
    
    Returns:
        Array de récompenses [num_players]
    """
    player_ids = jnp.arange(prev_state.num_players)
    
    def compute_for_player(player_id):
        temp_state = state.replace(current_player=player_id)
        temp_prev_state = prev_state.replace(current_player=player_id)
        return compute_reward(temp_state, temp_prev_state)
    
    rewards = jax.vmap(compute_for_player)(player_ids)
    
    return rewards
