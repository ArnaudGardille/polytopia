"""Fonctions utilitaires pour le calcul des scores."""

from typing import Tuple
import jax
import jax.numpy as jnp
from .state import GameState

# Pondérations inspirées des catégories de Polytopia
TERRITORY_POINTS = 100
POPULATION_POINTS = 5
MILITARY_POINTS = 20
RESOURCE_POINTS = 2
EXPLORATION_POINTS_PER_TILE = 5


def compute_score_components(state: GameState) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Retourne les totaux de score et leurs composantes par joueur."""
    num_players = state.player_stars.shape[0]
    player_ids = jnp.arange(num_players, dtype=jnp.int32)
    
    def per_player(player_id):
        owner_mask = state.city_owner == player_id
        has_city = owner_mask & (state.city_level > 0)
        territory = jnp.sum(has_city, dtype=jnp.int32) * TERRITORY_POINTS
        population = jnp.sum(jnp.where(owner_mask, state.city_population, 0), dtype=jnp.int32) * POPULATION_POINTS
        military = jnp.sum(
            (state.units_owner == player_id) & state.units_active,
            dtype=jnp.int32,
        ) * MILITARY_POINTS
        resources = state.player_stars[player_id] * RESOURCE_POINTS
        # Points d'exploration : nombre de cases explorées
        explored_count = jnp.sum(state.tiles_explored[player_id], dtype=jnp.int32)
        exploration = explored_count * EXPLORATION_POINTS_PER_TILE
        
        # Points pour temples : 100 + 50 * niveau par temple
        temple_points = jnp.sum(
            jnp.where(
                owner_mask & state.city_has_temple,
                100 + (state.city_temple_level * 50),
                0
            ),
            dtype=jnp.int32
        )
        
        # Points pour monuments : 400 pts par monument
        monument_points = jnp.sum(
            jnp.where(
                owner_mask & state.city_has_monument,
                400,
                0
            ),
            dtype=jnp.int32
        )
        
        # Points pour parcs : 250 pts par parc
        park_points = jnp.sum(
            jnp.where(
                owner_mask & state.city_has_park,
                250,
                0
            ),
            dtype=jnp.int32
        )
        
        total = territory + population + military + resources + exploration + temple_points + monument_points + park_points
        return total, territory, population, military, resources, exploration
    
    totals, territory, population, military, resources, exploration = jax.vmap(per_player)(player_ids)
    return totals, territory, population, military, resources, exploration


def update_scores(state: GameState) -> GameState:
    """Met à jour les champs de score d'un GameState."""
    totals, territory, population, military, resources, exploration = compute_score_components(state)
    return state.replace(
        player_score=totals,
        score_territory=territory,
        score_population=population,
        score_military=military,
        score_resources=resources,
        score_exploration=exploration,
    )
