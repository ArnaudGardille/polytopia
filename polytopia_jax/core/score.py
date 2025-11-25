"""Fonctions utilitaires pour le calcul des scores.

Système de score selon Polytopia :
- Armée : 5 pts par étoile de coût d'unité (Super Unit = 50 pts)
- Territoire : 20 pts par case contrôlée
- Exploration : 5 pts par case explorée
- Villes : Niveau 1 = 100 pts + 5 pts par population, Niveau 2+ = +50 pts par niveau
- Temples : 100 pts + 50 pts par niveau
- Monuments : 400 pts
- Parcs : 250 pts
- Science : 100 pts par tier de technologie
- Bonus difficulté : Score brut × (100% + bonus) pour mode Perfection
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from .state import GameState, GameMode, UnitType

# Constantes pour calcul de score (copiées depuis rules.py pour éviter dépendance circulaire)
# Coûts d'unités : NONE, WARRIOR, DEFENDER, ARCHER, RIDER, RAFT, KNIGHT, SWORDSMAN, CATAPULT, GIANT
UNIT_COST_SCORE = jnp.array([0, 2, 3, 3, 4, 0, 6, 5, 6, 20], dtype=jnp.int32)

# Tiers de technologies : 21 technologies au total
TECH_TIER_SCORE = jnp.array([
    0,  # NONE
    1,  # CLIMBING (T1)
    1,  # FISHING (T1)
    1,  # HUNTING (T1)
    1,  # ORGANIZATION (T1)
    1,  # RIDING (T1)
    2,  # ARCHERY (T2)
    2,  # RAMMING (T2)
    2,  # FARMING (T2)
    2,  # FORESTRY (T2)
    2,  # FREE_SPIRIT (T2)
    2,  # MEDITATION (T2)
    2,  # MINING (T2)
    2,  # ROADS (T2)
    2,  # SAILING (T2)
    2,  # STRATEGY (T2)
    3,  # AQUATISM (T3)
    3,  # PHILOSOPHY (T3)
    3,  # SMITHERY (T3)
    3,  # CHIVALRY (T3)
    3,  # MATHEMATICS (T3)
], dtype=jnp.int32)

# Constantes selon Polytopia
TERRITORY_POINTS_PER_TILE = 20  # 20 pts par case contrôlée
EXPLORATION_POINTS_PER_TILE = 5  # 5 pts par case explorée
UNIT_POINTS_PER_STAR = 5  # 5 pts par étoile de coût d'unité
SUPER_UNIT_POINTS = 50  # 50 pts pour Super Unit (Giant)
CITY_BASE_POINTS = 100  # 100 pts de base pour niveau 1
CITY_POPULATION_POINTS = 5  # 5 pts par point de population
CITY_LEVEL_BONUS = 50  # +50 pts par niveau au-dessus de 1
TEMPLE_BASE_POINTS = 100  # 100 pts de base pour temple
TEMPLE_LEVEL_POINTS = 50  # 50 pts par niveau de temple
MONUMENT_POINTS = 400  # 400 pts par monument
PARK_POINTS = 250  # 250 pts par parc
TECH_POINTS_PER_TIER = 100  # 100 pts par tier de technologie


def compute_score_components(state: GameState) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Retourne les totaux de score et leurs composantes par joueur selon les règles Polytopia."""
    num_players = state.player_stars.shape[0]
    player_ids = jnp.arange(num_players, dtype=jnp.int32)
    
    def per_player(player_id):
        owner_mask = state.city_owner == player_id
        has_city = owner_mask & (state.city_level > 0)
        
        # Territoire : 20 pts par case contrôlée
        territory_count = jnp.sum(has_city, dtype=jnp.int32)
        territory = territory_count * TERRITORY_POINTS_PER_TILE
        
        # Villes : Niveau 1 = 100 + 5*pop, Niveau 2+ = +50 par niveau
        def compute_city_points(y, x, acc):
            has_city_here = has_city[y, x]
            level = state.city_level[y, x]
            pop = state.city_population[y, x]
            
            # Points de base : 100 pour niveau 1+
            base = jnp.where(has_city_here, CITY_BASE_POINTS, 0)
            
            # Points population : 5 par point
            pop_points = jnp.where(has_city_here, pop * CITY_POPULATION_POINTS, 0)
            
            # Bonus niveau : +50 par niveau au-dessus de 1
            level_bonus = jnp.where(
                has_city_here & (level > 1),
                (level - 1) * CITY_LEVEL_BONUS,
                0
            )
            
            city_points = base + pop_points + level_bonus
            return acc + city_points
        
        # Parcourir toutes les cases pour calculer points villes
        h, w = state.terrain.shape[0], state.terrain.shape[1]
        city_points_total = jnp.array(0, dtype=jnp.int32)
        
        def scan_row(y, acc):
            def scan_col(x, a):
                return compute_city_points(y, x, a)
            return jax.lax.fori_loop(0, w, scan_col, acc)
        
        city_points_total = jax.lax.fori_loop(0, h, scan_row, city_points_total)
        
        # Militaire : 5 pts par étoile de coût d'unité
        def compute_unit_points(unit_id, acc):
            is_player_unit = (state.units_owner[unit_id] == player_id) & state.units_active[unit_id]
            unit_type = state.units_type[unit_id]
            # Utiliser l'index sécurisé pour éviter les erreurs hors limites
            cost = jnp.where(
                unit_type < UNIT_COST_SCORE.shape[0],
                UNIT_COST_SCORE[unit_type],
                0
            )
            
            # Super Unit (Giant) = 50 pts, sinon 5 pts par étoile
            is_giant = unit_type == UnitType.GIANT
            unit_points = jnp.where(
                is_giant,
                SUPER_UNIT_POINTS,
                cost * UNIT_POINTS_PER_STAR
            )
            
            return acc + jnp.where(is_player_unit, unit_points, 0)
        
        max_units = state.units_type.shape[0]
        military = jax.lax.fori_loop(0, max_units, compute_unit_points, jnp.array(0, dtype=jnp.int32))
        
        # Exploration : 5 pts par case explorée
        explored_count = jnp.sum(state.tiles_explored[player_id], dtype=jnp.int32)
        exploration = explored_count * EXPLORATION_POINTS_PER_TILE
        
        # Temples : 100 + 50 * niveau par temple
        temple_points = jnp.sum(
            jnp.where(
                owner_mask & state.city_has_temple,
                TEMPLE_BASE_POINTS + (state.city_temple_level * TEMPLE_LEVEL_POINTS),
                0
            ),
            dtype=jnp.int32
        )
        
        # Monuments : 400 pts par monument
        monument_points = jnp.sum(
            jnp.where(
                owner_mask & state.city_has_monument,
                MONUMENT_POINTS,
                0
            ),
            dtype=jnp.int32
        )
        
        # Parcs : 250 pts par parc
        park_points = jnp.sum(
            jnp.where(
                owner_mask & state.city_has_park,
                PARK_POINTS,
                0
            ),
            dtype=jnp.int32
        )
        
        # Science : 100 pts par tier de technologie
        player_techs = state.player_techs[player_id]
        # Calculer points pour chaque technologie
        def compute_tech_points(tech_id, acc):
            has_tech = player_techs[tech_id]
            tier = jnp.where(
                tech_id < TECH_TIER_SCORE.shape[0],
                TECH_TIER_SCORE[tech_id],
                0
            )
            points = jnp.where(has_tech, tier * TECH_POINTS_PER_TIER, 0)
            return acc + points
        
        num_techs = player_techs.shape[0]
        tech_points = jax.lax.fori_loop(0, num_techs, compute_tech_points, jnp.array(0, dtype=jnp.int32))
        
        # Score brut (sans bonus difficulté)
        base_score = (
            territory + city_points_total + military + exploration +
            temple_points + monument_points + park_points + tech_points
        )
        
        return base_score, territory, city_points_total, military, exploration, tech_points
    
    totals, territory, population, military, exploration, science = jax.vmap(per_player)(player_ids)
    return totals, territory, population, military, exploration, science


def _compute_difficulty_bonus_multiplier_jax(
    num_opponents: jnp.ndarray,
    difficulty_bonus_percent: jnp.ndarray
) -> jnp.ndarray:
    """Calcule le multiplicateur de bonus difficulté pour Perfection (compatible JAX).
    
    Formule Polytopia : 100% + 41% * ln(nb_adversaires) + bonus difficulté
    
    Args:
        num_opponents: Nombre d'adversaires (joueurs - 1)
        difficulty_bonus_percent: Bonus de difficulté en pourcentage (20, 40, ou 80)
    
    Returns:
        Multiplicateur de score (ex: 1.5 pour 150%)
    """
    # 100% + 41% * ln(nb_adversaires) + bonus difficulté
    base_multiplier = 1.0
    opponents_multiplier = 0.41 * jnp.log(jnp.maximum(num_opponents, 1).astype(jnp.float32))
    difficulty_multiplier = difficulty_bonus_percent.astype(jnp.float32) / 100.0
    
    return base_multiplier + opponents_multiplier + difficulty_multiplier


def update_scores(state: GameState) -> GameState:
    """Met à jour les champs de score d'un GameState.
    
    En mode Perfection, applique le bonus difficulté selon la formule Polytopia.
    """
    totals, territory, population, military, exploration, science = compute_score_components(state)
    
    # Appliquer le bonus difficulté en mode Perfection
    is_perfection = state.game_mode == GameMode.PERFECTION
    num_opponents = jnp.array(state.num_players - 1, dtype=jnp.int32)
    
    def compute_multiplier_per_player(player_id, base_score):
        # Calculer le bonus difficulté moyen des adversaires
        # Le joueur 0 est le joueur humain, les autres sont les adversaires
        has_opponents = num_opponents > 0
        
        def with_opponents(_):
            # Calculer la somme des bonus des adversaires (joueurs 1 à num_players-1)
            # Utiliser une boucle pour éviter jnp.arange avec tracer
            def sum_opponent_bonus(i, acc):
                is_opponent = (i > 0) & (i < state.num_players)
                bonus = jnp.where(is_opponent, state.player_income_bonus[i], 0)
                return acc + bonus
            
            total_bonus = jax.lax.fori_loop(0, state.num_players, sum_opponent_bonus, jnp.array(0, dtype=jnp.int32))
            avg_difficulty_bonus = jnp.where(
                num_opponents > 0,
                total_bonus.astype(jnp.float32) / num_opponents.astype(jnp.float32),
                0.0
            )
            
            # Convertir le bonus d'étoiles en pourcentage approximatif
            # easy: 1★ = 20%, normal: 2★ = 40%, hard: 3★ = 60%, crazy: 5★ = 80%
            difficulty_percent = jnp.where(
                avg_difficulty_bonus <= 1, 20,
                jnp.where(
                    avg_difficulty_bonus <= 2, 40,
                    jnp.where(
                        avg_difficulty_bonus <= 3, 60,
                        80  # Crazy
                    )
                )
            )
            
            # Calculer multiplicateur selon formule Polytopia
            multiplier = _compute_difficulty_bonus_multiplier_jax(
                jnp.array(num_opponents, dtype=jnp.int32),
                difficulty_percent.astype(jnp.int32)
            )
            
            return multiplier
        
        def no_opponents(_):
            return jnp.array(1.0, dtype=jnp.float32)
        
        multiplier = jax.lax.cond(
            has_opponents,
            with_opponents,
            no_opponents,
            operand=None
        )
        
        # Appliquer multiplicateur uniquement en mode Perfection
        final_score = jnp.where(
            is_perfection,
            (base_score.astype(jnp.float32) * multiplier).astype(jnp.int32),
            base_score
        )
        
        return final_score
    
    # Appliquer bonus pour chaque joueur
    player_ids = jnp.arange(state.num_players, dtype=jnp.int32)
    final_scores = jax.vmap(compute_multiplier_per_player)(player_ids, totals)
    
    return state.replace(
        player_score=final_scores,
        score_territory=territory,
        score_population=population,
        score_military=military,
        score_resources=jnp.zeros_like(totals),  # Plus utilisé dans nouveau système
        score_exploration=exploration,
    )
