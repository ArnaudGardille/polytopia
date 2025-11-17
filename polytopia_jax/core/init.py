"""Génération d'états initiaux du jeu."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from .state import GameState, TerrainType, UnitType, NO_OWNER
from .actions import Direction, get_action_direction_delta


class GameConfig(NamedTuple):
    """Configuration pour la génération d'un état initial."""
    height: int = 10
    width: int = 10
    num_players: int = 2
    max_units: int = 50
    # Probabilités de terrain (doivent sommer à 1.0)
    prob_plain: float = 0.6
    prob_forest: float = 0.2
    prob_water: float = 0.2


def init_random(key: jax.random.PRNGKey, config: GameConfig) -> GameState:
    """Génère un état initial aléatoire.
    
    Args:
        key: Clé aléatoire JAX
        config: Configuration du jeu
    
    Returns:
        GameState initialisé
    """
    key, terrain_key, capital_key, unit_key = jax.random.split(key, 4)
    
    # Créer état vide
    state = GameState.create_empty(
        height=config.height,
        width=config.width,
        max_units=config.max_units,
        num_players=config.num_players,
    )
    
    # Générer terrain
    state = _generate_terrain(state, terrain_key, config)
    
    # Placer les capitales
    state = _place_capitals(state, capital_key, config)
    
    # Initialiser les unités de départ
    state = _init_starting_units(state, unit_key, config)
    
    return state


def _generate_terrain(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère le terrain de manière procédurale."""
    # Générer des valeurs aléatoires uniformes
    rand = jax.random.uniform(key, shape=(state.height, state.width))
    
    # Assigner les types de terrain selon les probabilités
    terrain = jnp.zeros_like(state.terrain, dtype=jnp.int32)
    
    # Plaine par défaut
    terrain = jnp.where(rand < config.prob_plain, TerrainType.PLAIN, terrain)
    
    # Forêt
    threshold_forest = config.prob_plain + config.prob_forest
    terrain = jnp.where(
        (rand >= config.prob_plain) & (rand < threshold_forest),
        TerrainType.FOREST,
        terrain
    )
    
    # Eau peu profonde
    terrain = jnp.where(
        rand >= threshold_forest,
        TerrainType.WATER_SHALLOW,
        terrain
    )
    
    return state.replace(terrain=terrain)


def _place_capitals(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Place les capitales des joueurs sur la carte."""
    # Stratégie simple : diviser la carte en zones et placer une capitale par zone
    # Pour MVP, on place les capitales aux coins opposés
    
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    
    # Calculer les positions des capitales
    # Joueur 0 : coin supérieur gauche
    # Joueur 1 : coin inférieur droit
    # etc.
    
    positions = []
    for player_id in range(config.num_players):
        if player_id == 0:
            pos = (1, 1)  # Coin supérieur gauche
        elif player_id == 1:
            pos = (config.width - 2, config.height - 2)  # Coin inférieur droit
        else:
            # Pour plus de 2 joueurs, répartir sur les autres coins
            if player_id == 2:
                pos = (config.width - 2, 1)  # Coin supérieur droit
            else:  # player_id == 3
                pos = (1, config.height - 2)  # Coin inférieur gauche
        
        positions.append(pos)
    
    # Placer les capitales (niveau 1)
    for player_id, (x, y) in enumerate(positions):
        if x < config.width and y < config.height:
            city_owner = city_owner.at[y, x].set(player_id)
            city_level = city_level.at[y, x].set(1)
            # S'assurer que la case est une plaine
            state = state.replace(terrain=state.terrain.at[y, x].set(TerrainType.PLAIN))
    
    return state.replace(city_owner=city_owner, city_level=city_level)


def _init_starting_units(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Initialise les unités de départ (un guerrier par capitale)."""
    units_type = state.units_type.copy()
    units_pos = state.units_pos.copy()
    units_hp = state.units_hp.copy()
    units_owner = state.units_owner.copy()
    units_active = state.units_active.copy()
    
    unit_idx = 0
    
    # Pour chaque joueur, trouver sa capitale et y placer un guerrier
    # On utilise les positions calculées dans _place_capitals
    positions = []
    for player_id in range(config.num_players):
        if player_id == 0:
            pos = (1, 1)  # Coin supérieur gauche
        elif player_id == 1:
            pos = (config.width - 2, config.height - 2)  # Coin inférieur droit
        else:
            # Pour plus de 2 joueurs, répartir sur les autres coins
            if player_id == 2:
                pos = (config.width - 2, 1)  # Coin supérieur droit
            else:  # player_id == 3
                pos = (1, config.height - 2)  # Coin inférieur gauche
        positions.append((player_id, pos))
    
    # Placer les unités
    for player_id, (x, y) in positions:
        if unit_idx < config.max_units and x < config.width and y < config.height:
            units_type = units_type.at[unit_idx].set(UnitType.WARRIOR)
            units_pos = units_pos.at[unit_idx, 0].set(x)
            units_pos = units_pos.at[unit_idx, 1].set(y)
            units_hp = units_hp.at[unit_idx].set(10)  # 10 PV pour un guerrier
            units_owner = units_owner.at[unit_idx].set(player_id)
            units_active = units_active.at[unit_idx].set(True)
            unit_idx += 1
    
    return state.replace(
        units_type=units_type,
        units_pos=units_pos,
        units_hp=units_hp,
        units_owner=units_owner,
        units_active=units_active,
    )

