"""Génération d'états initiaux du jeu."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from .state import GameState, TerrainType, UnitType, NO_OWNER, GameMode, ResourceType
from .actions import Direction, get_action_direction_delta
from .score import update_scores

STARTING_PLAYER_STARS = 5
INITIAL_CITY_POPULATION = 1
FRUIT_PROBABILITY = 0.3
FISH_PROBABILITY = 0.35
ORE_PROBABILITY = 0.4


class GameConfig(NamedTuple):
    """Configuration pour la génération d'un état initial."""
    height: int = 10
    width: int = 10
    num_players: int = 2
    max_units: int = 50
    # Probabilités de terrain (doivent sommer à 1.0)
    prob_plain: float = 0.45
    prob_forest: float = 0.2
    prob_mountain: float = 0.15
    prob_water: float = 0.15
    prob_water_deep: float = 0.05
    # Probabilité d'apparition des animaux sauvages (pure esthétique)
    resource_prob_wild_animal: float = 0.3
    game_mode: GameMode = GameMode.DOMINATION
    max_turns: int = 30


def init_random(key: jax.random.PRNGKey, config: GameConfig) -> GameState:
    """Génère un état initial aléatoire.
    
    Args:
        key: Clé aléatoire JAX
        config: Configuration du jeu
    
    Returns:
        GameState initialisé
    """
    key, terrain_key, resource_key, overlay_key, capital_key, unit_key = jax.random.split(key, 6)
    
    # Créer état vide
    state = GameState.create_empty(
        height=config.height,
        width=config.width,
        max_units=config.max_units,
        num_players=config.num_players,
    )
    
    # Générer terrain
    state = _generate_terrain(state, terrain_key, config)
    
    # Générer les ressources naturelles
    state = _generate_resources(state, resource_key)
    state = _apply_resource_overlays(state, overlay_key, config)
    
    # Placer les capitales
    state = _place_capitals(state, capital_key, config)
    
    # Initialiser les unités de départ
    state = _init_starting_units(state, unit_key, config)
    
    # Donne des étoiles de départ à chaque joueur et configure le mode de jeu
    state = state.replace(
        player_stars=jnp.full(
            (config.num_players,),
            STARTING_PLAYER_STARS,
            dtype=jnp.int32,
        ),
        game_mode=jnp.array(int(config.game_mode), dtype=jnp.int32),
        max_turns=jnp.array(config.max_turns, dtype=jnp.int32),
    )
    
    state = update_scores(state)
    return state


def _generate_terrain(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère le terrain de manière procédurale."""
    rand = jax.random.uniform(key, shape=(state.height, state.width))
    
    terrain = jnp.zeros_like(state.terrain, dtype=jnp.int32)
    threshold_plain = config.prob_plain
    threshold_forest = threshold_plain + config.prob_forest
    threshold_mountain = threshold_forest + config.prob_mountain
    threshold_shallow = threshold_mountain + config.prob_water
    threshold_deep = threshold_shallow + config.prob_water_deep

    terrain = jnp.where(rand < threshold_plain, TerrainType.PLAIN, terrain)
    terrain = jnp.where(
        (rand >= threshold_plain) & (rand < threshold_forest),
        TerrainType.FOREST,
        terrain,
    )
    terrain = jnp.where(
        (rand >= threshold_forest) & (rand < threshold_mountain),
        TerrainType.MOUNTAIN,
        terrain,
    )
    terrain = jnp.where(
        (rand >= threshold_mountain) & (rand < threshold_shallow),
        TerrainType.WATER_SHALLOW,
        terrain,
    )
    terrain = jnp.where(
        (rand >= threshold_shallow) & (rand < threshold_deep),
        TerrainType.WATER_DEEP,
        terrain,
    )

    return state.replace(terrain=terrain)


def _generate_resources(
    state: GameState,
    key: jax.random.PRNGKey,
) -> GameState:
    """Génère les ressources naturelles sur la carte."""
    resource_type = state.resource_type
    resource_available = state.resource_available
    key_fruit, key_fish, key_ore = jax.random.split(key, 3)

    def _apply_mask(current_type, current_available, mask, value):
        placement = mask & (current_type == int(ResourceType.NONE))
        new_type = jnp.where(placement, int(value), current_type)
        new_available = jnp.where(placement, True, current_available)
        return new_type, new_available

    fruit_rand = jax.random.uniform(key_fruit, shape=state.terrain.shape)
    fruit_mask = (
        (state.terrain == TerrainType.PLAIN)
        & (fruit_rand < FRUIT_PROBABILITY)
    )
    resource_type, resource_available = _apply_mask(
        resource_type, resource_available, fruit_mask, ResourceType.FRUIT
    )

    fish_rand = jax.random.uniform(key_fish, shape=state.terrain.shape)
    fish_mask = (
        (state.terrain == TerrainType.WATER_SHALLOW)
        & (fish_rand < FISH_PROBABILITY)
    )
    resource_type, resource_available = _apply_mask(
        resource_type, resource_available, fish_mask, ResourceType.FISH
    )

    ore_rand = jax.random.uniform(key_ore, shape=state.terrain.shape)
    ore_mask = (
        (state.terrain == TerrainType.MOUNTAIN)
        & (ore_rand < ORE_PROBABILITY)
    )
    resource_type, resource_available = _apply_mask(
        resource_type, resource_available, ore_mask, ResourceType.ORE
    )

    return state.replace(
        resource_type=resource_type,
        resource_available=resource_available,
    )


def _apply_resource_overlays(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig,
) -> GameState:
    """Adapte les tuiles de terrain pour refléter visuellement les ressources."""
    terrain = state.terrain
    resource_type = state.resource_type
    resource_int = resource_type.astype(jnp.int32)

    fruit_mask = resource_int == int(ResourceType.FRUIT)
    fish_mask = resource_int == int(ResourceType.FISH)
    ore_mask = resource_int == int(ResourceType.ORE)

    terrain = jnp.where(fruit_mask, TerrainType.PLAIN_FRUIT, terrain)
    terrain = jnp.where(fish_mask, TerrainType.WATER_SHALLOW_WITH_FISH, terrain)
    terrain = jnp.where(ore_mask, TerrainType.MOUNTAIN_WITH_MINE, terrain)

    forest_mask = terrain == TerrainType.FOREST
    if config.resource_prob_wild_animal > 0:
        animal_roll = jax.random.uniform(key, shape=terrain.shape)
        spawn_mask = forest_mask & (animal_roll < config.resource_prob_wild_animal)
        terrain = jnp.where(spawn_mask, TerrainType.FOREST_WITH_WILD_ANIMAL, terrain)

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
    city_population = state.city_population.copy()
    resource_type = state.resource_type.copy()
    resource_available = state.resource_available.copy()
    
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
    
    # Initialiser l'exploration : toutes cases non explorées au départ
    tiles_explored = state.tiles_explored.copy()
    
    # Placer les capitales (niveau 1) et marquer vision initiale
    for player_id, (x, y) in enumerate(positions):
        if x < config.width and y < config.height:
            city_owner = city_owner.at[y, x].set(player_id)
            city_level = city_level.at[y, x].set(1)
            city_population = city_population.at[y, x].set(INITIAL_CITY_POPULATION)
            # S'assurer que la case est une plaine
            state = state.replace(terrain=state.terrain.at[y, x].set(TerrainType.PLAIN))
            resource_type = resource_type.at[y, x].set(int(ResourceType.NONE))
            resource_available = resource_available.at[y, x].set(False)
            
            # Marquer vision initiale autour de la capitale (3x3, rayon 1)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    vision_y = y + dy
                    vision_x = x + dx
                    if (0 <= vision_y < config.height and 0 <= vision_x < config.width):
                        tiles_explored = tiles_explored.at[player_id, vision_y, vision_x].set(True)
    
    return state.replace(
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
        resource_type=resource_type,
        resource_available=resource_available,
        tiles_explored=tiles_explored,
    )


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
    units_payload = state.units_payload_type.copy()
    
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
            units_payload = units_payload.at[unit_idx].set(UnitType.WARRIOR)
            unit_idx += 1
    
    return state.replace(
        units_type=units_type,
        units_pos=units_pos,
        units_hp=units_hp,
        units_owner=units_owner,
        units_active=units_active,
        units_payload_type=units_payload,
    )
