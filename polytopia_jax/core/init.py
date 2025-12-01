"""Génération d'états initiaux du jeu.

Respecte les règles de génération de carte de Polytopia :
https://polytopia.fandom.com/wiki/Map_Generation

Ordre de génération selon type de carte :
- Drylands/Lakes/Archipelago/Waterworld : Capitales → Villages primaires → Terrain → Villages secondaires
- Continents/Pangea : Masses terrestres → Capitales → Villages
"""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from .state import GameState, TerrainType, UnitType, NO_OWNER, GameMode, ResourceType, MapSize, MapType
from .actions import Direction, get_action_direction_delta
from .score import update_scores

STARTING_PLAYER_STARS = 5
INITIAL_CITY_POPULATION = 1
FRUIT_PROBABILITY = 0.3
FISH_PROBABILITY = 0.35
ORE_PROBABILITY = 0.4

# Nombre de ruines selon taille de carte
RUINS_COUNT = {
    MapSize.TINY: 4,
    MapSize.SMALL: 6,
    MapSize.NORMAL: 10,
    MapSize.LARGE: 15,
    MapSize.HUGE: 20,
    MapSize.MASSIVE: 23,
}

# Coefficient de densité de villages selon type de carte
# Formule villages primaires : floor(tiles * density_coefficient / 150)
VILLAGE_DENSITY = {
    MapType.DRYLANDS: 4.0,
    MapType.LAKES: 3.5,
    MapType.CONTINENTS: 2.0,  # Continents utilise moins de villages primaires
    MapType.PANGEA: 2.5,
    MapType.ARCHIPELAGO: 2.5,
    MapType.WATERWORLD: 1.5,
}

# Nombre de banlieues par capitale (pour Lakes/Archipelago)
SUBURBS_PER_CAPITAL = 2


def get_map_dimensions(map_size: MapSize) -> tuple[int, int]:
    """Retourne les dimensions (height, width) pour une taille de carte."""
    size_map = {
        MapSize.TINY: (11, 11),
        MapSize.SMALL: (14, 14),
        MapSize.NORMAL: (16, 16),
        MapSize.LARGE: (18, 18),
        MapSize.HUGE: (20, 20),
        MapSize.MASSIVE: (30, 30),
    }
    return size_map[map_size]


def get_map_type_probs(map_type: MapType) -> tuple[float, float, float, float, float]:
    """Retourne les probabilités de terrain selon le type de carte.
    
    Returns:
        (prob_plain, prob_forest, prob_mountain, prob_water, prob_water_deep)
    """
    # Probabilités de base pour chaque type
    # Format: (plain, forest, mountain, water_shallow, water_deep)
    type_probs = {
        MapType.DRYLANDS: (0.70, 0.20, 0.10, 0.0, 0.0),      # 0-10% eau
        MapType.LAKES: (0.50, 0.15, 0.10, 0.20, 0.05),       # 25-30% eau
        MapType.CONTINENTS: (0.35, 0.15, 0.10, 0.30, 0.10),  # 40-70% eau
        MapType.PANGEA: (0.40, 0.15, 0.10, 0.25, 0.10),      # 40-60% eau
        MapType.ARCHIPELAGO: (0.20, 0.10, 0.05, 0.40, 0.25), # 60-80% eau
        MapType.WATERWORLD: (0.05, 0.02, 0.01, 0.50, 0.42),  # 90-100% eau
    }
    return type_probs[map_type]


class GameConfig(NamedTuple):
    """Configuration pour la génération d'un état initial."""
    # Taille et type de carte (si spécifiés, remplacent height/width et probabilités)
    map_size: Optional[MapSize] = None  # Si None, utilise height/width
    map_type: Optional[MapType] = None  # Si None, utilise probabilités explicites
    
    # Dimensions (utilisées si map_size est None)
    height: int = 10
    width: int = 10
    num_players: int = 2
    max_units: int = 50
    
    # Probabilités de terrain (utilisées si map_type est None, doivent sommer à 1.0)
    prob_plain: float = 0.45
    prob_forest: float = 0.2
    prob_mountain: float = 0.15
    prob_water: float = 0.15
    prob_water_deep: float = 0.05
    
    # Probabilité d'apparition des animaux sauvages (pure esthétique)
    resource_prob_wild_animal: float = 0.3
    game_mode: GameMode = GameMode.DOMINATION
    max_turns: int = 30
    
    def get_dimensions(self) -> tuple[int, int]:
        """Retourne les dimensions effectives de la carte."""
        if self.map_size is not None:
            return get_map_dimensions(self.map_size)
        return (self.height, self.width)
    
    def get_terrain_probs(self) -> tuple[float, float, float, float, float]:
        """Retourne les probabilités de terrain effectives."""
        if self.map_type is not None:
            return get_map_type_probs(self.map_type)
        return (self.prob_plain, self.prob_forest, self.prob_mountain, 
                self.prob_water, self.prob_water_deep)


def init_random(key: jax.random.PRNGKey, config: GameConfig) -> GameState:
    """Génère un état initial aléatoire.
    
    Respecte l'ordre de génération Polytopia selon le type de carte :
    - Drylands/Lakes/Archipelago/Waterworld : Capitales → Villages primaires → Terrain → Villages secondaires
    - Continents/Pangea : Terrain (masses terrestres) → Capitales → Villages
    
    Args:
        key: Clé aléatoire JAX
        config: Configuration du jeu
    
    Returns:
        GameState initialisé
    """
    height, width = config.get_dimensions()
    keys = jax.random.split(key, 12)
    terrain_key, resource_key, overlay_key, capital_key = keys[0], keys[1], keys[2], keys[3]
    village_primary_key, village_secondary_key, suburb_key = keys[4], keys[5], keys[6]
    ruin_key, starfish_key, unit_key = keys[7], keys[8], keys[9]
    
    # Créer état vide
    state = GameState.create_empty(
        height=height,
        width=width,
        max_units=config.max_units,
        num_players=config.num_players,
    )
    
    # Ordre de génération selon type de carte
    map_type = config.map_type
    uses_quadrant_placement = map_type in (
        MapType.DRYLANDS, MapType.LAKES, MapType.ARCHIPELAGO, MapType.WATERWORLD, None
    )
    
    if uses_quadrant_placement:
        # Drylands/Lakes/Archipelago/Waterworld :
        # 1. Placer les capitales dans les quadrants
        state = _place_capitals(state, capital_key, config)
        
        # 2. Ajouter banlieues pour Lakes/Archipelago
        if map_type in (MapType.LAKES, MapType.ARCHIPELAGO):
            state = _generate_suburbs(state, suburb_key, config)
        
        # 3. Placer les villages primaires (avant terrain)
        state = _generate_villages_primary(state, village_primary_key, config)
        
        # 4. Générer le terrain
        state = _generate_terrain(state, terrain_key, config)
        
        # 5. Placer les villages secondaires (après terrain)
        state = _generate_villages_secondary(state, village_secondary_key, config)
    else:
        # Continents/Pangea :
        # 1. Générer le terrain (masses terrestres et eau)
        state = _generate_terrain(state, terrain_key, config)
        
        # 2. Placer les capitales (sur les masses terrestres)
        state = _place_capitals(state, capital_key, config)
        
        # 3. Générer tous les villages (post-terrain)
        state = _generate_villages_secondary(state, village_secondary_key, config)
    
    # Générer les ressources naturelles (uniquement près des villes/villages)
    state = _generate_resources(state, resource_key, config)
    state = _apply_resource_overlays(state, overlay_key, config)
    
    # Générer les ruines
    state = _generate_ruins(state, ruin_key, config)
    
    # Générer les starfish
    state = _generate_starfish(state, starfish_key, config)
    
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
    """Génère le terrain de manière procédurale selon le type de carte.
    
    Préserve le terrain sous les villes/villages existants (pour l'ordre
    de génération où les villages primaires sont placés avant le terrain).
    """
    rand = jax.random.uniform(key, shape=(state.height, state.width))
    
    prob_plain, prob_forest, prob_mountain, prob_water, prob_water_deep = config.get_terrain_probs()
    
    terrain = jnp.zeros_like(state.terrain, dtype=jnp.int32)
    threshold_plain = prob_plain
    threshold_forest = threshold_plain + prob_forest
    threshold_mountain = threshold_forest + prob_mountain
    threshold_shallow = threshold_mountain + prob_water
    threshold_deep = threshold_shallow + prob_water_deep

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
    
    # S'assurer que les villes/villages existants sont sur terrain terrestre
    # Les cases avec city_level > 0 doivent être PLAIN (pas d'eau)
    has_city = state.city_level > 0
    terrain = jnp.where(has_city, TerrainType.PLAIN, terrain)

    return state.replace(terrain=terrain)


def _generate_resources(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les ressources naturelles sur la carte.
    
    Les ressources sont uniquement placées à 2 cases de distance (Manhattan) 
    des villes ou villages, comme dans Polytopia.
    """
    resource_type = state.resource_type
    resource_available = state.resource_available
    key_fruit, key_fish, key_ore = jax.random.split(key, 3)
    
    height, width = state.height, state.width
    
    # Créer un masque des cases à 2 cases d'une ville/village
    # Une ville/village est identifiée par city_level > 0
    has_city_or_village = state.city_level > 0
    near_city_mask = jnp.zeros((height, width), dtype=jnp.bool_)
    
    # Pour chaque ville/village, marquer les cases à distance <= 2
    for y in range(height):
        for x in range(width):
            if bool(has_city_or_village[y, x]):
                for check_y in range(height):
                    for check_x in range(width):
                        dist = abs(check_x - x) + abs(check_y - y)
                        if dist <= 2:
                            near_city_mask = near_city_mask.at[check_y, check_x].set(True)

    def _apply_mask(current_type, current_available, mask, value):
        placement = mask & (current_type == int(ResourceType.NONE))
        new_type = jnp.where(placement, int(value), current_type)
        new_available = jnp.where(placement, True, current_available)
        return new_type, new_available

    # Fruits uniquement sur plaines près des villes/villages
    fruit_rand = jax.random.uniform(key_fruit, shape=state.terrain.shape)
    fruit_mask = (
        (state.terrain == TerrainType.PLAIN)
        & near_city_mask
        & (fruit_rand < FRUIT_PROBABILITY)
    )
    resource_type, resource_available = _apply_mask(
        resource_type, resource_available, fruit_mask, ResourceType.FRUIT
    )

    # Poissons uniquement sur eau peu profonde près des villes/villages
    fish_rand = jax.random.uniform(key_fish, shape=state.terrain.shape)
    fish_mask = (
        (state.terrain == TerrainType.WATER_SHALLOW)
        & near_city_mask
        & (fish_rand < FISH_PROBABILITY)
    )
    resource_type, resource_available = _apply_mask(
        resource_type, resource_available, fish_mask, ResourceType.FISH
    )

    # Minerais uniquement sur montagnes près des villes/villages
    ore_rand = jax.random.uniform(key_ore, shape=state.terrain.shape)
    ore_mask = (
        (state.terrain == TerrainType.MOUNTAIN)
        & near_city_mask
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


def _get_quadrant_grid(num_players: int) -> tuple[int, int]:
    """Retourne la grille de quadrants selon le nombre de joueurs.
    
    Règles Polytopia :
    - 1-4 joueurs : 4 quadrants (2x2)
    - 5-9 joueurs : 9 quadrants (3x3)
    - 10-16 joueurs : 16 quadrants (4x4)
    """
    if num_players <= 4:
        return 2, 2
    elif num_players <= 9:
        return 3, 3
    else:
        return 4, 4


def _place_capitals(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Place les capitales des joueurs sur la carte en utilisant des quadrants équitables.
    
    Respecte les règles Polytopia pour le placement en quadrants :
    - 1-4 joueurs : 4 quadrants (2x2)
    - 5-9 joueurs : 9 quadrants (3x3)
    - 10-16 joueurs : 16 quadrants (4x4)
    """
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    city_population = state.city_population.copy()
    resource_type = state.resource_type.copy()
    resource_available = state.resource_available.copy()
    terrain = state.terrain.copy()
    tiles_explored = state.tiles_explored.copy()
    
    num_players = config.num_players
    height, width = state.height, state.width
    
    # Obtenir la grille de quadrants selon les règles Polytopia
    grid_rows, grid_cols = _get_quadrant_grid(num_players)
    
    # Dimensions de chaque quadrant
    quadrant_width = width // grid_cols
    quadrant_height = height // grid_rows
    
    # Générer l'ordre des quadrants (mélangé pour l'assignation aux joueurs)
    num_quadrants = grid_rows * grid_cols
    key, subkey = jax.random.split(key)
    quadrant_order = jax.random.permutation(subkey, jnp.arange(num_quadrants))
    
    positions = []
    for player_id in range(num_players):
        # Obtenir le quadrant assigné à ce joueur
        quadrant_idx = int(quadrant_order[player_id])
        col = quadrant_idx % grid_cols
        row = quadrant_idx // grid_cols
        
        # Centre approximatif du quadrant avec randomisation dans le quadrant
        # On évite les bords du quadrant pour ne pas être trop près d'un autre joueur
        margin = max(1, min(quadrant_width, quadrant_height) // 4)
        
        key, subkey = jax.random.split(key)
        offset_x = jax.random.randint(subkey, (), -margin, margin + 1)
        key, subkey = jax.random.split(key)
        offset_y = jax.random.randint(subkey, (), -margin, margin + 1)
        
        center_x = col * quadrant_width + quadrant_width // 2 + int(offset_x)
        center_y = row * quadrant_height + quadrant_height // 2 + int(offset_y)
        
        x = max(1, min(width - 2, center_x))
        y = max(1, min(height - 2, center_y))
        
        # Chercher une case de plaine proche si la case choisie n'est pas une plaine
        # On cherche dans un rayon de 3 cases
        best_x, best_y = x, y
        best_dist = 999
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                check_x = x + dx
                check_y = y + dy
                if (0 <= check_x < width and 0 <= check_y < height):
                    # Vérifier que c'est terrestre (pas eau)
                    is_land = (terrain[check_y, check_x] != TerrainType.WATER_SHALLOW and 
                              terrain[check_y, check_x] != TerrainType.WATER_DEEP)
                    if is_land:
                        dist = abs(dx) + abs(dy)
                        if dist < best_dist:
                            best_x, best_y = check_x, check_y
                            best_dist = dist
        
        positions.append((int(best_x), int(best_y)))
    
    # Placer les capitales (niveau 1) et marquer vision initiale
    for player_id, (x, y) in enumerate(positions):
        if 0 <= x < width and 0 <= y < height:
            city_owner = city_owner.at[y, x].set(player_id)
            city_level = city_level.at[y, x].set(1)
            city_population = city_population.at[y, x].set(INITIAL_CITY_POPULATION)
            # S'assurer que la case est une plaine
            terrain = terrain.at[y, x].set(TerrainType.PLAIN)
            resource_type = resource_type.at[y, x].set(int(ResourceType.NONE))
            resource_available = resource_available.at[y, x].set(False)
            
            # Marquer vision initiale autour de la capitale (3x3, rayon 1)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    vision_y = y + dy
                    vision_x = x + dx
                    # Utiliser jnp pour les conditions pour compatibilité JIT
                    valid_y = (vision_y >= 0) & (vision_y < height)
                    valid_x = (vision_x >= 0) & (vision_x < width)
                    valid_pos = valid_y & valid_x
                    tiles_explored = jnp.where(
                        valid_pos,
                        tiles_explored.at[player_id, vision_y, vision_x].set(True),
                        tiles_explored
                    )
    
    return state.replace(
        terrain=terrain,
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
        resource_type=resource_type,
        resource_available=resource_available,
        tiles_explored=tiles_explored,
    )


def _get_num_primary_villages(config: GameConfig) -> int:
    """Calcule le nombre de villages primaires selon la formule Polytopia.
    
    Formule : floor(total_tiles * density_coefficient / 150)
    """
    height, width = config.get_dimensions()
    total_tiles = height * width
    
    if config.map_type is not None:
        density = VILLAGE_DENSITY.get(config.map_type, 3.0)
    else:
        density = 3.0  # Valeur par défaut
    
    return int(total_tiles * density / 150)


def _get_num_secondary_villages(config: GameConfig, num_existing: int) -> int:
    """Calcule le nombre de villages secondaires.
    
    Vise environ 2 villages par joueur au total.
    """
    target_total = config.num_players * 2
    return max(0, target_total - num_existing)


def _generate_suburbs(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les banlieues (suburbs) près des capitales pour Lakes/Archipelago.
    
    Les banlieues sont des villages placés à distance 2-3 des capitales,
    typiquement 2 par capitale.
    """
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    city_population = state.city_population.copy()
    terrain = state.terrain.copy()
    
    height, width = state.height, state.width
    
    # Trouver les positions des capitales
    capital_positions = []
    for y in range(height):
        for x in range(width):
            if bool(city_level[y, x] > 0) and bool(city_owner[y, x] >= 0):
                capital_positions.append((x, y))
    
    suburb_positions = []
    
    for cap_x, cap_y in capital_positions:
        # Pour chaque capitale, placer SUBURBS_PER_CAPITAL banlieues
        candidates = []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                dist = abs(dx) + abs(dy)
                if dist < 2 or dist > 3:  # Distance 2-3 de la capitale
                    continue
                sx, sy = cap_x + dx, cap_y + dy
                if 0 < sx < width - 1 and 0 < sy < height - 1:
                    # Pas de ville/village existant et terrain terrestre
                    is_land = (terrain[sy, sx] != TerrainType.WATER_SHALLOW and 
                              terrain[sy, sx] != TerrainType.WATER_DEEP)
                    no_city = city_level[sy, sx] == 0
                    if is_land and bool(no_city):
                        candidates.append((sx, sy))
        
        if candidates:
            key, subkey = jax.random.split(key)
            num_candidates = len(candidates)
            indices = jax.random.permutation(subkey, jnp.arange(num_candidates))
            
            placed = 0
            for idx in indices:
                if placed >= SUBURBS_PER_CAPITAL:
                    break
                sx, sy = candidates[int(idx)]
                # Vérifier distance aux autres banlieues
                too_close = False
                for bx, by in suburb_positions:
                    if abs(sx - bx) + abs(sy - by) <= 1:
                        too_close = True
                        break
                if not too_close:
                    suburb_positions.append((sx, sy))
                    placed += 1
    
    # Placer les banlieues
    for x, y in suburb_positions:
        city_owner = city_owner.at[y, x].set(NO_OWNER)
        city_level = city_level.at[y, x].set(1)
        city_population = city_population.at[y, x].set(0)
    
    return state.replace(
        terrain=terrain,
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )


def _generate_villages_primary(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les villages primaires (avant génération du terrain).
    
    Ces villages sont placés sur la carte avant que le terrain soit généré.
    Ils influencent ensuite la génération du terrain autour d'eux.
    """
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    city_population = state.city_population.copy()
    
    height, width = state.height, state.width
    
    # Calculer le nombre de villages primaires
    num_villages_target = _get_num_primary_villages(config)
    
    # Masque de validité
    valid_mask = jnp.ones((height, width), dtype=jnp.bool_)
    
    # Exclure les bords (2 cases de marge pour respecter wiki)
    valid_mask = valid_mask.at[:2, :].set(False)
    valid_mask = valid_mask.at[height-2:, :].set(False)
    valid_mask = valid_mask.at[:, :2].set(False)
    valid_mask = valid_mask.at[:, width-2:].set(False)
    
    # Exclure les cases déjà occupées (capitales, banlieues)
    has_city = city_level > 0
    valid_mask = valid_mask & ~has_city
    
    # Exclure les cases trop proches des villes existantes (distance 3)
    for y in range(height):
        for x in range(width):
            if bool(has_city[y, x]):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if abs(dx) + abs(dy) <= 3:
                            cy, cx = y + dy, x + dx
                            if 0 <= cy < height and 0 <= cx < width:
                                valid_mask = valid_mask.at[cy, cx].set(False)
    
    # Placer les villages
    village_positions = []
    key, subkey = jax.random.split(key)
    
    valid_positions = [(x, y) for y in range(height) for x in range(width) if bool(valid_mask[y, x])]
    
    if valid_positions:
        num_valid = len(valid_positions)
        shuffle_indices = jax.random.permutation(subkey, jnp.arange(num_valid))
        
        for idx in shuffle_indices:
            if len(village_positions) >= num_villages_target:
                break
            
            x, y = valid_positions[int(idx)]
            
            # Distance minimale entre villages primaires : 4 cases
            too_close = any(abs(x - vx) + abs(y - vy) < 4 for vx, vy in village_positions)
            
            if not too_close:
                village_positions.append((x, y))
    
    # Placer les villages
    for x, y in village_positions:
        city_owner = city_owner.at[y, x].set(NO_OWNER)
        city_level = city_level.at[y, x].set(1)
        city_population = city_population.at[y, x].set(0)
    
    return state.replace(
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )


def _generate_villages_secondary(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les villages secondaires (après génération du terrain).
    
    Ces villages sont placés après le terrain, sur des cases terrestres valides.
    Ils respectent les contraintes de distance par rapport aux autres villes/villages.
    """
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    city_population = state.city_population.copy()
    terrain = state.terrain.copy()
    
    height, width = state.height, state.width
    
    # Compter les villages existants
    num_existing = int(jnp.sum((city_level > 0) & (city_owner == NO_OWNER)))
    num_villages_target = _get_num_secondary_villages(config, num_existing)
    
    if num_villages_target <= 0:
        return state
    
    # Masque de validité
    valid_mask = jnp.ones((height, width), dtype=jnp.bool_)
    
    # Exclure les bords
    valid_mask = valid_mask.at[0, :].set(False)
    valid_mask = valid_mask.at[height-1, :].set(False)
    valid_mask = valid_mask.at[:, 0].set(False)
    valid_mask = valid_mask.at[:, width-1].set(False)
    
    # Exclure l'eau
    is_water = (terrain == TerrainType.WATER_SHALLOW) | (terrain == TerrainType.WATER_DEEP)
    valid_mask = valid_mask & ~is_water
    
    # Exclure les villes/villages existants et leurs alentours
    has_city = city_level > 0
    valid_mask = valid_mask & ~has_city
    
    for y in range(height):
        for x in range(width):
            if bool(has_city[y, x]):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if abs(dx) + abs(dy) <= 2:
                            cy, cx = y + dy, x + dx
                            if 0 <= cy < height and 0 <= cx < width:
                                valid_mask = valid_mask.at[cy, cx].set(False)
    
    # Placer les villages
    village_positions = []
    key, subkey = jax.random.split(key)
    
    valid_positions = [(x, y) for y in range(height) for x in range(width) if bool(valid_mask[y, x])]
    
    if valid_positions:
        num_valid = len(valid_positions)
        shuffle_indices = jax.random.permutation(subkey, jnp.arange(num_valid))
        
        for idx in shuffle_indices:
            if len(village_positions) >= num_villages_target:
                break
            
            x, y = valid_positions[int(idx)]
            
            # Distance minimale entre nouveaux villages : 3 cases
            too_close = any(abs(x - vx) + abs(y - vy) < 3 for vx, vy in village_positions)
            
            if not too_close:
                village_positions.append((x, y))
    
    # Placer les villages
    for x, y in village_positions:
        city_owner = city_owner.at[y, x].set(NO_OWNER)
        city_level = city_level.at[y, x].set(1)
        city_population = city_population.at[y, x].set(0)
        # S'assurer que c'est terrestre
        if bool(is_water[y, x]):
            terrain = terrain.at[y, x].set(TerrainType.PLAIN)
    
    return state.replace(
        terrain=terrain,
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )


def _generate_ruins(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les ruines sur la carte.
    
    Les ruines sont placées :
    - 4-23 selon taille de carte
    - Ne peuvent pas être à côté d'une capitale ou d'une autre ruine
    - Peuvent être sur plaine, forêt, montagne, ou océan profond
    - Ne peuvent pas être sur eau peu profonde
    """
    has_ruin = state.has_ruin.copy()
    height, width = state.height, state.width
    
    # Déterminer le nombre de ruines selon la taille
    if config.map_size is not None:
        num_ruins_target = RUINS_COUNT[config.map_size]
    else:
        # Estimation basée sur la taille de la carte
        total_tiles = height * width
        if total_tiles <= 121:  # Tiny
            num_ruins_target = 4
        elif total_tiles <= 196:  # Small
            num_ruins_target = 6
        elif total_tiles <= 256:  # Normal
            num_ruins_target = 10
        elif total_tiles <= 324:  # Large
            num_ruins_target = 15
        elif total_tiles <= 400:  # Huge
            num_ruins_target = 20
        else:  # Massive
            num_ruins_target = 23
    
    # Créer un masque de validité pour les ruines
    valid_mask = jnp.ones((height, width), dtype=jnp.bool_)
    
    # Exclure les cases avec villes/villages
    has_city_or_village = state.city_level > 0
    valid_mask = valid_mask & ~has_city_or_village
    
    # Exclure les cases à côté d'une capitale ou d'un village (distance 1)
    for y in range(height):
        for x in range(width):
            if bool(has_city_or_village[y, x]):
                # Marquer les cases adjacentes comme invalides
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        check_y = y + dy
                        check_x = x + dx
                        if 0 <= check_y < height and 0 <= check_x < width:
                            valid_mask = valid_mask.at[check_y, check_x].set(False)
    
    # Exclure l'eau peu profonde (seulement océan profond autorisé)
    is_shallow_water = state.terrain == TerrainType.WATER_SHALLOW
    valid_mask = valid_mask & ~is_shallow_water
    
    # Autoriser seulement : plaine, forêt, montagne, océan profond
    allowed_terrain = (
        (state.terrain == TerrainType.PLAIN) |
        (state.terrain == TerrainType.FOREST) |
        (state.terrain == TerrainType.MOUNTAIN) |
        (state.terrain == TerrainType.WATER_DEEP)
    )
    valid_mask = valid_mask & allowed_terrain
    
    # Placer les ruines de manière itérative
    ruin_positions = []
    key, subkey = jax.random.split(key)
    
    # Générer liste de toutes les positions valides
    valid_positions = []
    for y in range(height):
        for x in range(width):
            if bool(valid_mask[y, x]):
                valid_positions.append((x, y))
    
    # Mélanger avec JAX
    if len(valid_positions) > 0:
        # Générer indices aléatoires
        num_valid = len(valid_positions)
        shuffle_indices = jax.random.permutation(subkey, jnp.arange(num_valid))
        
        # Essayer de placer les ruines en respectant la distance minimale
        for idx in shuffle_indices:
            if len(ruin_positions) >= num_ruins_target:
                break
            
            x, y = valid_positions[int(idx)]
            
            # Vérifier distance aux ruines déjà placées (pas à côté = distance > 1)
            too_close = False
            for rx, ry in ruin_positions:
                dist = abs(x - rx) + abs(y - ry)
                if dist <= 1:
                    too_close = True
                    break
            
            if not too_close:
                ruin_positions.append((x, y))
    
    # Placer les ruines dans le state
    for x, y in ruin_positions:
        has_ruin = has_ruin.at[y, x].set(True)
    
    return state.replace(has_ruin=has_ruin)


def _generate_starfish(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les starfish sur la carte.
    
    Règles Polytopia :
    - 1 starfish pour 25 cases d'eau (shallow + deep)
    - Ne peuvent pas être adjacents à une autre starfish
    - Ne peuvent pas être adjacents à une ville (pas de lighthouse dans notre implémentation)
    - Placés sur shallow ou deep water
    """
    resource_type = state.resource_type.copy()
    resource_available = state.resource_available.copy()
    
    height, width = state.height, state.width
    
    # Compter les cases d'eau
    is_water = (state.terrain == TerrainType.WATER_SHALLOW) | (state.terrain == TerrainType.WATER_DEEP)
    water_count = jnp.sum(is_water)
    
    # Calculer le nombre de starfish (1 pour 25 cases d'eau)
    num_starfish = int(water_count) // 25
    if num_starfish == 0 and int(water_count) > 0:
        num_starfish = 1  # Au moins 1 si il y a de l'eau
    
    # Masque de validité pour starfish
    valid_mask = is_water.copy()
    
    # Exclure les cases adjacentes aux villes
    has_city = state.city_level > 0
    for y in range(height):
        for x in range(width):
            if bool(has_city[y, x]):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        cy, cx = y + dy, x + dx
                        if 0 <= cy < height and 0 <= cx < width:
                            valid_mask = valid_mask.at[cy, cx].set(False)
    
    # Générer liste de toutes les positions d'eau valides
    water_positions = [(x, y) for y in range(height) for x in range(width) if bool(valid_mask[y, x])]
    
    starfish_positions = []
    
    if water_positions and num_starfish > 0:
        key, subkey = jax.random.split(key)
        num_water = len(water_positions)
        shuffle_indices = jax.random.permutation(subkey, jnp.arange(num_water))
        
        for idx in shuffle_indices:
            if len(starfish_positions) >= num_starfish:
                break
            
            x, y = water_positions[int(idx)]
            
            # Vérifier non-adjacence aux autres starfish (distance > 1)
            too_close = any(
                abs(x - sx) <= 1 and abs(y - sy) <= 1
                for sx, sy in starfish_positions
            )
            
            if not too_close:
                starfish_positions.append((x, y))
    
    # Note : Les starfish sont purement décoratifs dans Polytopia
    # Pour l'instant, on ne stocke pas leur position car il n'y a pas de champ dédié
    # On pourrait ajouter un champ has_starfish si nécessaire pour le rendu
    # TODO: Ajouter has_starfish dans GameState si on veut les afficher
    
    return state.replace(
        resource_type=resource_type,
        resource_available=resource_available,
    )


def _init_starting_units(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Initialise les unités de départ (un guerrier par capitale).
    
    Recherche les vraies positions des capitales dans le GameState.
    """
    units_type = state.units_type.copy()
    units_pos = state.units_pos.copy()
    units_hp = state.units_hp.copy()
    units_owner = state.units_owner.copy()
    units_active = state.units_active.copy()
    units_payload = state.units_payload_type.copy()
    
    height, width = state.height, state.width
    unit_idx = 0
    
    # Trouver les positions réelles des capitales de chaque joueur
    capital_positions = {}
    for y in range(height):
        for x in range(width):
            owner = int(state.city_owner[y, x])
            level = int(state.city_level[y, x])
            if owner >= 0 and level > 0:
                # C'est une ville appartenant à un joueur
                if owner not in capital_positions:
                    capital_positions[owner] = (x, y)
    
    # Placer un guerrier sur chaque capitale et marquer leur vision comme explorée
    tiles_explored = state.tiles_explored.copy()
    for player_id in range(config.num_players):
        if player_id in capital_positions and unit_idx < config.max_units:
            x, y = capital_positions[player_id]
            units_type = units_type.at[unit_idx].set(UnitType.WARRIOR)
            units_pos = units_pos.at[unit_idx, 0].set(x)
            units_pos = units_pos.at[unit_idx, 1].set(y)
            units_hp = units_hp.at[unit_idx].set(10)  # 10 PV pour un guerrier
            units_owner = units_owner.at[unit_idx].set(player_id)
            units_active = units_active.at[unit_idx].set(True)
            units_payload = units_payload.at[unit_idx].set(UnitType.WARRIOR)
            
            # Marquer vision initiale autour de l'unité (3x3, rayon 1)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    vision_y = y + dy
                    vision_x = x + dx
                    valid_y = (vision_y >= 0) & (vision_y < height)
                    valid_x = (vision_x >= 0) & (vision_x < width)
                    valid_pos = valid_y & valid_x
                    tiles_explored = jnp.where(
                        valid_pos,
                        tiles_explored.at[player_id, vision_y, vision_x].set(True),
                        tiles_explored
                    )
            
            unit_idx += 1
    
    return state.replace(
        units_type=units_type,
        units_pos=units_pos,
        units_hp=units_hp,
        units_owner=units_owner,
        units_active=units_active,
        units_payload_type=units_payload,
        tiles_explored=tiles_explored,
    )
