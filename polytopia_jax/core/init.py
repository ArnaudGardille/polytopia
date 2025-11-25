"""Génération d'états initiaux du jeu."""

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
    
    Args:
        key: Clé aléatoire JAX
        config: Configuration du jeu
    
    Returns:
        GameState initialisé
    """
    height, width = config.get_dimensions()
    key, terrain_key, resource_key, overlay_key, capital_key, village_key, ruin_key, starfish_key, unit_key = jax.random.split(key, 9)
    
    # Créer état vide
    state = GameState.create_empty(
        height=height,
        width=width,
        max_units=config.max_units,
        num_players=config.num_players,
    )
    
    # Générer terrain selon type de carte
    state = _generate_terrain(state, terrain_key, config)
    
    # Placer les capitales (doit être fait avant villages et ressources)
    state = _place_capitals(state, capital_key, config)
    
    # Générer les villages neutres
    state = _generate_villages(state, village_key, config)
    
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
    """Génère le terrain de manière procédurale selon le type de carte."""
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


def _place_capitals(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Place les capitales des joueurs sur la carte en utilisant des quadrants équitables."""
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    city_population = state.city_population.copy()
    resource_type = state.resource_type.copy()
    resource_available = state.resource_available.copy()
    terrain = state.terrain.copy()
    tiles_explored = state.tiles_explored.copy()
    
    num_players = config.num_players
    height, width = state.height, state.width
    
    # Calculer le nombre de quadrants (arrondi vers le haut pour avoir au moins un quadrant par joueur)
    # Pour 2 joueurs : 2x1 ou 1x2, pour 4 joueurs : 2x2, etc.
    import math
    cols = math.ceil(math.sqrt(num_players))
    rows = math.ceil(num_players / cols)
    
    # Dimensions de chaque quadrant
    quadrant_width = width // cols
    quadrant_height = height // rows
    
    positions = []
    for player_id in range(num_players):
        # Calculer la position dans la grille de quadrants
        col = player_id % cols
        row = player_id // cols
        
        # Centre approximatif du quadrant avec un peu de randomisation
        center_x = col * quadrant_width + quadrant_width // 2
        center_y = row * quadrant_height + quadrant_height // 2
        
        # Ajouter un peu de randomisation (±1 case) pour éviter les positions trop prévisibles
        key, subkey = jax.random.split(key)
        offset_x = jax.random.randint(subkey, (), -1, 2)
        key, subkey = jax.random.split(key)
        offset_y = jax.random.randint(subkey, (), -1, 2)
        
        x = jnp.clip(center_x + offset_x, 1, width - 2)
        y = jnp.clip(center_y + offset_y, 1, height - 2)
        
        # Chercher une case de plaine proche si la case choisie n'est pas une plaine
        # On cherche dans un rayon de 2 cases
        best_x, best_y = x, y
        best_dist = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                check_x = x + dx
                check_y = y + dy
                if (0 <= check_x < width and 0 <= check_y < height):
                    if terrain[check_y, check_x] == TerrainType.PLAIN:
                        dist = abs(dx) + abs(dy)
                        if best_dist == 0 or dist < best_dist:
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


def _generate_villages(
    state: GameState,
    key: jax.random.PRNGKey,
    config: GameConfig
) -> GameState:
    """Génère les villages neutres sur la carte.
    
    Les villages sont placés :
    - À au moins 2 cases des capitales
    - À au moins 1 case des bords
    - À au moins 2 cases entre eux
    - Sur terrain terrestre (plaine, forêt, montagne)
    """
    city_owner = state.city_owner.copy()
    city_level = state.city_level.copy()
    city_population = state.city_population.copy()
    terrain = state.terrain.copy()
    
    height, width = state.height, state.width
    num_players = config.num_players
    
    # Calculer le nombre de villages souhaités (environ 2-3 par joueur)
    num_villages_target = num_players * 2 + int(jax.random.randint(key, (), 0, num_players + 1))
    
    # Créer une grille de validité pour les villages
    # Une case est valide si :
    # - Pas de capitale à moins de 2 cases
    # - À au moins 1 case du bord
    # - Terrain terrestre (pas eau)
    valid_mask = jnp.ones((height, width), dtype=jnp.bool_)
    
    # Exclure les bords (1 case de marge)
    valid_mask = valid_mask.at[0, :].set(False)
    valid_mask = valid_mask.at[height-1, :].set(False)
    valid_mask = valid_mask.at[:, 0].set(False)
    valid_mask = valid_mask.at[:, width-1].set(False)
    
    # Exclure les cases d'eau
    is_water = (terrain == TerrainType.WATER_SHALLOW) | (terrain == TerrainType.WATER_DEEP)
    valid_mask = valid_mask & ~is_water
    
    # Exclure les cases déjà occupées par une capitale
    has_capital = city_level > 0
    valid_mask = valid_mask & ~has_capital
    
    # Exclure les cases à moins de 2 cases d'une capitale (distance de Manhattan)
    # On crée une grille de distance minimale aux capitales
    capital_distances = jnp.full((height, width), 999, dtype=jnp.int32)
    for y in range(height):
        for x in range(width):
            if bool(has_capital[y, x]):
                # Calculer distance de Manhattan à toutes les cases
                for check_y in range(height):
                    for check_x in range(width):
                        dist = abs(check_x - x) + abs(check_y - y)
                        capital_distances = capital_distances.at[check_y, check_x].set(
                            jnp.minimum(capital_distances[check_y, check_x], dist)
                        )
    
    # Exclure les cases à moins de 2 cases d'une capitale
    valid_mask = valid_mask & (capital_distances >= 2)
    
    # Placer les villages de manière itérative
    # On utilise une approche simple avec boucles Python (acceptable pour la génération initiale)
    village_positions = []
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
        
        # Essayer de placer les villages en respectant la distance minimale
        for idx in shuffle_indices:
            if len(village_positions) >= num_villages_target:
                break
            
            x, y = valid_positions[int(idx)]
            
            # Vérifier distance aux villages déjà placés
            too_close = False
            for vx, vy in village_positions:
                dist = abs(x - vx) + abs(y - vy)
                if dist <= 2:
                    too_close = True
                    break
            
            if not too_close:
                village_positions.append((x, y))
    
    # Placer les villages dans le state
    for x, y in village_positions:
        city_owner = city_owner.at[y, x].set(NO_OWNER)
        city_level = city_level.at[y, x].set(1)
        city_population = city_population.at[y, x].set(0)  # Villages commencent à 0 pop
        # S'assurer que c'est un terrain terrestre (convertir en plaine si nécessaire)
        if terrain[y, x] == TerrainType.WATER_SHALLOW or terrain[y, x] == TerrainType.WATER_DEEP:
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
    
    Les starfish sont placés : 1 pour 25 cases d'eau (shallow + deep).
    """
    resource_type = state.resource_type.copy()
    resource_available = state.resource_available.copy()
    
    height, width = state.height, state.width
    
    # Compter les cases d'eau
    is_water = (state.terrain == TerrainType.WATER_SHALLOW) | (state.terrain == TerrainType.WATER_DEEP)
    water_count = jnp.sum(is_water)
    
    # Calculer le nombre de starfish (1 pour 25 cases d'eau)
    num_starfish = int(water_count) // 25
    if num_starfish == 0 and water_count > 0:
        num_starfish = 1  # Au moins 1 si il y a de l'eau
    
    # Générer liste de toutes les positions d'eau
    water_positions = []
    for y in range(height):
        for x in range(width):
            if bool(is_water[y, x]):
                water_positions.append((x, y))
    
    # Mélanger et sélectionner les positions pour les starfish
    if len(water_positions) > 0 and num_starfish > 0:
        key, subkey = jax.random.split(key)
        num_water = len(water_positions)
        shuffle_indices = jax.random.permutation(subkey, jnp.arange(num_water))
        
        # Placer les starfish (on utilise ResourceType pour stocker temporairement,
        # mais en réalité les starfish ne sont pas des ressources récoltables dans Polytopia)
        # Pour l'instant, on ne les implémente pas comme ressource, juste comme décoratif
        # Si besoin futur, on pourrait ajouter un champ séparé pour les starfish
        for i in range(min(num_starfish, len(shuffle_indices))):
            x, y = water_positions[int(shuffle_indices[i])]
            # Les starfish sont purement décoratifs dans Polytopia, donc on ne fait rien pour l'instant
            # Si on veut les implémenter, il faudrait ajouter un champ dans GameState
    
    return state.replace(
        resource_type=resource_type,
        resource_available=resource_available,
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
