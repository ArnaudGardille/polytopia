"""Logique de jeu - règles et transitions d'état.

Phase 0 (MVP) : périmètre fonctionnel
-------------------------------------
- Seules les unités ``WARRIOR`` existent, sans promotion ni compétences
  spéciales. Toutes les actions de combat sont donc résolues comme du corps à
  corps.
- Les villes sont capturées par simple entrée sur la case.
- La fin de partie repose uniquement sur l'élimination : dès qu'un joueur ne
  possède plus de capitale (``city_level > 0``), il est retiré et la partie se
  termine lorsqu'il ne reste qu'un propriétaire vivant.

Phase 1 : boucle économique minimale
------------------------------------
- Ajout d'une ressource `player_stars` et de la population par ville afin de
  limiter l'entraînement/les constructions.
- Les villes génèrent des étoiles en fin de tour selon leur niveau.
- Les bâtiments basiques (ferme, mine, hutte) augmentent la population d'une
  ville et peuvent faire monter son niveau.
"""

from enum import IntEnum
import jax
import jax.numpy as jnp
from .state import GameState, TerrainType, UnitType, NO_OWNER, GameMode, TechType, ResourceType
from .actions import ActionType, Direction, decode_action, get_action_direction_delta
from .score import update_scores


# Statistiques des unités (pour MVP, seulement guerrier)
# Converties en arrays JAX pour compatibilité avec le traçage
# Format: arrays indexés par UnitType
MAX_UNIT_TYPES = 16  # Taille maximale pour les types d'unités

def _pad_unit_array(values):
    return jnp.array(values + [0] * (MAX_UNIT_TYPES - len(values)), dtype=jnp.int32)

UNIT_HP_MAX = _pad_unit_array([0, 10, 15, 8, 10, 8, 15, 15, 8, 40])  # NONE, WARRIOR, DEFENDER, ARCHER, RIDER, RAFT, KNIGHT, SWORDSMAN, CATAPULT, GIANT
UNIT_ATTACK = _pad_unit_array([0, 2, 1, 2, 3, 2, 4, 3, 4, 5])
UNIT_DEFENSE = _pad_unit_array([0, 2, 3, 1, 1, 1, 1, 2, 1, 3])
UNIT_MOVEMENT = _pad_unit_array([0, 1, 1, 1, 2, 2, 3, 1, 1, 1])
UNIT_COST = _pad_unit_array([0, 2, 3, 3, 4, 0, 6, 5, 6, 20])  # Coûts approximatifs selon Polytopia
UNIT_ATTACK_RANGE = _pad_unit_array([0, 1, 1, 2, 1, 1, 1, 1, 3, 1])  # Catapult portée 3
UNIT_REQUIRED_TECH = _pad_unit_array([
    int(TechType.NONE),  # NONE
    int(TechType.NONE),  # WARRIOR (pas de tech requise)
    int(TechType.STRATEGY),  # DEFENDER (nécessite Strategy)
    int(TechType.ARCHERY),  # ARCHER
    int(TechType.RIDING),  # RIDER
    int(TechType.SAILING),  # RAFT
    int(TechType.CHIVALRY),  # KNIGHT (nécessite Chivalry)
    int(TechType.SMITHERY),  # SWORDSMAN (nécessite Smithery)
    int(TechType.MATHEMATICS),  # CATAPULT (nécessite Mathematics)
    int(TechType.NONE),  # GIANT (obtenu via amélioration ville niveau 5+, pas encore implémenté)
])
UNIT_IS_NAVAL = _pad_unit_array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).astype(jnp.bool_)
UNIT_CAN_ENTER_SHALLOW = _pad_unit_array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).astype(jnp.bool_)
UNIT_CAN_ENTER_DEEP = _pad_unit_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(jnp.bool_)
# Unités qui ne peuvent pas être promues (navales, super units)
UNIT_CAN_PROMOTE = _pad_unit_array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0]).astype(jnp.bool_)  # RAFT et GIANT ne peuvent pas être promues


class BuildingType(IntEnum):
    """Types de bâtiments basiques."""
    NONE = 0
    FARM = 1
    MINE = 2
    HUT = 3
    PORT = 4
    WINDMILL = 5
    FORGE = 6
    SAWMILL = 7
    MARKET = 8
    TEMPLE = 9
    MONUMENT = 10
    CITY_WALL = 11
    PARK = 12
    ROAD = 13
    BRIDGE = 14
    NUM_TYPES = 15


MAX_BUILDING_TYPES = 16  # Augmenté pour accommoder tous les nouveaux bâtiments
BUILDING_COST = jnp.array(
    [0, 3, 4, 2, 5, 6, 7, 5, 8, 10, 20, 5, 15, 3, 5] + [0] * (MAX_BUILDING_TYPES - 15),
    dtype=jnp.int32,
)
# POP_GAIN pour windmill/forge/sawmill sera calculé dynamiquement selon adjacents
# Pour les autres, 0 car ils donnent d'autres bonus
BUILDING_POP_GAIN = jnp.array(
    [0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * (MAX_BUILDING_TYPES - 15),
    dtype=jnp.int32,
)
BUILDING_REQUIRED_TECH = jnp.array(
    [
        int(TechType.NONE),  # NONE
        int(TechType.NONE),  # FARM
        int(TechType.MINING),  # MINE
        int(TechType.NONE),  # HUT
        int(TechType.SAILING),  # PORT
        int(TechType.NONE),  # WINDMILL (tier 2, nécessiterait Construction mais pas encore implémenté)
        int(TechType.MINING),  # FORGE (nécessite Mining)
        int(TechType.NONE),  # SAWMILL (tier 2, nécessiterait Forestry mais pas encore implémenté)
        int(TechType.NONE),  # MARKET (tier 2, nécessiterait Trade mais pas encore implémenté)
        int(TechType.NONE),  # TEMPLE (tier 3, nécessiterait Spiritualism mais pas encore implémenté)
        int(TechType.NONE),  # MONUMENT (tier 3, nécessiterait Mathematics mais pas encore implémenté)
        int(TechType.NONE),  # CITY_WALL (amélioration ville niveau 3)
        int(TechType.NONE),  # PARK (amélioration ville niveau 5)
        int(TechType.ROADS),  # ROAD (nécessite Roads)
        int(TechType.ROADS),  # BRIDGE (nécessite Roads)
    ] + [0] * (MAX_BUILDING_TYPES - 15),
    dtype=jnp.int32,
)

MAX_RESOURCE_TYPES = int(ResourceType.NUM_TYPES)
RESOURCE_COST = jnp.array(
    [0, 2, 2, 4] + [0] * (MAX_RESOURCE_TYPES - 4),
    dtype=jnp.int32,
)
RESOURCE_POP_GAIN = jnp.array(
    [0, 1, 1, 2] + [0] * (MAX_RESOURCE_TYPES - 4),
    dtype=jnp.int32,
)
RESOURCE_REQUIRED_TECH = jnp.array(
    [int(TechType.NONE), int(TechType.NONE), int(TechType.SAILING), int(TechType.MINING)] + [0] * (MAX_RESOURCE_TYPES - 4),
    dtype=jnp.int32,
)
RESOURCE_BASE_TERRAIN = jnp.array(
    [
        TerrainType.PLAIN,  # NONE (non utilisé)
        TerrainType.PLAIN,  # Fruit sur plaine
        TerrainType.WATER_SHALLOW,  # Poisson en eau peu profonde
        TerrainType.MOUNTAIN,  # Minerai sur montagne
    ] + [TerrainType.PLAIN] * (MAX_RESOURCE_TYPES - 4),
    dtype=jnp.int32,
)


NUM_TECHS = int(TechType.NUM_TECHS)

# Coûts de base selon tier (sans villes ni Philosophy)
# Formule: base_cost = tier * num_cities + 4
# Pour 0 villes: T1=4, T2=4, T3=4
# Pour 1 ville: T1=5, T2=6, T3=7
# On stocke le tier pour calculer dynamiquement
TECH_TIER = jnp.array([
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
], dtype=jnp.int32)

# Coûts de base pour compatibilité (coût pour 0 villes, base = 4)
# Pour calcul dynamique avec villes, utiliser TECH_TIER
# Coûts approximatifs selon tiers : T1=3-5, T2=4-6, T3=6-8
TECH_COST = jnp.array([
    0,  # NONE
    3,  # CLIMBING (T1)
    3,  # FISHING (T1)
    3,  # HUNTING (T1)
    3,  # ORGANIZATION (T1)
    3,  # RIDING (T1)
    5,  # ARCHERY (T2)
    4,  # RAMMING (T2)
    4,  # FARMING (T2)
    4,  # FORESTRY (T2)
    4,  # FREE_SPIRIT (T2)
    4,  # MEDITATION (T2)
    3,  # MINING (T2)
    4,  # ROADS (T2)
    4,  # SAILING (T2)
    4,  # STRATEGY (T2)
    6,  # AQUATISM (T3)
    6,  # PHILOSOPHY (T3)
    5,  # SMITHERY (T3)
    6,  # CHIVALRY (T3)
    7,  # MATHEMATICS (T3)
], dtype=jnp.int32)

# Construire les dépendances selon l'arbre de Polytopia
tech_deps_rows = []
for i in range(NUM_TECHS):
    deps = [False] * NUM_TECHS
    tech = TechType(i)
    
    if tech == TechType.SAILING:
        deps[TechType.CLIMBING] = True
    elif tech == TechType.ARCHERY:
        deps[TechType.HUNTING] = True
    elif tech == TechType.RAMMING:
        deps[TechType.FISHING] = True
    elif tech == TechType.FARMING:
        deps[TechType.ORGANIZATION] = True
    elif tech == TechType.FORESTRY:
        deps[TechType.HUNTING] = True
    elif tech == TechType.FREE_SPIRIT:
        deps[TechType.RIDING] = True
    elif tech == TechType.MEDITATION:
        deps[TechType.CLIMBING] = True
    elif tech == TechType.ROADS:
        deps[TechType.RIDING] = True
    elif tech == TechType.STRATEGY:
        deps[TechType.ORGANIZATION] = True
    elif tech == TechType.AQUATISM:
        deps[TechType.RAMMING] = True
    elif tech == TechType.PHILOSOPHY:
        deps[TechType.MEDITATION] = True
    
    tech_deps_rows.append(deps)

TECH_DEPENDENCIES = jnp.array(tech_deps_rows, dtype=jnp.bool_)

# Niveaux de ville : 1, 2, 3, 4, 5+ avec seuils croissants
# Niveau 4 : 7 pop, Niveau 5 : 9 pop, puis +2 par niveau supplémentaire
CITY_LEVEL_POP_THRESHOLDS = jnp.array([0, 1, 3, 5, 7, 9], dtype=jnp.int32)
CITY_STAR_INCOME_PER_LEVEL = jnp.array([0, 2, 4, 6, 6, 6], dtype=jnp.int32)  # Niveau 3+ = 6★
CITY_CAPTURE_POPULATION = CITY_LEVEL_POP_THRESHOLDS[1]
HARVEST_NEIGHBOR_DELTAS = jnp.array(
    [
        [-1, -1], [0, -1], [1, -1],
        [-1, 0],           [1, 0],
        [-1, 1],  [0, 1],  [1, 1],
    ],
    dtype=jnp.int32,
)


def step(state: GameState, action: int) -> GameState:
    """Applique une action et retourne le nouvel état.
    
    Args:
        state: État actuel du jeu
        action: Action encodée
    
    Returns:
        Nouvel état après l'action
    """
    decoded = decode_action(action)
    action_type = decoded["action_type"]
    
    # Vérifier que la partie n'est pas terminée
    state = jax.lax.cond(
        state.done,
        lambda s: s,  # Si terminé, retourner l'état inchangé
        lambda s: _apply_action(s, decoded),
        state
    )

    state = update_scores(state)
    
    return state


def _apply_action(state: GameState, decoded: dict) -> GameState:
    """Applique une action décodée."""
    action_type = decoded["action_type"]
    
    # Router vers la bonne fonction selon le type d'action
    state = jax.lax.switch(
        action_type,
        [
            lambda s: s,  # NO_OP
            lambda s: _apply_move(s, decoded),  # MOVE
            lambda s: _apply_attack(s, decoded),  # ATTACK
            lambda s: _apply_train_unit(s, decoded),  # TRAIN_UNIT
            lambda s: _apply_build(s, decoded),  # BUILD
            lambda s: _apply_research_tech(s, decoded),  # RESEARCH_TECH
            lambda s: _apply_end_turn(s),  # END_TURN
            lambda s: _apply_harvest_resource(s, decoded),  # HARVEST_RESOURCE
            lambda s: _apply_recover(s, decoded),  # RECOVER
        ],
        state
    )
    
    return state


def _apply_move(state: GameState, decoded: dict) -> GameState:
    """Applique un mouvement d'unité."""
    # Récupérer les valeurs (peuvent être des arrays JAX ou des int Python)
    unit_id_val = decoded.get("unit_id", -1)
    direction_val = decoded.get("direction", -1)
    
    # Convertir en arrays JAX si nécessaire et gérer les valeurs sentinelles
    unit_id = jnp.asarray(unit_id_val, dtype=jnp.int32) if not isinstance(unit_id_val, (int, type(None))) else (jnp.array(unit_id_val, dtype=jnp.int32) if unit_id_val is not None and unit_id_val >= 0 else jnp.array(-1, dtype=jnp.int32))
    direction = jnp.asarray(direction_val, dtype=jnp.int32) if not isinstance(direction_val, (int, type(None))) else (jnp.array(direction_val, dtype=jnp.int32) if direction_val is not None and direction_val >= 0 else jnp.array(-1, dtype=jnp.int32))
    
    # Vérifier que unit_id et direction sont valides
    has_unit_id = unit_id >= 0
    has_direction = direction >= 0
    both_valid = has_unit_id & has_direction

    def do_move(state):
        # Vérifier que l'unité existe et appartient au joueur actif
        unit_id_int = jnp.asarray(unit_id, dtype=jnp.int32)
        is_valid = (
            (unit_id_int < state.max_units)
            & state.units_active[unit_id_int]
            & (state.units_owner[unit_id_int] == state.current_player)
            & (~state.units_has_acted[unit_id_int])
        )

        return jax.lax.cond(
            is_valid,
            lambda s: _perform_move(s, unit_id_int, direction),
            lambda s: s,
            state
        )

    return jax.lax.cond(
        both_valid,
        do_move,
        lambda s: s,
        state
    )


def _perform_move(state: GameState, unit_id: jnp.ndarray, direction: jnp.ndarray) -> GameState:
    """Effectue le mouvement d'une unité (appelé après validation).
    
    Args:
        state: État du jeu
        unit_id: ID de l'unité (peut être int ou array JAX)
        direction: Direction (peut être int ou array JAX)
    """
    # Convertir en arrays JAX si nécessaire
    unit_id = jnp.asarray(unit_id, dtype=jnp.int32)
    direction = jnp.asarray(direction, dtype=jnp.int32)
    
    # Calculer la nouvelle position
    current_pos = state.units_pos[unit_id]
    from .actions import DIRECTION_DELTA
    delta = DIRECTION_DELTA[direction]
    new_pos = current_pos + delta
    new_x, new_y = new_pos[0], new_pos[1]
    
    is_in_bounds = (
        (new_x >= 0) & (new_x < state.width) &
        (new_y >= 0) & (new_y < state.height)
    )
    pos_occupied = _is_position_occupied(state, new_x, new_y, unit_id)
    terrain = state.terrain[new_y, new_x]
    unit_type = state.units_type[unit_id]
    has_climbing = _player_has_tech(state, TechType.CLIMBING)
    has_sailing = _player_has_tech(state, TechType.SAILING)
    origin_port = _is_friendly_port(state, current_pos[0], current_pos[1])
    target_port = jax.lax.cond(
        is_in_bounds,
        lambda _: _is_friendly_port(state, new_x, new_y),
        lambda _: jnp.array(False),
        operand=None,
    )
    is_land_tile = _is_land_terrain(terrain)
    is_mountain = _is_mountain_terrain(terrain)
    is_shallow_water = _is_shallow_water_terrain(terrain)
    is_deep_water = terrain == TerrainType.WATER_DEEP
    is_naval = UNIT_IS_NAVAL[unit_type]
    
    land_allowed = is_land_tile & jnp.where(is_mountain, has_climbing, True)
    
    naval_allowed = jnp.zeros_like(land_allowed)
    naval_allowed = jnp.where(
        is_shallow_water,
        jnp.where(is_naval, UNIT_CAN_ENTER_SHALLOW[unit_type], origin_port & has_sailing),
        naval_allowed,
    )
    naval_allowed = jnp.where(
        is_deep_water,
        jnp.where(is_naval, UNIT_CAN_ENTER_DEEP[unit_type], False),
        naval_allowed,
    )
    
    can_enter = jnp.where(is_land_tile, land_allowed, naval_allowed)
    
    # Vérifier si la case destination est explorée ou adjacente à une case explorée
    unit_owner = state.units_owner[unit_id]
    is_explored = jnp.where(
        is_in_bounds,
        state.tiles_explored[unit_owner, new_y, new_x],
        False
    )
    
    # Vérifier si adjacente à une case explorée (8 directions)
    def check_adjacent_explored():
        # Vérifier les 8 cases adjacentes
        adjacent_deltas = jnp.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ], dtype=jnp.int32)
        
        def check_delta(delta_idx, has_adjacent):
            dy, dx = adjacent_deltas[delta_idx]
            adj_y = new_y + dy
            adj_x = new_x + dx
            
            in_bounds_adj = (adj_y >= 0) & (adj_y < state.height) & (adj_x >= 0) & (adj_x < state.width)
            is_explored_adj = jnp.where(
                in_bounds_adj,
                state.tiles_explored[unit_owner, adj_y, adj_x],
                False
            )
            return has_adjacent | is_explored_adj
        
        has_adjacent = jax.lax.fori_loop(0, 8, check_delta, False)
        return has_adjacent
    
    is_adjacent_to_explored = jnp.where(is_in_bounds, check_adjacent_explored(), False)
    can_see_destination = is_explored | is_adjacent_to_explored
    
    can_move = is_in_bounds & ~pos_occupied & can_enter & can_see_destination
    
    convert_to_raft = (~is_naval) & is_shallow_water & origin_port & has_sailing
    disembark_to_land = is_naval & is_land_tile
    
    def move_unit(state):
        new_units_pos = state.units_pos.at[unit_id, 0].set(new_x)
        new_units_pos = new_units_pos.at[unit_id, 1].set(new_y)
        new_units_has_acted = state.units_has_acted.at[unit_id].set(True)
        state = state.replace(
            units_pos=new_units_pos,
            units_has_acted=new_units_has_acted,
        )
        state = jax.lax.cond(
            convert_to_raft,
            lambda s: _embark_unit(s, unit_id),
            lambda s: s,
            state
        )
        state = jax.lax.cond(
            disembark_to_land,
            lambda s: _disembark_unit(s, unit_id),
            lambda s: s,
            state
        )
        state = _check_city_capture(state, unit_id, new_x, new_y)
        
        # Mettre à jour l'exploration : marquer les cases vues comme explorées
        unit_owner = state.units_owner[unit_id]
        unit_vision = _compute_unit_vision(state, unit_id, new_pos)
        new_tiles_explored = state.tiles_explored.at[unit_owner].set(
            state.tiles_explored[unit_owner] | unit_vision
        )
        state = state.replace(tiles_explored=new_tiles_explored)
        
        return state
    
    return jax.lax.cond(
        can_move,
        move_unit,
        lambda s: s,
        state
    )


def _apply_build(state: GameState, decoded: dict) -> GameState:
    """Applique la construction d'un bâtiment basique."""
    building_type_val = decoded.get("unit_type", -1)
    target_pos_val = decoded.get("target_pos")
    
    if target_pos_val is None:
        return state
    
    if isinstance(target_pos_val, tuple):
        target_x = jnp.array(target_pos_val[0], dtype=jnp.int32)
        target_y = jnp.array(target_pos_val[1], dtype=jnp.int32)
    else:
        target_x = target_pos_val[0] if hasattr(target_pos_val, '__getitem__') else jnp.array(-1, dtype=jnp.int32)
        target_y = target_pos_val[1] if hasattr(target_pos_val, '__getitem__') and len(target_pos_val) > 1 else jnp.array(-1, dtype=jnp.int32)
    
    building_type = jnp.asarray(building_type_val, dtype=jnp.int32) if not isinstance(building_type_val, (int, type(None))) else (jnp.array(building_type_val, dtype=jnp.int32) if building_type_val is not None and building_type_val >= 0 else jnp.array(-1, dtype=jnp.int32))
    
    has_building = building_type > 0
    has_target = (target_x >= 0) & (target_y >= 0)
    
    def do_build(state):
        is_city = state.city_level[target_y, target_x] > 0
        is_owner = state.city_owner[target_y, target_x] == state.current_player
        cost = BUILDING_COST[building_type]
        has_stars = state.player_stars[state.current_player] >= cost
        required_tech = BUILDING_REQUIRED_TECH[building_type]
        has_required = jnp.where(
            required_tech == int(TechType.NONE),
            True,
            state.player_techs[state.current_player, required_tech],
        )
        # Vérifier que le bâtiment n'existe pas déjà (pour les bâtiments uniques)
        is_windmill = building_type == BuildingType.WINDMILL
        is_forge = building_type == BuildingType.FORGE
        is_sawmill = building_type == BuildingType.SAWMILL
        is_market = building_type == BuildingType.MARKET
        is_temple = building_type == BuildingType.TEMPLE
        is_monument = building_type == BuildingType.MONUMENT
        is_wall = building_type == BuildingType.CITY_WALL
        is_park = building_type == BuildingType.PARK
        is_road = building_type == BuildingType.ROAD
        is_bridge = building_type == BuildingType.BRIDGE
        
        already_has_windmill = jnp.where(is_windmill, state.city_has_windmill[target_y, target_x], False)
        already_has_forge = jnp.where(is_forge, state.city_has_forge[target_y, target_x], False)
        already_has_sawmill = jnp.where(is_sawmill, state.city_has_sawmill[target_y, target_x], False)
        already_has_market = jnp.where(is_market, state.city_has_market[target_y, target_x], False)
        already_has_temple = jnp.where(is_temple, state.city_has_temple[target_y, target_x], False)
        already_has_monument = jnp.where(is_monument, state.city_has_monument[target_y, target_x], False)
        already_has_wall = jnp.where(is_wall, state.city_has_wall[target_y, target_x], False)
        already_has_park = jnp.where(is_park, state.city_has_park[target_y, target_x], False)
        already_has_road = jnp.where(is_road, state.has_road[target_y, target_x], False)
        already_has_bridge = jnp.where(is_bridge, state.has_bridge[target_y, target_x], False)
        
        already_built = (already_has_windmill | already_has_forge | already_has_sawmill | 
                         already_has_market | already_has_temple | already_has_monument | 
                         already_has_wall | already_has_park | already_has_road | already_has_bridge)
        
        # Routes : doivent être sur plaine ou forêt, pas besoin d'être dans une ville
        # Ponts : doivent être sur eau peu profonde
        terrain = state.terrain[target_y, target_x]
        is_plain_or_forest = _is_plain_terrain(terrain) | _is_forest_terrain(terrain)
        is_shallow_water = _is_shallow_water_terrain(terrain)
        
        # Pour routes/ponts, on peut construire en dehors des villes (mais doit être territoire ami)
        # Vérifier si territoire ami (en ville amie ou adjacente à ville amie)
        is_in_city = is_city & is_owner
        has_adjacent_city, _, _ = _find_adjacent_friendly_city(
            state, target_x, target_y, state.current_player
        )
        is_friendly_territory = is_in_city | has_adjacent_city
        
        can_build_road = is_road & is_plain_or_forest & ~already_has_road & is_friendly_territory
        can_build_bridge = is_bridge & is_shallow_water & ~already_has_bridge & is_friendly_territory
        
        # Pour les autres bâtiments, nécessite une ville du joueur
        can_build_other = is_city & is_owner & ~already_built & has_stars & has_required
        can_build_road_or_bridge = (can_build_road | can_build_bridge) & has_stars & has_required
        
        can_build = can_build_other | can_build_road_or_bridge
        
        return jax.lax.cond(
            can_build,
            lambda s: _perform_build(s, building_type, target_x, target_y),
            lambda s: s,
            state
        )
    
    return jax.lax.cond(
        has_building & has_target,
        do_build,
        lambda s: s,
        state
    )


def _count_adjacent_farms(state: GameState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compte les fermes adjacentes (villes avec FARM construite)."""
    # Pour simplifier, on compte les villes adjacentes (chaque ville a au moins une ferme potentielle)
    count = jnp.array(0, dtype=jnp.int32)
    
    def check_delta(i, acc):
        dx = HARVEST_NEIGHBOR_DELTAS[i, 0]
        dy = HARVEST_NEIGHBOR_DELTAS[i, 1]
        nx = x + dx
        ny = y + dy
        
        # Utiliser la forme du terrain pour vérifier les limites
        h, w = state.terrain.shape[0], state.terrain.shape[1]
        in_bounds = (
            (nx >= 0) & (nx < w) &
            (ny >= 0) & (ny < h)
        )
        
        # Une ville adjacente compte comme ayant une ferme (simplification MVP)
        has_city = jnp.where(in_bounds, state.city_level[ny, nx] > 0, False)
        
        return jnp.where(has_city, acc + 1, acc)
    
    return jax.lax.fori_loop(0, HARVEST_NEIGHBOR_DELTAS.shape[0], check_delta, count)


def _count_adjacent_mines(state: GameState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compte les mines adjacentes."""
    count = jnp.array(0, dtype=jnp.int32)
    
    def check_delta(i, acc):
        dx = HARVEST_NEIGHBOR_DELTAS[i, 0]
        dy = HARVEST_NEIGHBOR_DELTAS[i, 1]
        nx = x + dx
        ny = y + dy
        
        # Utiliser la forme du terrain pour vérifier les limites
        h, w = state.terrain.shape[0], state.terrain.shape[1]
        in_bounds = (
            (nx >= 0) & (nx < w) &
            (ny >= 0) & (ny < h)
        )
        
        # Vérifier si terrain montagne avec mine OU ville avec mine construite
        is_mountain_mine = jnp.where(
            in_bounds,
            state.terrain[ny, nx] == TerrainType.MOUNTAIN_WITH_MINE,
            False
        )
        # Pour simplifier MVP, on compte aussi les villes adjacentes comme ayant potentiellement une mine
        # (car les mines sont construites dans les villes)
        has_city = jnp.where(in_bounds, state.city_level[ny, nx] > 0, False)
        
        has_mine = is_mountain_mine | has_city
        
        return jnp.where(has_mine, acc + 1, acc)
    
    return jax.lax.fori_loop(0, HARVEST_NEIGHBOR_DELTAS.shape[0], check_delta, count)


def _count_adjacent_huts(state: GameState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compte les huttes adjacentes."""
    count = jnp.array(0, dtype=jnp.int32)
    
    def check_delta(i, acc):
        dx = HARVEST_NEIGHBOR_DELTAS[i, 0]
        dy = HARVEST_NEIGHBOR_DELTAS[i, 1]
        nx = x + dx
        ny = y + dy
        
        # Utiliser la forme du terrain pour vérifier les limites
        h, w = state.terrain.shape[0], state.terrain.shape[1]
        in_bounds = (
            (nx >= 0) & (nx < w) &
            (ny >= 0) & (ny < h)
        )
        
        # Pour simplifier MVP, on compte les villes adjacentes (chaque ville peut avoir une hutte)
        has_city = jnp.where(in_bounds, state.city_level[ny, nx] > 0, False)
        
        return jnp.where(has_city, acc + 1, acc)
    
    return jax.lax.fori_loop(0, HARVEST_NEIGHBOR_DELTAS.shape[0], check_delta, count)


def _perform_build(state: GameState, building_type: jnp.ndarray, target_x: jnp.ndarray, target_y: jnp.ndarray) -> GameState:
    """Applique les effets d'un bâtiment validé."""
    cost = BUILDING_COST[building_type]
    new_player_stars = state.player_stars.at[state.current_player].add(-cost)
    
    # Identifier le type de bâtiment
    is_port = building_type == BuildingType.PORT
    is_windmill = building_type == BuildingType.WINDMILL
    is_forge = building_type == BuildingType.FORGE
    is_sawmill = building_type == BuildingType.SAWMILL
    is_market = building_type == BuildingType.MARKET
    is_temple = building_type == BuildingType.TEMPLE
    is_monument = building_type == BuildingType.MONUMENT
    is_wall = building_type == BuildingType.CITY_WALL
    is_park = building_type == BuildingType.PARK
    
    # Appliquer les effets selon le type
    state = state.replace(player_stars=new_player_stars)
    
    # Port
    state = jax.lax.cond(
        is_port,
        lambda s: _update_city_connections(s.replace(city_has_port=s.city_has_port.at[target_y, target_x].set(True))),
        lambda s: s,
        state
    )
    
    # Windmill : +1 pop par ferme adjacente
    def apply_windmill(s):
        farm_count = _count_adjacent_farms(s, target_x, target_y)
        pop_gain = farm_count  # +1 pop par ferme adjacente
        new_pop = s.city_population[target_y, target_x] + pop_gain
        updated = _set_city_population(s, target_x, target_y, new_pop)
        return updated.replace(city_has_windmill=updated.city_has_windmill.at[target_y, target_x].set(True))
    
    state = jax.lax.cond(is_windmill, apply_windmill, lambda s: s, state)
    
    # Forge : +2 pop par mine adjacente
    def apply_forge(s):
        mine_count = _count_adjacent_mines(s, target_x, target_y)
        pop_gain = mine_count * 2  # +2 pop par mine adjacente
        new_pop = s.city_population[target_y, target_x] + pop_gain
        updated = _set_city_population(s, target_x, target_y, new_pop)
        return updated.replace(city_has_forge=updated.city_has_forge.at[target_y, target_x].set(True))
    
    state = jax.lax.cond(is_forge, apply_forge, lambda s: s, state)
    
    # Sawmill : +1 pop par hutte adjacente
    def apply_sawmill(s):
        hut_count = _count_adjacent_huts(s, target_x, target_y)
        pop_gain = hut_count  # +1 pop par hutte adjacente
        new_pop = s.city_population[target_y, target_x] + pop_gain
        updated = _set_city_population(s, target_x, target_y, new_pop)
        return updated.replace(city_has_sawmill=updated.city_has_sawmill.at[target_y, target_x].set(True))
    
    state = jax.lax.cond(is_sawmill, apply_sawmill, lambda s: s, state)
    
    # Market : génère des étoiles chaque tour (sera géré dans _compute_income_for_player)
    state = jax.lax.cond(
        is_market,
        lambda s: s.replace(city_has_market=s.city_has_market.at[target_y, target_x].set(True)),
        lambda s: s,
        state
    )
    
    # Temple : niveau 1 initial
    state = jax.lax.cond(
        is_temple,
        lambda s: s.replace(
            city_has_temple=s.city_has_temple.at[target_y, target_x].set(True),
            city_temple_level=s.city_temple_level.at[target_y, target_x].set(1)
        ),
        lambda s: s,
        state
    )
    
    # Monument
    state = jax.lax.cond(
        is_monument,
        lambda s: s.replace(city_has_monument=s.city_has_monument.at[target_y, target_x].set(True)),
        lambda s: s,
        state
    )
    
    # City Wall
    state = jax.lax.cond(
        is_wall,
        lambda s: s.replace(city_has_wall=s.city_has_wall.at[target_y, target_x].set(True)),
        lambda s: s,
        state
    )
    
    # Park
    state = jax.lax.cond(
        is_park,
        lambda s: s.replace(city_has_park=s.city_has_park.at[target_y, target_x].set(True)),
        lambda s: s,
        state
    )
    
    # Bâtiments avec pop_gain direct (FARM, MINE, HUT)
    pop_gain = BUILDING_POP_GAIN[building_type]
    has_direct_pop = pop_gain > 0
    
    def apply_population_build(s):
        new_population_value = s.city_population[target_y, target_x] + pop_gain
        return _set_city_population(s, target_x, target_y, new_population_value)
    
    state = jax.lax.cond(
        has_direct_pop & ~is_port & ~is_windmill & ~is_forge & ~is_sawmill,
        apply_population_build,
        lambda s: s,
        state
    )
    
    return state


def _apply_harvest_resource(state: GameState, decoded: dict) -> GameState:
    """Récolte une ressource adjacente à une ville contrôlée."""
    target_pos_val = decoded.get("target_pos")
    if target_pos_val is None:
        return state
    
    if isinstance(target_pos_val, tuple):
        target_x = jnp.array(target_pos_val[0], dtype=jnp.int32)
        target_y = jnp.array(target_pos_val[1], dtype=jnp.int32)
    else:
        target_x = target_pos_val[0] if hasattr(target_pos_val, '__getitem__') else jnp.array(-1, dtype=jnp.int32)
        target_y = target_pos_val[1] if hasattr(target_pos_val, '__getitem__') and len(target_pos_val) > 1 else jnp.array(-1, dtype=jnp.int32)
    
    in_bounds = (
        (target_x >= 0) & (target_x < state.width) &
        (target_y >= 0) & (target_y < state.height)
    )
    
    def do_harvest(state):
        resource_type = state.resource_type[target_y, target_x]
        available = state.resource_available[target_y, target_x]
        has_resource = available & (resource_type > int(ResourceType.NONE))
        cost = RESOURCE_COST[resource_type]
        player_stars = state.player_stars[state.current_player]
        has_stars = player_stars >= cost
        required_tech = RESOURCE_REQUIRED_TECH[resource_type]
        has_required = jnp.where(
            required_tech == int(TechType.NONE),
            True,
            state.player_techs[state.current_player, required_tech],
        )
        has_city, city_x, city_y = _find_adjacent_friendly_city(
            state, target_x, target_y, state.current_player
        )
        can_harvest = has_resource & has_stars & has_required & has_city
        
        def perform(state):
            pop_gain = RESOURCE_POP_GAIN[resource_type]
            new_available = state.resource_available.at[target_y, target_x].set(False)
            new_resource_type = state.resource_type.at[target_y, target_x].set(int(ResourceType.NONE))
            new_player_stars = state.player_stars.at[state.current_player].add(-cost)
            should_update_terrain = resource_type > int(ResourceType.NONE)
            base_terrain = RESOURCE_BASE_TERRAIN[resource_type]

            def update_terrain(_: None) -> jnp.ndarray:
                return state.terrain.at[target_y, target_x].set(base_terrain)

            new_terrain = jax.lax.cond(
                should_update_terrain,
                update_terrain,
                lambda _: state.terrain,
                operand=None,
            )

            updated_state = state.replace(
                resource_available=new_available,
                resource_type=new_resource_type,
                player_stars=new_player_stars,
                terrain=new_terrain,
            )
            return _add_population_to_city(updated_state, city_x, city_y, pop_gain)
        
        return jax.lax.cond(
            can_harvest,
            perform,
            lambda s: s,
            state
        )
    
    return jax.lax.cond(
        in_bounds,
        do_harvest,
        lambda s: s,
        state
    )


def _apply_recover(state: GameState, decoded: dict) -> GameState:
    """Applique l'action de guérison (Recover) d'une unité."""
    unit_id_val = decoded.get("unit_id", -1)
    
    # Convertir unit_id
    unit_id = jnp.asarray(unit_id_val, dtype=jnp.int32) if not isinstance(unit_id_val, (int, type(None))) else (jnp.array(unit_id_val, dtype=jnp.int32) if unit_id_val is not None and unit_id_val >= 0 else jnp.array(-1, dtype=jnp.int32))
    
    # Vérifier que l'unité est valide
    is_unit_valid = (
        (unit_id >= 0)
        & (unit_id < state.max_units)
        & state.units_active[unit_id]
        & (state.units_owner[unit_id] == state.current_player)
        & (~state.units_has_acted[unit_id])
    )
    
    def do_recover(state):
        unit_hp = state.units_hp[unit_id]
        unit_type = state.units_type[unit_id]
        unit_max_hp = UNIT_HP_MAX[unit_type]
        unit_pos = state.units_pos[unit_id]
        x, y = unit_pos[0], unit_pos[1]
        
        # Vérifier que l'unité n'est pas à HP max
        needs_healing = unit_hp < unit_max_hp
        
        # Déterminer si territoire ami (en ville amie ou adjacente à ville amie)
        in_bounds = (x >= 0) & (x < state.width) & (y >= 0) & (y < state.height)
        
        # Vérifier si unité est en ville amie
        x_safe = jnp.clip(x, 0, state.width - 1)
        y_safe = jnp.clip(y, 0, state.height - 1)
        is_in_city = (state.city_level[y_safe, x_safe] > 0) & (state.city_owner[y_safe, x_safe] == state.current_player)
        
        # Vérifier si adjacente à ville amie
        has_adjacent_city, _, _ = _find_adjacent_friendly_city(
            state, x, y, state.current_player
        )
        
        is_friendly_territory = (is_in_city | has_adjacent_city) & in_bounds
        
        # Guérir 4 HP en territoire ami, 2 HP ailleurs
        heal_amount = jnp.where(is_friendly_territory, 4, 2)
        
        # Calculer nouveau HP (limité à maxHP)
        new_hp = jnp.minimum(unit_hp + heal_amount, unit_max_hp)
        
        can_recover = needs_healing & in_bounds
        
        def perform_heal(state):
            new_units_hp = state.units_hp.at[unit_id].set(new_hp)
            new_units_has_acted = state.units_has_acted.at[unit_id].set(True)
            return state.replace(
                units_hp=new_units_hp,
                units_has_acted=new_units_has_acted,
            )
        
        return jax.lax.cond(
            can_recover,
            perform_heal,
            lambda s: s,
            state
        )
    
    return jax.lax.cond(
        is_unit_valid,
        do_recover,
        lambda s: s,
        state
    )


def _apply_research_tech(state: GameState, decoded: dict) -> GameState:
    """Applique une recherche technologique."""
    tech_id_val = decoded.get("unit_type", -1)
    tech_id = (
        jnp.array(tech_id_val, dtype=jnp.int32)
        if isinstance(tech_id_val, int) and tech_id_val >= 0
        else jnp.array(-1, dtype=jnp.int32)
    )
    
    def do_research(state):
        player_id = state.current_player
        already_unlocked = state.player_techs[player_id, tech_id]
        cost = _compute_tech_cost(state, tech_id, player_id)
        has_cost = cost > 0
        has_stars = state.player_stars[player_id] >= cost
        deps_met = _tech_dependencies_met_state(state, tech_id)
        can_research = (~already_unlocked) & has_cost & has_stars & deps_met
        
        def perform(state):
            new_player_techs = state.player_techs.at[player_id, tech_id].set(True)
            new_player_stars = state.player_stars.at[player_id].add(-cost)
            return state.replace(
                player_techs=new_player_techs,
                player_stars=new_player_stars,
            )
        
        return jax.lax.cond(
            can_research,
            perform,
            lambda s: s,
            state
        )
    
    return jax.lax.cond(
        tech_id >= 0,
        do_research,
        lambda s: s,
        state
    )


def _compute_defense_bonus(state: GameState, unit_id: int, unit_pos: jnp.ndarray) -> jnp.ndarray:
    """Calcule le bonus de défense d'une unité selon son terrain et les technologies.
    
    Args:
        state: État du jeu
        unit_id: ID de l'unité
        unit_pos: Position de l'unité [x, y]
    
    Returns:
        Bonus de défense (1.0 = aucun, 1.5 = standard, 4.0 = murs de ville)
    """
    x = unit_pos[0]
    y = unit_pos[1]
    unit_owner = state.units_owner[unit_id]
    
    # Vérifier les limites
    in_bounds = (x >= 0) & (x < state.width) & (y >= 0) & (y < state.height)
    
    # Utiliser des indices sécurisés pour éviter les erreurs hors limites
    x_safe = jnp.clip(x, 0, state.width - 1)
    y_safe = jnp.clip(y, 0, state.height - 1)
    
    terrain = state.terrain[y_safe, x_safe]
    is_forest = _is_forest_terrain(terrain)
    is_mountain = _is_mountain_terrain(terrain)
    is_water = _is_shallow_water_terrain(terrain) | (terrain == TerrainType.WATER_DEEP)
    is_city = (state.city_level[y_safe, x_safe] > 0) & (state.city_owner[y_safe, x_safe] == unit_owner)
    
    # Vérifier les technologies du propriétaire de l'unité
    has_archery = state.player_techs[unit_owner, TechType.ARCHERY]
    has_climbing = state.player_techs[unit_owner, TechType.CLIMBING]
    has_aquatism = state.player_techs[unit_owner, TechType.AQUATISM]
    
    # Bonus selon terrain et technologies
    # Forêt : 1.5x si ARCHERY recherchée
    forest_bonus = jnp.where(is_forest & has_archery, 1.5, 1.0)
    
    # Montagne : 1.5x si CLIMBING recherchée
    mountain_bonus = jnp.where(is_mountain & has_climbing, 1.5, 1.0)
    
    # Eau : 1.5x si AQUATISM recherchée
    water_bonus = jnp.where(is_water & has_aquatism, 1.5, 1.0)
    
    # Ville : 1.5x si unité en ville amie (simplification MVP : toutes unités ont Fortify)
    # TODO: Vérifier compétence Fortify quand système de compétences sera implémenté
    city_bonus = jnp.where(is_city, 1.5, 1.0)
    
    # Ville avec murs : 4.0x
    has_wall = state.city_has_wall[y_safe, x_safe]
    city_wall_bonus = jnp.where(is_city & has_wall, 4.0, city_bonus)
    
    # Prendre le bonus le plus élevé applicable
    # Ordre de priorité : ville avec murs > ville > terrain spécial
    bonus = jnp.maximum(
        jnp.maximum(forest_bonus, mountain_bonus),
        jnp.maximum(water_bonus, city_wall_bonus)
    )
    
    # Retourner 1.0 si hors limites, sinon le bonus calculé
    return jnp.where(in_bounds, bonus, 1.0)


def _apply_attack(state: GameState, decoded: dict) -> GameState:
    """Applique une attaque."""
    attacker_id_val = decoded.get("unit_id", -1)
    target_pos_val = decoded.get("target_pos")
    
    # Gérer target_pos (peut être None, tuple, ou array JAX)
    if target_pos_val is None:
        return state
    
    # Extraire target_x et target_y
    if isinstance(target_pos_val, tuple):
        target_x = jnp.array(target_pos_val[0], dtype=jnp.int32)
        target_y = jnp.array(target_pos_val[1], dtype=jnp.int32)
    else:
        # C'est un array JAX
        target_x = target_pos_val[0] if hasattr(target_pos_val, '__getitem__') else jnp.array(-1, dtype=jnp.int32)
        target_y = target_pos_val[1] if hasattr(target_pos_val, '__getitem__') and len(target_pos_val) > 1 else jnp.array(-1, dtype=jnp.int32)
    
    # Convertir attacker_id
    attacker_id = jnp.asarray(attacker_id_val, dtype=jnp.int32) if not isinstance(attacker_id_val, (int, type(None))) else (jnp.array(attacker_id_val, dtype=jnp.int32) if attacker_id_val is not None and attacker_id_val >= 0 else jnp.array(-1, dtype=jnp.int32))
    
    # Vérifier que l'attaquant est valide
    is_attacker_valid = (
        (attacker_id >= 0)
        & (attacker_id < state.max_units)
        & state.units_active[attacker_id]
        & (state.units_owner[attacker_id] == state.current_player)
        & (~state.units_has_acted[attacker_id])
    )
    
    # Trouver l'unité cible à la position
    target_id = _get_unit_at_position(state, target_x, target_y)
    has_target = target_id >= 0
    
    # Vérifier que la cible est ennemie
    attacker_owner = state.units_owner[attacker_id]
    target_owner = state.units_owner[target_id]
    is_enemy = attacker_owner != target_owner
    
    # Vérifier la portée (mêlée = distance de Chebyshev <= 1)
    attacker_pos = state.units_pos[attacker_id]
    dx = jnp.abs(attacker_pos[0] - target_x)
    dy = jnp.abs(attacker_pos[1] - target_y)
    distance = jnp.maximum(dx, dy)
    attacker_type = state.units_type[attacker_id]
    attacker_range = UNIT_ATTACK_RANGE[attacker_type]
    in_range = distance <= attacker_range
    
    can_attack = is_attacker_valid & has_target & is_enemy & in_range
    
    return jax.lax.cond(
        can_attack,
        lambda s: _perform_combat(s, attacker_id, target_id, distance),
        lambda s: s,
        state
    )


def _can_unit_promote(unit_type: jnp.ndarray) -> jnp.ndarray:
    """Vérifie si une unité peut être promue.
    
    Les unités navales et super units ne peuvent pas être promues.
    
    Args:
        unit_type: Type d'unité
    
    Returns:
        Booléen indiquant si l'unité peut être promue
    """
    return UNIT_CAN_PROMOTE[unit_type]


def _get_unit_max_hp(state: GameState, unit_id: int) -> jnp.ndarray:
    """Retourne le HP maximum d'une unité (base + 5 si vétérane).
    
    Args:
        state: État du jeu
        unit_id: ID de l'unité
    
    Returns:
        HP maximum de l'unité
    """
    unit_type = state.units_type[unit_id]
    base_max_hp = UNIT_HP_MAX[unit_type]
    is_veteran = state.units_veteran[unit_id]
    return base_max_hp + jnp.where(is_veteran, 5, 0)


def _promote_unit(state: GameState, unit_id: int) -> GameState:
    """Promouvoit une unité au rang de vétéran.
    
    Les vétérans gagnent +5 HP maximum et sont complètement guéris.
    
    Args:
        state: État du jeu
        unit_id: ID de l'unité à promouvoir
    
    Returns:
        Nouvel état avec l'unité promue
    """
    unit_type = state.units_type[unit_id]
    base_max_hp = UNIT_HP_MAX[unit_type]
    veteran_max_hp = base_max_hp + 5
    
    # Guérir complètement l'unité
    new_units_hp = state.units_hp.at[unit_id].set(veteran_max_hp)
    
    # Marquer comme vétéran
    new_units_veteran = state.units_veteran.at[unit_id].set(True)
    
    return state.replace(
        units_hp=new_units_hp,
        units_veteran=new_units_veteran,
    )


def _perform_combat(state: GameState, attacker_id: int, target_id: int, distance: jnp.ndarray) -> GameState:
    """Effectue un combat entre deux unités selon la formule complète de Polytopia."""
    attacker_type = state.units_type[attacker_id]
    target_type = state.units_type[target_id]
    
    # Récupérer les stats (utiliser les arrays JAX)
    attacker_attack = UNIT_ATTACK[attacker_type]
    attacker_defense = UNIT_DEFENSE[attacker_type]
    attacker_max_hp = UNIT_HP_MAX[attacker_type]
    attacker_hp = state.units_hp[attacker_id]
    
    target_attack = UNIT_ATTACK[target_type]
    target_defense = UNIT_DEFENSE[target_type]
    target_max_hp = UNIT_HP_MAX[target_type]
    target_hp = state.units_hp[target_id]
    
    # Calculer le bonus de défense de la cible
    target_pos = state.units_pos[target_id]
    defense_bonus = _compute_defense_bonus(state, target_id, target_pos)
    
    # Formule de combat complète de Polytopia
    # attackForce = attacker.attack * (attacker.health / attacker.maxHealth)
    attacker_hp_ratio = jnp.where(attacker_max_hp > 0, attacker_hp / attacker_max_hp, 0.0)
    attack_force = attacker_attack.astype(jnp.float32) * attacker_hp_ratio
    
    # defenseForce = defender.defense * (defender.health / defender.maxHealth) * defenseBonus
    target_hp_ratio = jnp.where(target_max_hp > 0, target_hp / target_max_hp, 0.0)
    defense_force = target_defense.astype(jnp.float32) * target_hp_ratio * defense_bonus
    
    # totalDamage = attackForce + defenseForce
    total_damage = attack_force + defense_force
    
    # Gérer division par zéro
    safe_total = jnp.where(total_damage > 0, total_damage, 1.0)
    
    # attackResult = round((attackForce / totalDamage) * attacker.attack * 4.5)
    attack_result = jnp.round((attack_force / safe_total) * attacker_attack.astype(jnp.float32) * 4.5).astype(jnp.int32)
    
    # defenseResult = round((defenseForce / totalDamage) * defender.defense * 4.5)
    defense_result = jnp.round((defense_force / safe_total) * target_defense.astype(jnp.float32) * 4.5).astype(jnp.int32)
    
    # Vérifier si contre-attaque autorisée (pas si unité à distance tue la cible)
    attacker_range = UNIT_ATTACK_RANGE[attacker_type]
    allow_retaliation = jnp.where(attacker_range > 1, distance <= 1, True)
    
    # Appliquer les dégâts à la cible
    new_target_hp = state.units_hp[target_id] - attack_result
    target_dead_after_attack = new_target_hp <= 0
    
    # Pas de contre-attaque si la cible est tuée (selon règles Polytopia)
    # Mais on vérifie aussi la portée pour les unités à distance
    counter_damage = jnp.where(
        allow_retaliation & ~target_dead_after_attack,
        defense_result,
        0
    )
    
    # Appliquer les dégâts
    new_units_hp = state.units_hp.at[target_id].set(jnp.maximum(0, new_target_hp))
    new_units_hp = new_units_hp.at[attacker_id].add(-counter_damage)
    
    # Vérifier si des unités sont détruites
    target_dead = new_units_hp[target_id] <= 0
    attacker_dead = new_units_hp[attacker_id] <= 0
    
    # Incrémenter les kills pour l'attaquant si la cible est morte
    new_units_kills = state.units_kills.at[attacker_id].add(
        jnp.where(target_dead & ~attacker_dead, 1, 0)
    )
    
    # Incrémenter les kills pour la cible si l'attaquant est mort (contre-attaque)
    new_units_kills = new_units_kills.at[target_id].add(
        jnp.where(attacker_dead & ~target_dead & (counter_damage > 0), 1, 0)
    )
    
    # Désactiver les unités mortes
    new_units_active = state.units_active.at[target_id].set(
        state.units_active[target_id] & ~target_dead
    )
    new_units_active = new_units_active.at[attacker_id].set(
        state.units_active[attacker_id] & ~attacker_dead
    )
    
    # Si la cible est morte et l'attaquant survit, l'attaquant prend sa place
    def move_attacker(state, new_units_pos):
        target_pos = state.units_pos[target_id]
        new_units_pos = new_units_pos.at[attacker_id, 0].set(target_pos[0])
        new_units_pos = new_units_pos.at[attacker_id, 1].set(target_pos[1])
        return state.replace(units_pos=new_units_pos)
    
    state = jax.lax.cond(
        target_dead & ~attacker_dead & (distance <= 1),
        lambda s: move_attacker(s, state.units_pos),
        lambda s: s,
        state
    )
    
    new_units_has_acted = state.units_has_acted.at[attacker_id].set(True)
    state = state.replace(
        units_hp=new_units_hp,
        units_active=new_units_active,
        units_has_acted=new_units_has_acted,
        units_kills=new_units_kills,
    )
    
    # Vérifier les promotions (3 kills = vétéran)
    # Promouvoir l'attaquant s'il a atteint 3 kills et peut être promu
    attacker_can_promote = _can_unit_promote(state.units_type[attacker_id])
    attacker_has_3_kills = new_units_kills[attacker_id] >= 3
    attacker_not_veteran = ~state.units_veteran[attacker_id]
    should_promote_attacker = attacker_can_promote & attacker_has_3_kills & attacker_not_veteran & ~attacker_dead
    
    state = jax.lax.cond(
        should_promote_attacker,
        lambda s: _promote_unit(s, attacker_id),
        lambda s: s,
        state
    )
    
    # Promouvoir la cible si elle a atteint 3 kills (contre-attaque)
    target_can_promote = _can_unit_promote(state.units_type[target_id])
    target_has_3_kills = new_units_kills[target_id] >= 3
    target_not_veteran = ~state.units_veteran[target_id]
    should_promote_target = target_can_promote & target_has_3_kills & target_not_veteran & ~target_dead & (counter_damage > 0)
    
    state = jax.lax.cond(
        should_promote_target,
        lambda s: _promote_unit(s, target_id),
        lambda s: s,
        state
    )
    
    # Vérifier la victoire
    state = _check_victory(state)
    
    return state


def _apply_train_unit(state: GameState, decoded: dict) -> GameState:
    """Entraîne une nouvelle unité dans une ville."""
    unit_type_val = decoded.get("unit_type")
    target_pos_val = decoded.get("target_pos")
    
    # Gérer les valeurs None ou arrays JAX
    if target_pos_val is None:
        return state
    
    # Extraire target_x et target_y
    if isinstance(target_pos_val, tuple):
        target_x = jnp.array(target_pos_val[0], dtype=jnp.int32)
        target_y = jnp.array(target_pos_val[1], dtype=jnp.int32)
    else:
        # C'est un array JAX
        target_x = target_pos_val[0] if hasattr(target_pos_val, '__getitem__') else jnp.array(-1, dtype=jnp.int32)
        target_y = target_pos_val[1] if hasattr(target_pos_val, '__getitem__') and len(target_pos_val) > 1 else jnp.array(-1, dtype=jnp.int32)
    
    # Convertir unit_type
    unit_type = jnp.asarray(unit_type_val, dtype=jnp.int32) if not isinstance(unit_type_val, (int, type(None))) else (jnp.array(unit_type_val, dtype=jnp.int32) if unit_type_val is not None and unit_type_val >= 0 else jnp.array(-1, dtype=jnp.int32))
    
    # Vérifier que unit_type et target_pos sont valides
    has_unit_type = unit_type >= 0
    has_target = (target_x >= 0) & (target_y >= 0)
    both_valid = has_unit_type & has_target
    
    def do_train(state):
        # Vérifier que la position est une ville du joueur actif
        is_city = state.city_level[target_y, target_x] > 0
        is_owner = state.city_owner[target_y, target_x] == state.current_player
        cost = UNIT_COST[unit_type]
        has_cost = cost > 0
        has_stars = state.player_stars[state.current_player] >= cost
        # Vérifier la technologie requise
        required_tech = UNIT_REQUIRED_TECH[unit_type]
        has_required_tech = jnp.where(
            required_tech == int(TechType.NONE),
            True,
            state.player_techs[state.current_player, required_tech],
        )
        can_train = is_city & is_owner & has_cost & has_stars & has_required_tech
        
        return jax.lax.cond(
            can_train,
            lambda s: _perform_train_unit(s, unit_type, target_x, target_y),
            lambda s: s,
            state
        )
    
    return jax.lax.cond(
        both_valid,
        do_train,
        lambda s: s,
        state
    )


def _perform_train_unit(state: GameState, unit_type: jnp.ndarray, target_x: jnp.ndarray, target_y: jnp.ndarray) -> GameState:
    """Effectue l'entraînement d'une unité (appelé après validation)."""
    
    # Trouver un slot d'unité libre
    free_slot = _find_free_unit_slot(state)
    has_free_slot = free_slot >= 0
    
    # Vérifier que la case n'est pas occupée
    pos_occupied = _is_position_occupied(state, target_x, target_y, -1)
    can_place = has_free_slot & ~pos_occupied
    
    def create_unit(state):
        # Créer l'unité
        unit_hp_max = UNIT_HP_MAX[unit_type]
        new_units_type = state.units_type.at[free_slot].set(unit_type)
        new_units_pos = state.units_pos.at[free_slot, 0].set(target_x)
        new_units_pos = new_units_pos.at[free_slot, 1].set(target_y)
        new_units_hp = state.units_hp.at[free_slot].set(unit_hp_max)
        new_units_owner = state.units_owner.at[free_slot].set(state.current_player)
        new_units_active = state.units_active.at[free_slot].set(True)
        new_payload = state.units_payload_type.at[free_slot].set(unit_type)
        
        new_player_stars = state.player_stars.at[state.current_player].add(-UNIT_COST[unit_type])
        
        return state.replace(
            units_type=new_units_type,
            units_pos=new_units_pos,
            units_hp=new_units_hp,
            units_owner=new_units_owner,
            units_active=new_units_active,
            units_has_acted=state.units_has_acted.at[free_slot].set(False),
            units_payload_type=new_payload,
            player_stars=new_player_stars,
        )
    
    return jax.lax.cond(
        can_place,
        create_unit,
        lambda s: s,
        state
    )


def _apply_end_turn(state: GameState) -> GameState:
    """Termine le tour du joueur actif."""
    income = _compute_income_for_player(state, state.current_player)
    new_player_stars = state.player_stars.at[state.current_player].add(income)
    
    # Passer au joueur suivant
    next_player = (state.current_player + 1) % state.num_players
    
    # Si on revient au joueur 0, incrémenter le tour
    new_turn = jnp.where(next_player == 0, state.turn + 1, state.turn)
    
    # Mettre à jour la vision de tous les joueurs
    def update_player_vision(i, tiles_visible):
        player_vision = _compute_player_vision(state, i)
        return tiles_visible.at[i].set(player_vision)
    
    new_tiles_visible = jax.lax.fori_loop(
        0, state.num_players,
        update_player_vision,
        state.tiles_visible
    )
    
    state = state.replace(
        player_stars=new_player_stars,
        current_player=next_player,
        turn=new_turn,
        units_has_acted=jnp.zeros_like(state.units_has_acted),
        tiles_visible=new_tiles_visible,
    )
    
    state = _check_victory(state)
    return state


def _check_city_capture(state: GameState, unit_id: int, x: int, y: int) -> GameState:
    """Vérifie et effectue la capture d'une ville."""
    # Vérifier s'il y a une ville à cette position
    has_city = state.city_level[y, x] > 0
    is_enemy_city = state.city_owner[y, x] != state.current_player
    is_neutral = state.city_owner[y, x] == NO_OWNER
    
    should_capture = has_city & (is_enemy_city | is_neutral)
    
    # Capturer la ville
    new_city_owner = jnp.where(
        should_capture,
        state.current_player,
        state.city_owner[y, x]
    )
    new_city_owner_array = state.city_owner.at[y, x].set(new_city_owner)
    
    state = state.replace(city_owner=new_city_owner_array)
    
    def reset_population(s):
        return _set_city_population(s, x, y, CITY_CAPTURE_POPULATION)
    
    state = jax.lax.cond(
        should_capture,
        reset_population,
        lambda s: s,
        state
    )
    
    # Mettre à jour les connexions après capture
    state = jax.lax.cond(
        should_capture,
        lambda s: _update_city_connections(s),
        lambda s: s,
        state
    )
    
    # Vérifier la victoire
    state = _check_victory(state)
    
    return state


def _check_victory(state: GameState) -> GameState:
    """Vérifie les conditions de victoire selon le mode de jeu."""
    # Pour chaque joueur, vérifier s'il a encore une capitale
    # Au lieu d'utiliser jnp.arange, on vérifie directement chaque joueur possible
    # Pour le MVP, on suppose max 4 joueurs (peut être étendu si nécessaire)
    max_players_check = 4
    
    def check_player(player_id):
        has_capital = jnp.any(
            (state.city_owner == player_id) & (state.city_level > 0)
        )
        return has_capital
    
    # Vérifier les joueurs 0 à max_players_check-1
    players_to_check = jnp.arange(max_players_check)
    players_alive = jax.vmap(check_player)(players_to_check)
    
    # Masquer les joueurs au-delà de state.num_players
    valid_players = players_to_check < state.num_players
    players_alive = players_alive & valid_players
    
    num_alive = jnp.sum(players_alive)
    
    # Conditions de victoire selon le mode (utiliser jax.lax.switch pour compatibilité JAX)
    def check_domination():
        # Domination : élimination complète (un seul joueur reste)
        return num_alive <= 1
    
    def check_perfection():
        # Perfection : limite de tours atteinte
        return state.turn >= state.max_turns
    
    def check_glory():
        # Glory : premier joueur à atteindre 10,000 points
        glory_threshold = 10000
        return jnp.any(state.player_score >= glory_threshold)
    
    def check_might():
        # Might : capturer toutes les capitales ennemies
        # Un joueur gagne s'il possède toutes les capitales (tous les autres n'en ont plus)
        def player_has_all_capitals(player_id):
            own_capitals = jnp.sum(
                (state.city_owner == player_id) & (state.city_level > 0)
            )
            other_capitals = jnp.sum(
                (state.city_owner != player_id) & (state.city_owner >= 0) & (state.city_level > 0)
            )
            return (own_capitals > 0) & (other_capitals == 0)
        
        players_winning = jax.vmap(player_has_all_capitals)(players_to_check)
        players_winning = players_winning & valid_players
        return jnp.any(players_winning)
    
    def check_creative():
        # Creative : pas de limite, partie continue jusqu'à élimination ou limite manuelle
        # Par défaut, utiliser la même logique que Domination
        return num_alive <= 1
    
    # Utiliser jax.lax.switch pour sélectionner la fonction selon le mode
    # Convertir game_mode en int pour jax.lax.switch (mais sans utiliser int() directement)
    game_mode_int = jnp.asarray(state.game_mode, dtype=jnp.int32)
    is_done = jax.lax.switch(
        game_mode_int,
        [check_domination, check_perfection, check_creative, check_glory, check_might]
    )
    
    return state.replace(done=is_done)


def _is_position_occupied(state: GameState, x: int, y: int, exclude_unit_id: int = -1) -> bool:
    """Vérifie si une position est occupée par une unité."""
    # Vérifier toutes les unités actives
    unit_x = state.units_pos[:, 0]
    unit_y = state.units_pos[:, 1]
    
    # Vérifier si une unité active est à cette position
    # Note: Pour simplifier et éviter les problèmes de traçage, on ignore exclude_unit_id
    # dans le contexte JAX. C'est acceptable pour le MVP car cette fonction est principalement
    # utilisée pour vérifier si une case est libre avant un mouvement.
    at_position = (unit_x == x) & (unit_y == y) & state.units_active
    
    return jnp.any(at_position)


def _get_unit_at_position(state: GameState, x: int, y: int) -> int:
    """Retourne l'ID de l'unité à une position, ou -1 si aucune."""
    # Trouver toutes les unités à cette position
    unit_x = state.units_pos[:, 0]
    unit_y = state.units_pos[:, 1]
    at_position = (unit_x == x) & (unit_y == y) & state.units_active
    
    # Utiliser jnp.argmax pour trouver le premier match (ou 0 si aucun)
    # Si aucun match, argmax retourne 0, donc on doit vérifier
    first_match_idx = jnp.argmax(at_position.astype(jnp.int32))
    has_match = jnp.any(at_position)
    
    # Retourner l'index si match, sinon -1
    return jnp.where(has_match, first_match_idx, -1)


def _find_free_unit_slot(state: GameState) -> int:
    """Trouve un slot d'unité libre, ou -1 si aucun."""
    # Trouver les slots libres
    is_free = ~state.units_active
    
    # Utiliser argmax pour trouver le premier slot libre
    first_free_idx = jnp.argmax(is_free.astype(jnp.int32))
    has_free = jnp.any(is_free)
    
    # Retourner l'index si libre, sinon -1
    return jnp.where(has_free, first_free_idx, -1)


def _player_has_tech(state: GameState, tech_type: int) -> jnp.ndarray:
    """Retourne True si le joueur actif possède la techno donnée."""
    tech_idx = jnp.asarray(tech_type, dtype=jnp.int32)
    in_bounds = (tech_idx >= 0) & (tech_idx < NUM_TECHS)
    return jnp.where(
        in_bounds,
        state.player_techs[state.current_player, tech_idx],
        False,
    )


def _tech_dependencies_met_state(state: GameState, tech_id: jnp.ndarray) -> jnp.ndarray:
    """Vérifie que les dépendances d'une techno sont satisfaites."""
    deps = TECH_DEPENDENCIES[tech_id]
    player_row = state.player_techs[state.current_player]
    return jnp.all(jnp.where(deps, player_row, True))


def _tech_dependencies_mask(player_row: jnp.ndarray) -> jnp.ndarray:
    """Retourne un bool par techno indiquant si les dépendances sont satisfaites."""
    deps = TECH_DEPENDENCIES
    player_broadcast = jnp.broadcast_to(player_row, deps.shape)
    deps_satisfied = jnp.where(deps, player_broadcast, True)
    return jnp.all(deps_satisfied, axis=1)


def _is_plain_terrain(terrain_value: jnp.ndarray) -> jnp.ndarray:
    return (
        (terrain_value == TerrainType.PLAIN)
        | (terrain_value == TerrainType.PLAIN_FRUIT)
    )


def _is_forest_terrain(terrain_value: jnp.ndarray) -> jnp.ndarray:
    return (
        (terrain_value == TerrainType.FOREST)
        | (terrain_value == TerrainType.FOREST_WITH_WILD_ANIMAL)
    )


def _is_mountain_terrain(terrain_value: jnp.ndarray) -> jnp.ndarray:
    return (
        (terrain_value == TerrainType.MOUNTAIN)
        | (terrain_value == TerrainType.MOUNTAIN_WITH_MINE)
    )


def _is_shallow_water_terrain(terrain_value: jnp.ndarray) -> jnp.ndarray:
    return (
        (terrain_value == TerrainType.WATER_SHALLOW)
        | (terrain_value == TerrainType.WATER_SHALLOW_WITH_FISH)
    )


def _is_land_terrain(terrain_value: jnp.ndarray) -> jnp.ndarray:
    return (
        _is_plain_terrain(terrain_value)
        | _is_forest_terrain(terrain_value)
        | _is_mountain_terrain(terrain_value)
    )


def _compute_unit_vision(state: GameState, unit_id: int, unit_pos: jnp.ndarray) -> jnp.ndarray:
    """Calcule le masque de vision d'une unité.
    
    Args:
        state: État du jeu
        unit_id: ID de l'unité
        unit_pos: Position de l'unité [x, y]
    
    Returns:
        Masque de vision [H, W] (True = visible)
    """
    x = unit_pos[0]
    y = unit_pos[1]
    
    # Créer grille de coordonnées
    # Utiliser la forme du terrain directement (compatible avec contexte tracé)
    h, w = state.terrain.shape[0], state.terrain.shape[1]
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.int32),
        jnp.arange(w, dtype=jnp.int32),
        indexing='ij'
    )
    
    # Calculer distance de Chebyshev depuis l'unité
    dx = jnp.abs(x_coords - x)
    dy = jnp.abs(y_coords - y)
    distance = jnp.maximum(dx, dy)
    
    # Déterminer rayon de vision selon terrain
    # Vision standard : 3x3 (rayon 1)
    # Vision montagne : 5x5 (rayon 2)
    x_safe = jnp.clip(x, 0, w - 1)
    y_safe = jnp.clip(y, 0, h - 1)
    terrain = state.terrain[y_safe, x_safe]
    is_mountain = _is_mountain_terrain(terrain)
    
    # Rayon = 1 (standard) ou 2 (montagne)
    vision_radius = jnp.where(is_mountain, 2, 1)
    
    # Masque de vision : distance <= rayon
    vision_mask = distance <= vision_radius
    
    return vision_mask


def _compute_city_vision(state: GameState, x: int, y: int) -> jnp.ndarray:
    """Calcule le masque de vision d'une ville (vision 3x3 autour).
    
    Args:
        state: État du jeu
        x, y: Position de la ville
    
    Returns:
        Masque de vision [H, W] (True = visible)
    """
    # Utiliser la forme du terrain directement (compatible avec contexte tracé)
    h, w = state.terrain.shape[0], state.terrain.shape[1]
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.int32),
        jnp.arange(w, dtype=jnp.int32),
        indexing='ij'
    )
    
    # Distance de Chebyshev depuis la ville
    dx = jnp.abs(x_coords - x)
    dy = jnp.abs(y_coords - y)
    distance = jnp.maximum(dx, dy)
    
    # Vision standard 3x3 (rayon 1) pour villes
    vision_mask = distance <= 1
    
    return vision_mask


def _compute_player_vision(state: GameState, player_id: int) -> jnp.ndarray:
    """Calcule la vision totale d'un joueur (combine unités et villes).
    
    Args:
        state: État du jeu
        player_id: ID du joueur
    
    Returns:
        Masque de vision [H, W] (True = visible)
    """
    # Utiliser la forme du terrain directement (compatible avec contexte tracé)
    h, w = state.terrain.shape[0], state.terrain.shape[1]
    total_vision = jnp.zeros((h, w), dtype=jnp.bool_)
    
    # Vision des unités actives du joueur
    def add_unit_vision(unit_id, vision):
        unit_pos = state.units_pos[unit_id]
        unit_vision = _compute_unit_vision(state, unit_id, unit_pos)
        return vision | unit_vision
    
    # Parcourir toutes les unités actives du joueur
    player_units_mask = (state.units_owner == player_id) & state.units_active
    # Utiliser la longueur de units_type directement (compatible avec contexte tracé)
    max_units = state.units_type.shape[0]
    unit_ids = jnp.arange(max_units, dtype=jnp.int32)
    
    def scan_unit(unit_id, vision):
        has_unit = player_units_mask[unit_id]
        unit_pos = state.units_pos[unit_id]
        unit_vision = _compute_unit_vision(state, unit_id, unit_pos)
        return jnp.where(has_unit, vision | unit_vision, vision)
    
    # Utiliser la longueur de units_type directement (compatible avec contexte tracé)
    max_units = state.units_type.shape[0]
    total_vision = jax.lax.fori_loop(0, max_units, lambda i, v: scan_unit(i, v), total_vision)
    
    # Vision des villes du joueur
    def add_city_vision(y, x, vision):
        has_city = (state.city_level[y, x] > 0) & (state.city_owner[y, x] == player_id)
        city_vision = _compute_city_vision(state, x, y)
        return jnp.where(has_city, vision | city_vision, vision)
    
    # Parcourir toutes les cases pour trouver les villes
    def scan_city_row(y, vision):
        def scan_city_col(x, v):
            return add_city_vision(y, x, v)
        return jax.lax.fori_loop(0, w, scan_city_col, vision)
    
    total_vision = jax.lax.fori_loop(0, h, scan_city_row, total_vision)
    
    return total_vision


def _is_friendly_port(state: GameState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    has_port = state.city_has_port[y, x]
    owner = state.city_owner[y, x]
    return has_port & (owner == state.current_player)


def _embark_unit(state: GameState, unit_id: jnp.ndarray) -> GameState:
    unit_type = state.units_type[unit_id]
    payload = state.units_payload_type.at[unit_id].set(unit_type)
    new_units_type = state.units_type.at[unit_id].set(UnitType.RAFT)
    return state.replace(
        units_type=new_units_type,
        units_payload_type=payload,
    )


def _disembark_unit(state: GameState, unit_id: jnp.ndarray) -> GameState:
    payload = state.units_payload_type[unit_id]
    restored_type = jnp.where(payload > 0, payload, UnitType.WARRIOR)
    new_units_type = state.units_type.at[unit_id].set(restored_type)
    new_payloads = state.units_payload_type.at[unit_id].set(0)
    return state.replace(
        units_type=new_units_type,
        units_payload_type=new_payloads,
    )


def _population_to_level(population: jnp.ndarray) -> jnp.ndarray:
    """Calcule le niveau d'une ville en fonction de sa population.
    
    Supporte les niveaux 1-5+ avec seuils croissants.
    """
    thresholds = CITY_LEVEL_POP_THRESHOLDS
    max_level_idx = len(thresholds) - 1
    
    # Commencer par le niveau 0 (pas de ville)
    level = jnp.array(0, dtype=jnp.int32)
    
    # Monter les niveaux jusqu'à trouver le bon seuil
    def check_level(i, current_level):
        threshold = thresholds[i]
        is_above_or_equal = population >= threshold
        return jnp.where(is_above_or_equal, i, current_level)
    
    # Vérifier chaque niveau depuis le plus bas (1) jusqu'au plus haut
    level = jax.lax.fori_loop(1, max_level_idx + 1, check_level, level)
    
    return level


def _set_city_population(state: GameState, x: int, y: int, new_population: jnp.ndarray) -> GameState:
    """Met à jour la population (et donc le niveau) d'une ville."""
    new_level = _population_to_level(new_population)
    new_city_population = state.city_population.at[y, x].set(new_population)
    new_city_level = state.city_level.at[y, x].set(new_level)
    
    return state.replace(
        city_population=new_city_population,
        city_level=new_city_level,
    )


def _add_population_to_city(state: GameState, x: jnp.ndarray, y: jnp.ndarray, pop_gain: jnp.ndarray) -> GameState:
    """Ajoute de la population à une ville spécifique."""
    current_population = state.city_population[y, x]
    new_population = current_population + pop_gain
    return _set_city_population(state, x, y, new_population)


def _compute_income_for_player(state: GameState, player_id: jnp.ndarray) -> jnp.ndarray:
    """Calcule le revenu total généré par les villes d'un joueur."""
    tile_income = CITY_STAR_INCOME_PER_LEVEL[state.city_level]
    owned_income = jnp.where(state.city_owner == player_id, tile_income, 0)
    base_income = jnp.sum(owned_income, dtype=jnp.int32)
    
    # Ajouter revenus des marchés (+1★ par marché)
    market_income = jnp.sum(
        jnp.where(
            (state.city_owner == player_id) & state.city_has_market,
            1,
            0
        ),
        dtype=jnp.int32
    )
    
    bonus = state.player_income_bonus[player_id]
    return base_income + market_income + bonus


def _is_connected_tile(state: GameState, x: jnp.ndarray, y: jnp.ndarray, player_id: jnp.ndarray) -> jnp.ndarray:
    """Vérifie si une case est connectée (route, pont, ou port)."""
    has_road = state.has_road[y, x]
    has_bridge = state.has_bridge[y, x]
    has_port = state.city_has_port[y, x] & (state.city_owner[y, x] == player_id)
    is_city = (state.city_level[y, x] > 0) & (state.city_owner[y, x] == player_id)
    
    # Une ville compte comme connectée (point de connexion)
    return has_road | has_bridge | has_port | is_city


def _find_capital_city(state: GameState, player_id: jnp.ndarray) -> tuple:
    """Trouve la capitale d'un joueur (première ville trouvée).
    
    Returns:
        (found, capital_x, capital_y)
    """
    h, w = state.terrain.shape[0], state.terrain.shape[1]
    
    def scan_row(y, carry):
        found, cap_x, cap_y = carry
        
        def scan_col(x, c):
            f, cx, cy = c
            is_city = (state.city_level[y, x] > 0) & (state.city_owner[y, x] == player_id)
            new_found = f | is_city
            new_cx = jnp.where(~f & is_city, x, cx)
            new_cy = jnp.where(~f & is_city, y, cy)
            return (new_found, new_cx, new_cy)
        
        return jax.lax.fori_loop(0, w, scan_col, carry)
    
    init = (jnp.array(False, dtype=jnp.bool_), jnp.array(-1, dtype=jnp.int32), jnp.array(-1, dtype=jnp.int32))
    found, cap_x, cap_y = jax.lax.fori_loop(0, h, scan_row, init)
    
    return (found, cap_x, cap_y)


def _are_cities_connected(state: GameState, x1: jnp.ndarray, y1: jnp.ndarray, x2: jnp.ndarray, y2: jnp.ndarray, player_id: jnp.ndarray) -> jnp.ndarray:
    """Vérifie si deux villes sont connectées via routes/ponts/ports.
    
    Utilise un parcours en largeur simplifié (BFS) pour trouver un chemin.
    Limité à une distance raisonnable pour éviter la complexité.
    """
    # Si les villes sont adjacentes et connectées, elles sont connectées
    dx = jnp.abs(x1 - x2)
    dy = jnp.abs(y2 - y1)
    is_adjacent = (dx <= 1) & (dy <= 1) & ((dx + dy) > 0)
    
    # Vérifier si les deux cases sont connectées
    tile1_connected = _is_connected_tile(state, x1, y1, player_id)
    tile2_connected = _is_connected_tile(state, x2, y2, player_id)
    
    # Si adjacentes et toutes deux connectées, elles sont connectées
    directly_connected = is_adjacent & tile1_connected & tile2_connected
    
    # Pour simplifier, on considère que deux villes sont connectées si :
    # 1. Elles sont directement adjacentes et connectées, OU
    # 2. Il existe un chemin via cases adjacentes connectées (BFS simplifié)
    # Pour l'instant, on se limite aux connexions directes pour éviter la complexité
    return directly_connected


def _update_city_connections(state: GameState) -> GameState:
    """Met à jour les bonus de population selon les connexions de villes.
    
    Chaque connexion entre une ville et la capitale donne +1 population à chaque ville connectée.
    """
    # Pour chaque joueur, mettre à jour les connexions
    def update_player(player_id, current_state):
        # Trouver la capitale
        has_capital, cap_x, cap_y = _find_capital_city(current_state, player_id)
        
        # Si pas de capitale, pas de connexions
        def no_capital(s):
            return s
        
        def with_capital(s):
            h, w = s.terrain.shape[0], s.terrain.shape[1]
            
            # Pour chaque ville du joueur, vérifier si connectée à la capitale
            def check_city_row(y, pop_bonus_map):
                def check_city_col(x, bonus_map):
                    is_city = (s.city_level[y, x] > 0) & (s.city_owner[y, x] == player_id)
                    is_capital = (x == cap_x) & (y == cap_y)
                    
                    # La capitale est toujours connectée à elle-même
                    is_connected = is_capital | _are_cities_connected(s, x, y, cap_x, cap_y, player_id)
                    
                    # Bonus de +1 population si connectée
                    bonus = jnp.where(is_city & is_connected, 1, 0)
                    return bonus_map.at[y, x].set(bonus)
                
                return jax.lax.fori_loop(0, w, check_city_col, pop_bonus_map)
            
            # Calculer les bonus de connexion
            connection_bonus = jnp.zeros((h, w), dtype=jnp.int32)
            connection_bonus = jax.lax.fori_loop(0, h, check_city_row, connection_bonus)
            
            # Appliquer les bonus de population (seulement si la ville n'avait pas déjà ce bonus)
            # Pour simplifier, on ajoute toujours le bonus (sera recalculé à chaque fois)
            def apply_bonus_row(y, updated_state):
                def apply_bonus_col(x, s):
                    bonus = connection_bonus[y, x]
                    has_bonus = bonus > 0
                    
                    def add_bonus(st):
                        current_pop = st.city_population[y, x]
                        # Ne pas ajouter si déjà au maximum (éviter accumulation infinie)
                        # Pour simplifier, on ajoute toujours +1 si connectée
                        new_pop = current_pop + bonus
                        return _set_city_population(st, x, y, new_pop)
                    
                    return jax.lax.cond(has_bonus, add_bonus, lambda st: st, s)
                
                return jax.lax.fori_loop(0, w, apply_bonus_col, updated_state)
            
            return jax.lax.fori_loop(0, h, apply_bonus_row, s)
        
        return jax.lax.cond(has_capital, with_capital, no_capital, current_state)
    
    # Mettre à jour pour tous les joueurs
    def update_all_players(i, s):
        return update_player(i, s)
    
    return jax.lax.fori_loop(0, state.num_players, update_all_players, state)


def _player_can_harvest(state: GameState) -> jnp.ndarray:
    """Retourne True si le joueur courant peut récolter au moins une ressource."""
    total_tiles = state.height * state.width
    width = jnp.array(state.width, dtype=jnp.int32)
    
    def body(idx, found):
        x = jax.lax.rem(idx, width)
        y = jax.lax.div(idx, width)
        tile_available = _can_harvest_tile(state, x, y)
        return found | tile_available
    
    return jax.lax.fori_loop(
        0,
        total_tiles,
        body,
        jnp.array(False, dtype=jnp.bool_),
    )


def _can_harvest_tile(state: GameState, x: int, y: int) -> jnp.ndarray:
    """Vérifie si une ressource peut être récoltée sur une case donnée."""
    if isinstance(x, int):
        x_idx = jnp.array(x, dtype=jnp.int32)
    else:
        x_idx = jnp.asarray(x, dtype=jnp.int32)
    if isinstance(y, int):
        y_idx = jnp.array(y, dtype=jnp.int32)
    else:
        y_idx = jnp.asarray(y, dtype=jnp.int32)
    
    resource_type = state.resource_type[y_idx, x_idx]
    available = state.resource_available[y_idx, x_idx]
    has_resource = available & (resource_type > int(ResourceType.NONE))
    cost = RESOURCE_COST[resource_type]
    player_stars = state.player_stars[state.current_player]
    has_stars = player_stars >= cost
    required = RESOURCE_REQUIRED_TECH[resource_type]
    has_required = jnp.where(
        required == int(TechType.NONE),
        True,
        state.player_techs[state.current_player, required],
    )
    has_city, _, _ = _find_adjacent_friendly_city(
        state, x_idx, y_idx, state.current_player
    )
    return has_resource & has_stars & has_required & has_city


def _find_adjacent_friendly_city(
    state: GameState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    player_id: jnp.ndarray,
) -> tuple:
    """Recherche une ville amie adjacente à une case."""
    x_idx = jnp.asarray(x, dtype=jnp.int32)
    y_idx = jnp.asarray(y, dtype=jnp.int32)
    player = jnp.asarray(player_id, dtype=jnp.int32)
    offsets = HARVEST_NEIGHBOR_DELTAS
    
    def body(i, carry):
        found, city_x, city_y = carry
        dx = offsets[i, 0]
        dy = offsets[i, 1]
        nx = x_idx + dx
        ny = y_idx + dy
        in_bounds = (
            (nx >= 0) & (nx < state.width) &
            (ny >= 0) & (ny < state.height)
        )
        owner = jnp.where(in_bounds, state.city_owner[ny, nx], NO_OWNER)
        has_city = jnp.where(in_bounds, state.city_level[ny, nx] > 0, False)
        is_friendly = has_city & (owner == player)
        city_x = jnp.where(~found & is_friendly, nx, city_x)
        city_y = jnp.where(~found & is_friendly, ny, city_y)
        found = found | is_friendly
        return (found, city_x, city_y)
    
    init = (
        jnp.array(False, dtype=jnp.bool_),
        jnp.array(-1, dtype=jnp.int32),
        jnp.array(-1, dtype=jnp.int32),
    )
    return jax.lax.fori_loop(
        0,
        HARVEST_NEIGHBOR_DELTAS.shape[0],
        body,
        init,
    )


def _compute_tech_cost(state: GameState, tech_id: jnp.ndarray, player_id: jnp.ndarray) -> jnp.ndarray:
    """Calcule le coût dynamique d'une technologie selon nombre de villes et Philosophy.
    
    Formule Polytopia: cost = tier * num_cities + 4
    Avec Philosophy: cost = ceil(cost * 0.67) (réduction de 33%)
    
    Args:
        state: État du jeu
        tech_id: ID de la technologie
        player_id: ID du joueur
    
    Returns:
        Coût de la technologie (en étoiles)
    """
    tier = TECH_TIER[tech_id]
    
    # Compter le nombre de villes du joueur
    num_cities = jnp.sum(
        (state.city_owner == player_id) & (state.city_level > 0),
        dtype=jnp.int32
    )
    
    # Coût de base selon formule Polytopia: tier * num_cities + 4
    base_cost = tier * num_cities + 4
    
    # Vérifier si le joueur a Philosophy
    has_philosophy = state.player_techs[player_id, TechType.PHILOSOPHY]
    
    # Appliquer réduction de 33% si Philosophy (arrondi vers le haut)
    discounted_cost = jnp.ceil(base_cost * 0.67).astype(jnp.int32)
    
    # Utiliser le coût réduit si Philosophy, sinon coût de base
    final_cost = jnp.where(has_philosophy, discounted_cost, base_cost)
    
    # Retourner 0 si tech_id invalide
    return jnp.where(tech_id > 0, final_cost, 0)


def _min_positive_cost(costs: jnp.ndarray) -> jnp.ndarray:
    """Retourne le coût positif minimal d'un tableau (ou un entier max sinon)."""
    max_int = jnp.iinfo(jnp.int32).max
    positive = jnp.where(costs > 0, costs, max_int)
    return jnp.min(positive)


def legal_actions_mask(state: GameState) -> jnp.ndarray:
    """Retourne un masque simplifié des actions légales par type."""
    num_actions = int(ActionType.NUM_ACTIONS)
    mask = jnp.ones(num_actions, dtype=jnp.bool_)
    
    stars = state.player_stars[state.current_player]
    
    train_costs = UNIT_COST[1:]
    min_train_cost = _min_positive_cost(train_costs)
    can_train = stars >= min_train_cost
    mask = mask.at[ActionType.TRAIN_UNIT].set(can_train)
    
    build_costs = BUILDING_COST[1:int(BuildingType.NUM_TYPES)]
    min_build_cost = _min_positive_cost(build_costs)
    can_build = stars >= min_build_cost
    mask = mask.at[ActionType.BUILD].set(can_build)
    
    player_techs_row = state.player_techs[state.current_player]
    deps_met = _tech_dependencies_mask(player_techs_row)
    
    # Calculer les coûts dynamiques pour toutes les technologies
    player_id = state.current_player
    tech_ids = jnp.arange(NUM_TECHS, dtype=jnp.int32)
    tech_costs = jax.vmap(lambda tid: _compute_tech_cost(state, tid, player_id))(tech_ids)
    
    available_techs = (
        (tech_costs > 0)
        & (~player_techs_row)
        & deps_met
        & (stars >= tech_costs)
    )
    can_research = jnp.any(available_techs)
    mask = mask.at[ActionType.RESEARCH_TECH].set(can_research)
    
    can_harvest = _player_can_harvest(state)
    mask = mask.at[ActionType.HARVEST_RESOURCE].set(can_harvest)
    
    def done_mask():
        base = jnp.zeros_like(mask)
        return base.at[ActionType.NO_OP].set(True)
    
    return jax.lax.cond(
        state.done,
        lambda _: done_mask(),
        lambda _: mask,
        operand=None
    )
