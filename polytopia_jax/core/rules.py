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

UNIT_HP_MAX = _pad_unit_array([0, 10, 15, 8, 10, 8])
UNIT_ATTACK = _pad_unit_array([0, 2, 1, 2, 3, 2])
UNIT_DEFENSE = _pad_unit_array([0, 2, 3, 1, 1, 1])
UNIT_MOVEMENT = _pad_unit_array([0, 1, 1, 1, 2, 2])
UNIT_COST = _pad_unit_array([0, 2, 3, 3, 4, 0])
UNIT_ATTACK_RANGE = _pad_unit_array([0, 1, 1, 2, 1, 1])
UNIT_IS_NAVAL = _pad_unit_array([0, 0, 0, 0, 0, 1]).astype(jnp.bool_)
UNIT_CAN_ENTER_SHALLOW = _pad_unit_array([0, 0, 0, 0, 0, 1]).astype(jnp.bool_)
UNIT_CAN_ENTER_DEEP = _pad_unit_array([0, 0, 0, 0, 0, 0]).astype(jnp.bool_)


class BuildingType(IntEnum):
    """Types de bâtiments basiques."""
    NONE = 0
    FARM = 1
    MINE = 2
    HUT = 3
    PORT = 4
    NUM_TYPES = 5


MAX_BUILDING_TYPES = 8
BUILDING_COST = jnp.array(
    [0, 3, 4, 2, 5] + [0] * (MAX_BUILDING_TYPES - 5),
    dtype=jnp.int32,
)
BUILDING_POP_GAIN = jnp.array(
    [0, 1, 2, 1, 0] + [0] * (MAX_BUILDING_TYPES - 5),
    dtype=jnp.int32,
)
BUILDING_REQUIRED_TECH = jnp.array(
    [int(TechType.NONE), int(TechType.NONE), int(TechType.MINING), int(TechType.NONE), int(TechType.SAILING)] + [0] * (MAX_BUILDING_TYPES - 5),
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

TECH_COST = jnp.array(
    [0, 3, 4, 3] + [0] * (NUM_TECHS - 4),
    dtype=jnp.int32,
)

tech_deps_rows = [
    [False] * NUM_TECHS,
    [False] * NUM_TECHS,  # CLIMBING
    [False, True, False, False] + [False] * (NUM_TECHS - 4),  # SAILING requires CLIMBING
    [False] * NUM_TECHS,  # MINING
]
if NUM_TECHS > 4:
    tech_deps_rows.extend([[False] * NUM_TECHS for _ in range(NUM_TECHS - 4)])

TECH_DEPENDENCIES = jnp.array(tech_deps_rows, dtype=jnp.bool_)

CITY_LEVEL_POP_THRESHOLDS = jnp.array([0, 1, 3, 5], dtype=jnp.int32)
CITY_STAR_INCOME_PER_LEVEL = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
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
    can_move = is_in_bounds & ~pos_occupied & can_enter
    
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
        pop_gain = BUILDING_POP_GAIN[building_type]
        is_port = building_type == BuildingType.PORT
        has_effect = jnp.where(is_port, True, pop_gain > 0)
        has_stars = state.player_stars[state.current_player] >= cost
        required_tech = BUILDING_REQUIRED_TECH[building_type]
        has_required = jnp.where(
            required_tech == int(TechType.NONE),
            True,
            state.player_techs[state.current_player, required_tech],
        )
        can_build = is_city & is_owner & has_effect & has_stars & has_required
        
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


def _perform_build(state: GameState, building_type: jnp.ndarray, target_x: jnp.ndarray, target_y: jnp.ndarray) -> GameState:
    """Applique les effets d'un bâtiment validé."""
    pop_gain = BUILDING_POP_GAIN[building_type]
    cost = BUILDING_COST[building_type]
    new_population_value = state.city_population[target_y, target_x] + pop_gain
    
    is_port = building_type == BuildingType.PORT
    
    def apply_port(state):
        new_player_stars = state.player_stars.at[state.current_player].add(-cost)
        new_ports = state.city_has_port.at[target_y, target_x].set(True)
        return state.replace(
            player_stars=new_player_stars,
            city_has_port=new_ports,
        )
    
    def apply_population_build(state):
        updated_state = _set_city_population(state, target_x, target_y, new_population_value)
        new_player_stars = updated_state.player_stars.at[updated_state.current_player].add(-cost)
        return updated_state.replace(player_stars=new_player_stars)
    
    return jax.lax.cond(
        is_port,
        apply_port,
        apply_population_build,
        state
    )


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


def _apply_research_tech(state: GameState, decoded: dict) -> GameState:
    """Applique une recherche technologique."""
    tech_id_val = decoded.get("unit_type", -1)
    tech_id = (
        jnp.array(tech_id_val, dtype=jnp.int32)
        if isinstance(tech_id_val, int) and tech_id_val >= 0
        else jnp.array(-1, dtype=jnp.int32)
    )
    
    def do_research(state):
        already_unlocked = state.player_techs[state.current_player, tech_id]
        cost = TECH_COST[tech_id]
        has_cost = cost > 0
        has_stars = state.player_stars[state.current_player] >= cost
        deps_met = _tech_dependencies_met_state(state, tech_id)
        can_research = (~already_unlocked) & has_cost & has_stars & deps_met
        
        def perform(state):
            new_player_techs = state.player_techs.at[state.current_player, tech_id].set(True)
            new_player_stars = state.player_stars.at[state.current_player].add(-cost)
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


def _perform_combat(state: GameState, attacker_id: int, target_id: int, distance: jnp.ndarray) -> GameState:
    """Effectue un combat entre deux unités."""
    attacker_type = state.units_type[attacker_id]
    target_type = state.units_type[target_id]
    
    # Récupérer les stats (utiliser les arrays JAX)
    attacker_attack = UNIT_ATTACK[attacker_type]
    attacker_defense = UNIT_DEFENSE[attacker_type]
    target_attack = UNIT_ATTACK[target_type]
    target_defense = UNIT_DEFENSE[target_type]
    
    # Calculer les dégâts
    # Dégâts = Attaque - Défense (minimum 1)
    damage_to_target = jnp.maximum(1, attacker_attack - target_defense)
    damage_to_attacker = jnp.maximum(1, target_attack - attacker_defense)
    attacker_range = UNIT_ATTACK_RANGE[attacker_type]
    allow_retaliation = jnp.where(attacker_range > 1, distance <= 1, True)
    counter_damage = jnp.where(allow_retaliation, damage_to_attacker, 0)
    
    # Appliquer les dégâts
    new_units_hp = state.units_hp.at[target_id].add(-damage_to_target)
    new_units_hp = new_units_hp.at[attacker_id].add(-counter_damage)
    
    # Vérifier si des unités sont détruites
    target_dead = new_units_hp[target_id] <= 0
    attacker_dead = new_units_hp[attacker_id] <= 0
    
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
        can_train = is_city & is_owner & has_cost & has_stars
        
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
    
    state = state.replace(
        player_stars=new_player_stars,
        current_player=next_player,
        turn=new_turn,
        units_has_acted=jnp.zeros_like(state.units_has_acted),
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
    
    # Vérifier la victoire
    state = _check_victory(state)
    
    return state


def _check_victory(state: GameState) -> GameState:
    """Vérifie les conditions de victoire (élimination)."""
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
    domination_done = num_alive <= 1
    
    perfection_limit_reached = (
        (state.game_mode == GameMode.PERFECTION)
        & (state.turn >= state.max_turns)
    )
    
    is_done = domination_done | perfection_limit_reached
    
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
    """Calcule le niveau d'une ville en fonction de sa population."""
    thresholds = CITY_LEVEL_POP_THRESHOLDS
    level_three = thresholds[3]
    level_two = thresholds[2]
    level_one = thresholds[1]
    
    level = jnp.where(population >= level_three, 3, 0)
    level = jnp.where((population >= level_two) & (population < level_three), 2, level)
    level = jnp.where((population >= level_one) & (population < level_two), 1, level)
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
    bonus = state.player_income_bonus[player_id]
    return base_income + bonus


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
    available_techs = (
        (TECH_COST > 0)
        & (~player_techs_row)
        & deps_met
        & (stars >= TECH_COST)
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
