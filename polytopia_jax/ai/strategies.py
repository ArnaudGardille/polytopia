"""IA basées sur des stratégies configurables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import random

import numpy as np

from polytopia_jax.core.actions import (
    ActionType,
    Direction,
    DIRECTION_DELTA,
    encode_action,
    END_TURN_ACTION,
)
from polytopia_jax.core.rules import (
    UNIT_ATTACK_RANGE,
    UNIT_COST,
    BuildingType,
    BUILDING_COST,
    TECH_COST,
)
from polytopia_jax.core.state import GameState, TerrainType, UnitType, TechType


StrategyName = str


@dataclass(frozen=True)
class DifficultyPreset:
    """Paramètres de bonus par niveau de difficulté."""

    name: str
    star_bonus: int


DIFFICULTY_PRESETS: Dict[str, DifficultyPreset] = {
    "easy": DifficultyPreset("easy", 0),
    "normal": DifficultyPreset("normal", 0),
    "hard": DifficultyPreset("hard", 2),
    "crazy": DifficultyPreset("crazy", 4),
}


def resolve_difficulty(name: Optional[str]) -> DifficultyPreset:
    if not name:
        return DIFFICULTY_PRESETS["normal"]
    preset = DIFFICULTY_PRESETS.get(name.lower())
    return preset if preset else DIFFICULTY_PRESETS["normal"]

AVAILABLE_STRATEGIES: Dict[StrategyName, str] = {
    "idle": "DoNothingStrategy",
    "random": "RandomStrategy",
    "rush": "RushStrategy",
    "economy": "EconomyTechStrategy",
}


@dataclass
class AIContext:
    """Données pré-calculées pour accélérer les décisions."""

    player_units: List[int]
    unit_positions: Dict[int, Tuple[int, int]]
    occupancy: Dict[Tuple[int, int], int]
    player_cities: List[Tuple[int, int]]
    enemy_cities: List[Tuple[int, int]]
    enemy_units: List[Tuple[int, int]]


def resolve_strategy_name(strategy_name: Optional[str]) -> StrategyName:
    if not strategy_name:
        return "rush"
    name = strategy_name.lower()
    return name if name in AVAILABLE_STRATEGIES else "rush"


class AIStrategy:
    """Interface des stratégies."""

    name: StrategyName = "rush"

    def select_action(
        self,
        state: GameState,
        context: AIContext,
        player_id: int,
        rng: random.Random,
    ) -> int:
        raise NotImplementedError


class DoNothingStrategy(AIStrategy):
    name = "idle"

    def select_action(self, state, context, player_id, rng):
        return END_TURN_ACTION


class RandomStrategy(AIStrategy):
    name = "random"

    def select_action(self, state, context, player_id, rng):
        candidates: List[int] = []
        candidates.extend(_gather_all_attack_actions(state, context, player_id))
        candidates.extend(_gather_all_move_actions(state, context, player_id, rng))
        candidates.extend(_gather_all_train_actions(state, context, player_id, rng))
        candidates.extend(_gather_all_build_actions(state, context, player_id))
        candidates.extend(_gather_all_research_actions(state, context, player_id))
        candidates.append(END_TURN_ACTION)
        return rng.choice(candidates)


class RushStrategy(AIStrategy):
    name = "rush"

    def select_action(self, state, context, player_id, rng):
        attack = _find_first_attack(state, context, player_id)
        if attack is not None:
            return attack
        move = _move_towards_enemy(state, context, player_id, rng)
        if move is not None:
            return move
        train = _train_basic_unit(state, context, player_id, rng)
        if train is not None:
            return train
        random_move = _random_move(state, context, player_id, rng)
        if random_move is not None:
            return random_move
        return END_TURN_ACTION


class EconomyTechStrategy(AIStrategy):
    name = "economy"
    TECH_ORDER = [TechType.MINING, TechType.CLIMBING, TechType.SAILING]
    BUILD_ORDER = [BuildingType.FARM, BuildingType.HUT, BuildingType.MINE]

    def select_action(self, state, context, player_id, rng):
        research = _research_next(state, player_id, self.TECH_ORDER)
        if research is not None:
            return research
        build = _build_population_structure(
            state,
            context,
            player_id,
            self.BUILD_ORDER,
        )
        if build is not None:
            return build
        attack = _find_first_attack(state, context, player_id)
        if attack is not None:
            return attack
        move = _move_towards_enemy(state, context, player_id, rng)
        if move is not None:
            return move
        train = _train_basic_unit(state, context, player_id, rng)
        if train is not None:
            return train
        return END_TURN_ACTION


STRATEGY_CLASSES: Dict[StrategyName, AIStrategy] = {
    "idle": DoNothingStrategy(),
    "random": RandomStrategy(),
    "rush": RushStrategy(),
    "economy": EconomyTechStrategy(),
}


class StrategyAI:
    """IA paramétrable par stratégie."""

    def __init__(
        self,
        player_id: int,
        strategy_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.player_id = player_id
        self.strategy_name = resolve_strategy_name(strategy_name)
        self.strategy = STRATEGY_CLASSES[self.strategy_name]
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState, context: Optional[AIContext] = None) -> int:
        if bool(np.asarray(state.done)):
            return END_TURN_ACTION
        current_player = int(np.asarray(state.current_player))
        if current_player != self.player_id:
            return END_TURN_ACTION
        if context is None:
            context = _build_context(state, self.player_id)
        if not context.player_units:
            train = _train_basic_unit(state, context, self.player_id, self._rng)
            return train if train is not None else END_TURN_ACTION
        return self.strategy.select_action(
            state,
            context,
            self.player_id,
            self._rng,
        )


def _build_context(state: GameState, player_id: int) -> AIContext:
    """Construit le contexte de l'IA de manière optimisée."""
    units_active = np.array(state.units_active, copy=False)
    units_owner = np.array(state.units_owner, copy=False)
    units_pos = np.array(state.units_pos, copy=False)

    # Trouver les indices des unités actives (optimisé avec np.where)
    active_indices = np.where(units_active)[0]
    
    player_units: List[int] = []
    unit_positions: Dict[int, Tuple[int, int]] = {}
    occupancy: Dict[Tuple[int, int], int] = {}
    enemy_units: List[Tuple[int, int]] = []

    # Parcourir uniquement les unités actives
    for unit_id in active_indices:
        pos = (
            int(units_pos[unit_id, 0]),
            int(units_pos[unit_id, 1]),
        )
        unit_positions[unit_id] = pos
        occupancy[pos] = unit_id
        owner = int(units_owner[unit_id])
        if owner == player_id:
            player_units.append(unit_id)
        else:
            enemy_units.append(pos)

    # Optimiser la recherche des villes avec np.where
    city_owner = np.array(state.city_owner, copy=False)
    city_level = np.array(state.city_level, copy=False)
    
    # Trouver les positions des villes (niveau > 0)
    city_y, city_x = np.where(city_level > 0)
    
    player_cities: List[Tuple[int, int]] = []
    enemy_cities: List[Tuple[int, int]] = []

    for idx in range(len(city_y)):
        y, x = int(city_y[idx]), int(city_x[idx])
        owner = int(city_owner[y, x])
        if owner == player_id:
            player_cities.append((x, y))
        elif owner >= 0:
            enemy_cities.append((x, y))

    return AIContext(
        player_units=player_units,
        unit_positions=unit_positions,
        occupancy=occupancy,
        player_cities=player_cities,
        enemy_cities=enemy_cities,
        enemy_units=enemy_units,
    )


def _gather_all_attack_actions(
    state: GameState,
    context: AIContext,
    player_id: int,
) -> List[int]:
    actions = []
    for unit_id in context.player_units:
        action = _attack_with_unit(state, context, player_id, unit_id)
        if action is not None:
            actions.append(action)
    return actions


def _gather_all_move_actions(
    state: GameState,
    context: AIContext,
    player_id: int,
    rng: random.Random,
) -> List[int]:
    actions = []
    directions = [d for d in Direction if d != Direction.NUM_DIRECTIONS]
    for unit_id in context.player_units:
        shuffled = directions[:]
        rng.shuffle(shuffled)
        for direction in shuffled:
            candidate = _move_in_direction(state, context, unit_id, int(direction))
            if candidate is not None:
                actions.append(candidate)
                break
    return actions


def _gather_all_train_actions(
    state: GameState,
    context: AIContext,
    player_id: int,
    rng: random.Random,
) -> List[int]:
    action = _train_basic_unit(state, context, player_id, rng)
    return [action] if action is not None else []


def _gather_all_build_actions(
    state: GameState,
    context: AIContext,
    player_id: int,
) -> List[int]:
    action = _build_population_structure(
        state,
        context,
        player_id,
        [BuildingType.FARM, BuildingType.HUT],
    )
    return [action] if action is not None else []


def _gather_all_research_actions(
    state: GameState,
    context: AIContext,
    player_id: int,
) -> List[int]:
    action = _research_next(state, player_id, [TechType.MINING, TechType.CLIMBING])
    return [action] if action is not None else []


def _find_first_attack(
    state: GameState,
    context: AIContext,
    player_id: int,
) -> Optional[int]:
    for unit_id in context.player_units:
        action = _attack_with_unit(state, context, player_id, unit_id)
        if action is not None:
            return action
    return None


def _attack_with_unit(
    state: GameState,
    context: AIContext,
    player_id: int,
    unit_id: int,
) -> Optional[int]:
    unit_type = int(np.asarray(state.units_type[unit_id]))
    attack_range = max(1, int(np.asarray(UNIT_ATTACK_RANGE[unit_type])))
    unit_pos = context.unit_positions.get(unit_id)
    if unit_pos is None:
        return None

    for rng in range(1, attack_range + 1):
        for dx, dy in DIRECTION_DELTA:
            target = (unit_pos[0] + int(dx) * rng, unit_pos[1] + int(dy) * rng)
            if not _is_in_bounds(state, target[0], target[1]):
                continue
            occupant = context.occupancy.get(target)
            if occupant is None:
                continue
            owner = int(np.asarray(state.units_owner[occupant]))
            if owner == player_id:
                continue
            return encode_action(
                ActionType.ATTACK,
                unit_id=unit_id,
                target_pos=target,
            )
    return None


def _move_towards_enemy(
    state: GameState,
    context: AIContext,
    player_id: int,
    rng: random.Random,
) -> Optional[int]:
    targets = context.enemy_cities or context.enemy_units
    if not targets:
        return None
    for unit_id in context.player_units:
        unit_pos = context.unit_positions[unit_id]
        target = _nearest_target(unit_pos, targets)
        if target is None:
            continue
        action = _move_unit_towards(state, context, unit_id, target, rng)
        if action is not None:
            return action
    return None


def _nearest_target(
    unit_pos: Tuple[int, int],
    targets: Iterable[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    best_target = None
    best_distance = None
    for target in targets:
        distance = abs(unit_pos[0] - target[0]) + abs(unit_pos[1] - target[1])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_target = target
    return best_target


def _move_unit_towards(
    state: GameState,
    context: AIContext,
    unit_id: int,
    target: Tuple[int, int],
    rng: random.Random,
) -> Optional[int]:
    unit_pos = context.unit_positions.get(unit_id)
    if unit_pos is None:
        return None
    directions = list(Direction)
    directions = [d for d in directions if d != Direction.NUM_DIRECTIONS]

    def sort_key(direction):
        dx, dy = DIRECTION_DELTA[int(direction)]
        new_pos = (unit_pos[0] + int(dx), unit_pos[1] + int(dy))
        return abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])

    directions.sort(key=sort_key)
    for direction in directions:
        candidate = _move_in_direction(state, context, unit_id, int(direction))
        if candidate is not None:
            return candidate
    rng.shuffle(directions)
    for direction in directions:
        candidate = _move_in_direction(state, context, unit_id, int(direction))
        if candidate is not None:
            return candidate
    return None


def _move_in_direction(
    state: GameState,
    context: AIContext,
    unit_id: int,
    direction: int,
) -> Optional[int]:
    unit_pos = context.unit_positions.get(unit_id)
    if unit_pos is None:
        return None
    dx, dy = DIRECTION_DELTA[direction]
    target = (unit_pos[0] + int(dx), unit_pos[1] + int(dy))
    if not _is_in_bounds(state, target[0], target[1]):
        return None
    if target in context.occupancy:
        return None
    if not _is_traversable(state, target[0], target[1]):
        return None
    return encode_action(
        ActionType.MOVE,
        unit_id=unit_id,
        direction=direction,
    )


def _random_move(
    state: GameState,
    context: AIContext,
    player_id: int,
    rng: random.Random,
) -> Optional[int]:
    directions = [d for d in Direction if d != Direction.NUM_DIRECTIONS]
    rng.shuffle(context.player_units)
    for unit_id in context.player_units:
        rng.shuffle(directions)
        for direction in directions:
            candidate = _move_in_direction(state, context, unit_id, int(direction))
            if candidate is not None:
                return candidate
    return None


def _train_basic_unit(
    state: GameState,
    context: AIContext,
    player_id: int,
    rng: random.Random,
) -> Optional[int]:
    stars = int(np.asarray(state.player_stars[player_id]))
    cost = int(np.asarray(UNIT_COST[int(UnitType.WARRIOR)]))
    if stars < cost or not context.player_cities:
        return None
    rng.shuffle(context.player_cities)
    for city in context.player_cities:
        if city in context.occupancy:
            continue
        return encode_action(
            ActionType.TRAIN_UNIT,
            unit_type=int(UnitType.WARRIOR),
            target_pos=city,
        )
    return None


def _research_next(
    state: GameState,
    player_id: int,
    tech_order: List[TechType],
) -> Optional[int]:
    stars = int(np.asarray(state.player_stars[player_id]))
    player_row = np.array(state.player_techs[player_id], copy=True)
    for tech in tech_order:
        tech_idx = int(tech)
        if player_row[tech_idx]:
            continue
        cost = int(np.asarray(TECH_COST[tech_idx]))
        if cost <= 0 or stars < cost:
            continue
        return encode_action(
            ActionType.RESEARCH_TECH,
            unit_type=tech_idx,
        )
    return None


def _build_population_structure(
    state: GameState,
    context: AIContext,
    player_id: int,
    building_order: List[BuildingType],
) -> Optional[int]:
    if not context.player_cities:
        return None
    stars = int(np.asarray(state.player_stars[player_id]))
    for building in building_order:
        building_idx = int(building)
        cost = int(np.asarray(BUILDING_COST[building_idx]))
        if cost <= 0 or stars < cost:
            continue
        for city in context.player_cities:
            return encode_action(
                ActionType.BUILD,
                unit_type=building_idx,
                target_pos=city,
            )
    return None


def _is_in_bounds(state: GameState, x: int, y: int) -> bool:
    return 0 <= x < state.width and 0 <= y < state.height


def _is_traversable(state: GameState, x: int, y: int) -> bool:
    terrain = int(np.asarray(state.terrain[y, x]))
    return terrain in (TerrainType.PLAIN, TerrainType.FOREST)
