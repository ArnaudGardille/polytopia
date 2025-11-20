"""Heuristiques basiques pour piloter les IA."""

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
from polytopia_jax.core.rules import UNIT_ATTACK_RANGE, UNIT_COST
from polytopia_jax.core.state import GameState, TerrainType, UnitType


@dataclass(frozen=True)
class DifficultyPreset:
    """Décrit un niveau de difficulté IA."""

    name: str
    star_bonus: int  # bonus de revenu fixe par tour


DIFFICULTY_PRESETS: Dict[str, DifficultyPreset] = {
    "easy": DifficultyPreset(name="easy", star_bonus=0),
    "normal": DifficultyPreset(name="normal", star_bonus=0),
    "hard": DifficultyPreset(name="hard", star_bonus=2),
    "crazy": DifficultyPreset(name="crazy", star_bonus=4),
}


def resolve_difficulty(name: str) -> DifficultyPreset:
    """Retourne le preset correspondant (normal par défaut)."""
    if not name:
        return DIFFICULTY_PRESETS["normal"]
    return DIFFICULTY_PRESETS.get(name.lower(), DIFFICULTY_PRESETS["normal"])


class HeuristicAI:
    """Agent IA très simple priorisant combat/expansion."""

    def __init__(self, player_id: int, seed: Optional[int] = None):
        self.player_id = player_id
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState) -> int:
        """Choisit la prochaine action à jouer."""
        if bool(np.asarray(state.done)):
            return END_TURN_ACTION
        current_player = int(np.asarray(state.current_player))
        if current_player != self.player_id:
            return END_TURN_ACTION

        units_context = self._build_units_context(state)
        player_units = units_context["available_units"]
        occupancy = units_context["occupancy"]

        # 1) attaques prioritaires
        for unit_id in player_units:
            attack_action = self._try_attack(state, unit_id, occupancy)
            if attack_action is not None:
                return attack_action

        # 2) déplacements vers une ville ennemie ou neutre
        for unit_id in player_units:
            move_action = self._try_move_to_city(state, unit_id, occupancy)
            if move_action is not None:
                return move_action

        # 3) recruter une unité si possible pour renforcer le front
        train_action = self._try_train_unit(state, occupancy)
        if train_action is not None:
            return train_action

        # 4) déplacement opportuniste
        for unit_id in player_units:
            random_move = self._try_random_move(state, unit_id, occupancy)
            if random_move is not None:
                return random_move

        return END_TURN_ACTION

    def _build_units_context(self, state: GameState) -> dict:
        """Pré-calculs utilitaires pour inspecter l'état."""
        units_active = np.array(state.units_active, copy=True)
        units_owner = np.array(state.units_owner, copy=True)
        units_has_acted = np.array(state.units_has_acted, copy=True)
        units_pos = np.array(state.units_pos, copy=True)

        available_units: List[int] = []
        occupancy: Dict[Tuple[int, int], int] = {}

        for unit_id in range(state.max_units):
            if not bool(units_active[unit_id]):
                continue
            pos = (
                int(units_pos[unit_id, 0]),
                int(units_pos[unit_id, 1]),
            )
            occupancy[pos] = unit_id
            owner = int(units_owner[unit_id])
            if (
                owner == self.player_id
                and not bool(units_has_acted[unit_id])
            ):
                available_units.append(unit_id)

        return {
            "available_units": available_units,
            "occupancy": occupancy,
        }

    def _try_attack(
        self,
        state: GameState,
        unit_id: int,
        occupancy: Dict[Tuple[int, int], int],
    ) -> Optional[int]:
        unit_type_val = int(np.asarray(state.units_type[unit_id]))
        attack_range = int(np.asarray(UNIT_ATTACK_RANGE[unit_type_val]))
        if attack_range <= 0:
            attack_range = 1
        unit_pos = np.array(state.units_pos[unit_id])

        for target in self._iter_positions_in_range(unit_pos, attack_range):
            if not self._is_in_bounds(state, target[0], target[1]):
                continue
            target_id = occupancy.get(target, -1)
            if target_id < 0:
                continue
            owner = int(np.asarray(state.units_owner[target_id]))
            if owner == self.player_id:
                continue
            return encode_action(
                ActionType.ATTACK,
                unit_id=unit_id,
                target_pos=target,
            )
        return None

    def _iter_positions_in_range(
        self,
        unit_pos: np.ndarray,
        attack_range: int,
    ) -> Iterable[Tuple[int, int]]:
        """Génère les positions atteignables pour une attaque."""
        base_dirs = [tuple(map(int, delta)) for delta in DIRECTION_DELTA]
        ux, uy = int(unit_pos[0]), int(unit_pos[1])

        for rng in range(1, max(1, attack_range) + 1):
            for dx, dy in base_dirs:
                yield (ux + dx * rng, uy + dy * rng)

    def _try_move_to_city(
        self,
        state: GameState,
        unit_id: int,
        occupancy: Dict[Tuple[int, int], int],
    ) -> Optional[int]:
        """Essaie d'avancer vers la ville ennemie/neutre la plus proche."""
        unit_pos = np.array(state.units_pos[unit_id], copy=True)
        target_city = self._find_target_city(state, unit_pos)
        if target_city is None:
            return None
        return self._move_towards(state, unit_id, target_city, occupancy)

    def _find_target_city(
        self,
        state: GameState,
        unit_pos: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        city_owner = np.array(state.city_owner, copy=True)
        city_level = np.array(state.city_level, copy=True)

        best_city = None
        best_score = None
        ux, uy = int(unit_pos[0]), int(unit_pos[1])

        for y in range(state.height):
            for x in range(state.width):
                if city_level[y, x] <= 0:
                    continue
                owner = int(city_owner[y, x])
                if owner == self.player_id:
                    continue
                distance = abs(x - ux) + abs(y - uy)
                score = distance
                if best_score is None or score < best_score:
                    best_score = score
                    best_city = (x, y)
        return best_city

    def _move_towards(
        self,
        state: GameState,
        unit_id: int,
        target: Tuple[int, int],
        occupancy: Dict[Tuple[int, int], int],
    ) -> Optional[int]:
        unit_pos = np.array(state.units_pos[unit_id], copy=True)
        ux, uy = int(unit_pos[0]), int(unit_pos[1])
        tx, ty = target
        best_dir = None
        best_distance = None
        dirs = list(Direction)
        self._rng.shuffle(dirs)

        for direction in dirs:
            if direction == Direction.NUM_DIRECTIONS:
                continue
            delta = tuple(int(v) for v in DIRECTION_DELTA[int(direction)])
            nx, ny = ux + delta[0], uy + delta[1]
            if not self._is_in_bounds(state, nx, ny):
                continue
            if (nx, ny) in occupancy:
                continue
            if not self._is_traversable(state, nx, ny):
                continue
            distance = abs(tx - nx) + abs(ty - ny)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_dir = direction

        if best_dir is None:
            return None

        return encode_action(
            ActionType.MOVE,
            unit_id=unit_id,
            direction=int(best_dir),
        )

    def _try_random_move(
        self,
        state: GameState,
        unit_id: int,
        occupancy: Dict[Tuple[int, int], int],
    ) -> Optional[int]:
        dirs = list(Direction)
        self._rng.shuffle(dirs)
        unit_pos = np.array(state.units_pos[unit_id], copy=True)
        ux, uy = int(unit_pos[0]), int(unit_pos[1])

        for direction in dirs:
            if direction == Direction.NUM_DIRECTIONS:
                continue
            delta = tuple(int(v) for v in DIRECTION_DELTA[int(direction)])
            nx, ny = ux + delta[0], uy + delta[1]
            if not self._is_in_bounds(state, nx, ny):
                continue
            if (nx, ny) in occupancy:
                continue
            if not self._is_traversable(state, nx, ny):
                continue
            return encode_action(
                ActionType.MOVE,
                unit_id=unit_id,
                direction=int(direction),
            )
        return None

    def _try_train_unit(
        self,
        state: GameState,
        occupancy: Dict[Tuple[int, int], int],
    ) -> Optional[int]:
        """Essaie d'entraîner un guerrier sur une ville disponible."""
        stars = int(np.asarray(state.player_stars[self.player_id]))
        warrior_cost = int(np.asarray(UNIT_COST[int(UnitType.WARRIOR)]))
        if stars < warrior_cost:
            return None

        city_owner = np.array(state.city_owner, copy=True)
        city_level = np.array(state.city_level, copy=True)

        player_cities: List[Tuple[int, int]] = []
        for y in range(state.height):
            for x in range(state.width):
                if city_level[y, x] > 0 and int(city_owner[y, x]) == self.player_id:
                    player_cities.append((x, y))

        self._rng.shuffle(player_cities)
        for pos in player_cities:
            if pos in occupancy:
                continue
            return encode_action(
                ActionType.TRAIN_UNIT,
                target_pos=pos,
                unit_type=int(UnitType.WARRIOR),
            )
        return None

    def _is_in_bounds(self, state: GameState, x: int, y: int) -> bool:
        return 0 <= x < state.width and 0 <= y < state.height

    def _is_traversable(self, state: GameState, x: int, y: int) -> bool:
        """Autorise la traversée sur les cases terrestres basiques."""
        terrain = int(np.asarray(state.terrain[y, x]))
        return terrain in (
            TerrainType.PLAIN,
            TerrainType.PLAIN_FRUIT,
            TerrainType.FOREST,
            TerrainType.FOREST_WITH_WILD_ANIMAL,
        )
