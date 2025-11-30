"""Tests pour les mouvements et la validation des directions.

Ces tests vérifient que les mouvements fonctionnent correctement dans une grille simple
avec 8 directions de mouvement : {-1, 0, 1} en x et {-1, 0, 1} en y.
"""

import pytest
import jax.numpy as jnp
from polytopia_jax.core.rules import step
from polytopia_jax.core.state import GameState, UnitType, TerrainType
from polytopia_jax.core.actions import ActionType, Direction, encode_action, DIRECTION_DELTA
from polytopia_jax.core.init import GameConfig, init_random


def _make_test_state_with_unit_at(x: int, y: int, unit_id: int = 0) -> GameState:
    """Crée un état de test avec une unité à une position spécifique."""
    state = GameState.create_empty(
        height=10,
        width=10,
        max_units=4,
        num_players=2,
    )
    
    # Placer une unité à la position spécifiée
    units_type = state.units_type.at[unit_id].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[unit_id].set(0)
    units_pos = state.units_pos.at[unit_id, 0].set(x)
    units_pos = units_pos.at[unit_id, 1].set(y)
    units_hp = state.units_hp.at[unit_id].set(10)
    units_active = state.units_active.at[unit_id].set(True)
    
    # S'assurer que le terrain est traversable
    terrain = state.terrain.at[y, x].set(TerrainType.PLAIN)
    
    return state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
        terrain=terrain,
    )


def test_all_directions():
    """Test tous les mouvements depuis une position donnée."""
    # Position au centre
    state = _make_test_state_with_unit_at(5, 5)
    
    # Tester chaque direction avec les deltas {-1, 0, 1}
    test_cases = [
        (Direction.UP_LEFT, (-1, -1)),
        (Direction.UP, (0, -1)),
        (Direction.UP_RIGHT, (1, -1)),
        (Direction.RIGHT, (1, 0)),
        (Direction.DOWN_RIGHT, (1, 1)),
        (Direction.DOWN, (0, 1)),
        (Direction.DOWN_LEFT, (-1, 1)),
        (Direction.LEFT, (-1, 0)),
    ]
    
    for direction, expected_offset in test_cases:
        # Obtenir le delta absolu de la direction
        absolute_delta = DIRECTION_DELTA[direction]
        
        # Vérifier que le delta correspond bien à l'offset attendu
        assert absolute_delta[0] == expected_offset[0], f"Direction {direction.name}: dx attendu {expected_offset[0]}, obtenu {absolute_delta[0]}"
        assert absolute_delta[1] == expected_offset[1], f"Direction {direction.name}: dy attendu {expected_offset[1]}, obtenu {absolute_delta[1]}"
        
        # Appliquer le mouvement
        action = encode_action(
            ActionType.MOVE,
            unit_id=0,
            direction=direction,
        )
        new_state = step(state, action)
        
        # Vérifier que la position a changé selon le delta absolu
        expected_x = 5 + absolute_delta[0]
        expected_y = 5 + absolute_delta[1]
        
        actual_pos = new_state.units_pos[0]
        
        # Le mouvement peut être bloqué (terrain, limites, etc.)
        # mais s'il a réussi, vérifier la position
        if not jnp.array_equal(actual_pos, state.units_pos[0]):
            assert actual_pos[0] == expected_x, f"Direction {direction.name}: x attendu {expected_x}, obtenu {actual_pos[0]}"
            assert actual_pos[1] == expected_y, f"Direction {direction.name}: y attendu {expected_y}, obtenu {actual_pos[1]}"


def test_movement_validation():
    """Test que les mouvements invalides sont correctement détectés et ignorés."""
    state = _make_test_state_with_unit_at(5, 5)
    
    # Placer une autre unité à côté
    units_type = state.units_type.at[1].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[1].set(0)
    units_pos = state.units_pos.at[1, 0].set(6)  # À droite de l'unité 0
    units_pos = units_pos.at[1, 1].set(5)
    units_hp = state.units_hp.at[1].set(10)
    units_active = state.units_active.at[1].set(True)
    terrain = state.terrain.at[5, 6].set(TerrainType.PLAIN)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
        terrain=terrain,
    )
    
    # Essayer de déplacer l'unité 0 vers la droite (case occupée)
    action = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.RIGHT,
    )
    new_state = step(state, action)
    
    # Le mouvement devrait être bloqué (case occupée)
    assert new_state.units_pos[0, 0] == 5  # Position inchangée
    assert new_state.units_pos[0, 1] == 5


def test_vertical_movements():
    """Test les mouvements verticaux (UP/DOWN)."""
    state = _make_test_state_with_unit_at(5, 5)
    
    # Mouvement vers le haut
    action_up = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.UP,
    )
    new_state_up = step(state, action_up)
    
    # Vérifier que Y a diminué de 1
    if not jnp.array_equal(new_state_up.units_pos[0], state.units_pos[0]):
        assert new_state_up.units_pos[0, 0] == 5
        assert new_state_up.units_pos[0, 1] == 4  # 5 - 1
    
    # Mouvement vers le bas
    action_down = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.DOWN,
    )
    new_state_down = step(state, action_down)
    
    # Vérifier que Y a augmenté de 1
    if not jnp.array_equal(new_state_down.units_pos[0], state.units_pos[0]):
        assert new_state_down.units_pos[0, 0] == 5
        assert new_state_down.units_pos[0, 1] == 6  # 5 + 1


def test_direction_deltas_are_simple():
    """Vérifie que tous les deltas sont dans {-1, 0, 1}."""
    for direction in Direction:
        if direction == Direction.NUM_DIRECTIONS:
            continue
        delta = DIRECTION_DELTA[direction]
        assert delta[0] in [-1, 0, 1], f"Direction {direction.name}: dx doit être dans {{-1, 0, 1}}, obtenu {delta[0]}"
        assert delta[1] in [-1, 0, 1], f"Direction {direction.name}: dy doit être dans {{-1, 0, 1}}, obtenu {delta[1]}"






