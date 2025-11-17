"""Tests pour core/actions.py."""

import pytest
import jax.numpy as jnp
from polytopia_jax.core.actions import (
    ActionType,
    Direction,
    encode_action,
    decode_action,
    get_action_direction_delta,
    DIRECTION_DELTA,
    NO_ACTION,
    END_TURN_ACTION,
)


def test_encode_decode_action():
    """Test l'encodage et décodage d'actions."""
    # Test NO_OP
    action = encode_action(ActionType.NO_OP)
    decoded = decode_action(action)
    assert decoded["action_type"] == ActionType.NO_OP
    # unit_id peut être 0 ou None selon l'encodage
    assert decoded["unit_id"] is None or decoded["unit_id"] == 0
    
    # Test MOVE avec unit_id et direction
    action = encode_action(
        ActionType.MOVE,
        unit_id=5,
        direction=Direction.RIGHT
    )
    decoded = decode_action(action)
    assert decoded["action_type"] == ActionType.MOVE
    assert decoded["unit_id"] == 5
    assert decoded["direction"] == Direction.RIGHT
    
    # Test ATTACK avec target_pos
    action = encode_action(
        ActionType.ATTACK,
        unit_id=10,
        target_pos=(15, 20)
    )
    decoded = decode_action(action)
    assert decoded["action_type"] == ActionType.ATTACK
    assert decoded["unit_id"] == 10
    assert decoded["target_pos"] == (15, 20)
    
    # Test TRAIN_UNIT avec unit_type
    action = encode_action(
        ActionType.TRAIN_UNIT,
        unit_type=1
    )
    decoded = decode_action(action)
    assert decoded["action_type"] == ActionType.TRAIN_UNIT
    assert decoded["unit_type"] == 1


def test_all_action_types():
    """Vérifie que toutes les actions peuvent être encodées."""
    for action_type in ActionType:
        if action_type == ActionType.NUM_ACTIONS:
            continue
        action = encode_action(action_type)
        decoded = decode_action(action)
        assert decoded["action_type"] == action_type


def test_directions():
    """Test les directions et leurs deltas."""
    # Test toutes les directions
    for direction in Direction:
        if direction == Direction.NUM_DIRECTIONS:
            continue
        delta = get_action_direction_delta(direction)
        assert delta.shape == (2,)
        assert delta.dtype == jnp.int32
    
    # Vérifier les valeurs spécifiques
    assert jnp.array_equal(
        get_action_direction_delta(Direction.UP),
        jnp.array([0, -1])
    )
    assert jnp.array_equal(
        get_action_direction_delta(Direction.RIGHT),
        jnp.array([1, 0])
    )
    assert jnp.array_equal(
        get_action_direction_delta(Direction.DOWN),
        jnp.array([0, 1])
    )
    assert jnp.array_equal(
        get_action_direction_delta(Direction.LEFT),
        jnp.array([-1, 0])
    )


def test_action_with_parameters():
    """Test les actions avec différents paramètres."""
    # MOVE avec tous les paramètres
    action = encode_action(
        ActionType.MOVE,
        unit_id=42,
        direction=Direction.DOWN,
        target_pos=(10, 15)
    )
    decoded = decode_action(action)
    assert decoded["action_type"] == ActionType.MOVE
    assert decoded["unit_id"] == 42
    assert decoded["direction"] == Direction.DOWN
    assert decoded["target_pos"] == (10, 15)
    
    # ATTACK avec unit_id et target_pos
    action = encode_action(
        ActionType.ATTACK,
        unit_id=7,
        target_pos=(5, 5)
    )
    decoded = decode_action(action)
    assert decoded["action_type"] == ActionType.ATTACK
    assert decoded["unit_id"] == 7
    assert decoded["target_pos"] == (5, 5)


def test_constants():
    """Test les constantes d'actions."""
    assert decode_action(NO_ACTION)["action_type"] == ActionType.NO_OP
    assert decode_action(END_TURN_ACTION)["action_type"] == ActionType.END_TURN


def test_direction_delta_array():
    """Test que DIRECTION_DELTA est correct."""
    assert DIRECTION_DELTA.shape == (4, 2)
    assert DIRECTION_DELTA.dtype == jnp.int32
    
    # Vérifier chaque direction
    expected = jnp.array([
        [0, -1],  # UP
        [1, 0],   # RIGHT
        [0, 1],   # DOWN
        [-1, 0],  # LEFT
    ], dtype=jnp.int32)
    
    assert jnp.array_equal(DIRECTION_DELTA, expected)

