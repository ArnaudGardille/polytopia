"""Encodage des actions discrètes."""

from enum import IntEnum
from typing import Tuple, Optional
import jax.numpy as jnp


class ActionType(IntEnum):
    """Types d'actions possibles."""
    NO_OP = 0
    MOVE = 1
    ATTACK = 2
    TRAIN_UNIT = 3
    BUILD = 4
    RESEARCH_TECH = 5
    END_TURN = 6
    HARVEST_RESOURCE = 7
    NUM_ACTIONS = 8


# Directions pour les mouvements (8 directions)
class Direction(IntEnum):
    """Directions de mouvement."""
    UP = 0  # (0, -1)
    UP_RIGHT = 1  # (1, -1)
    RIGHT = 2  # (1, 0)
    DOWN_RIGHT = 3  # (1, 1)
    DOWN = 4  # (0, 1)
    DOWN_LEFT = 5  # (-1, 1)
    LEFT = 6  # (-1, 0)
    UP_LEFT = 7  # (-1, -1)
    NUM_DIRECTIONS = 8


# Mapping direction -> (dx, dy)
# Système de grille simple avec 8 directions : {-1, 0, 1} en x et {-1, 0, 1} en y
DIRECTION_DELTA = jnp.array([
    [0, -1],   # UP
    [1, -1],   # UP_RIGHT
    [1, 0],    # RIGHT
    [1, 1],    # DOWN_RIGHT
    [0, 1],    # DOWN
    [-1, 1],   # DOWN_LEFT
    [-1, 0],   # LEFT
    [-1, -1],  # UP_LEFT
], dtype=jnp.int32)


def encode_action(
    action_type: int,
    unit_id: Optional[int] = None,
    direction: Optional[int] = None,
    target_pos: Optional[Tuple[int, int]] = None,
    unit_type: Optional[int] = None,
) -> int:
    """Encode une action en un entier.
    
    Format d'encodage simple:
    - ActionType: 3 bits (0-6)
    - unit_id: 8 bits (0-255)
    - direction: 3 bits (0-7)
    - target_x: 6 bits (0-63)
    - target_y: 6 bits (0-63)
    - unit_type: 4 bits (0-15)
    
    Total: 30 bits (fits in int32)
    
    Args:
        action_type: Type d'action (ActionType)
        unit_id: ID de l'unité concernée (None pour actions globales)
        direction: Direction pour MOVE (None si non applicable)
        target_pos: Position cible (x, y) pour certaines actions
        unit_type: Type d'unité pour TRAIN_UNIT
    
    Returns:
        Action encodée comme entier
    """
    encoded = action_type
    
    if unit_id is not None:
        encoded |= (unit_id & 0xFF) << 3
    
    if direction is not None:
        encoded |= (direction & 0x7) << 11
    
    if target_pos is not None:
        x, y = target_pos
        encoded |= (x & 0x3F) << 14
        encoded |= (y & 0x3F) << 20
    
    if unit_type is not None:
        encoded |= (unit_type & 0xF) << 26
    
    return encoded


def decode_action(action_id: int) -> dict:
    """Décode une action encodée.
    
    Cette fonction essaie de détecter si elle est appelée dans un contexte tracé.
    Si oui, elle utilise des opérations JAX pures. Sinon, elle utilise du Python normal.
    
    Args:
        action_id: Action encodée (peut être int Python ou array JAX)
    
    Returns:
        Dictionnaire avec les champs de l'action
    """
    # Convertir en array JAX pour les opérations
    action_id_array = jnp.asarray(action_id, dtype=jnp.int32)
    
    # Extraire les champs avec opérations JAX pures
    action_type = action_id_array & 0x7
    unit_id_raw = (action_id_array >> 3) & 0xFF
    direction_raw = (action_id_array >> 11) & 0x7
    target_x_raw = (action_id_array >> 14) & 0x3F
    target_y_raw = (action_id_array >> 20) & 0x3F
    unit_type_raw = (action_id_array >> 26) & 0xF
    
    # Utiliser jnp.where pour gérer les valeurs invalides (valeurs sentinelles -1)
    # Ces valeurs seront converties en None dans le dict si elles sont invalides
    unit_id_val = jnp.where(unit_id_raw < 255, unit_id_raw, -1)
    direction_val = jnp.where(direction_raw < Direction.NUM_DIRECTIONS, direction_raw, -1)
    target_x_val = jnp.where((target_x_raw < 64) & (target_y_raw < 64), target_x_raw, -1)
    target_y_val = jnp.where((target_x_raw < 64) & (target_y_raw < 64), target_y_raw, -1)
    unit_type_val = jnp.where(unit_type_raw > 0, unit_type_raw, -1)
    
    # Essayer de convertir en Python pour le dict
    # Si on est dans un contexte tracé, cela échouera mais on peut utiliser une approche différente
    try:
        # Conversion normale (hors traçage)
        action_type_int = int(action_type.item() if hasattr(action_type, 'item') else action_type)
        unit_id_int = int(unit_id_val.item() if hasattr(unit_id_val, 'item') else unit_id_val)
        direction_int = int(direction_val.item() if hasattr(direction_val, 'item') else direction_val)
        target_x_int = int(target_x_val.item() if hasattr(target_x_val, 'item') else target_x_val)
        target_y_int = int(target_y_val.item() if hasattr(target_y_val, 'item') else target_y_val)
        unit_type_int = int(unit_type_val.item() if hasattr(unit_type_val, 'item') else unit_type_val)
        
        result = {
            "action_type": action_type_int,
            "unit_id": unit_id_int if unit_id_int >= 0 else None,
            "direction": direction_int if direction_int >= 0 else None,
            "target_pos": (target_x_int, target_y_int) if target_x_int >= 0 and target_y_int >= 0 else None,
            "unit_type": unit_type_int if unit_type_int >= 0 else None,
        }
    except (AttributeError, TypeError):
        # On est dans un contexte tracé, retourner les valeurs JAX directement
        # Le code appelant devra gérer les valeurs sentinelles
        result = {
            "action_type": action_type,
            "unit_id": unit_id_val,
            "direction": direction_val,
            "target_pos": (target_x_val, target_y_val),
            "unit_type": unit_type_val,
        }
    
    return result


def get_action_direction_delta(direction: int) -> jnp.ndarray:
    """Retourne le delta (dx, dy) pour une direction donnée.
    
    Args:
        direction: Direction (Direction enum value)
    
    Returns:
        Array [dx, dy]
    """
    return DIRECTION_DELTA[direction]


# Constantes pour les actions spéciales
NO_ACTION = encode_action(ActionType.NO_OP)
END_TURN_ACTION = encode_action(ActionType.END_TURN)
