"""Définition de GameState - État du jeu comme pytree JAX."""

from typing import Optional
import jax
import jax.numpy as jnp
from flax import struct


# Constantes de terrain
class TerrainType:
    """Types de terrain."""
    PLAIN = 0
    FOREST = 1
    MOUNTAIN = 2
    WATER_SHALLOW = 3
    WATER_DEEP = 4
    NUM_TYPES = 5


# Constantes de types d'unités
class UnitType:
    """Types d'unités."""
    NONE = 0
    WARRIOR = 1
    NUM_TYPES = 2


# Constantes de propriétaires
NO_OWNER = -1


@struct.dataclass
class GameState:
    """État du jeu représenté comme un pytree JAX.
    
    Tous les champs sont des arrays JAX pour permettre jit et vmap.
    """
    # Terrain
    terrain: jnp.ndarray  # [H, W] - types de terrain
    
    # Villes
    city_owner: jnp.ndarray  # [H, W] - propriétaire de chaque ville (-1 si pas de ville)
    city_level: jnp.ndarray  # [H, W] - niveau de chaque ville (0 si pas de ville)
    
    # Unités
    units_type: jnp.ndarray  # [N_units_max] - types d'unités
    units_pos: jnp.ndarray  # [N_units_max, 2] - positions (x, y)
    units_hp: jnp.ndarray  # [N_units_max] - points de vie
    units_owner: jnp.ndarray  # [N_units_max] - propriétaire de chaque unité
    units_active: jnp.ndarray  # [N_units_max] - booléen indiquant si l'unité existe
    
    # État du jeu
    current_player: jnp.ndarray  # joueur actif (scalaire)
    turn: jnp.ndarray  # numéro de tour (scalaire)
    done: jnp.ndarray  # partie terminée (scalaire booléen)
    
    # Dimensions
    height: int
    width: int
    max_units: int
    num_players: int

    @classmethod
    def create_empty(
        cls,
        height: int,
        width: int,
        max_units: int,
        num_players: int,
    ) -> "GameState":
        """Crée un état de jeu vide."""
        return cls(
            terrain=jnp.zeros((height, width), dtype=jnp.int32),
            city_owner=jnp.full((height, width), NO_OWNER, dtype=jnp.int32),
            city_level=jnp.zeros((height, width), dtype=jnp.int32),
            units_type=jnp.zeros(max_units, dtype=jnp.int32),
            units_pos=jnp.zeros((max_units, 2), dtype=jnp.int32),
            units_hp=jnp.zeros(max_units, dtype=jnp.int32),
            units_owner=jnp.full(max_units, NO_OWNER, dtype=jnp.int32),
            units_active=jnp.zeros(max_units, dtype=jnp.bool_),
            current_player=jnp.array(0, dtype=jnp.int32),
            turn=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            height=height,
            width=width,
            max_units=max_units,
            num_players=num_players,
        )


