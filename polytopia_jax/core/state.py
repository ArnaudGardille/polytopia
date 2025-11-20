"""Définition de GameState - État du jeu comme pytree JAX.

Phase 0 (MVP) : limitations connues
-----------------------------------
- Une seule unité active (``UnitType.WARRIOR``) avec des statistiques fixes.
- Pas d'économie ni d'arbre technologique : les villes se réduisent à un
  propriétaire et à un niveau booléen (>0 signifie qu'une ville existe).
- Les villes sont capturées instantanément par mouvement et une capture d'une
  capitale ennemie supprime immédiatement ce joueur de la partie.
- La seule condition de victoire est l'élimination (un unique joueur possède
  au moins une ville). Les autres modes seront ajoutés dans les phases
  ultérieures.
"""

from enum import IntEnum
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
    PLAIN_FRUIT = 5
    FOREST_WITH_WILD_ANIMAL = 6
    MOUNTAIN_WITH_MINE = 7
    WATER_SHALLOW_WITH_FISH = 8
    NUM_TYPES = 9


# Constantes de types d'unités - limité à WARRIOR pour le MVP Phase 0
class UnitType(IntEnum):
    """Types d'unités."""
    NONE = 0
    WARRIOR = 1
    DEFENDER = 2
    ARCHER = 3
    RIDER = 4
    RAFT = 5
    NUM_TYPES = 6


# Constantes de propriétaires
NO_OWNER = -1


class GameMode(IntEnum):
    """Modes de jeu disponibles."""
    DOMINATION = 0
    PERFECTION = 1


class TechType(IntEnum):
    """Technologies disponibles dans l'arbre."""
    NONE = 0
    CLIMBING = 1
    SAILING = 2
    MINING = 3
    NUM_TECHS = 4


class ResourceType(IntEnum):
    """Types de ressources récoltables."""
    NONE = 0
    FRUIT = 1
    FISH = 2
    ORE = 3
    NUM_TYPES = 4


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
    city_population: jnp.ndarray  # [H, W] - population stockée pour calculer les niveaux
    city_has_port: jnp.ndarray  # [H, W] - présence d'un port (booléen)
    
    # Unités
    units_type: jnp.ndarray  # [N_units_max] - types d'unités
    units_pos: jnp.ndarray  # [N_units_max, 2] - positions (x, y)
    units_hp: jnp.ndarray  # [N_units_max] - points de vie
    units_owner: jnp.ndarray  # [N_units_max] - propriétaire de chaque unité
    units_active: jnp.ndarray  # [N_units_max] - booléen indiquant si l'unité existe
    units_has_acted: jnp.ndarray  # [N_units_max] - booléen action effectuée ce tour
    units_payload_type: jnp.ndarray  # [N_units_max] - type original lorsqu'en radeau/naval
    
    # Économie
    player_stars: jnp.ndarray  # [num_players] - ressources économiques de chaque joueur
    player_techs: jnp.ndarray  # [num_players, num_techs] - booléen techs débloquées
    player_score: jnp.ndarray  # [num_players] - score global (mode Perfection)
    score_territory: jnp.ndarray  # [num_players] - composante territoire
    score_population: jnp.ndarray  # [num_players] - composante population
    score_military: jnp.ndarray  # [num_players] - composante militaire
    score_resources: jnp.ndarray  # [num_players] - composante ressources
    player_income_bonus: jnp.ndarray  # [num_players] - bonus d'étoiles par tour (difficulté)
    resource_type: jnp.ndarray  # [H, W] - type de ressource présente sur la case
    resource_available: jnp.ndarray  # [H, W] - True si la ressource n'a pas encore été récoltée
    
    # État du jeu
    current_player: jnp.ndarray  # joueur actif (scalaire)
    turn: jnp.ndarray  # numéro de tour (scalaire)
    done: jnp.ndarray  # partie terminée (scalaire booléen)
    game_mode: jnp.ndarray  # mode de jeu (DOMINATION, PERFECTION…)
    max_turns: jnp.ndarray  # nombre maximum de tours (pour Perfection)
    
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
        num_techs = int(TechType.NUM_TECHS)
        return cls(
            terrain=jnp.zeros((height, width), dtype=jnp.int32),
            city_owner=jnp.full((height, width), NO_OWNER, dtype=jnp.int32),
            city_level=jnp.zeros((height, width), dtype=jnp.int32),
            city_population=jnp.zeros((height, width), dtype=jnp.int32),
            city_has_port=jnp.zeros((height, width), dtype=jnp.bool_),
            units_type=jnp.zeros(max_units, dtype=jnp.int32),
            units_pos=jnp.zeros((max_units, 2), dtype=jnp.int32),
            units_hp=jnp.zeros(max_units, dtype=jnp.int32),
            units_owner=jnp.full(max_units, NO_OWNER, dtype=jnp.int32),
            units_active=jnp.zeros(max_units, dtype=jnp.bool_),
            units_has_acted=jnp.zeros(max_units, dtype=jnp.bool_),
            units_payload_type=jnp.zeros(max_units, dtype=jnp.int32),
            player_stars=jnp.zeros(num_players, dtype=jnp.int32),
            player_techs=jnp.zeros((num_players, num_techs), dtype=jnp.bool_),
            player_score=jnp.zeros(num_players, dtype=jnp.int32),
            score_territory=jnp.zeros(num_players, dtype=jnp.int32),
            score_population=jnp.zeros(num_players, dtype=jnp.int32),
            score_military=jnp.zeros(num_players, dtype=jnp.int32),
            score_resources=jnp.zeros(num_players, dtype=jnp.int32),
            player_income_bonus=jnp.zeros(num_players, dtype=jnp.int32),
            resource_type=jnp.zeros((height, width), dtype=jnp.int32),
            resource_available=jnp.zeros((height, width), dtype=jnp.bool_),
            current_player=jnp.array(0, dtype=jnp.int32),
            turn=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            game_mode=jnp.array(GameMode.DOMINATION, dtype=jnp.int32),
            max_turns=jnp.array(30, dtype=jnp.int32),
            height=height,
            width=width,
            max_units=max_units,
            num_players=num_players,
        )
