"""Tests pour le système de promotions (vétérans)."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.rules import (
    step,
    UNIT_HP_MAX,
    UNIT_CAN_PROMOTE,
    _can_unit_promote,
    _promote_unit,
    _get_unit_max_hp,
    _perform_combat,
)
from polytopia_jax.core.state import GameState, UnitType, NO_OWNER
from polytopia_jax.core.actions import ActionType, encode_action


def _make_empty_state(height: int = 4, width: int = 4, max_units: int = 10) -> GameState:
    """Crée un GameState vide et déterministe pour les tests unitaires."""
    from polytopia_jax.core.state import GameMode
    state = GameState.create_empty(
        height=height,
        width=width,
        max_units=max_units,
        num_players=2,
    )
    stars = jnp.full((state.num_players,), 100, dtype=jnp.int32)
    city_owner = state.city_owner.at[0, 0].set(0)
    city_level = state.city_level.at[0, 0].set(1)
    city_population = state.city_population.at[0, 0].set(1)
    city_owner = city_owner.at[height - 1, width - 1].set(1)
    city_level = city_level.at[height - 1, width - 1].set(1)
    city_population = city_population.at[height - 1, width - 1].set(1)
    state = state.replace(
        player_stars=stars,
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
        game_mode=jnp.array(GameMode.DOMINATION, dtype=jnp.int32),
    )
    return state


def _with_unit(
    state: GameState,
    unit_id: int,
    unit_type: UnitType,
    x: int,
    y: int,
    owner: int,
    hp: int = None,
    kills: int = 0,
    veteran: bool = False,
) -> GameState:
    """Ajoute ou met à jour une unité."""
    if hp is None:
        hp = UNIT_HP_MAX[unit_type]
    
    units_type = state.units_type.at[unit_id].set(unit_type)
    units_pos = state.units_pos.at[unit_id, 0].set(x)
    units_pos = units_pos.at[unit_id, 1].set(y)
    units_hp = state.units_hp.at[unit_id].set(hp)
    units_owner = state.units_owner.at[unit_id].set(owner)
    units_active = state.units_active.at[unit_id].set(True)
    units_kills = state.units_kills.at[unit_id].set(kills)
    units_veteran = state.units_veteran.at[unit_id].set(veteran)
    
    return state.replace(
        units_type=units_type,
        units_pos=units_pos,
        units_hp=units_hp,
        units_owner=units_owner,
        units_active=units_active,
        units_kills=units_kills,
        units_veteran=units_veteran,
    )


def test_can_unit_promote():
    """Test que seules les unités terrestres peuvent être promues."""
    assert bool(UNIT_CAN_PROMOTE[UnitType.WARRIOR])
    assert bool(UNIT_CAN_PROMOTE[UnitType.DEFENDER])
    assert bool(UNIT_CAN_PROMOTE[UnitType.ARCHER])
    assert bool(UNIT_CAN_PROMOTE[UnitType.RIDER])
    assert bool(UNIT_CAN_PROMOTE[UnitType.KNIGHT])
    assert bool(UNIT_CAN_PROMOTE[UnitType.SWORDSMAN])
    assert bool(UNIT_CAN_PROMOTE[UnitType.CATAPULT])
    
    # RAFT et GIANT ne peuvent pas être promues
    assert not bool(UNIT_CAN_PROMOTE[UnitType.RAFT])
    assert not bool(UNIT_CAN_PROMOTE[UnitType.GIANT])


def test_kills_increment_on_kill():
    """Test que les kills sont incrémentés quand une unité tue une autre."""
    state = _make_empty_state()
    
    # Attaquant (joueur 0) avec 0 kills, position (1, 1)
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=10, kills=0)
    # Cible faible (joueur 1) avec 1 HP, position adjacente (2, 1)
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    # Utiliser directement _perform_combat pour éviter les problèmes de mouvement
    distance = jnp.array(1, dtype=jnp.int32)  # Distance de Chebyshev = 1 (adjacent)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # L'attaquant devrait avoir 1 kill
    assert int(new_state.units_kills[0]) == 1
    # La cible devrait être morte
    assert not bool(new_state.units_active[1])


def test_kills_increment_on_counter_attack_kill():
    """Test que les kills sont incrémentés lors d'une contre-attaque qui tue."""
    state = _make_empty_state()
    
    # Attaquant faible (joueur 0) avec 1 HP
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=1, kills=0)
    # Cible forte (joueur 1) avec 10 HP
    state = _with_unit(state, 1, UnitType.DEFENDER, 2, 1, owner=1, hp=10, kills=0)
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # La cible devrait avoir 1 kill (contre-attaque)
    assert int(new_state.units_kills[1]) == 1
    # L'attaquant devrait être mort
    assert not bool(new_state.units_active[0])


def test_promotion_after_3_kills():
    """Test qu'une unité est promue après 3 kills."""
    state = _make_empty_state()
    
    # Attaquant avec 2 kills déjà
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=10, kills=2, veteran=False)
    # Cible faible
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    base_max_hp = UNIT_HP_MAX[UnitType.WARRIOR]
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # L'attaquant devrait avoir 3 kills
    assert int(new_state.units_kills[0]) == 3
    # L'attaquant devrait être vétéran
    assert bool(new_state.units_veteran[0])
    # L'attaquant devrait avoir +5 HP max et être complètement guéri
    assert int(new_state.units_hp[0]) == base_max_hp + 5


def test_promotion_heals_unit():
    """Test qu'une unité blessée est complètement guérie lors de la promotion."""
    state = _make_empty_state()
    
    # Attaquant blessé avec 2 kills
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=3, kills=2, veteran=False)
    # Cible faible
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    base_max_hp = UNIT_HP_MAX[UnitType.WARRIOR]
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # L'attaquant devrait être complètement guéri (HP max vétéran)
    assert int(new_state.units_hp[0]) == base_max_hp + 5


def test_promotion_only_once():
    """Test qu'une unité ne peut être promue qu'une seule fois."""
    state = _make_empty_state()
    
    # Attaquant déjà vétéran avec 3 kills
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=15, kills=3, veteran=True)
    # Cible faible
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    base_max_hp = UNIT_HP_MAX[UnitType.WARRIOR]
    veteran_max_hp = base_max_hp + 5
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # L'attaquant devrait avoir 4 kills
    assert int(new_state.units_kills[0]) == 4
    # L'attaquant devrait toujours être vétéran
    assert bool(new_state.units_veteran[0])
    # L'attaquant devrait toujours avoir le même HP max (pas +10)
    assert int(new_state.units_hp[0]) <= veteran_max_hp


def test_naval_unit_cannot_promote():
    """Test que les unités navales ne peuvent pas être promues."""
    state = _make_empty_state()
    
    # RAFT avec 2 kills
    state = _with_unit(state, 0, UnitType.RAFT, 1, 1, owner=0, hp=8, kills=2, veteran=False)
    # Cible faible
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    base_max_hp = UNIT_HP_MAX[UnitType.RAFT]
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # Le RAFT devrait avoir 3 kills
    assert int(new_state.units_kills[0]) == 3
    # Mais ne devrait PAS être vétéran
    assert not bool(new_state.units_veteran[0])
    # Et ne devrait PAS avoir +5 HP max
    assert int(new_state.units_hp[0]) <= base_max_hp


def test_giant_cannot_promote():
    """Test que les Giants ne peuvent pas être promues."""
    state = _make_empty_state()
    
    # GIANT avec 2 kills
    state = _with_unit(state, 0, UnitType.GIANT, 1, 1, owner=0, hp=40, kills=2, veteran=False)
    # Cible faible
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    base_max_hp = UNIT_HP_MAX[UnitType.GIANT]
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # Le GIANT devrait avoir 3 kills
    assert int(new_state.units_kills[0]) == 3
    # Mais ne devrait PAS être vétéran
    assert not bool(new_state.units_veteran[0])
    # Et ne devrait PAS avoir +5 HP max
    assert int(new_state.units_hp[0]) <= base_max_hp


def test_get_unit_max_hp():
    """Test que _get_unit_max_hp retourne le bon HP max selon le statut vétéran."""
    state = _make_empty_state()
    
    # Unité normale
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=10, kills=0, veteran=False)
    base_max_hp = UNIT_HP_MAX[UnitType.WARRIOR]
    
    max_hp_normal = _get_unit_max_hp(state, 0)
    assert int(max_hp_normal) == base_max_hp
    
    # Même unité promue vétéran
    state = state.replace(units_veteran=state.units_veteran.at[0].set(True))
    max_hp_veteran = _get_unit_max_hp(state, 0)
    assert int(max_hp_veteran) == base_max_hp + 5


def test_veteran_hp_in_combat():
    """Test que le HP max vétéran est utilisé dans les calculs de combat."""
    state = _make_empty_state()
    
    # Attaquant vétéran avec HP max 15 (10 + 5)
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=15, kills=3, veteran=True)
    # Cible normale avec 10 HP
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=10, kills=0)
    
    # Vérifier que le HP max vétéran est utilisé
    max_hp = _get_unit_max_hp(state, 0)
    assert int(max_hp) == 15  # 10 base + 5 vétéran
    
    # Utiliser directement _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # L'attaquant vétéran devrait toujours avoir un HP max de 15
    # (même s'il prend des dégâts, le max reste 15)
    max_hp_after = _get_unit_max_hp(new_state, 0)
    assert int(max_hp_after) == 15


def test_multiple_kills_same_turn():
    """Test qu'une unité peut obtenir plusieurs kills dans le même tour (si possible)."""
    state = _make_empty_state()
    
    # Attaquant avec 0 kills
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=10, kills=0)
    # Première cible faible
    state = _with_unit(state, 1, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    # Tuer la première cible avec _perform_combat
    distance = jnp.array(1, dtype=jnp.int32)
    state = _perform_combat(state, 0, 1, distance)
    
    assert int(state.units_kills[0]) == 1
    
    # Placer une deuxième cible faible adjacente
    state = _with_unit(state, 2, UnitType.WARRIOR, 2, 1, owner=1, hp=1, kills=0)
    
    # Tuer la deuxième cible
    distance = jnp.array(1, dtype=jnp.int32)
    state = _perform_combat(state, 0, 2, distance)
    
    assert int(state.units_kills[0]) == 2


def test_promotion_on_counter_attack():
    """Test qu'une unité peut être promue lors d'une contre-attaque."""
    state = _make_empty_state()
    
    # Attaquant faible avec 1 HP
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0, hp=1, kills=0)
    # Cible avec 2 kills déjà
    state = _with_unit(state, 1, UnitType.DEFENDER, 2, 1, owner=1, hp=15, kills=2, veteran=False)
    
    base_max_hp = UNIT_HP_MAX[UnitType.DEFENDER]
    
    # Utiliser directement _perform_combat (l'attaquant va mourir, la cible obtient son 3ème kill)
    distance = jnp.array(1, dtype=jnp.int32)
    new_state = _perform_combat(state, 0, 1, distance)
    
    # La cible devrait avoir 3 kills
    assert int(new_state.units_kills[1]) == 3
    # La cible devrait être promue vétéran
    assert bool(new_state.units_veteran[1])
    # La cible devrait être complètement guérie avec +5 HP max
    assert int(new_state.units_hp[1]) == base_max_hp + 5

