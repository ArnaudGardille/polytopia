"""Tests pour les nouvelles unités : Knight, Swordsman, Catapult, Giant."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.rules import (
    step,
    UNIT_HP_MAX,
    UNIT_ATTACK,
    UNIT_DEFENSE,
    UNIT_MOVEMENT,
    UNIT_COST,
    UNIT_ATTACK_RANGE,
    UNIT_REQUIRED_TECH,
    _compute_tech_cost,
)
from polytopia_jax.core.state import GameState, UnitType, TechType, NO_OWNER
from polytopia_jax.core.actions import ActionType, Direction, encode_action


def _make_empty_state(height: int = 4, width: int = 4, max_units: int = 10, stars_per_player: int = 0) -> GameState:
    """Crée un GameState vide et déterministe pour les tests unitaires."""
    state = GameState.create_empty(
        height=height,
        width=width,
        max_units=max_units,
        num_players=2,
    )
    stars = jnp.full((state.num_players,), stars_per_player, dtype=jnp.int32)
    city_owner = state.city_owner.at[0, 0].set(0)
    city_level = state.city_level.at[0, 0].set(1)
    city_population = state.city_population.at[0, 0].set(1)
    state = state.replace(
        player_stars=stars,
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )
    return state


def _with_city(state: GameState, x: int, y: int, owner: int = 0, level: int = 1, population: int = 1) -> GameState:
    """Ajoute ou met à jour une ville sur une case donnée."""
    city_owner = state.city_owner.at[y, x].set(owner)
    city_level = state.city_level.at[y, x].set(level)
    city_population = state.city_population.at[y, x].set(population)
    return state.replace(
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )


def _unlock_tech(state: GameState, tech: TechType, player: int = 0) -> GameState:
    """Débloque une technologie pour un joueur."""
    new_techs = state.player_techs.at[player, tech].set(True)
    return state.replace(player_techs=new_techs)


def test_unit_stats_knight():
    """Vérifie les stats de Knight."""
    assert UNIT_HP_MAX[UnitType.KNIGHT] == 15
    assert UNIT_ATTACK[UnitType.KNIGHT] == 4
    assert UNIT_DEFENSE[UnitType.KNIGHT] == 1
    assert UNIT_MOVEMENT[UnitType.KNIGHT] == 3
    assert UNIT_ATTACK_RANGE[UnitType.KNIGHT] == 1
    assert UNIT_COST[UnitType.KNIGHT] == 6


def test_unit_stats_swordsman():
    """Vérifie les stats de Swordsman."""
    assert UNIT_HP_MAX[UnitType.SWORDSMAN] == 15
    assert UNIT_ATTACK[UnitType.SWORDSMAN] == 3
    assert UNIT_DEFENSE[UnitType.SWORDSMAN] == 2
    assert UNIT_MOVEMENT[UnitType.SWORDSMAN] == 1
    assert UNIT_ATTACK_RANGE[UnitType.SWORDSMAN] == 1
    assert UNIT_COST[UnitType.SWORDSMAN] == 5


def test_unit_stats_catapult():
    """Vérifie les stats de Catapult."""
    assert UNIT_HP_MAX[UnitType.CATAPULT] == 8
    assert UNIT_ATTACK[UnitType.CATAPULT] == 4
    assert UNIT_DEFENSE[UnitType.CATAPULT] == 1
    assert UNIT_MOVEMENT[UnitType.CATAPULT] == 1
    assert UNIT_ATTACK_RANGE[UnitType.CATAPULT] == 3  # Portée 3
    assert UNIT_COST[UnitType.CATAPULT] == 6


def test_unit_stats_giant():
    """Vérifie les stats de Giant."""
    assert UNIT_HP_MAX[UnitType.GIANT] == 40
    assert UNIT_ATTACK[UnitType.GIANT] == 5
    assert UNIT_DEFENSE[UnitType.GIANT] == 3
    assert UNIT_MOVEMENT[UnitType.GIANT] == 1
    assert UNIT_ATTACK_RANGE[UnitType.GIANT] == 1
    assert UNIT_COST[UnitType.GIANT] == 20


def test_unit_required_techs():
    """Vérifie que les technologies requises sont correctes."""
    assert UNIT_REQUIRED_TECH[UnitType.WARRIOR] == TechType.NONE
    # DEFENDER nécessite STRATEGY selon l'implémentation actuelle
    assert UNIT_REQUIRED_TECH[UnitType.DEFENDER] == TechType.STRATEGY
    assert UNIT_REQUIRED_TECH[UnitType.ARCHER] == TechType.ARCHERY
    assert UNIT_REQUIRED_TECH[UnitType.RIDER] == TechType.RIDING
    assert UNIT_REQUIRED_TECH[UnitType.RAFT] == TechType.SAILING
    assert UNIT_REQUIRED_TECH[UnitType.KNIGHT] == TechType.CHIVALRY
    assert UNIT_REQUIRED_TECH[UnitType.SWORDSMAN] == TechType.SMITHERY
    assert UNIT_REQUIRED_TECH[UnitType.CATAPULT] == TechType.MATHEMATICS
    assert UNIT_REQUIRED_TECH[UnitType.GIANT] == TechType.NONE  # Pas encore implémenté via amélioration


def test_train_knight_requires_chivalry():
    """L'entraînement d'un Knight nécessite Chivalry."""
    state = _make_empty_state(stars_per_player=20)
    state = _with_city(state, 1, 1, owner=0, level=1, population=1)
    
    # Essayer d'entraîner sans la tech
    action = encode_action(
        ActionType.TRAIN_UNIT,
        unit_type=UnitType.KNIGHT,
        target_pos=(1, 1),
    )
    blocked_state = step(state, action)
    assert jnp.sum(blocked_state.units_active) == 0
    assert int(blocked_state.player_stars[0]) == 20
    
    # Débloquer les prérequis : RIDING + FORESTRY
    state = _unlock_tech(state, TechType.RIDING)
    state = _unlock_tech(state, TechType.FORESTRY)
    
    # Rechercher Chivalry
    research_action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CHIVALRY)
    state = step(state, research_action)
    
    # Maintenant on peut entraîner
    trained_state = step(state, action)
    assert jnp.sum(trained_state.units_active) == 1
    assert trained_state.units_type[jnp.where(trained_state.units_active)[0][0]] == UnitType.KNIGHT


def test_train_swordsman_requires_smithery():
    """L'entraînement d'un Swordsman nécessite Smithery."""
    state = _make_empty_state(stars_per_player=20)
    state = _with_city(state, 1, 1, owner=0, level=1, population=1)
    
    # Essayer d'entraîner sans la tech
    action = encode_action(
        ActionType.TRAIN_UNIT,
        unit_type=UnitType.SWORDSMAN,
        target_pos=(1, 1),
    )
    blocked_state = step(state, action)
    assert jnp.sum(blocked_state.units_active) == 0
    
    # Débloquer Mining (prérequis de Smithery)
    state = _unlock_tech(state, TechType.MINING)
    
    # Rechercher Smithery
    research_action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.SMITHERY)
    state = step(state, research_action)
    
    # Maintenant on peut entraîner
    trained_state = step(state, action)
    assert jnp.sum(trained_state.units_active) == 1
    assert trained_state.units_type[jnp.where(trained_state.units_active)[0][0]] == UnitType.SWORDSMAN


def test_train_catapult_requires_mathematics():
    """L'entraînement d'une Catapult nécessite Mathematics."""
    state = _make_empty_state(stars_per_player=20)
    state = _with_city(state, 1, 1, owner=0, level=1, population=1)
    
    # Essayer d'entraîner sans la tech
    action = encode_action(
        ActionType.TRAIN_UNIT,
        unit_type=UnitType.CATAPULT,
        target_pos=(1, 1),
    )
    blocked_state = step(state, action)
    assert jnp.sum(blocked_state.units_active) == 0
    
    # Débloquer les prérequis : ORGANIZATION + FORESTRY
    state = _unlock_tech(state, TechType.ORGANIZATION)
    state = _unlock_tech(state, TechType.FORESTRY)
    
    # Rechercher Mathematics
    research_action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.MATHEMATICS)
    state = step(state, research_action)
    
    # Maintenant on peut entraîner
    trained_state = step(state, action)
    assert jnp.sum(trained_state.units_active) == 1
    assert trained_state.units_type[jnp.where(trained_state.units_active)[0][0]] == UnitType.CATAPULT


def test_catapult_ranged_attack():
    """Une Catapult peut attaquer à distance 3."""
    state = _make_empty_state(stars_per_player=20)
    
    # Créer une Catapult
    units_type = state.units_type.at[0].set(UnitType.CATAPULT)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(8)
    units_active = state.units_active.at[0].set(True)
    
    # Créer une cible à distance 3
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = units_owner.at[1].set(1)
    units_pos = units_pos.at[1].set(jnp.array([4, 1]))  # Distance 3 en x
    units_hp = units_hp.at[1].set(10)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    # Attaquer à distance 3
    action = encode_action(
        ActionType.ATTACK,
        unit_id=0,
        target_pos=(4, 1),
    )
    new_state = step(state, action)
    
    # La cible devrait avoir perdu des PV
    assert new_state.units_hp[1] < units_hp[1]
    # La Catapult ne devrait pas avoir subi de contre-attaque (distance > 1)
    assert new_state.units_hp[0] == units_hp[0]


def test_catapult_cannot_attack_beyond_range():
    """Une Catapult ne peut pas attaquer au-delà de la portée 3."""
    state = _make_empty_state(stars_per_player=20)
    
    # Créer une Catapult
    units_type = state.units_type.at[0].set(UnitType.CATAPULT)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(8)
    units_active = state.units_active.at[0].set(True)
    
    # Créer une cible à distance 4 (hors portée)
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = units_owner.at[1].set(1)
    units_pos = units_pos.at[1].set(jnp.array([5, 1]))  # Distance 4 en x
    units_hp = units_hp.at[1].set(10)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    # Essayer d'attaquer à distance 4
    action = encode_action(
        ActionType.ATTACK,
        unit_id=0,
        target_pos=(5, 1),
    )
    new_state = step(state, action)
    
    # La cible ne devrait pas avoir perdu de PV
    assert new_state.units_hp[1] == units_hp[1]


def test_knight_high_movement():
    """Un Knight peut se déplacer de 3 cases."""
    state = _make_empty_state(stars_per_player=20)
    
    # Créer un Knight
    units_type = state.units_type.at[0].set(UnitType.KNIGHT)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(15)
    units_active = state.units_active.at[0].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    # Déplacer vers la droite (1 case)
    action1 = encode_action(ActionType.MOVE, unit_id=0, direction=Direction.RIGHT)
    state = step(state, action1)
    assert tuple(state.units_pos[0].tolist()) == (2, 1)
    
    # Réinitialiser has_acted pour permettre un autre mouvement
    state = state.replace(units_has_acted=jnp.zeros_like(state.units_has_acted))
    
    # Déplacer encore vers la droite (2e case)
    action2 = encode_action(ActionType.MOVE, unit_id=0, direction=Direction.RIGHT)
    state = step(state, action2)
    assert tuple(state.units_pos[0].tolist()) == (3, 1)
    
    # Réinitialiser has_acted
    state = state.replace(units_has_acted=jnp.zeros_like(state.units_has_acted))
    
    # Déplacer encore vers la droite (3e case)
    action3 = encode_action(ActionType.MOVE, unit_id=0, direction=Direction.RIGHT)
    state = step(state, action3)
    assert tuple(state.units_pos[0].tolist()) == (4, 1)


def test_swordsman_balanced_stats():
    """Un Swordsman a des stats équilibrées (attaque et défense)."""
    state = _make_empty_state(stars_per_player=20)
    
    # Créer un Swordsman et un Warrior
    units_type = state.units_type.at[0].set(UnitType.SWORDSMAN)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(15)
    units_active = state.units_active.at[0].set(True)
    
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = units_owner.at[1].set(1)
    units_pos = units_pos.at[1].set(jnp.array([2, 1]))
    units_hp = units_hp.at[1].set(10)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    # Le Swordsman attaque
    action = encode_action(ActionType.ATTACK, unit_id=0, target_pos=(2, 1))
    new_state = step(state, action)
    
    # Le Warrior devrait avoir perdu des PV
    assert new_state.units_hp[1] < units_hp[1]
    # Le Swordsman devrait avoir subi une contre-attaque mais moins de dégâts grâce à sa défense
    assert new_state.units_hp[0] < units_hp[0]


def test_giant_high_hp():
    """Un Giant a beaucoup de PV (40)."""
    state = _make_empty_state(stars_per_player=20)
    
    # Créer un Giant
    units_type = state.units_type.at[0].set(UnitType.GIANT)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(40)  # HP max
    units_active = state.units_active.at[0].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    # Vérifier que le Giant a bien 40 HP
    assert state.units_hp[0] == 40
    
    # Créer un Warrior ennemi
    units_type = state.units_type.at[1].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[1].set(1)
    units_pos = state.units_pos.at[1].set(jnp.array([2, 1]))
    units_hp = state.units_hp.at[1].set(10)
    units_active = state.units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    # Le Warrior attaque le Giant
    action = encode_action(ActionType.ATTACK, unit_id=1, target_pos=(1, 1))
    new_state = step(state, action)
    
    # Le Giant devrait avoir perdu peu de PV grâce à sa défense élevée
    assert new_state.units_hp[0] > 35  # Moins de 5 PV perdus


def test_tech_cost_dynamic():
    """Le coût des technologies augmente avec le nombre de villes."""
    # Créer un état avec 0 villes pour le joueur 0
    state = GameState.create_empty(height=4, width=4, max_units=10, num_players=2)
    state = state.replace(player_stars=jnp.array([100, 0], dtype=jnp.int32))
    
    # 0 villes
    cost_0_cities = _compute_tech_cost(state, TechType.CHIVALRY, 0)
    assert int(cost_0_cities) == 4  # tier 2 * 0 villes + 4
    
    # Ajouter 1 ville
    city_owner = state.city_owner.at[0, 0].set(0)
    city_level = state.city_level.at[0, 0].set(1)
    state = state.replace(city_owner=city_owner, city_level=city_level)
    cost_1_city = _compute_tech_cost(state, TechType.CHIVALRY, 0)
    assert int(cost_1_city) == 6  # tier 2 * 1 ville + 4
    
    # Ajouter une 2e ville
    city_owner = city_owner.at[2, 2].set(0)
    city_level = city_level.at[2, 2].set(1)
    state = state.replace(city_owner=city_owner, city_level=city_level)
    cost_2_cities = _compute_tech_cost(state, TechType.CHIVALRY, 0)
    assert int(cost_2_cities) == 8  # tier 2 * 2 villes + 4


def test_tech_dependencies_smithery():
    """Smithery nécessite Mining."""
    state = _make_empty_state(stars_per_player=20)
    
    # Essayer de rechercher Smithery sans Mining
    action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.SMITHERY)
    blocked_state = step(state, action)
    assert not bool(blocked_state.player_techs[0, TechType.SMITHERY])
    
    # Débloquer Mining
    state = _unlock_tech(state, TechType.MINING)
    
    # Maintenant on peut rechercher Smithery
    researched_state = step(state, action)
    assert bool(researched_state.player_techs[0, TechType.SMITHERY])


def test_tech_dependencies_chivalry():
    """Chivalry nécessite RIDING + FORESTRY."""
    state = _make_empty_state(stars_per_player=20)
    
    # Essayer de rechercher Chivalry sans prérequis
    action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CHIVALRY)
    blocked_state = step(state, action)
    assert not bool(blocked_state.player_techs[0, TechType.CHIVALRY])
    
    # Débloquer seulement RIDING
    state = _unlock_tech(state, TechType.RIDING)
    blocked_state2 = step(state, action)
    assert not bool(blocked_state2.player_techs[0, TechType.CHIVALRY])
    
    # Débloquer aussi FORESTRY
    state = _unlock_tech(state, TechType.FORESTRY)
    researched_state = step(state, action)
    assert bool(researched_state.player_techs[0, TechType.CHIVALRY])


def test_tech_dependencies_mathematics():
    """Mathematics nécessite ORGANIZATION + FORESTRY."""
    state = _make_empty_state(stars_per_player=20)
    
    # Essayer de rechercher Mathematics sans prérequis
    action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.MATHEMATICS)
    blocked_state = step(state, action)
    assert not bool(blocked_state.player_techs[0, TechType.MATHEMATICS])
    
    # Débloquer seulement ORGANIZATION
    state = _unlock_tech(state, TechType.ORGANIZATION)
    blocked_state2 = step(state, action)
    assert not bool(blocked_state2.player_techs[0, TechType.MATHEMATICS])
    
    # Débloquer aussi FORESTRY
    state = _unlock_tech(state, TechType.FORESTRY)
    researched_state = step(state, action)
    assert bool(researched_state.player_techs[0, TechType.MATHEMATICS])

