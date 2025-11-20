"""Tests pour polytopia_jax/core/rules.py."""

import pytest
import jax
import jax.numpy as jnp
from polytopia_jax.core.rules import (
    step,
    legal_actions_mask,
    _check_victory,
    BuildingType,
    CITY_CAPTURE_POPULATION,
    CITY_STAR_INCOME_PER_LEVEL,
    RESOURCE_COST,
)
from polytopia_jax.core.state import GameState, UnitType, NO_OWNER, GameMode, TechType, TerrainType, ResourceType
from polytopia_jax.core.actions import ActionType, Direction, encode_action
from polytopia_jax.core.init import init_random, GameConfig
from polytopia_jax.core.score import update_scores


def _make_empty_state(height: int = 4, width: int = 4, max_units: int = 4, stars_per_player: int = 0) -> GameState:
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
    city_owner = city_owner.at[height - 1, width - 1].set(1)
    city_level = city_level.at[height - 1, width - 1].set(1)
    city_population = city_population.at[height - 1, width - 1].set(1)
    state = state.replace(
        player_stars=stars,
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )
    return update_scores(state)


def _with_city(state: GameState, x: int, y: int, owner: int = 0, level: int = 1, population: int = 1, has_port: bool = False) -> GameState:
    """Ajoute ou met à jour une ville sur une case donnée."""
    city_owner = state.city_owner.at[y, x].set(owner)
    city_level = state.city_level.at[y, x].set(level)
    city_population = state.city_population.at[y, x].set(population)
    city_port = state.city_has_port.at[y, x].set(has_port)
    return update_scores(state.replace(
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
        city_has_port=city_port,
    ))


def _with_resource(state: GameState, x: int, y: int, resource_type: ResourceType) -> GameState:
    """Place une ressource récoltable sur la case spécifiée."""
    types = state.resource_type.at[y, x].set(int(resource_type))
    available = state.resource_available.at[y, x].set(True)
    terrain = state.terrain
    if resource_type == ResourceType.FRUIT:
        terrain = terrain.at[y, x].set(TerrainType.PLAIN_FRUIT)
    elif resource_type == ResourceType.FISH:
        terrain = terrain.at[y, x].set(TerrainType.WATER_SHALLOW_WITH_FISH)
    elif resource_type == ResourceType.ORE:
        terrain = terrain.at[y, x].set(TerrainType.MOUNTAIN_WITH_MINE)
    return state.replace(
        resource_type=types,
        resource_available=available,
        terrain=terrain,
    )


def _clear_cities(state: GameState) -> GameState:
    """Supprime toutes les villes du plateau."""
    city_owner = state.city_owner.at[:, :].set(NO_OWNER)
    city_level = state.city_level.at[:, :].set(0)
    city_population = state.city_population.at[:, :].set(0)
    city_port = state.city_has_port.at[:, :].set(False)
    return update_scores(state.replace(
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
        city_has_port=city_port,
    ))


def test_step_no_op():
    """Test que NO_OP ne change pas l'état."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    action = encode_action(ActionType.NO_OP)
    new_state = step(state, action)
    
    # L'état ne devrait pas changer
    assert jnp.array_equal(new_state.terrain, state.terrain)
    assert new_state.current_player == state.current_player


def test_step_move_valid():
    """Test un mouvement valide."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Trouver la première unité du joueur 0
    player_0_units = jnp.where(
        (state.units_owner == 0) & state.units_active
    )[0]
    
    if len(player_0_units) > 0:
        unit_id = int(player_0_units[0])
        original_pos = state.units_pos[unit_id].copy()
        
        # Déplacer vers la droite
        action = encode_action(
            ActionType.MOVE,
            unit_id=unit_id,
            direction=Direction.RIGHT
        )
        
        new_state = step(state, action)
        new_pos = new_state.units_pos[unit_id]
        
        # La position devrait avoir changé
        assert new_pos[0] == original_pos[0] + 1
        assert new_pos[1] == original_pos[1]


def test_step_move_invalid_out_of_bounds():
    """Test qu'un mouvement hors limites est ignoré."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=5, width=5, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Placer une unité au bord
    new_units_pos = state.units_pos.at[0, 0].set(0)
    new_units_pos = new_units_pos.at[0, 1].set(0)
    state = state.replace(units_pos=new_units_pos)
    
    # Essayer de bouger vers le haut (hors limites)
    action = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.UP
    )
    
    new_state = step(state, action)
    
    # La position ne devrait pas avoir changé
    assert new_state.units_pos[0, 0] == 0
    assert new_state.units_pos[0, 1] == 0


def test_unit_single_action_per_turn():
    """Vérifie qu'une unité ne peut agir qu'une fois par tour."""
    key = jax.random.PRNGKey(0)
    config = GameConfig(height=8, width=8, num_players=2, max_units=20)
    state = init_random(key, config)

    # Forcer une unité contrôlée par le joueur 0 au centre
    player_units = jnp.where(
        (state.units_owner == 0) & state.units_active
    )[0]
    assert len(player_units) > 0
    unit_id = int(player_units[0])
    state = state.replace(
        units_pos=state.units_pos.at[unit_id, 0].set(2)
    )
    state = state.replace(
        units_pos=state.units_pos.at[unit_id, 1].set(2)
    )
    terrain = state.terrain.at[2, 3].set(TerrainType.PLAIN)
    terrain = terrain.at[2, 4].set(TerrainType.PLAIN)
    state = state.replace(terrain=terrain)

    move_action = encode_action(
        ActionType.MOVE,
        unit_id=unit_id,
        direction=Direction.RIGHT,
    )

    first_state = step(state, move_action)
    second_state = step(first_state, move_action)

    # Deuxième mouvement doit être ignoré
    assert jnp.array_equal(
        second_state.units_pos[unit_id],
        first_state.units_pos[unit_id],
    )

    # Passer un tour complet (joueur 0 -> 1 -> 0)
    after_end_turn = step(second_state, encode_action(ActionType.END_TURN))
    back_to_player = step(after_end_turn, encode_action(ActionType.END_TURN))
    third_state = step(back_to_player, move_action)

    # Le mouvement redevient possible
    assert third_state.units_pos[unit_id, 0] == first_state.units_pos[unit_id, 0] + 1


def test_harvest_resource_adds_population_and_consumes_tile():
    """Une récolte valide dépense des étoiles et augmente la population."""
    state = _make_empty_state(stars_per_player=5)
    state = _clear_cities(state)
    state = _with_city(state, 1, 1, owner=0, level=1, population=1)
    state = _with_resource(state, 2, 2, ResourceType.FRUIT)
    action = encode_action(
        ActionType.HARVEST_RESOURCE,
        target_pos=(2, 2),
    )
    new_state = step(state, action)
    assert not bool(new_state.resource_available[2, 2])
    assert int(new_state.resource_type[2, 2]) == int(ResourceType.NONE)
    assert int(new_state.terrain[2, 2]) == int(TerrainType.PLAIN)
    assert int(new_state.city_population[1, 1]) == int(state.city_population[1, 1]) + 1
    expected_cost = int(RESOURCE_COST[int(ResourceType.FRUIT)])
    assert int(new_state.player_stars[0]) == int(state.player_stars[0]) - expected_cost


def test_harvest_requires_required_tech():
    """Les ressources soumises à une techno ne sont récoltables qu'après déblocage."""
    state = _make_empty_state(stars_per_player=5)
    state = _clear_cities(state)
    state = _with_city(state, 1, 1, owner=0, level=1, population=1)
    terrain = state.terrain.at[0, 1].set(TerrainType.WATER_SHALLOW)
    state = state.replace(terrain=terrain)
    state = _with_resource(state, 1, 0, ResourceType.FISH)
    action = encode_action(
        ActionType.HARVEST_RESOURCE,
        target_pos=(1, 0),
    )
    blocked_state = step(state, action)
    assert int(blocked_state.player_stars[0]) == int(state.player_stars[0])
    assert bool(blocked_state.resource_available[0, 1])
    assert int(blocked_state.resource_type[0, 1]) == int(ResourceType.FISH)
    techs = state.player_techs.at[0, TechType.SAILING].set(True)
    state = state.replace(player_techs=techs)
    allowed_state = step(state, action)
    assert not bool(allowed_state.resource_available[0, 1])
    assert int(allowed_state.resource_type[0, 1]) == int(ResourceType.NONE)
    assert int(allowed_state.terrain[0, 1]) == int(TerrainType.WATER_SHALLOW)
    assert int(allowed_state.city_population[1, 1]) == int(state.city_population[1, 1]) + 1


def test_income_bonus_added_on_end_turn():
    """Vérifie que le bonus de difficulté est ajouté aux revenus."""
    state = _make_empty_state()
    state = state.replace(
        player_stars=jnp.array([0, 0], dtype=jnp.int32),
        player_income_bonus=jnp.array([3, 0], dtype=jnp.int32),
    )
    end_state = step(state, encode_action(ActionType.END_TURN))
    # Ville niveau 1 = 2 ★ + bonus 3 ★
    assert int(end_state.player_stars[0]) == 5
def test_move_blocked_by_occupied_tile():
    """Une unité ne peut pas entrer sur une case occupée."""
    state = _make_empty_state()
    
    units_type = state.units_type.at[0].set(UnitType.WARRIOR)
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[0].set(0)
    units_owner = units_owner.at[1].set(0)
    units_pos = state.units_pos.at[0, 0].set(1)
    units_pos = units_pos.at[0, 1].set(1)
    units_pos = units_pos.at[1, 0].set(2)
    units_pos = units_pos.at[1, 1].set(1)
    units_hp = state.units_hp.at[0].set(10)
    units_hp = units_hp.at[1].set(10)
    units_active = state.units_active.at[0].set(True)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    action = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.RIGHT,
    )
    new_state = step(state, action)
    
    assert tuple(new_state.units_pos[0].tolist()) == (1, 1)


def test_step_attack():
    """Test une attaque."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Placer deux unités adjacentes (joueur 0 et joueur 1)
    new_units_pos = state.units_pos.at[0, 0].set(5)
    new_units_pos = new_units_pos.at[0, 1].set(5)
    state = state.replace(
        units_pos=new_units_pos,
        units_owner=state.units_owner.at[0].set(0),
        units_active=state.units_active.at[0].set(True),
        units_hp=state.units_hp.at[0].set(10),
    )
    
    # Trouver une unité du joueur 1 ou en créer une
    player_1_units = jnp.where(
        (state.units_owner == 1) & state.units_active
    )[0]
    
    if len(player_1_units) > 0:
        target_id = int(player_1_units[0])
        # Placer la cible à côté de l'attaquant
        new_units_pos = state.units_pos.at[target_id, 0].set(6)
        new_units_pos = new_units_pos.at[target_id, 1].set(5)
        state = state.replace(units_pos=new_units_pos)
        
        original_hp_target = state.units_hp[target_id]
        
        # Attaquer
        action = encode_action(
            ActionType.ATTACK,
            unit_id=0,
            target_pos=(6, 5)
        )
        
        new_state = step(state, action)
        
        # La cible devrait avoir perdu des PV
        assert new_state.units_hp[target_id] < original_hp_target


def test_attack_removes_defeated_unit_and_moves_attacker():
    """L'attaquant occupe la case de la cible détruite."""
    state = _make_empty_state()
    
    units_type = state.units_type.at[0].set(UnitType.WARRIOR)
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[0].set(0)
    units_owner = units_owner.at[1].set(1)
    units_pos = state.units_pos.at[0, 0].set(1)
    units_pos = units_pos.at[0, 1].set(1)
    units_pos = units_pos.at[1, 0].set(2)
    units_pos = units_pos.at[1, 1].set(1)
    units_hp = state.units_hp.at[0].set(10)
    units_hp = units_hp.at[1].set(1)
    units_active = state.units_active.at[0].set(True)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    action = encode_action(
        ActionType.ATTACK,
        unit_id=0,
        target_pos=(2, 1),
    )
    new_state = step(state, action)
    
    assert tuple(new_state.units_pos[0].tolist()) == (2, 1)
    assert not bool(new_state.units_active[1])


def test_city_capture_changes_owner_and_finishes_game():
    """Capturer la dernière ville ennemie déclenche la victoire."""
    state = _make_empty_state()
    state = _clear_cities(state)
    
    units_type = state.units_type.at[0].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0, 0].set(1)
    units_pos = units_pos.at[0, 1].set(1)
    units_hp = state.units_hp.at[0].set(10)
    units_active = state.units_active.at[0].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    state = _with_city(state, 2, 1, owner=1, level=1, population=1)
    
    action = encode_action(
        ActionType.MOVE,
        unit_id=0,
        direction=Direction.RIGHT,
    )
    new_state = step(state, action)
    
    assert new_state.city_owner[1, 2] == 0
    assert new_state.city_level[1, 2] == 1
    assert new_state.city_population[1, 2] == CITY_CAPTURE_POPULATION
    assert bool(new_state.done.item())


def test_train_unit_requires_sufficient_stars():
    """L'entraînement d'une unité consomme des étoiles."""
    rich_state = _with_city(_make_empty_state(stars_per_player=2), 1, 1, owner=0, level=1, population=1)
    action = encode_action(
        ActionType.TRAIN_UNIT,
        unit_type=UnitType.WARRIOR,
        target_pos=(1, 1),
    )
    trained_state = step(rich_state, action)
    assert int(trained_state.player_stars[0]) == 0
    assert jnp.sum(trained_state.units_active) == 1
    
    poor_state = _with_city(_make_empty_state(stars_per_player=1), 1, 1, owner=0, level=1, population=1)
    blocked_state = step(poor_state, action)
    assert jnp.sum(blocked_state.units_active) == 0
    assert int(blocked_state.player_stars[0]) == 1


def test_build_increases_population_and_consumes_stars():
    """Construire un bâtiment augmente la population et facture des étoiles."""
    state = _with_city(_make_empty_state(stars_per_player=6), 1, 1, owner=0, level=1, population=1)
    action = encode_action(
        ActionType.BUILD,
        unit_type=BuildingType.FARM,
        target_pos=(1, 1),
    )
    new_state = step(state, action)
    assert int(new_state.city_population[1, 1]) == 2  # +1 population
    assert int(new_state.city_level[1, 1]) == 1  # seuil maintenu
    assert int(new_state.player_stars[0]) == 3  # 6 - coût 3


def test_end_turn_awards_city_income():
    """Le revenu des villes est ajouté lors de la fin de tour."""
    base_state = _clear_cities(_make_empty_state(stars_per_player=0))
    state = _with_city(base_state, 1, 1, owner=0, level=2, population=3)
    action = encode_action(ActionType.END_TURN)
    new_state = step(state, action)
    expected_income = int(CITY_STAR_INCOME_PER_LEVEL[2])
    assert int(new_state.player_stars[0]) == expected_income
    assert new_state.current_player == 1
    assert int(new_state.player_score[0]) > int(state.player_score[0])


def test_perfection_mode_turn_limit_sets_done():
    """Le mode Perfection se termine quand max_turns est atteint."""
    state = _with_city(_make_empty_state(), 1, 1, owner=0, level=1, population=1)
    state = state.replace(
        game_mode=jnp.array(GameMode.PERFECTION, dtype=jnp.int32),
        max_turns=jnp.array(1, dtype=jnp.int32),
        turn=jnp.array(1, dtype=jnp.int32),
    )
    state = _check_victory(state)
    assert bool(state.done)


def test_archer_ranged_attack_without_counter():
    """Un archer peut tirer à distance sans subir de riposte."""
    state = _clear_cities(_make_empty_state())
    units_type = state.units_type.at[0].set(UnitType.ARCHER)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(8)
    units_active = state.units_active.at[0].set(True)
    
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = units_owner.at[1].set(1)
    units_pos = units_pos.at[1].set(jnp.array([1, 3]))
    units_hp = units_hp.at[1].set(10)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    action = encode_action(
        ActionType.ATTACK,
        unit_id=0,
        target_pos=(1, 3),
    )
    new_state = step(state, action)
    assert new_state.units_hp[1] < units_hp[1]
    assert new_state.units_hp[0] == units_hp[0]


def test_warrior_cannot_attack_out_of_range():
    """Un guerrier ne peut pas attaquer à distance 2."""
    state = _clear_cities(_make_empty_state())
    
    units_type = state.units_type.at[0].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = state.units_hp.at[0].set(10)
    units_active = state.units_active.at[0].set(True)
    
    units_type = units_type.at[1].set(UnitType.WARRIOR)
    units_owner = units_owner.at[1].set(1)
    units_pos = units_pos.at[1].set(jnp.array([1, 3]))
    units_hp = units_hp.at[1].set(10)
    units_active = units_active.at[1].set(True)
    
    state = state.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
    )
    
    action = encode_action(
        ActionType.ATTACK,
        unit_id=0,
        target_pos=(1, 3),
    )
    new_state = step(state, action)
    assert jnp.array_equal(new_state.units_hp, units_hp)


def test_research_tech_unlocks_mountain_movement():
    """Une techno débloque l'accès aux montagnes."""
    state = _clear_cities(_make_empty_state(stars_per_player=5))
    terrain = state.terrain.at[1, 2].set(TerrainType.MOUNTAIN)
    units_type = state.units_type.at[0].set(UnitType.WARRIOR)
    units_owner = state.units_owner.at[0].set(0)
    units_pos = state.units_pos.at[0].set(jnp.array([1, 1]))
    units_active = state.units_active.at[0].set(True)
    state = state.replace(
        terrain=terrain,
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_active=units_active,
    )
    
    move_action = encode_action(ActionType.MOVE, unit_id=0, direction=Direction.RIGHT)
    blocked_state = step(state, move_action)
    assert tuple(blocked_state.units_pos[0].tolist()) == (1, 1)
    
    research_action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CLIMBING)
    researched_state = step(state, research_action)
    moved_state = step(researched_state, move_action)
    assert tuple(moved_state.units_pos[0].tolist()) == (2, 1)


def test_research_tech_respects_dependencies():
    """Impossible de rechercher Sailing sans Climbing."""
    state = _make_empty_state(stars_per_player=10)
    sailing_action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.SAILING)
    blocked_state = step(state, sailing_action)
    assert not bool(blocked_state.player_techs[0, TechType.SAILING])
    assert int(blocked_state.player_stars[0]) == int(state.player_stars[0])
    
    climbing_action = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CLIMBING)
    state_with_climbing = step(state, climbing_action)
    final_state = step(state_with_climbing, sailing_action)
    assert bool(final_state.player_techs[0, TechType.SAILING])


def test_build_mine_requires_mining():
    """Les mines exigent la techno Mining."""
    base_state = _clear_cities(_make_empty_state(stars_per_player=8))
    state = _with_city(base_state, 1, 1, owner=0, level=1, population=1)
    mine_action = encode_action(
        ActionType.BUILD,
        unit_type=BuildingType.MINE,
        target_pos=(1, 1),
    )
    blocked_state = step(state, mine_action)
    assert int(blocked_state.city_population[1, 1]) == 1
    
    research_mining = encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.MINING)
    state_with_mining = step(state, research_mining)
    built_state = step(state_with_mining, mine_action)
    assert int(built_state.city_population[1, 1]) == 3


def test_build_port_requires_sailing():
    """La construction d'un port nécessite Sailing."""
    state = _with_city(_make_empty_state(stars_per_player=10), 1, 1, owner=0, level=1, population=2)
    port_action = encode_action(
        ActionType.BUILD,
        unit_type=BuildingType.PORT,
        target_pos=(1, 1),
    )
    blocked = step(state, port_action)
    assert not bool(blocked.city_has_port[1, 1])
    
    state = state.replace(player_stars=state.player_stars.at[0].set(10))
    state = step(state, encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CLIMBING))
    state = state.replace(player_stars=state.player_stars.at[0].set(10))
    state = step(state, encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.SAILING))
    state = state.replace(player_stars=state.player_stars.at[0].set(10))
    built = step(state, port_action)
    assert bool(built.city_has_port[1, 1])


def test_embark_requires_port():
    """Impossible d'entrer dans l'eau sans port ami."""
    base = _with_city(_make_empty_state(stars_per_player=5), 1, 1, owner=0, level=2, population=2, has_port=False)
    units_type = base.units_type.at[0].set(UnitType.WARRIOR)
    units_owner = base.units_owner.at[0].set(0)
    units_pos = base.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = base.units_hp.at[0].set(10)
    units_active = base.units_active.at[0].set(True)
    state = base.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
        terrain=base.terrain.at[1, 2].set(TerrainType.WATER_SHALLOW),
    )
    move_action = encode_action(ActionType.MOVE, unit_id=0, direction=Direction.RIGHT)
    blocked = step(state, move_action)
    assert tuple(blocked.units_pos[0].tolist()) == (1, 1)
    
    state_with_port = _with_city(blocked, 1, 1, owner=0, level=2, population=2, has_port=True)
    state_with_port = state_with_port.replace(player_stars=state_with_port.player_stars.at[0].set(10))
    state_with_port = step(state_with_port, encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CLIMBING))
    state_with_port = state_with_port.replace(player_stars=state_with_port.player_stars.at[0].set(10))
    state_with_port = step(state_with_port, encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.SAILING))
    embarked = step(state_with_port, move_action)
    assert tuple(embarked.units_pos[0].tolist()) == (2, 1)
    assert embarked.units_type[0] == UnitType.RAFT


def test_disembark_on_any_land():
    """Un radeau peut débarquer sur n'importe quelle case terrestre libre."""
    base = _with_city(_make_empty_state(stars_per_player=10), 1, 1, owner=0, level=2, population=2, has_port=True)
    units_type = base.units_type.at[0].set(UnitType.WARRIOR)
    units_owner = base.units_owner.at[0].set(0)
    units_pos = base.units_pos.at[0].set(jnp.array([1, 1]))
    units_hp = base.units_hp.at[0].set(10)
    units_active = base.units_active.at[0].set(True)
    terrain = base.terrain.at[1, 2].set(TerrainType.WATER_SHALLOW)
    state = base.replace(
        units_type=units_type,
        units_owner=units_owner,
        units_pos=units_pos,
        units_hp=units_hp,
        units_active=units_active,
        terrain=terrain,
    )
    state = state.replace(player_stars=state.player_stars.at[0].set(10))
    state = step(state, encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.CLIMBING))
    state = state.replace(player_stars=state.player_stars.at[0].set(10))
    state = step(state, encode_action(ActionType.RESEARCH_TECH, unit_type=TechType.SAILING))
    move_right = encode_action(ActionType.MOVE, unit_id=0, direction=Direction.RIGHT)
    state = step(state, move_right)
    assert state.units_type[0] == UnitType.RAFT
    assert tuple(state.units_pos[0].tolist()) == (2, 1)
    
    state = state.replace(units_has_acted=jnp.zeros_like(state.units_has_acted))
    landed = step(state, move_right)
    assert landed.units_type[0] == UnitType.WARRIOR
    assert tuple(landed.units_pos[0].tolist()) == (3, 1)


def test_step_end_turn():
    """Test la fin de tour."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    original_player = state.current_player
    original_turn = state.turn
    
    # Fin de tour
    action = encode_action(ActionType.END_TURN)
    new_state = step(state, action)
    
    # Le joueur devrait avoir changé
    assert new_state.current_player != original_player
    assert new_state.current_player == (original_player + 1) % config.num_players


def test_step_end_turn_increments_turn():
    """Test que le tour s'incrémente quand on revient au joueur 0."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # S'assurer qu'on est au joueur 0
    state = state.replace(current_player=jnp.array(1))
    
    # Fin de tour (passe au joueur 0)
    action = encode_action(ActionType.END_TURN)
    new_state = step(state, action)
    
    # Le tour devrait s'être incrémenté
    assert new_state.turn == state.turn + 1
    assert new_state.current_player == 0


def test_end_turn_cycles_players_and_turn_counter():
    """Les fins de tour avancent les joueurs et comptent les tours complets."""
    state = _make_empty_state()
    action = encode_action(ActionType.END_TURN)
    
    mid_state = step(state, action)
    assert mid_state.current_player.item() == 1
    assert mid_state.turn.item() == 0
    
    full_round_state = step(mid_state, action)
    assert full_round_state.current_player.item() == 0
    assert full_round_state.turn.item() == 1


def test_legal_actions_mask():
    """Test le masque d'actions légales."""
    state = _with_city(_make_empty_state(stars_per_player=10), 1, 1, owner=0, level=1, population=1)
    mask = legal_actions_mask(state)
    
    assert mask.dtype == jnp.bool_
    assert mask.shape[0] == int(ActionType.NUM_ACTIONS)
    assert bool(mask[ActionType.TRAIN_UNIT])
    assert bool(mask[ActionType.BUILD])
    assert bool(mask[ActionType.RESEARCH_TECH])


def test_legal_actions_mask_blocks_when_no_stars():
    """TRAIN_UNIT et BUILD deviennent illégaux sans ressources."""
    state = _with_city(_make_empty_state(stars_per_player=0), 1, 1, owner=0, level=1, population=1)
    mask = legal_actions_mask(state)
    assert not bool(mask[ActionType.TRAIN_UNIT])
    assert not bool(mask[ActionType.BUILD])
    assert not bool(mask[ActionType.RESEARCH_TECH])


def test_legal_actions_mask_game_over():
    """Test que le masque est vide quand la partie est terminée."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Marquer la partie comme terminée
    state = state.replace(done=jnp.array(True))
    
    mask = legal_actions_mask(state)
    
    # Seul NO_OP doit rester possible
    assert bool(mask[ActionType.NO_OP])
    remaining = mask.at[ActionType.NO_OP].set(False)
    assert jnp.all(~remaining)


def test_check_victory():
    """Test la détection de victoire."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Initialement, la partie ne devrait pas être terminée
    state = _check_victory(state)
    assert not state.done
    
    # Éliminer le joueur 1 en supprimant sa capitale
    state = state.replace(
        city_owner=state.city_owner.at[:, :].set(0),
        city_level=state.city_level.at[:, :].set(1),
    )
    
    state = _check_victory(state)
    # Maintenant la partie devrait être terminée
    assert state.done


def test_step_jit_compatible():
    """Test que step est compatible avec jax.jit."""
    @jax.jit
    def step_jitted(state, action):
        return step(state, action)
    
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=8, width=8, num_players=2, max_units=20)
    state = init_random(key, config)
    
    action = encode_action(ActionType.END_TURN)
    new_state = step_jitted(state, action)
    
    # Vérifier que l'état est valide
    assert new_state.height == config.height
    assert new_state.width == config.width


def test_step_invalid_action_ignored():
    """Test qu'une action invalide est ignorée."""
    key = jax.random.PRNGKey(42)
    config = GameConfig(height=10, width=10, num_players=2, max_units=20)
    state = init_random(key, config)
    
    # Action avec unit_id invalide
    action = encode_action(
        ActionType.MOVE,
        unit_id=999,  # ID invalide
        direction=Direction.RIGHT
    )
    
    new_state = step(state, action)
    
    # L'état ne devrait pas avoir changé
    assert jnp.array_equal(new_state.units_pos, state.units_pos)
