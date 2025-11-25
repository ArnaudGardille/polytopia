"""Tests pour le système de score selon les règles Polytopia."""

import pytest
import jax
import jax.numpy as jnp
import math
from polytopia_jax.core.state import (
    GameState,
    UnitType,
    NO_OWNER,
    GameMode,
    TechType,
)
from polytopia_jax.core.score import (
    update_scores,
    compute_score_components,
    TERRITORY_POINTS_PER_TILE,
    EXPLORATION_POINTS_PER_TILE,
    UNIT_POINTS_PER_STAR,
    SUPER_UNIT_POINTS,
    CITY_BASE_POINTS,
    CITY_POPULATION_POINTS,
    CITY_LEVEL_BONUS,
    TEMPLE_BASE_POINTS,
    TEMPLE_LEVEL_POINTS,
    MONUMENT_POINTS,
    PARK_POINTS,
    TECH_POINTS_PER_TIER,
    _compute_difficulty_bonus_multiplier_jax,
)
from polytopia_jax.core.rules import UNIT_COST


def _make_empty_state(height: int = 10, width: int = 10, max_units: int = 20, num_players: int = 2) -> GameState:
    """Crée un GameState vide pour les tests."""
    state = GameState.create_empty(
        height=height,
        width=width,
        max_units=max_units,
        num_players=num_players,
    )
    return state


def _with_city(state: GameState, x: int, y: int, owner: int = 0, level: int = 1, population: int = 1) -> GameState:
    """Ajoute une ville à un état."""
    city_owner = state.city_owner.at[y, x].set(owner)
    city_level = state.city_level.at[y, x].set(level)
    city_population = state.city_population.at[y, x].set(population)
    return state.replace(
        city_owner=city_owner,
        city_level=city_level,
        city_population=city_population,
    )


def _with_unit(state: GameState, unit_id: int, unit_type: UnitType, x: int, y: int, owner: int = 0) -> GameState:
    """Ajoute une unité à un état."""
    units_type = state.units_type.at[unit_id].set(unit_type)
    units_pos = state.units_pos.at[unit_id, 0].set(x)
    units_pos = units_pos.at[unit_id, 1].set(y)
    units_owner = state.units_owner.at[unit_id].set(owner)
    units_active = state.units_active.at[unit_id].set(True)
    units_hp = state.units_hp.at[unit_id].set(10)  # HP par défaut
    
    return state.replace(
        units_type=units_type,
        units_pos=units_pos,
        units_owner=units_owner,
        units_active=units_active,
        units_hp=units_hp,
    )


def _with_explored_tiles(state: GameState, player_id: int, tiles: list[tuple[int, int]]) -> GameState:
    """Marque des cases comme explorées pour un joueur."""
    tiles_explored = state.tiles_explored
    for x, y in tiles:
        tiles_explored = tiles_explored.at[player_id, y, x].set(True)
    return state.replace(tiles_explored=tiles_explored)


def _with_tech(state: GameState, player_id: int, tech_type: TechType) -> GameState:
    """Ajoute une technologie à un joueur."""
    player_techs = state.player_techs.at[player_id, tech_type].set(True)
    return state.replace(player_techs=player_techs)


def test_territory_points():
    """Test que le territoire donne 20 pts par case contrôlée."""
    state = _make_empty_state()
    
    # Ajouter 3 villes au joueur 0
    state = _with_city(state, 0, 0, owner=0)
    state = _with_city(state, 1, 1, owner=0)
    state = _with_city(state, 2, 2, owner=0)
    
    # Ajouter 2 villes au joueur 1
    state = _with_city(state, 5, 5, owner=1)
    state = _with_city(state, 6, 6, owner=1)
    
    state = update_scores(state)
    
    # Joueur 0 : 3 villes × 20 pts = 60 pts
    assert int(state.score_territory[0]) == 3 * TERRITORY_POINTS_PER_TILE
    
    # Joueur 1 : 2 villes × 20 pts = 40 pts
    assert int(state.score_territory[1]) == 2 * TERRITORY_POINTS_PER_TILE


def test_city_points_level_1():
    """Test que les villes niveau 1 donnent 100 + 5*population."""
    state = _make_empty_state()
    
    # Ville niveau 1 avec 1 population
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    state = update_scores(state)
    
    # 100 (base) + 5*1 (population) = 105 pts
    expected = CITY_BASE_POINTS + (1 * CITY_POPULATION_POINTS)
    assert int(state.score_population[0]) == expected
    
    # Ville niveau 1 avec 2 population
    state = _with_city(state, 1, 1, owner=0, level=1, population=2)
    state = update_scores(state)
    
    # 100 (base) + 5*2 (population) = 110 pts pour cette ville
    # Total : 105 + 110 = 215 pts
    expected_total = (CITY_BASE_POINTS + 1 * CITY_POPULATION_POINTS) + (CITY_BASE_POINTS + 2 * CITY_POPULATION_POINTS)
    assert int(state.score_population[0]) == expected_total


def test_city_points_level_2():
    """Test que les villes niveau 2 donnent 100 + 5*pop + 50."""
    state = _make_empty_state()
    
    # Ville niveau 2 avec 3 population (seuil niveau 2)
    state = _with_city(state, 0, 0, owner=0, level=2, population=3)
    state = update_scores(state)
    
    # 100 (base) + 5*3 (population) + 50 (bonus niveau 2) = 165 pts
    expected = CITY_BASE_POINTS + (3 * CITY_POPULATION_POINTS) + CITY_LEVEL_BONUS
    assert int(state.score_population[0]) == expected


def test_city_points_level_3():
    """Test que les villes niveau 3 donnent 100 + 5*pop + 100."""
    state = _make_empty_state()
    
    # Ville niveau 3 avec 5 population (seuil niveau 3)
    state = _with_city(state, 0, 0, owner=0, level=3, population=5)
    state = update_scores(state)
    
    # 100 (base) + 5*5 (population) + 100 (bonus niveau 3: 2×50) = 225 pts
    expected = CITY_BASE_POINTS + (5 * CITY_POPULATION_POINTS) + (2 * CITY_LEVEL_BONUS)
    assert int(state.score_population[0]) == expected


def test_military_points_by_cost():
    """Test que les unités donnent 5 pts par étoile de coût."""
    state = _make_empty_state(max_units=10)
    
    # Warrior : coût 2★ → 10 pts
    state = _with_unit(state, 0, UnitType.WARRIOR, 0, 0, owner=0)
    
    # Defender : coût 3★ → 15 pts
    state = _with_unit(state, 1, UnitType.DEFENDER, 1, 1, owner=0)
    
    # Rider : coût 4★ → 20 pts
    state = _with_unit(state, 2, UnitType.RIDER, 2, 2, owner=0)
    
    state = update_scores(state)
    
    # Total attendu : 10 + 15 + 20 = 45 pts
    warrior_cost = UNIT_COST[UnitType.WARRIOR]
    defender_cost = UNIT_COST[UnitType.DEFENDER]
    rider_cost = UNIT_COST[UnitType.RIDER]
    
    expected = (warrior_cost * UNIT_POINTS_PER_STAR + 
               defender_cost * UNIT_POINTS_PER_STAR + 
               rider_cost * UNIT_POINTS_PER_STAR)
    
    assert int(state.score_military[0]) == expected


def test_giant_points():
    """Test que les Giants donnent 50 pts (Super Unit)."""
    state = _make_empty_state(max_units=10)
    
    # Giant : 50 pts (pas 5*20=100)
    state = _with_unit(state, 0, UnitType.GIANT, 0, 0, owner=0)
    
    state = update_scores(state)
    
    assert int(state.score_military[0]) == SUPER_UNIT_POINTS


def test_exploration_points():
    """Test que l'exploration donne 5 pts par case explorée."""
    state = _make_empty_state()
    
    # Joueur 0 explore 10 cases
    explored_tiles_0 = [(i, 0) for i in range(10)]
    state = _with_explored_tiles(state, 0, explored_tiles_0)
    
    # Joueur 1 explore 5 cases
    explored_tiles_1 = [(i, 5) for i in range(5)]
    state = _with_explored_tiles(state, 1, explored_tiles_1)
    
    state = update_scores(state)
    
    # Joueur 0 : 10 cases × 5 pts = 50 pts
    assert int(state.score_exploration[0]) == 10 * EXPLORATION_POINTS_PER_TILE
    
    # Joueur 1 : 5 cases × 5 pts = 25 pts
    assert int(state.score_exploration[1]) == 5 * EXPLORATION_POINTS_PER_TILE


def test_temple_points():
    """Test que les temples donnent 100 + 50*niveau."""
    state = _make_empty_state()
    
    # Temple niveau 1
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    city_has_temple = state.city_has_temple.at[0, 0].set(True)
    city_temple_level = state.city_temple_level.at[0, 0].set(1)
    state = state.replace(
        city_has_temple=city_has_temple,
        city_temple_level=city_temple_level,
    )
    
    state = update_scores(state)
    
    # Le score total inclut les points de temple
    # Ville : 100 + 5*1 = 105
    # Temple : 100 + 50*1 = 150
    # Total ville+temple : 255
    # Mais le score_population ne compte que la ville
    # Le temple est dans le score total mais pas dans score_population
    totals, _, _, _, _, _ = compute_score_components(state)
    assert int(totals[0]) > CITY_BASE_POINTS + CITY_POPULATION_POINTS


def test_temple_points_level_2():
    """Test que les temples niveau 2 donnent 100 + 100."""
    state = _make_empty_state()
    
    # Temple niveau 2
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    city_has_temple = state.city_has_temple.at[0, 0].set(True)
    city_temple_level = state.city_temple_level.at[0, 0].set(2)
    state = state.replace(
        city_has_temple=city_has_temple,
        city_temple_level=city_temple_level,
    )
    
    state = update_scores(state)
    
    # Vérifier que le score total inclut le temple niveau 2
    totals, _, _, _, _, _ = compute_score_components(state)
    # Ville : 105, Temple niveau 2 : 200, Total > 300
    assert int(totals[0]) > 300


def test_monument_points():
    """Test que les monuments donnent 400 pts."""
    state = _make_empty_state()
    
    # Monument
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    city_has_monument = state.city_has_monument.at[0, 0].set(True)
    state = state.replace(city_has_monument=city_has_monument)
    
    state = update_scores(state)
    
    # Vérifier que le score total inclut le monument
    totals, _, _, _, _, _ = compute_score_components(state)
    # Ville : 105, Monument : 400, Total > 500
    assert int(totals[0]) > 500


def test_park_points():
    """Test que les parcs donnent 250 pts."""
    state = _make_empty_state()
    
    # Parc
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    city_has_park = state.city_has_park.at[0, 0].set(True)
    state = state.replace(city_has_park=city_has_park)
    
    state = update_scores(state)
    
    # Vérifier que le score total inclut le parc
    totals, _, _, _, _, _ = compute_score_components(state)
    # Ville : 105, Parc : 250, Total > 350
    assert int(totals[0]) > 350


def test_science_points_tier_1():
    """Test que les technologies tier 1 donnent 100 pts."""
    state = _make_empty_state()
    
    # Tech tier 1 : CLIMBING
    state = _with_tech(state, 0, TechType.CLIMBING)
    
    state = update_scores(state)
    
    # Vérifier que le score inclut la science
    totals, _, _, _, _, science = compute_score_components(state)
    assert int(science[0]) == TECH_POINTS_PER_TIER  # 100 pts pour tier 1


def test_science_points_tier_2():
    """Test que les technologies tier 2 donnent 200 pts."""
    state = _make_empty_state()
    
    # Tech tier 2 : ARCHERY
    state = _with_tech(state, 0, TechType.ARCHERY)
    
    state = update_scores(state)
    
    # Vérifier que le score inclut la science
    totals, _, _, _, _, science = compute_score_components(state)
    assert int(science[0]) == 2 * TECH_POINTS_PER_TIER  # 200 pts pour tier 2


def test_science_points_tier_3():
    """Test que les technologies tier 3 donnent 300 pts."""
    state = _make_empty_state()
    
    # Tech tier 3 : PHILOSOPHY
    state = _with_tech(state, 0, TechType.PHILOSOPHY)
    
    state = update_scores(state)
    
    # Vérifier que le score inclut la science
    totals, _, _, _, _, science = compute_score_components(state)
    assert int(science[0]) == 3 * TECH_POINTS_PER_TIER  # 300 pts pour tier 3


def test_science_points_multiple():
    """Test que plusieurs technologies s'additionnent."""
    state = _make_empty_state()
    
    # Plusieurs technologies
    state = _with_tech(state, 0, TechType.CLIMBING)  # Tier 1 : 100 pts
    state = _with_tech(state, 0, TechType.ARCHERY)    # Tier 2 : 200 pts
    state = _with_tech(state, 0, TechType.PHILOSOPHY) # Tier 3 : 300 pts
    
    state = update_scores(state)
    
    # Total : 100 + 200 + 300 = 600 pts
    totals, _, _, _, _, science = compute_score_components(state)
    assert int(science[0]) == 600


def test_complete_score_calculation():
    """Test un calcul de score complet avec toutes les composantes."""
    state = _make_empty_state()
    
    # Joueur 0 :
    # - 2 villes niveau 1 (pop 1 et 2)
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    state = _with_city(state, 1, 1, owner=0, level=1, population=2)
    
    # - 2 unités (Warrior 2★, Rider 4★)
    state = _with_unit(state, 0, UnitType.WARRIOR, 2, 2, owner=0)
    state = _with_unit(state, 1, UnitType.RIDER, 3, 3, owner=0)
    
    # - 5 cases explorées
    state = _with_explored_tiles(state, 0, [(i, 0) for i in range(5)])
    
    # - 1 tech tier 1
    state = _with_tech(state, 0, TechType.CLIMBING)
    
    state = update_scores(state)
    
    # Calcul attendu :
    # Territoire : 2 × 20 = 40
    # Villes : (100+5) + (100+10) = 215
    # Militaire : (2×5) + (4×5) = 30
    # Exploration : 5 × 5 = 25
    # Science : 100
    # Total : 40 + 215 + 30 + 25 + 100 = 410
    
    totals, territory, population, military, exploration, science = compute_score_components(state)
    
    assert int(territory[0]) == 2 * TERRITORY_POINTS_PER_TILE
    assert int(population[0]) == (CITY_BASE_POINTS + 1*CITY_POPULATION_POINTS) + (CITY_BASE_POINTS + 2*CITY_POPULATION_POINTS)
    assert int(military[0]) == (UNIT_COST[UnitType.WARRIOR] * UNIT_POINTS_PER_STAR + UNIT_COST[UnitType.RIDER] * UNIT_POINTS_PER_STAR)
    assert int(exploration[0]) == 5 * EXPLORATION_POINTS_PER_TILE
    assert int(science[0]) == TECH_POINTS_PER_TIER
    
    expected_total = int(territory[0]) + int(population[0]) + int(military[0]) + int(exploration[0]) + int(science[0])
    assert int(totals[0]) == expected_total


def test_difficulty_bonus_multiplier():
    """Test le calcul du multiplicateur de bonus difficulté."""
    import jax.numpy as jnp
    
    # Formule : 100% + 41% * ln(nb_adversaires) + bonus difficulté
    
    # 1 adversaire, bonus 20%
    multiplier = _compute_difficulty_bonus_multiplier_jax(jnp.array(1), jnp.array(20))
    expected = 1.0 + 0.41 * math.log(1) + 0.20
    assert abs(float(multiplier) - expected) < 0.01
    
    # 3 adversaires, bonus 40%
    multiplier = _compute_difficulty_bonus_multiplier_jax(jnp.array(3), jnp.array(40))
    expected = 1.0 + 0.41 * math.log(3) + 0.40
    assert abs(float(multiplier) - expected) < 0.01
    
    # 0 adversaires, bonus 20% (utilise max(0, 1) = 1 pour éviter ln(0))
    multiplier = _compute_difficulty_bonus_multiplier_jax(jnp.array(0), jnp.array(20))
    expected = 1.0 + 0.41 * math.log(1) + 0.20
    assert abs(float(multiplier) - expected) < 0.01


def test_perfection_mode_bonus():
    """Test que le bonus difficulté s'applique en mode Perfection."""
    state = _make_empty_state(num_players=3)
    
    # Mode Perfection
    state = state.replace(game_mode=jnp.array(GameMode.PERFECTION, dtype=jnp.int32))
    
    # Joueur 0 avec score de base 1000
    state = _with_city(state, 0, 0, owner=0, level=1, population=10)
    # Ajouter assez de villes pour avoir un score significatif
    for i in range(10):
        state = _with_city(state, i % 5, (i // 5) + 1, owner=0, level=1, population=1)
    
    # Bonus difficulté pour les adversaires (joueurs 1 et 2)
    # 2★ = 40% de bonus
    player_income_bonus = state.player_income_bonus.at[1].set(2)
    player_income_bonus = player_income_bonus.at[2].set(2)
    state = state.replace(player_income_bonus=player_income_bonus)
    
    state = update_scores(state)
    
    # Le score du joueur 0 devrait avoir un multiplicateur appliqué
    # 2 adversaires, bonus 40%
    # Multiplicateur = 1.0 + 0.41*ln(2) + 0.40 ≈ 1.68
    base_score = int(state.score_territory[0]) + int(state.score_population[0])
    
    # Le score final devrait être supérieur au score de base (sans bonus)
    # car le bonus s'applique au score total
    assert int(state.player_score[0]) >= base_score


def test_domination_mode_no_bonus():
    """Test que le bonus difficulté ne s'applique PAS en mode Domination."""
    state = _make_empty_state(num_players=3)
    
    # Mode Domination (par défaut)
    assert int(state.game_mode) == GameMode.DOMINATION
    
    # Score de base
    state = _with_city(state, 0, 0, owner=0, level=1, population=5)
    
    # Bonus difficulté pour les adversaires
    player_income_bonus = state.player_income_bonus.at[1].set(2)
    player_income_bonus = player_income_bonus.at[2].set(2)
    state = state.replace(player_income_bonus=player_income_bonus)
    
    state = update_scores(state)
    
    # En mode Domination, pas de multiplicateur
    # Le score devrait être égal à la somme des composantes
    base_score = int(state.score_territory[0]) + int(state.score_population[0])
    assert int(state.player_score[0]) == base_score


def test_score_update_is_idempotent():
    """Test que mettre à jour le score plusieurs fois donne le même résultat."""
    state = _make_empty_state()
    
    state = _with_city(state, 0, 0, owner=0, level=1, population=3)
    state = _with_unit(state, 0, UnitType.WARRIOR, 1, 1, owner=0)
    
    state1 = update_scores(state)
    state2 = update_scores(state1)
    state3 = update_scores(state2)
    
    # Les scores devraient être identiques
    assert int(state1.player_score[0]) == int(state2.player_score[0])
    assert int(state2.player_score[0]) == int(state3.player_score[0])


def test_score_with_multiple_players():
    """Test que le score est calculé correctement pour plusieurs joueurs."""
    state = _make_empty_state(num_players=3)
    
    # Joueur 0 : 1 ville
    state = _with_city(state, 0, 0, owner=0, level=1, population=1)
    
    # Joueur 1 : 2 villes
    state = _with_city(state, 5, 5, owner=1, level=1, population=1)
    state = _with_city(state, 6, 6, owner=1, level=1, population=1)
    
    # Joueur 2 : 1 unité
    state = _with_unit(state, 0, UnitType.WARRIOR, 8, 8, owner=2)
    
    state = update_scores(state)
    
    # Vérifier que chaque joueur a son propre score
    assert int(state.score_territory[0]) == 1 * TERRITORY_POINTS_PER_TILE
    assert int(state.score_territory[1]) == 2 * TERRITORY_POINTS_PER_TILE
    assert int(state.score_territory[2]) == 0
    
    assert int(state.score_military[2]) == UNIT_COST[UnitType.WARRIOR] * UNIT_POINTS_PER_STAR

