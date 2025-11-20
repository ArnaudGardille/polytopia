"""Tests pour les modèles Pydantic."""

import pytest
from polytopia_jax.web.models import (
    UnitView,
    CityView,
    GameStateView,
    ReplayMetadata,
    GameInfo,
    ReplayResponse,
    StateResponse,
)


def test_unit_view():
    """Test la validation du modèle UnitView."""
    unit = UnitView(
        type=1,
        pos=[3, 4],
        hp=10,
        owner=0
    )
    
    assert unit.type == 1
    assert unit.pos == [3, 4]
    assert unit.hp == 10
    assert unit.owner == 0
    
    # Test sérialisation
    data = unit.model_dump()
    assert data["type"] == 1
    assert data["pos"] == [3, 4]


def test_city_view():
    """Test la validation du modèle CityView."""
    city = CityView(
        owner=0,
        level=2,
        pos=[5, 6]
    )
    
    assert city.owner == 0
    assert city.level == 2
    assert city.pos == [5, 6]


def test_game_state_view():
    """Test la validation du modèle GameStateView."""
    state = GameStateView(
        terrain=[[0, 1], [1, 0]],
        cities=[CityView(owner=0, level=1, pos=[0, 0])],
        units=[UnitView(type=1, pos=[1, 1], hp=10, owner=0)],
        city_population=[[0, 0], [0, 1]],
        player_stars=[5, 3],
        current_player=0,
        turn=5,
        done=False
    )
    
    assert len(state.terrain) == 2
    assert len(state.cities) == 1
    assert len(state.units) == 1
    assert state.current_player == 0
    assert state.turn == 5
    assert state.done is False


def test_game_state_view_from_raw_state():
    """Test la conversion depuis un état brut."""
    raw_state = {
        "terrain": [[0, 1], [1, 0]],
        "city_owner": [[-1, 0], [-1, -1]],
        "city_level": [[0, 1], [0, 0]],
        "city_population": [[0, 1], [0, 0]],
        "units": [
            {"type": 1, "pos": [1, 0], "hp": 10, "owner": 0}
        ],
        "player_stars": [6, 4],
        "current_player": 0,
        "turn": 1,
        "done": False
    }
    
    state_view = GameStateView.from_raw_state(raw_state)
    
    assert len(state_view.terrain) == 2
    assert len(state_view.cities) == 1
    assert state_view.cities[0].owner == 0
    assert state_view.cities[0].level == 1
    assert state_view.cities[0].pos == [1, 0]
    assert len(state_view.units) == 1
    assert state_view.units[0].type == 1
    assert state_view.city_population[0][1] == 1
    assert state_view.player_stars == [6, 4]
    assert state_view.current_player == 0
    assert state_view.turn == 1


def test_game_state_view_from_raw_state_no_cities():
    """Test la conversion avec aucun ville."""
    raw_state = {
        "terrain": [[0, 0], [0, 0]],
        "city_owner": [[-1, -1], [-1, -1]],
        "city_level": [[0, 0], [0, 0]],
        "units": [],
        "current_player": 0,
        "turn": 0,
        "done": False
    }
    
    state_view = GameStateView.from_raw_state(raw_state)
    assert len(state_view.cities) == 0
    assert len(state_view.units) == 0


def test_replay_metadata():
    """Test la validation du modèle ReplayMetadata."""
    metadata = ReplayMetadata(
        height=10,
        width=10,
        num_players=2,
        max_turns=100,
        seed=42,
        final_turn=50,
        total_actions=200,
        game_done=False
    )
    
    assert metadata.height == 10
    assert metadata.width == 10
    assert metadata.num_players == 2
    assert metadata.seed == 42


def test_replay_metadata_optional():
    """Test que les champs optionnels fonctionnent."""
    metadata = ReplayMetadata(
        height=8,
        width=8,
        num_players=2
    )
    
    assert metadata.height == 8
    assert metadata.max_turns is None
    assert metadata.seed is None


def test_game_info():
    """Test la validation du modèle GameInfo."""
    metadata = ReplayMetadata(height=10, width=10, num_players=2)
    game_info = GameInfo(
        id="test_game",
        metadata=metadata
    )
    
    assert game_info.id == "test_game"
    assert game_info.metadata.height == 10


def test_replay_response():
    """Test la validation du modèle ReplayResponse."""
    metadata = ReplayMetadata(height=10, width=10, num_players=2)
    state = GameStateView(
        terrain=[[0]],
        cities=[],
        units=[],
        current_player=0,
        turn=0,
        done=False
    )
    
    response = ReplayResponse(
        id="test_game",
        metadata=metadata,
        states=[state]
    )
    
    assert response.id == "test_game"
    assert len(response.states) == 1


def test_state_response():
    """Test la validation du modèle StateResponse."""
    state = GameStateView(
        terrain=[[0]],
        cities=[],
        units=[],
        current_player=0,
        turn=5,
        done=False
    )
    
    response = StateResponse(turn=5, state=state)
    
    assert response.turn == 5
    assert response.state.turn == 5



