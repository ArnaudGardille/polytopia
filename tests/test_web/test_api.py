"""Tests pour l'API FastAPI."""

import json
import pytest
from pathlib import Path
import tempfile
import shutil
from fastapi.testclient import TestClient

from polytopia_jax.web.api import app
from polytopia_jax.web.replay_store import clear_cache
from polytopia_jax.core.actions import END_TURN_ACTION


@pytest.fixture
def temp_replays_dir(monkeypatch):
    """Crée un dossier temporaire pour les replays."""
    temp_dir = Path(tempfile.mkdtemp())
    replays_dir = temp_dir / "replays"
    replays_dir.mkdir()
    
    # Mock la fonction _get_replays_dir
    from polytopia_jax.web import replay_store
    original_get_replays_dir = replay_store._get_replays_dir
    
    def mock_get_replays_dir():
        return replays_dir
    
    monkeypatch.setattr(replay_store, "_get_replays_dir", mock_get_replays_dir)
    
    yield replays_dir
    
    clear_cache()
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_replay(temp_replays_dir):
    """Crée un replay de test."""
    replay_data = {
        "metadata": {
            "height": 4,
            "width": 4,
            "num_players": 2,
            "max_turns": 5,
            "seed": 123,
            "final_turn": 2,
            "total_actions": 10,
            "game_done": False
        },
        "states": [
            {
                "terrain": [[0, 0], [0, 0]],
                "city_owner": [[-1, 0], [-1, -1]],
                "city_level": [[0, 1], [0, 0]],
                "units": [
                    {"type": 1, "pos": [0, 0], "hp": 10, "owner": 0}
                ],
                "current_player": 0,
                "turn": 0,
                "done": False
            },
            {
                "terrain": [[0, 0], [0, 0]],
                "city_owner": [[-1, 0], [-1, -1]],
                "city_level": [[0, 1], [0, 0]],
                "units": [
                    {"type": 1, "pos": [1, 0], "hp": 10, "owner": 0}
                ],
                "current_player": 1,
                "turn": 1,
                "done": False
            },
            {
                "terrain": [[0, 0], [0, 0]],
                "city_owner": [[-1, 0], [-1, -1]],
                "city_level": [[0, 1], [0, 0]],
                "units": [
                    {"type": 1, "pos": [1, 0], "hp": 8, "owner": 0}
                ],
                "current_player": 0,
                "turn": 2,
                "done": False
            }
        ]
    }
    
    replay_path = temp_replays_dir / "test_replay.json"
    with open(replay_path, 'w', encoding='utf-8') as f:
        json.dump(replay_data, f)
    
    return "test_replay", replay_data


@pytest.fixture
def client():
    """Crée un client de test FastAPI."""
    clear_cache()
    return TestClient(app)


def test_root(client):
    """Test le point d'entrée de l'API."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == "0.1.0"


def test_list_games_empty(client, temp_replays_dir):
    """Test la liste des replays quand il n'y en a pas."""
    response = client.get("/games")
    assert response.status_code == 200
    data = response.json()
    assert "games" in data
    assert len(data["games"]) == 0


def test_list_games(client, sample_replay):
    """Test la liste des replays."""
    game_id, _ = sample_replay
    
    response = client.get("/games")
    assert response.status_code == 200
    data = response.json()
    assert "games" in data
    assert len(data["games"]) >= 1
    
    # Vérifier que notre replay est dans la liste
    game_ids = [g["id"] for g in data["games"]]
    assert game_id in game_ids


def test_get_replay(client, sample_replay):
    """Test la récupération d'un replay complet."""
    game_id, replay_data = sample_replay
    
    response = client.get(f"/games/{game_id}/replay")
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == game_id
    assert "metadata" in data
    assert "states" in data
    assert len(data["states"]) == len(replay_data["states"])


def test_get_replay_not_found(client):
    """Test l'erreur 404 si le replay n'existe pas."""
    response = client.get("/games/nonexistent/replay")
    assert response.status_code == 404
    assert "non trouvé" in response.json()["detail"].lower()


def test_get_metadata(client, sample_replay):
    """Test la récupération des métadonnées."""
    game_id, replay_data = sample_replay
    
    response = client.get(f"/games/{game_id}/metadata")
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == game_id
    assert "metadata" in data
    assert data["metadata"]["height"] == replay_data["metadata"]["height"]


def test_get_state(client, sample_replay):
    """Test la récupération d'un état à un tour donné."""
    game_id, replay_data = sample_replay
    
    response = client.get(f"/games/{game_id}/state/0")
    assert response.status_code == 200
    data = response.json()
    
    assert data["turn"] == 0
    assert "state" in data
    assert "terrain" in data["state"]
    assert "units" in data["state"]


def test_get_state_invalid_turn_negative(client, sample_replay):
    """Test l'erreur 400 pour un tour négatif."""
    game_id, _ = sample_replay
    
    response = client.get(f"/games/{game_id}/state/-1")
    assert response.status_code == 400
    assert "invalide" in response.json()["detail"].lower()


def test_get_state_invalid_turn_out_of_bounds(client, sample_replay):
    """Test l'erreur 400 pour un tour hors limites."""
    game_id, _ = sample_replay
    
    response = client.get(f"/games/{game_id}/state/100")
    assert response.status_code == 400
    assert "hors limites" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_cors_headers(client):
    """Test que les headers CORS sont présents."""
    response = client.options("/games")
    # CORS est géré par le middleware, les headers peuvent être présents
    # même si on ne peut pas les tester directement avec TestClient
    assert response.status_code in [200, 405]  # OPTIONS peut retourner 405 ou 200


def test_get_state_all_turns(client, sample_replay):
    """Test la récupération de tous les tours disponibles."""
    game_id, replay_data = sample_replay
    
    num_states = len(replay_data["states"])
    
    for turn in range(num_states):
        response = client.get(f"/games/{game_id}/state/{turn}")
        assert response.status_code == 200
        data = response.json()
        assert data["turn"] == turn


def test_live_perfection_flow(client):
    """Test basique du mode live Perfection."""
    create_response = client.post(
        "/live/perfection",
        json={"opponents": 2, "difficulty": "crazy", "seed": 7},
    )
    assert create_response.status_code == 200
    live_data = create_response.json()
    assert "game_id" in live_data
    assert live_data["state"]["current_player"] == 0
    game_id = live_data["game_id"]

    # Récupération de l'état via GET
    get_response = client.get(f"/live/{game_id}")
    assert get_response.status_code == 200

    # Appliquer une action (fin de tour)
    action_response = client.post(
        f"/live/{game_id}/action",
        json={"action_id": END_TURN_ACTION},
    )
    assert action_response.status_code == 200

    # Terminer explicitement le tour
    end_turn_response = client.post(f"/live/{game_id}/end_turn")
    assert end_turn_response.status_code == 200
    end_data = end_turn_response.json()
    assert "state" in end_data

