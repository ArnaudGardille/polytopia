"""Tests pour le module replay_store."""

import json
import pytest
from pathlib import Path
import tempfile
import shutil

from polytopia_jax.web.replay_store import (
    list_replays,
    load_replay,
    get_replay_metadata,
    get_state_at_turn,
    _validate_game_id,
    ReplayNotFoundError,
    InvalidTurnError,
    clear_cache,
)


@pytest.fixture
def temp_replays_dir(monkeypatch):
    """Crée un dossier temporaire pour les replays."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Créer un sous-dossier replays
    replays_dir = temp_dir / "replays"
    replays_dir.mkdir()
    
    # Mock la fonction _get_replays_dir pour utiliser le dossier temporaire
    from polytopia_jax.web import replay_store
    original_get_replays_dir = replay_store._get_replays_dir
    
    def mock_get_replays_dir():
        return replays_dir
    
    monkeypatch.setattr(replay_store, "_get_replays_dir", mock_get_replays_dir)
    
    yield replays_dir
    
    # Nettoyer
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
            "final_turn": 3,
            "total_actions": 10,
            "game_done": False
        },
        "states": [
            {
                "terrain": [[0, 0], [0, 0]],
                "city_owner": [[-1, -1], [-1, 0]],
                "city_level": [[0, 0], [0, 1]],
                "units": [{"type": 1, "pos": [0, 0], "hp": 10, "owner": 0}],
                "current_player": 0,
                "turn": 0,
                "done": False
            },
            {
                "terrain": [[0, 0], [0, 0]],
                "city_owner": [[-1, -1], [-1, 0]],
                "city_level": [[0, 0], [0, 1]],
                "units": [{"type": 1, "pos": [1, 0], "hp": 10, "owner": 0}],
                "current_player": 1,
                "turn": 1,
                "done": False
            }
        ]
    }
    
    replay_path = temp_replays_dir / "test_replay.json"
    with open(replay_path, 'w', encoding='utf-8') as f:
        json.dump(replay_data, f)
    
    return "test_replay", replay_data


def test_validate_game_id():
    """Test la validation des IDs de replay."""
    # ID valide
    assert _validate_game_id("test_game") == "test_game"
    assert _validate_game_id("  test_game  ") == "test_game"
    
    # IDs invalides
    with pytest.raises(ValueError, match="path traversal"):
        _validate_game_id("../test")
    
    with pytest.raises(ValueError, match="path traversal"):
        _validate_game_id("test/../other")
    
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        _validate_game_id("")
    
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        _validate_game_id("   ")


def test_list_replays_empty(temp_replays_dir):
    """Test la liste des replays quand le dossier est vide."""
    replays = list_replays()
    assert replays == []


def test_list_replays(sample_replay, temp_replays_dir):
    """Test la liste des replays."""
    game_id, _ = sample_replay
    
    replays = list_replays()
    assert len(replays) == 1
    assert replays[0]["id"] == game_id
    assert "metadata" in replays[0]


def test_load_replay(sample_replay):
    """Test le chargement d'un replay existant."""
    game_id, replay_data = sample_replay
    clear_cache()
    
    loaded = load_replay(game_id)
    assert loaded["metadata"] == replay_data["metadata"]
    assert len(loaded["states"]) == len(replay_data["states"])


def test_load_replay_not_found():
    """Test l'erreur si le replay n'existe pas."""
    clear_cache()
    
    with pytest.raises(ReplayNotFoundError):
        load_replay("nonexistent_replay")


def test_load_replay_cache(sample_replay):
    """Test que le cache fonctionne."""
    game_id, _ = sample_replay
    clear_cache()
    
    # Premier chargement
    replay1 = load_replay(game_id)
    
    # Deuxième chargement (devrait utiliser le cache)
    replay2 = load_replay(game_id)
    
    assert replay1 == replay2


def test_get_replay_metadata(sample_replay):
    """Test l'extraction des métadonnées."""
    game_id, replay_data = sample_replay
    clear_cache()
    
    metadata = get_replay_metadata(game_id)
    assert metadata == replay_data["metadata"]


def test_get_state_at_turn(sample_replay):
    """Test la récupération d'un état à un tour valide."""
    game_id, replay_data = sample_replay
    clear_cache()
    
    state = get_state_at_turn(game_id, 0)
    assert state == replay_data["states"][0]
    
    state = get_state_at_turn(game_id, 1)
    assert state == replay_data["states"][1]


def test_get_state_at_turn_invalid(sample_replay):
    """Test l'erreur si le tour est invalide."""
    game_id, _ = sample_replay
    clear_cache()
    
    # Tour négatif
    with pytest.raises(InvalidTurnError):
        get_state_at_turn(game_id, -1)
    
    # Tour hors limites
    with pytest.raises(InvalidTurnError):
        get_state_at_turn(game_id, 100)


def test_list_replays_ignores_invalid(temp_replays_dir):
    """Test que les fichiers invalides sont ignorés."""
    # Créer un fichier JSON invalide
    invalid_path = temp_replays_dir / "invalid.json"
    with open(invalid_path, 'w') as f:
        f.write("{ invalid json }")
    
    # Créer un fichier qui n'est pas un replay valide
    not_replay_path = temp_replays_dir / "not_replay.json"
    with open(not_replay_path, 'w') as f:
        json.dump({"not": "a replay"}, f)
    
    replays = list_replays()
    # Ne devrait pas inclure les fichiers invalides
    assert len(replays) == 0

