"""Gestion du stockage et lecture des replays JSON."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class ReplayNotFoundError(Exception):
    """Exception levée quand un replay n'est pas trouvé."""
    pass


class InvalidTurnError(Exception):
    """Exception levée quand un tour est invalide."""
    pass


# Cache simple en mémoire pour les replays chargés
_replay_cache: Dict[str, dict] = {}


def _get_replays_dir() -> Path:
    """Retourne le chemin du dossier replays."""
    # Le dossier replays est à la racine du projet
    # polytopia_jax/web/replay_store.py -> polytopia_jax -> projet root
    project_root = Path(__file__).parent.parent.parent
    return project_root / "replays"


def _validate_game_id(game_id: str) -> str:
    """Valide et nettoie l'ID d'un replay (sécurité contre path traversal).
    
    Args:
        game_id: ID du replay à valider
        
    Returns:
        ID nettoyé
        
    Raises:
        ValueError: Si l'ID contient des caractères invalides
    """
    # Nettoyer l'ID
    game_id = game_id.strip()
    
    # Vérifier qu'il n'y a pas de path traversal
    if ".." in game_id or "/" in game_id or "\\" in game_id:
        raise ValueError(f"path traversal détecté pour l'ID: {game_id}")
    
    # Vérifier que l'ID n'est pas vide
    if not game_id:
        raise ValueError("ID de replay ne peut pas être vide")
    
    return game_id


def _get_replay_path(game_id: str) -> Path:
    """Retourne le chemin complet vers un fichier replay.
    
    Args:
        game_id: ID du replay (nom sans extension)
        
    Returns:
        Chemin vers le fichier JSON
    """
    game_id = _validate_game_id(game_id)
    replays_dir = _get_replays_dir()
    return replays_dir / f"{game_id}.json"


def list_replays() -> List[Dict]:
    """Liste tous les replays disponibles dans le dossier replays/.
    
    Returns:
        Liste de dictionnaires contenant 'id' et 'metadata' pour chaque replay
    """
    replays_dir = _get_replays_dir()
    
    if not replays_dir.exists():
        return []
    
    replays = []
    
    # Parcourir tous les fichiers JSON dans le dossier replays
    for json_file in replays_dir.glob("*.json"):
        game_id = json_file.stem  # Nom sans extension
        
        try:
            # Charger uniquement les métadonnées
            metadata = get_replay_metadata(game_id)
            replays.append({
                "id": game_id,
                "metadata": metadata
            })
        except (ReplayNotFoundError, json.JSONDecodeError, KeyError):
            # Ignorer les fichiers corrompus ou invalides
            continue
    
    return replays


def load_replay(game_id: str, use_cache: bool = True) -> dict:
    """Charge un replay depuis un fichier.
    
    Args:
        game_id: ID du replay (nom sans extension)
        use_cache: Utiliser le cache en mémoire si disponible
        
    Returns:
        Dictionnaire complet du replay avec 'metadata' et 'states'
        
    Raises:
        ReplayNotFoundError: Si le replay n'existe pas
    """
    game_id = _validate_game_id(game_id)
    
    # Vérifier le cache
    if use_cache and game_id in _replay_cache:
        return _replay_cache[game_id]
    
    replay_path = _get_replay_path(game_id)
    
    if not replay_path.exists():
        raise ReplayNotFoundError(f"Replay '{game_id}' non trouvé")
    
    try:
        with open(replay_path, 'r', encoding='utf-8') as f:
            replay = json.load(f)
        
        # Valider la structure
        if "metadata" not in replay or "states" not in replay:
            raise ValueError(f"Format de replay invalide pour '{game_id}'")
        
        # Mettre en cache
        if use_cache:
            _replay_cache[game_id] = replay
        
        return replay
    
    except json.JSONDecodeError as e:
        raise ReplayNotFoundError(f"Erreur de lecture du replay '{game_id}': {e}")
    except Exception as e:
        raise ReplayNotFoundError(f"Erreur lors du chargement du replay '{game_id}': {e}")


def get_replay_metadata(game_id: str) -> dict:
    """Extrait les métadonnées d'un replay.
    
    Args:
        game_id: ID du replay
        
    Returns:
        Dictionnaire des métadonnées
        
    Raises:
        ReplayNotFoundError: Si le replay n'existe pas
    """
    replay = load_replay(game_id)
    return replay.get("metadata", {})


def get_state_at_turn(game_id: str, turn: int) -> dict:
    """Récupère l'état du jeu à un tour donné.
    
    Args:
        game_id: ID du replay
        turn: Numéro de tour (>= 0)
        
    Returns:
        Dictionnaire représentant l'état du jeu à ce tour
        
    Raises:
        ReplayNotFoundError: Si le replay n'existe pas
        InvalidTurnError: Si le tour est invalide (hors limites)
    """
    if turn < 0:
        raise InvalidTurnError(f"Tour invalide: {turn} (doit être >= 0)")
    
    replay = load_replay(game_id)
    states = replay.get("states", [])
    
    if not states:
        raise InvalidTurnError(f"Aucun état disponible dans le replay '{game_id}'")
    
    # Vérifier que le tour demandé existe
    if turn >= len(states):
        max_turn = len(states) - 1
        raise InvalidTurnError(
            f"Tour {turn} hors limites pour le replay '{game_id}' "
            f"(tours disponibles: 0-{max_turn})"
        )
    
    return states[turn]


def clear_cache():
    """Vide le cache des replays (utile pour les tests)."""
    global _replay_cache
    _replay_cache.clear()
