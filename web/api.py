"""Application FastAPI pour visualiser les replays de parties."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    GamesListResponse,
    GameInfo,
    ReplayMetadata,
    ReplayResponse,
    StateResponse,
    GameStateView,
)
from .replay_store import (
    list_replays,
    load_replay,
    get_replay_metadata,
    get_state_at_turn,
    ReplayNotFoundError,
    InvalidTurnError,
)

# Créer l'application FastAPI
app = FastAPI(
    title="Polytopia-JAX API",
    description="API pour visualiser les replays de parties Polytopia",
    version="0.1.0",
)

# Configurer CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En développement, autoriser toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée de l'API."""
    return {
        "message": "Polytopia-JAX API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/games", response_model=GamesListResponse, tags=["Games"])
async def list_games():
    """Liste tous les replays disponibles.
    
    Returns:
        Liste des replays avec leurs métadonnées
    """
    replays = list_replays()
    
    # Convertir en modèles Pydantic
    games = []
    for replay_data in replays:
        try:
            game_info = GameInfo(
                id=replay_data["id"],
                metadata=ReplayMetadata(**replay_data["metadata"])
            )
            games.append(game_info)
        except Exception:
            # Ignorer les replays avec des métadonnées invalides
            continue
    
    return GamesListResponse(games=games)


@app.get("/games/{game_id}/metadata", response_model=dict, tags=["Games"])
async def get_metadata(game_id: str):
    """Récupère uniquement les métadonnées d'une partie.
    
    Args:
        game_id: Identifiant du replay (nom sans extension)
        
    Returns:
        Métadonnées du replay
        
    Raises:
        404: Si le replay n'est pas trouvé
    """
    try:
        metadata = get_replay_metadata(game_id)
        return {
            "id": game_id,
            "metadata": metadata
        }
    except ReplayNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.get("/games/{game_id}/replay", response_model=ReplayResponse, tags=["Games"])
async def get_replay(game_id: str):
    """Récupère le replay complet d'une partie.
    
    Args:
        game_id: Identifiant du replay (nom sans extension)
        
    Returns:
        Replay complet avec métadonnées et tous les états
        
    Raises:
        404: Si le replay n'est pas trouvé
    """
    try:
        replay = load_replay(game_id)
        
        # Convertir les états en GameStateView
        states = [
            GameStateView.from_raw_state(state)
            for state in replay.get("states", [])
        ]
        
        return ReplayResponse(
            id=game_id,
            metadata=ReplayMetadata(**replay.get("metadata", {})),
            states=states
        )
    except ReplayNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du chargement du replay: {str(e)}"
        )


@app.get("/games/{game_id}/state/{turn}", response_model=StateResponse, tags=["Games"])
async def get_state(game_id: str, turn: int):
    """Récupère l'état du jeu à un tour spécifique.
    
    Args:
        game_id: Identifiant du replay (nom sans extension)
        turn: Numéro de tour (>= 0)
        
    Returns:
        État du jeu à ce tour
        
    Raises:
        404: Si le replay n'est pas trouvé
        400: Si le tour est invalide (hors limites)
    """
    try:
        raw_state = get_state_at_turn(game_id, turn)
        state_view = GameStateView.from_raw_state(raw_state)
        
        return StateResponse(
            turn=turn,
            state=state_view
        )
    except ReplayNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except InvalidTurnError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du chargement de l'état: {str(e)}"
        )

