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
    LiveGameConfig,
    LiveGameResponse,
    LiveActionPayload,
)
from .replay_store import (
    list_replays,
    load_replay,
    get_replay_metadata,
    get_state_at_turn,
    ReplayNotFoundError,
    InvalidTurnError,
)
from .live_game_store import (
    create_perfection_game,
    get_game as get_live_game,
    apply_action as apply_live_action,
    end_turn as end_live_turn,
    serialize_session,
    LiveGameNotFound,
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


@app.post("/live/perfection", response_model=LiveGameResponse, tags=["Live"])
async def start_live_perfection_game(config: LiveGameConfig):
    """Crée une nouvelle partie live du mode Perfection."""
    session = create_perfection_game(
        opponents=config.opponents,
        difficulty=config.difficulty,
        seed=config.seed,
    )
    return _session_to_response(session)


@app.get("/live/{game_id}", response_model=LiveGameResponse, tags=["Live"])
async def get_live_game_state(game_id: str):
    """Retourne l'état courant d'une partie live."""
    try:
        session = get_live_game(game_id)
        return _session_to_response(session)
    except LiveGameNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.post("/live/{game_id}/action", response_model=LiveGameResponse, tags=["Live"])
async def post_live_action(game_id: str, payload: LiveActionPayload):
    """Applique une action encodée et retourne le nouvel état."""
    try:
        session = apply_live_action(game_id, payload.action_id)
        return _session_to_response(session)
    except LiveGameNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.post("/live/{game_id}/end_turn", response_model=LiveGameResponse, tags=["Live"])
async def post_live_end_turn(game_id: str):
    """Termine explicitement le tour du joueur humain."""
    try:
        session = end_live_turn(game_id)
        return _session_to_response(session)
    except LiveGameNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


def _session_to_response(session):
    serialized = serialize_session(session)
    state_view = GameStateView.from_raw_state(serialized["state"])
    return LiveGameResponse(
        game_id=serialized["game_id"],
        max_turns=serialized["max_turns"],
        opponents=serialized["opponents"],
        difficulty=serialized["difficulty"],
        state=state_view,
    )

