"""Stockage en mémoire des parties live (mode Perfection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import random
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import jax
import jax.numpy as jnp

from polytopia_jax.core.init import init_random, GameConfig as EngineGameConfig
from polytopia_jax.core.rules import step
from polytopia_jax.core.actions import END_TURN_ACTION, decode_action, ActionType
from polytopia_jax.ai.strategies import _build_context
from polytopia_jax.core.state import GameState, GameMode
from polytopia_jax.ai import (
    DifficultyPreset,
    StrategyAI,
    resolve_difficulty,
    resolve_strategy_name,
)

from .serialize import state_to_dict, dict_to_state
from .view_options import ViewOptions, resolve_view_options


HUMAN_PLAYER_ID = 0
PERFECTION_MAX_TURNS = 30


@dataclass
class LiveGameSession:
    """Représente une partie live maintenue en mémoire."""

    id: str
    state: GameState
    max_turns: int = PERFECTION_MAX_TURNS
    opponents: int = 3
    difficulty: str = "crazy"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ai_agents: Dict[int, StrategyAI] = field(default_factory=dict)
    strategy: str = "rush"
    view_options: ViewOptions = field(
        default_factory=lambda: resolve_view_options()
    )
    initial_seed: Optional[int] = None  # Seed initial utilisé pour générer la carte


_LIVE_GAMES: Dict[str, LiveGameSession] = {}


def _get_live_games_dir() -> Path:
    """Retourne le chemin du dossier live_games."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "live_games"


def _get_live_game_path(game_id: str) -> Path:
    """Retourne le chemin complet vers un fichier de partie live."""
    live_games_dir = _get_live_games_dir()
    live_games_dir.mkdir(exist_ok=True)
    return live_games_dir / f"{game_id}.json"


def _save_session(session: LiveGameSession) -> None:
    """Sauvegarde une session sur disque."""
    try:
        game_path = _get_live_game_path(session.id)
        state_dict = state_to_dict(session.state)
        
        # Sauvegarder les métadonnées nécessaires pour recréer les agents IA
        data = {
            "id": session.id,
            "state": state_dict,
            "max_turns": session.max_turns,
            "opponents": session.opponents,
            "difficulty": session.difficulty,
            "strategy": session.strategy,
            "created_at": session.created_at.isoformat(),
            "ai_agents": {
                str(player_id): {
                    "player_id": player_id,
                    "strategy_name": session.ai_agents[player_id].strategy_name,
                }
                for player_id in session.ai_agents.keys()
            },
            "initial_seed": session.initial_seed,
            "view_options": {
                "reveal_map": session.view_options.reveal_map,
                "unlock_all_techs": session.view_options.unlock_all_techs,
            },
        }
        
        with open(game_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Erreur lors de la sauvegarde de la partie {session.id}: {e}")


def _load_session(game_id: str) -> Optional[LiveGameSession]:
    """Charge une session depuis le disque."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        game_path = _get_live_game_path(game_id)
        if not game_path.exists():
            return None
        
        try:
            with open(game_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                f"Fichier JSON corrompu pour la partie {game_id}: {e}. "
                f"Ligne {e.lineno}, colonne {e.colno}. "
                f"Le fichier sera ignoré."
            )
            # Optionnel: déplacer le fichier corrompu dans un dossier backup
            _backup_corrupted_file(game_path)
            return None
        
        # Reconstruire l'état
        state = dict_to_state(data["state"])
        
        # Restaurer le bonus de difficulté
        difficulty_preset = resolve_difficulty(data.get("difficulty", "crazy"))
        bonus = jnp.zeros(state.num_players, dtype=jnp.int32)
        for player_id in range(1, state.num_players):
            bonus = bonus.at[player_id].set(difficulty_preset.star_bonus)
        state = state.replace(player_income_bonus=bonus)
        
        # Recréer les agents IA
        ai_agents = {}
        ai_agents_data = data.get("ai_agents", {})
        strategy_name = resolve_strategy_name(data.get("strategy", "rush"))
        initial_seed = data.get("initial_seed")
        
        for player_id_str, agent_data in ai_agents_data.items():
            player_id = int(player_id_str)
            # Utiliser le seed initial + player_id pour recréer l'agent de manière cohérente
            if initial_seed is not None:
                seed = (initial_seed + player_id) % (2**31)
            else:
                # Fallback si le seed initial n'est pas disponible
                seed = hash(game_id[:8] + str(player_id)) % (2**31)
            ai_agents[player_id] = StrategyAI(
                player_id,
                strategy_name=resolve_strategy_name(agent_data.get("strategy_name", strategy_name)),
                seed=seed,
            )
        
        # Si aucun agent n'a été chargé, les recréer depuis les métadonnées
        if not ai_agents:
            num_players = int(state.num_players)
            for player_id in range(1, num_players):
                if initial_seed is not None:
                    seed = (initial_seed + player_id) % (2**31)
                else:
                    seed = hash(game_id[:8] + str(player_id)) % (2**31)
                ai_agents[player_id] = StrategyAI(
                    player_id,
                    strategy_name=strategy_name,
                    seed=seed,
                )
        
        # Restaurer les view_options
        view_options_data = data.get("view_options", {})
        view_options = ViewOptions(
            reveal_map=view_options_data.get("reveal_map", False),
            unlock_all_techs=view_options_data.get("unlock_all_techs", False),
        )
        
        # Restaurer la date de création
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)
        
        session = LiveGameSession(
            id=data["id"],
            state=state,
            max_turns=data.get("max_turns", PERFECTION_MAX_TURNS),
            opponents=data.get("opponents", 3),
            difficulty=data.get("difficulty", "crazy"),
            ai_agents=ai_agents,
            strategy=strategy_name,
            view_options=view_options,
            created_at=created_at,
            initial_seed=initial_seed,
        )
        
        return session
    except Exception as e:
        logger.warning(f"Erreur lors du chargement de la partie {game_id}: {e}")
        return None


def _backup_corrupted_file(file_path: Path) -> None:
    """Déplace un fichier corrompu dans un dossier backup."""
    try:
        backup_dir = file_path.parent / "corrupted_backup"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / file_path.name
        file_path.rename(backup_path)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Fichier corrompu déplacé vers {backup_path}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Impossible de sauvegarder le fichier corrompu {file_path}: {e}")


def load_all_sessions() -> None:
    """Charge toutes les parties sauvegardées depuis le disque."""
    live_games_dir = _get_live_games_dir()
    if not live_games_dir.exists():
        return
    
    loaded_count = 0
    for json_file in live_games_dir.glob("*.json"):
        game_id = json_file.stem
        if game_id in _LIVE_GAMES:
            continue  # Déjà chargé
        
        session = _load_session(game_id)
        if session:
            _LIVE_GAMES[game_id] = session
            loaded_count += 1
    
    if loaded_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Chargement de {loaded_count} partie(s) live depuis le disque")


class LiveGameNotFound(Exception):
    """Levée quand une partie live est introuvable."""


def create_perfection_game(
    opponents: int = 3,
    difficulty: str = "crazy",
    strategy: Optional[str] = None,
    seed: Optional[int] = None,
    view_options: Optional[ViewOptions] = None,
) -> LiveGameSession:
    """Crée une partie Perfection live et la stocke."""
    num_players = _clamp_players(opponents + 1)
    board_size = _compute_board_size(opponents)
    key_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    key = jax.random.PRNGKey(key_seed)

    engine_config = EngineGameConfig(
        height=board_size,
        width=board_size,
        num_players=num_players,
        max_units=max(64, board_size * 2),
        game_mode=GameMode.PERFECTION,
        max_turns=PERFECTION_MAX_TURNS,
    )

    state = init_random(key, engine_config)
    difficulty_preset = resolve_difficulty(difficulty)
    strategy_name = resolve_strategy_name(strategy)
    state = _apply_difficulty_bonuses(state, difficulty_preset)
    ai_agents = {
        player_id: StrategyAI(
            player_id,
            strategy_name=strategy_name,
            seed=(key_seed + player_id),
        )
        for player_id in range(1, num_players)
    }
    state = _enforce_turn_limit(state, PERFECTION_MAX_TURNS)

    game_id = uuid4().hex
    session = LiveGameSession(
        id=game_id,
        state=state,
        max_turns=PERFECTION_MAX_TURNS,
        opponents=opponents,
        difficulty=difficulty_preset.name,
        ai_agents=ai_agents,
        strategy=strategy_name,
        view_options=view_options or resolve_view_options(),
        initial_seed=key_seed,
    )
    session.state = _advance_ai_turns(session, PERFECTION_MAX_TURNS)
    _LIVE_GAMES[game_id] = session
    _save_session(session)
    return session


def get_game(game_id: str) -> LiveGameSession:
    """Retourne une partie live par ID."""
    # Si la partie n'est pas en mémoire, essayer de la charger depuis le disque
    if game_id not in _LIVE_GAMES:
        session = _load_session(game_id)
        if session:
            _LIVE_GAMES[game_id] = session
        else:
            raise LiveGameNotFound(f"Partie live '{game_id}' introuvable")
    
    try:
        return _LIVE_GAMES[game_id]
    except KeyError as exc:
        raise LiveGameNotFound(f"Partie live '{game_id}' introuvable") from exc


def remove_game(game_id: str) -> None:
    """Supprime une partie live (utilisé pour le nettoyage)."""
    _LIVE_GAMES.pop(game_id, None)


def apply_action(game_id: str, action_id: int) -> LiveGameSession:
    """Applique une action utilisateur sur la partie live."""
    session = get_game(game_id)
    state = step(session.state, action_id)
    state = _enforce_turn_limit(state, session.max_turns)
    session.state = state
    session.state = _advance_ai_turns(session, session.max_turns)
    _save_session(session)
    return session


def end_turn(game_id: str) -> LiveGameSession:
    """Termine explicitement le tour du joueur humain."""
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    logger.info(f"[END_TURN] Début de la fin de tour pour game_id={game_id}")
    start = time.time()
    
    session = get_game(game_id)
    logger.info(f"[END_TURN] Session récupérée en {time.time()-start:.3f}s")
    
    step_start = time.time()
    state = step(session.state, END_TURN_ACTION)
    logger.info(f"[END_TURN] Step END_TURN_ACTION en {time.time()-step_start:.3f}s")
    
    state = _enforce_turn_limit(state, session.max_turns)
    session.state = state
    
    ai_start = time.time()
    logger.info(f"[END_TURN] Début _advance_ai_turns...")
    session.state = _advance_ai_turns(session, session.max_turns)
    logger.info(f"[END_TURN] _advance_ai_turns terminé en {time.time()-ai_start:.3f}s")
    
    _save_session(session)
    logger.info(f"[END_TURN] Total: {time.time()-start:.3f}s")
    return session


def serialize_session(session: LiveGameSession) -> dict:
    """Retourne le payload sérialisé pour l'API."""
    return {
        "game_id": session.id,
        "state": state_to_dict(session.state),
        "max_turns": session.max_turns,
        "opponents": session.opponents,
        "difficulty": session.difficulty,
        "strategy": session.strategy,
    }


def _clamp_players(num_players: int) -> int:
    return max(2, min(4, num_players))


def _compute_board_size(opponents: int) -> int:
    opponents = max(1, opponents)
    base = 10
    return base + min(5, opponents)


def _advance_ai_turns(session: LiveGameSession, max_turns: int) -> GameState:
    import logging
    logger = logging.getLogger(__name__)
    
    state = session.state
    loop_guard = 0
    while not _is_done(state) and _current_player(state) != HUMAN_PLAYER_ID:
        player_id = _current_player(state)
        logger.info(f"[AI Turn] Joueur {player_id} commence son tour (loop_guard={loop_guard})")
        
        agent = session.ai_agents.get(player_id)
        if agent is None:
            logger.warning(f"[AI Turn] Aucun agent trouvé pour le joueur {player_id}, fin de tour forcée")
            state = step(state, END_TURN_ACTION)
        else:
            previous_player = _current_player(state)
            state = _play_ai_turn(state, agent)
            # Vérifier que le joueur a changé après le tour de l'IA
            if _current_player(state) == previous_player and not _is_done(state):
                logger.warning(f"[AI Turn] Le joueur {player_id} n'a pas changé après son tour, forcer END_TURN")
                state = step(state, END_TURN_ACTION)
        
        state = _enforce_turn_limit(state, max_turns)
        loop_guard += 1
        
        if loop_guard > 32:
            logger.error(f"[AI Turn] Loop guard dépassé ! Arrêt forcé après {loop_guard} itérations")
            # Forcer la fin du tour pour éviter une boucle infinie
            if _current_player(state) != HUMAN_PLAYER_ID and not _is_done(state):
                logger.error(f"[AI Turn] Forcer END_TURN pour éviter la boucle infinie")
                state = step(state, END_TURN_ACTION)
            break
            
    logger.info(f"[AI Turn] Fin des tours IA, retour au joueur humain (total iterations: {loop_guard})")
    return state


def _enforce_turn_limit(state: GameState, max_turns: int) -> GameState:
    current_turn = int(jnp.asarray(state.turn))
    if current_turn >= max_turns:
        return state.replace(done=jnp.array(True, dtype=jnp.bool_))
    return state


def _is_done(state: GameState) -> bool:
    return bool(jnp.asarray(state.done))


def _current_player(state: GameState) -> int:
    return int(jnp.asarray(state.current_player))


def _play_ai_turn(state: GameState, agent: StrategyAI) -> GameState:
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    local_guard = 0
    # Limiter le nombre d'actions pour éviter les boucles infinies ou les tests trop longs
    # Utiliser un maximum raisonnable plutôt que max_units * 2 qui peut être très grand
    max_actions = min(64, int(state.max_units * 2))  # Maximum de 64 actions par tour pour éviter les timeouts
    start_time = time.time()
    max_time = 5.0  # Maximum 5 secondes par tour d'IA (réduit grâce à la mise en cache du contexte)
    
    logger.info(f"[AI Turn] Joueur {agent.player_id} joue, max_actions={max_actions}")
    
    # Construire le contexte une seule fois au début du tour
    context = _build_context(state, agent.player_id)
    
    while not _is_done(state) and _current_player(state) == agent.player_id:
        # Vérifier le timeout
        if time.time() - start_time > max_time:
            logger.warning(f"[AI Turn] Joueur {agent.player_id} timeout après {time.time() - start_time:.2f}s, fin de tour forcée")
            state = step(state, END_TURN_ACTION)
            break
        
        if local_guard >= max_actions:
            # Forcer la fin du tour si trop d'actions
            logger.warning(f"[AI Turn] Joueur {agent.player_id} a atteint la limite de {max_actions} actions, fin de tour forcée")
            state = step(state, END_TURN_ACTION)
            # Vérifier que le joueur a bien changé, sinon sortir immédiatement
            if _current_player(state) == agent.player_id:
                logger.error(f"[AI Turn] END_TURN_ACTION n'a pas changé le joueur courant ! Sortie forcée.")
            break
        
        # Utiliser le contexte mis en cache
        action = agent.choose_action(state, context)
        logger.debug(f"[AI Turn] Joueur {agent.player_id} action #{local_guard}: {action}")
        state = step(state, action)
        
        # Reconstruire le contexte après les actions qui modifient les positions des unités
        # (MOVE et ATTACK modifient les positions, donc le contexte doit être mis à jour)
        decoded = decode_action(action)
        action_type = decoded.get("action_type")
        if action_type in (ActionType.MOVE.value, ActionType.ATTACK.value):
            context = _build_context(state, agent.player_id)
        
        local_guard += 1
    
    elapsed = time.time() - start_time
    logger.info(f"[AI Turn] Joueur {agent.player_id} a terminé après {local_guard} actions en {elapsed:.2f}s")
    return state


def _apply_difficulty_bonuses(state: GameState, preset: DifficultyPreset) -> GameState:
    """Injecte les bonus de revenu dans l'état."""
    bonus = jnp.zeros(state.num_players, dtype=jnp.int32)
    for player_id in range(1, state.num_players):
        bonus = bonus.at[player_id].set(preset.star_bonus)
    return state.replace(player_income_bonus=bonus)
