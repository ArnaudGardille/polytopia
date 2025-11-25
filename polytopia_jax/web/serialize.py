"""Sérialisation de GameState en format JSON."""

from polytopia_jax.core.state import GameState, NO_OWNER
from polytopia_jax.core.rules import CITY_STAR_INCOME_PER_LEVEL


def state_to_dict(state: GameState) -> dict:
    """Convertit un GameState JAX en dictionnaire Python sérialisable.
    
    Args:
        state: État du jeu à sérialiser
    
    Returns:
        Dictionnaire contenant toutes les données de l'état en format JSON-compatible
    """
    # Convertir les arrays JAX en arrays numpy puis en listes Python
    import numpy as np
    
    # Forcer la conversion en array numpy puis en liste Python native
    terrain_arr = np.array(state.terrain, copy=True)
    city_owner_arr = np.array(state.city_owner, copy=True)
    city_level_arr = np.array(state.city_level, copy=True)
    city_population_arr = np.array(state.city_population, copy=True)
    city_has_port_arr = np.array(state.city_has_port, copy=True)
    resource_type_arr = np.array(state.resource_type, copy=True)
    resource_available_arr = np.array(state.resource_available, copy=True)

    terrain = terrain_arr.tolist()
    city_owner = city_owner_arr.tolist()
    city_level = city_level_arr.tolist()
    city_population = city_population_arr.tolist()
    city_has_port = city_has_port_arr.tolist()
    resource_type = resource_type_arr.tolist()
    resource_available = resource_available_arr.astype(bool).tolist()
    
    # Extraire uniquement les unités actives (optimisé)
    active_units = []
    units_active_arr = np.array(state.units_active, copy=True)
    
    # Trouver les indices des unités actives pour éviter de boucler sur toutes les unités
    active_indices = np.where(units_active_arr)[0]
    
    if len(active_indices) > 0:
        units_type_arr = np.array(state.units_type, copy=True)
        units_pos_arr = np.array(state.units_pos, copy=True)
        units_hp_arr = np.array(state.units_hp, copy=True)
        units_owner_arr = np.array(state.units_owner, copy=True)
        units_has_acted_arr = np.array(state.units_has_acted, copy=True)
        units_payload_arr = np.array(state.units_payload_type, copy=True)
        
        for i in active_indices:
            unit = {
                "id": int(i),
                "type": int(units_type_arr[i]),
                "pos": [int(units_pos_arr[i, 0]), int(units_pos_arr[i, 1])],
                "hp": int(units_hp_arr[i]),
                "owner": int(units_owner_arr[i]),
                "has_acted": bool(units_has_acted_arr[i]),
                "payload_type": int(units_payload_arr[i]),
            }
            active_units.append(unit)
    
    # Sérialiser uniquement les payloads des unités actives (optimisation)
    units_payload_serialized = []
    if len(active_indices) > 0:
        units_payload_arr = np.array(state.units_payload_type, copy=True)
        units_payload_serialized = [int(units_payload_arr[i]) for i in active_indices]
    
    # Convertir les scalaires JAX en types Python natifs
    # Utiliser .item() pour extraire la valeur scalaire
    current_player = int(np.asarray(state.current_player).item())
    turn = int(np.asarray(state.turn).item())
    done = bool(np.asarray(state.done).item())
    player_stars = np.array(state.player_stars, copy=True).tolist()
    player_score = np.array(state.player_score, copy=True).tolist()
    player_techs = np.array(state.player_techs, copy=True).astype(int).tolist()
    tile_income_lookup = np.array(CITY_STAR_INCOME_PER_LEVEL, copy=True)
    city_income_grid = np.array(tile_income_lookup[city_level_arr], copy=True)
    num_players = int(state.num_players)
    income_bonus = np.array(state.player_income_bonus, copy=True)
    player_income = []
    for player_id in range(num_players):
        owned_income = np.array(np.where(city_owner_arr == player_id, city_income_grid, 0), copy=True)
        bonus_value = int(np.asarray(income_bonus[player_id]).item())
        player_income.append(int(np.sum(owned_income)) + bonus_value)
    score_breakdown = {
        "territory": np.array(state.score_territory, copy=True).tolist(),
        "population": np.array(state.score_population, copy=True).tolist(),
        "military": np.array(state.score_military, copy=True).tolist(),
        "economy": np.array(state.score_resources, copy=True).tolist(),
        "exploration": np.array(state.score_exploration, copy=True).tolist(),
    }
    game_mode = int(np.asarray(state.game_mode).item())
    max_turns = int(np.asarray(state.max_turns).item())
    
    # Vision et exploration
    tiles_explored_arr = np.array(state.tiles_explored, copy=True)
    tiles_visible_arr = np.array(state.tiles_visible, copy=True)
    tiles_explored = tiles_explored_arr.astype(int).tolist()
    tiles_visible = tiles_visible_arr.astype(int).tolist()
    
    # Métadonnées de configuration
    config = {
        "height": int(state.height),
        "width": int(state.width),
        "max_units": int(state.max_units),
        "num_players": int(state.num_players),
    }
    
    return {
        "terrain": terrain,
        "city_owner": city_owner,
        "city_level": city_level,
        "city_population": city_population,
        "city_has_port": city_has_port,
        "resource_type": resource_type,
        "resource_available": resource_available,
        "units": active_units,
        "current_player": current_player,
        "turn": turn,
        "done": done,
        "player_stars": player_stars,
        "player_score": player_score,
        "player_income": player_income,
        "score_breakdown": score_breakdown,
        "player_techs": player_techs,
        "units_payload_type": units_payload_serialized,
        "game_mode": game_mode,
        "max_turns": max_turns,
        "tiles_explored": tiles_explored,
        "tiles_visible": tiles_visible,
        "config": config,
    }


def dict_to_state(data: dict) -> GameState:
    """Reconstruit un GameState depuis un dictionnaire sérialisé.
    
    Args:
        data: Dictionnaire contenant l'état sérialisé
    
    Returns:
        GameState reconstruit
    """
    import jax.numpy as jnp
    import numpy as np
    from polytopia_jax.core.state import GameMode
    
    config = data.get("config", {})
    height = config.get("height", 10)
    width = config.get("width", 10)
    max_units = config.get("max_units", 64)
    num_players = config.get("num_players", 2)
    
    # Créer un état vide
    state = GameState.create_empty(height, width, max_units, num_players)
    
    # Restaurer le terrain
    if "terrain" in data:
        state = state.replace(terrain=jnp.array(data["terrain"], dtype=jnp.int32))
    
    # Restaurer les villes
    if "city_owner" in data:
        state = state.replace(city_owner=jnp.array(data["city_owner"], dtype=jnp.int32))
    if "city_level" in data:
        state = state.replace(city_level=jnp.array(data["city_level"], dtype=jnp.int32))
    if "city_population" in data:
        state = state.replace(city_population=jnp.array(data["city_population"], dtype=jnp.int32))
    if "city_has_port" in data:
        state = state.replace(city_has_port=jnp.array(data["city_has_port"], dtype=jnp.bool_))
    
    # Restaurer les ressources
    if "resource_type" in data:
        state = state.replace(resource_type=jnp.array(data["resource_type"], dtype=jnp.int32))
    if "resource_available" in data:
        state = state.replace(resource_available=jnp.array(data["resource_available"], dtype=jnp.bool_))
    
    # Restaurer les unités
    if "units" in data:
        units = data["units"]
        units_type = jnp.zeros(max_units, dtype=jnp.int32)
        units_pos = jnp.zeros((max_units, 2), dtype=jnp.int32)
        units_hp = jnp.zeros(max_units, dtype=jnp.int32)
        units_owner = jnp.full(max_units, NO_OWNER, dtype=jnp.int32)
        units_active = jnp.zeros(max_units, dtype=jnp.bool_)
        units_has_acted = jnp.zeros(max_units, dtype=jnp.bool_)
        units_payload_type = jnp.zeros(max_units, dtype=jnp.int32)
        
        for unit in units:
            unit_id = unit.get("id", 0)
            if 0 <= unit_id < max_units:
                units_type = units_type.at[unit_id].set(unit.get("type", 0))
                pos = unit.get("pos", [0, 0])
                units_pos = units_pos.at[unit_id].set(jnp.array(pos, dtype=jnp.int32))
                units_hp = units_hp.at[unit_id].set(unit.get("hp", 0))
                units_owner = units_owner.at[unit_id].set(unit.get("owner", NO_OWNER))
                units_active = units_active.at[unit_id].set(True)
                units_has_acted = units_has_acted.at[unit_id].set(unit.get("has_acted", False))
                units_payload_type = units_payload_type.at[unit_id].set(unit.get("payload_type", 0))
        
        state = state.replace(
            units_type=units_type,
            units_pos=units_pos,
            units_hp=units_hp,
            units_owner=units_owner,
            units_active=units_active,
            units_has_acted=units_has_acted,
            units_payload_type=units_payload_type,
        )
    
    # Restaurer l'état du jeu
    if "current_player" in data:
        state = state.replace(current_player=jnp.array(data["current_player"], dtype=jnp.int32))
    if "turn" in data:
        state = state.replace(turn=jnp.array(data["turn"], dtype=jnp.int32))
    if "done" in data:
        state = state.replace(done=jnp.array(data["done"], dtype=jnp.bool_))
    if "game_mode" in data:
        state = state.replace(game_mode=jnp.array(data["game_mode"], dtype=jnp.int32))
    if "max_turns" in data:
        state = state.replace(max_turns=jnp.array(data["max_turns"], dtype=jnp.int32))
    
    # Restaurer les statistiques des joueurs
    if "player_stars" in data:
        state = state.replace(player_stars=jnp.array(data["player_stars"], dtype=jnp.int32))
    if "player_score" in data:
        state = state.replace(player_score=jnp.array(data["player_score"], dtype=jnp.int32))
    if "player_techs" in data:
        state = state.replace(player_techs=jnp.array(data["player_techs"], dtype=jnp.bool_))
    
    # Restaurer le breakdown des scores
    if "score_breakdown" in data:
        breakdown = data["score_breakdown"]
        if "territory" in breakdown:
            state = state.replace(score_territory=jnp.array(breakdown["territory"], dtype=jnp.int32))
        if "population" in breakdown:
            state = state.replace(score_population=jnp.array(breakdown["population"], dtype=jnp.int32))
        if "military" in breakdown:
            state = state.replace(score_military=jnp.array(breakdown["military"], dtype=jnp.int32))
        if "economy" in breakdown:
            state = state.replace(score_resources=jnp.array(breakdown["economy"], dtype=jnp.int32))
    
    # Restaurer le bonus de revenu (nécessaire pour la difficulté)
    # On doit le recalculer depuis la difficulté, mais pour l'instant on le met à zéro
    # Il sera restauré lors du chargement de la session
    state = state.replace(player_income_bonus=jnp.zeros(num_players, dtype=jnp.int32))
    
    return state

