"""Sérialisation de GameState en format JSON."""

from polytopia_jax.core.state import GameState
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
    
    # Extraire uniquement les unités actives
    active_units = []
    units_active_arr = np.array(state.units_active, copy=True)
    units_type_arr = np.array(state.units_type, copy=True)
    units_pos_arr = np.array(state.units_pos, copy=True)
    units_hp_arr = np.array(state.units_hp, copy=True)
    units_owner_arr = np.array(state.units_owner, copy=True)
    units_has_acted_arr = np.array(state.units_has_acted, copy=True)
    units_payload_arr = np.array(state.units_payload_type, copy=True)
    
    for i in range(state.max_units):
        if bool(units_active_arr[i]):
            unit = {
                "id": int(i),
                "type": int(units_type_arr[i].item()),
                "pos": [
                    int(units_pos_arr[i, 0].item()),
                    int(units_pos_arr[i, 1].item())
                ],
                "hp": int(units_hp_arr[i].item()),
                "owner": int(units_owner_arr[i].item()),
                "has_acted": bool(units_has_acted_arr[i]),
                "payload_type": int(units_payload_arr[i].item()),
            }
            active_units.append(unit)
    
    # Convertir les scalaires JAX en types Python natifs
    # Utiliser .item() pour extraire la valeur scalaire
    current_player = int(np.asarray(state.current_player).item())
    turn = int(np.asarray(state.turn).item())
    done = bool(np.asarray(state.done).item())
    player_stars = np.array(state.player_stars, copy=True).tolist()
    player_score = np.array(state.player_score, copy=True).tolist()
    player_techs = np.array(state.player_techs, copy=True).astype(int).tolist()
    tile_income_lookup = np.array(CITY_STAR_INCOME_PER_LEVEL, copy=True)
    city_income_grid = tile_income_lookup[city_level_arr]
    num_players = int(state.num_players)
    income_bonus = np.array(state.player_income_bonus, copy=True)
    player_income = []
    for player_id in range(num_players):
        owned_income = np.where(city_owner_arr == player_id, city_income_grid, 0)
        player_income.append(int(np.sum(owned_income) + income_bonus[player_id]))
    score_breakdown = {
        "territory": np.array(state.score_territory, copy=True).tolist(),
        "population": np.array(state.score_population, copy=True).tolist(),
        "military": np.array(state.score_military, copy=True).tolist(),
        "economy": np.array(state.score_resources, copy=True).tolist(),
    }
    game_mode = int(np.asarray(state.game_mode).item())
    max_turns = int(np.asarray(state.max_turns).item())
    
    # Métadonnées de configuration
    config = {
        "height": state.height,
        "width": state.width,
        "max_units": state.max_units,
        "num_players": state.num_players,
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
        "units_payload_type": units_payload_arr.tolist(),
        "game_mode": game_mode,
        "max_turns": max_turns,
        "config": config,
    }
