"""Sérialisation de GameState en format JSON."""

import jax.numpy as jnp
from polytopia_jax.core.state import GameState


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
    terrain = np.array(state.terrain, copy=True).tolist()
    city_owner = np.array(state.city_owner, copy=True).tolist()
    city_level = np.array(state.city_level, copy=True).tolist()
    
    # Extraire uniquement les unités actives
    active_units = []
    units_active_arr = np.array(state.units_active, copy=True)
    units_type_arr = np.array(state.units_type, copy=True)
    units_pos_arr = np.array(state.units_pos, copy=True)
    units_hp_arr = np.array(state.units_hp, copy=True)
    units_owner_arr = np.array(state.units_owner, copy=True)
    
    for i in range(state.max_units):
        if bool(units_active_arr[i]):
            unit = {
                "type": int(units_type_arr[i].item()),
                "pos": [
                    int(units_pos_arr[i, 0].item()),
                    int(units_pos_arr[i, 1].item())
                ],
                "hp": int(units_hp_arr[i].item()),
                "owner": int(units_owner_arr[i].item()),
            }
            active_units.append(unit)
    
    # Convertir les scalaires JAX en types Python natifs
    # Utiliser .item() pour extraire la valeur scalaire
    current_player = int(np.asarray(state.current_player).item())
    turn = int(np.asarray(state.turn).item())
    done = bool(np.asarray(state.done).item())
    
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
        "units": active_units,
        "current_player": current_player,
        "turn": turn,
        "done": done,
        "config": config,
    }

