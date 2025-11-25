"""Gestion des options d'affichage des états renvoyés par le backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from polytopia_jax.core.state import TechType


def _env_flag(name: str, default: bool) -> bool:
    """Convertit une variable d'environnement en booléen."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ViewOptions:
    """Options régissant la présentation des états de jeu."""

    reveal_map: bool = True
    unlock_all_techs: bool = True


def resolve_view_options(
    reveal_map: Optional[bool] = None,
    unlock_all_techs: Optional[bool] = None,
    base: Optional[ViewOptions] = None,
) -> ViewOptions:
    """Fusionne les overrides éventuels avec la configuration par défaut.

    Les valeurs par défaut proviennent des variables d'environnement :
    - ``POLYTOPIA_VIEW_REVEAL_MAP`` (defaut = True)
    - ``POLYTOPIA_VIEW_UNLOCK_ALL_TECHS`` (defaut = True)
    """

    default_options = base or ViewOptions(
        reveal_map=_env_flag("POLYTOPIA_VIEW_REVEAL_MAP", True),
        unlock_all_techs=_env_flag("POLYTOPIA_VIEW_UNLOCK_ALL_TECHS", True),
    )

    return ViewOptions(
        reveal_map=default_options.reveal_map if reveal_map is None else reveal_map,
        unlock_all_techs=default_options.unlock_all_techs
        if unlock_all_techs is None
        else unlock_all_techs,
    )


def apply_view_overrides(state: dict, options: ViewOptions) -> dict:
    """Applique les options d'affichage sur un état sérialisé."""
    new_state = dict(state)

    if options.unlock_all_techs:
        new_state["player_techs"] = _build_full_tech_matrix(state)

    if options.reveal_map:
        # Révéler toute la carte
        new_state["visibility_mask"] = _build_full_visibility(state)
    else:
        # Utiliser la vision du joueur actif
        current_player = state.get("current_player", 0)
        tiles_visible = state.get("tiles_visible", [])
        if tiles_visible and current_player < len(tiles_visible):
            # Convertir booléens en int (1 = visible, 0 = non visible)
            visibility_mask = [[int(cell) for cell in row] for row in tiles_visible[current_player]]
            new_state["visibility_mask"] = visibility_mask
        else:
            # Fallback : toute la carte visible si pas de données de vision
            new_state["visibility_mask"] = _build_full_visibility(state)

    return new_state


def _build_full_tech_matrix(state: dict) -> list[list[int]]:
    """Construit une matrice entièrement débloquée pour les techno."""
    player_techs = state.get("player_techs") or []
    num_players = len(player_techs)
    if num_players == 0:
        num_players = state.get("config", {}).get("num_players") or len(
            state.get("player_stars", [])
        )

    sample_row = player_techs[0] if player_techs else []
    num_techs = len(sample_row) if sample_row else int(TechType.NUM_TECHS)

    if num_players <= 0 or num_techs <= 0:
        return []

    return [[1] * num_techs for _ in range(num_players)]


def _build_full_visibility(state: dict) -> list[list[int]]:
    """Construit un masque de visibilité complet (toutes les cases révélées)."""
    terrain = state.get("terrain") or []
    if terrain and isinstance(terrain, list):
        height = len(terrain)
        first_row = terrain[0] if terrain else []
        width = len(first_row) if isinstance(first_row, list) else 0
    else:
        config = state.get("config", {})
        height = int(config.get("height", 0))
        width = int(config.get("width", 0))

    if height <= 0 or width <= 0:
        return []

    return [[1] * width for _ in range(height)]
