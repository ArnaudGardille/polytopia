"""Module core - Moteur de jeu pur en JAX."""

from .state import GameState
from .actions import ActionType, encode_action, decode_action
from .init import init_random, GameConfig
from .rules import step, legal_actions_mask
from .reward import compute_reward, compute_reward_all_players

__all__ = [
    "GameState",
    "ActionType",
    "encode_action",
    "decode_action",
    "init_random",
    "GameConfig",
    "step",
    "legal_actions_mask",
    "compute_reward",
    "compute_reward_all_players",
]

