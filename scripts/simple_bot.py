"""Bot simple pour simulation de parties.

Ce module conserve l'interface historique mais délègue désormais
la logique à ``polytopia_jax.ai.StrategyAI`` afin de partager le même
comportement entre les replays, le backend live et les futurs wrappers RL.
"""

from typing import Optional

from polytopia_jax.ai import StrategyAI


class SimpleBot(StrategyAI):
    """Wrapper rétro-compatible autour de l'IA heuristique."""

    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__(player_id=player_id, strategy_name="rush", seed=seed)
