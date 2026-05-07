"""Wrappers RL pour Polytopia.

`gymnasium` et `pettingzoo` sont des dépendances optionnelles. Importer
`SimulationConfig` / `SimulationSession` ne doit jamais échouer ; les
wrappers Gym/PettingZoo ne sont exposés que s'ils sont installés.
"""

from .session import SimulationConfig, SimulationSession

__all__ = ["SimulationConfig", "SimulationSession"]

try:
    from .gym_env import PolytopiaEnv  # noqa: F401
    __all__.append("PolytopiaEnv")
except ImportError:
    pass

try:
    from .pettingzoo_env import PolytopiaAECEnv  # noqa: F401
    __all__.append("PolytopiaAECEnv")
except ImportError:
    pass
