# Polytopia-JAX

Environnement de simulation inspiré de Polytopia, construit en JAX pour la **synchronisation d'agents d'IA** (apprentissage par renforcement, self-play).

L'objectif est de fournir un moteur de jeu performant, vectorisable et compatible avec `jit` et `vmap`, exposé sous une API minimale pour le RL.

---

## 1. Objectifs du projet

- Simulation de masse de parties pour l'apprentissage par renforcement.
- Séparation stricte entre :
  - `polytopia_jax/core` — simulation pure en JAX (pas d'IO, pas de mutation),
  - `polytopia_jax/ai` — stratégies IA scriptées pour warm-start et adversaires fixes,
  - `rl` — pilote `SimulationSession` partagé par les boucles RL.

---

## 2. Structure du projet

```
polytopia-jax/
├─ polytopia_jax/
│  ├─ core/
│  │  ├─ state.py        # GameState (pytree JAX, @struct.dataclass)
│  │  ├─ rules.py        # step(), legal_actions_mask(), combat, économie
│  │  ├─ actions.py      # Encodage/décodage des actions (30-bit packed int)
│  │  ├─ init.py         # Génération des états initiaux
│  │  ├─ reward.py       # Récompense RL
│  │  └─ score.py        # Score Perfection (optionnel, lu par reward.py)
│  └─ ai/
│     └─ strategies.py   # Stratégies IA (rush, economy, random, idle)
│
├─ rl/
│  └─ session.py         # SimulationSession + SimulationConfig
│
├─ tests/
│  ├─ test_state.py / test_actions.py / test_movement.py
│  ├─ test_init.py / test_rules.py / test_reward.py / test_score.py
│  ├─ test_new_units.py / test_promotions.py
│  ├─ test_ai/test_strategies.py
│  └─ test_rl/test_session.py
│
├─ pyproject.toml
├─ SETUP.md
├─ CLAUDE.md
└─ README.md
```

Le moteur est écrit en JAX. L'état du jeu est un pytree statique (`@struct.dataclass` de `flax.struct`) afin de permettre `jit` et `vmap`.

---

## 3. Coordonnées et actions

- Grille `(x, y)` avec coordonnées alignées (pas d'offset hex).
- 8 directions de mouvement (`Direction` enum) avec deltas dans `{-1, 0, 1}` × `{-1, 0, 1}`.
- Actions encodées en 30 bits dans un `int32` ; voir `core/actions.py` (`encode_action` / `decode_action`).

### Fonctions principales

```python
from polytopia_jax.core.init import init_random, GameConfig
from polytopia_jax.core.rules import step, legal_actions_mask

state = init_random(key, GameConfig(...))
mask = legal_actions_mask(state)
state = step(state, action_id)
```

Aucune fonction dans `polytopia_jax/core/` ne doit avoir d'effet de bord (pas d'IO, pas de mutation, pas de RNG global — passez les `key` en argument).

---

## 4. Pilote de simulation (`rl/`)

`rl.SimulationSession` est l'abstraction principale pour entraîner ou évaluer des agents. Il fait avancer le moteur, applique automatiquement les tours d'IA adverses (selon `ai_strategy`), et applique les bonus de difficulté.

```python
from rl import SimulationConfig, SimulationSession

session = SimulationSession(SimulationConfig(
    height=12, width=12,
    opponents=1,
    max_turns=30,
    difficulty="normal",
    ai_strategy="rush",
))
state = session.reset(seed=42)
state = session.apply_player_action(action_id)
```

`ai_strategy` accepte `rush | economy | random | idle`.

---

## 5. Installation

Voir [`SETUP.md`](SETUP.md).

```bash
pip install "jax[cpu]"      # ou jax[cuda12] selon plateforme
pip install -e .
```

Dépendances : `jax`, `jaxlib`, `flax`, `numpy`, `pytest`. Aucun serveur web, frontend ou outil de scraping ne fait partie du projet.

Pour activer les wrappers Gymnasium / PettingZoo, ajoutez `gymnasium` et/ou `pettingzoo` à votre environnement (imports conditionnels).

---

## 6. Tests

```bash
pytest tests/ -v
pytest tests/test_rules.py -v
```

Le pipeline de test couvre l'encodage d'actions, l'initialisation, les règles (combat, économie, capture, fin de partie), les récompenses RL, et la session RL.
