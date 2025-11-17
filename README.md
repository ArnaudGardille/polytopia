# Polytopia-JAX

Environnement de simulation inspiré de Polytopia, construit en JAX pour l'apprentissage par renforcement et doté d'une interface web permettant de visualiser les parties et leurs replays.

L'objectif est de fournir :

- un moteur de jeu performant, vectorisable et compatible avec jit et vmap ;
- une infrastructure propre pour l'entraînement RL ;
- un système de visualisation clair et indépendant du moteur.

---

## 1. Objectifs du projet

- Reproduire un ensemble de mécaniques proches de Polytopia : cartes en grille, villes, unités, production, combat, tours alternés.
- Permettre la simulation en masse de parties pour l'apprentissage par renforcement.
- Garantir une séparation stricte entre :
  - `core` (simulation pure en JAX),
  - `rl` (wrappers Gymnasium / PettingZoo),
  - `web` (API FastAPI),
  - `frontend` (visualisation React + PixiJS).

---

## 2. Structure générale du projet

```
polytopia-jax/
├─ core/
│  ├─ state.py           # Définition de GameState (pytree JAX)
│  ├─ rules.py           # Déplacements, combats, production, capture
│  ├─ actions.py         # Encodage des actions discrètes
│  ├─ init.py            # Génération des états initiaux
│  └─ reward.py          # Fonctions de récompense
│
├─ rl/
│  ├─ gym_env.py         # Wrapper Gymnasium
│  └─ pettingzoo_env.py  # Wrapper PettingZoo
│
├─ training/
│  └─ example_dqn.py     # Exemple d'entraînement (à compléter)
│
├─ web/
│  ├─ api.py             # Backend FastAPI
│  ├─ models.py          # Conversion GameState → GameStateView
│  └─ replay_store.py    # Lecture et écriture des replays
│
├─ frontend/             # Interface web (React + TypeScript)
│  ├─ src/
│  │  ├─ App.tsx
│  │  ├─ api.ts
│  │  ├─ components/
│  │  │  ├─ Board.tsx
│  │  │  └─ HUD.tsx
│  │  └─ types.ts
│  └─ ...
│
├─ scripts/
│  ├─ generate_replay.py # Génère un replay bot vs bot
│  └─ run_web_demo.py    # Démo front + backend
│
├─ tests/
│  └─ test_core.py       # Tests unitaires
│
├─ pyproject.toml
└─ README.md
```

---

## 3. Cœur de simulation (module `core/`)

Le moteur est écrit en JAX. L'état du jeu est représenté sous forme de pytree statique afin de permettre jit et vmap.

### Exemple de structure d'état

```python
@dataclass
class GameState:
    terrain: jnp.ndarray        # [H, W]
    city_owner: jnp.ndarray     # [H, W]
    city_level: jnp.ndarray     # [H, W]
    units_type: jnp.ndarray     # [N_units_max]
    units_pos: jnp.ndarray      # [N_units_max, 2]
    units_hp: jnp.ndarray       # [N_units_max]
    units_owner: jnp.ndarray    # [N_units_max]
    current_player: jnp.ndarray
    turn: jnp.ndarray
    done: jnp.ndarray
```

### Fonctions principales

- `init_random(key, config)` : génération d'un état initial.
- `step(state, action)` : transition d'état pure et JIT-compatible.
- `legal_actions_mask(state)` : masque des actions valides.

Aucune fonction dans `core/` ne doit interagir avec l'extérieur (pas d'IO, pas d'état mutable).

---

## 4. Environnements RL (`rl/`)

Deux wrappers sont prévus :

### Gymnasium (single-agent)

Implémente l'API standard :

```python
env = PolytopiaEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

L'observation correspond à une vue normalisée du plateau pour le joueur courant.

### PettingZoo (multi-agent)

Permet le self-play et les agents indépendants. Support des modes AEC et Parallel.

---

## 5. Backend web (`web/`)

Backend FastAPI exposant plusieurs endpoints :

- `GET /games/{id}/state/{turn}` — état d'un tour donné.
- `GET /games/{id}/replay` — récupération du replay complet.
- `GET /games` — liste des replays disponibles.
- `WS /games/{id}/live` — diffusion en direct (optionnel).

`models.py` contient la version sérialisée de l'état (`GameStateView`), optimisée pour l'affichage.

---

## 6. Frontend (`frontend/`)

Frontend en React + TypeScript :

- `Board.tsx` : affichage du plateau via Canvas ou PixiJS.
- `HUD.tsx` : informations principales (tours, scores, joueur actif).
- `api.ts` : communication avec FastAPI.
- `types.ts` : types TypeScript pour `GameStateView`.

Le frontend ne contient aucune logique de jeu : il se contente d'afficher.

---

## 7. Installation

### 7.1 Backend et simulation

Installer JAX selon la plateforme :

```bash
pip install "jax[cpu]"
```

Installer le projet :

```bash
pip install -e .
```

Dépendances principales : `jax`, `gymnasium`, `pettingzoo`, `fastapi`, `uvicorn`, `pydantic`.

### 7.2 Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 8. Flux de travail recommandé

### Étape 1 : Développer un sous-ensemble minimal du jeu

- petite carte fixe ;
- unités simples ;
- villes basiques ;
- victoire par élimination.

### Étape 2 : Ajouter des tests unitaires

Tester les transitions élémentaires (step, mouvements, combats, fin de partie).

### Étape 3 : Générer des replays

```bash
python scripts/generate_replay.py --output replays/game_001.json
```

### Étape 4 : Visualisation

**Backend :**

```bash
uvicorn polytopia_jax.web.api:app --reload
```

**Frontend :**

```bash
npm run dev
```

---

## 9. Documentation utile

- [JAX](https://docs.jax.dev)
- [Gymnasium](https://gymnasium.farama.org/)
- [PettingZoo](https://pettingzoo.farama.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 10. Évolutions prévues

- Extension du gameplay (types d'unités, tech tree, diplomatie).
- Optimisation du batching (simulation massive sur GPU/TPU).
- Mode spectateur live via WebSocket.
- Enregistrement compact des replays.
