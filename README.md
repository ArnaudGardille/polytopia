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
  - `polytopia_jax/core` (simulation pure en JAX),
  - `polytopia_jax/ai` (stratégies IA),
  - `rl` (wrappers Gymnasium / PettingZoo),
  - `polytopia_jax/web` (API FastAPI),
  - `frontend` (visualisation React + SVG).

---

## 2. Structure générale du projet

```
polytopia-jax/
├─ polytopia_jax/
│  ├─ core/
│  │  ├─ state.py        # Définition de GameState (pytree JAX)
│  │  ├─ rules.py        # Déplacements, combats, production, capture
│  │  ├─ actions.py      # Encodage des actions discrètes
│  │  ├─ init.py         # Génération des états initiaux
│  │  ├─ reward.py       # Fonctions de récompense
│  │  └─ score.py        # Calcul des scores
│  │
│  ├─ ai/
│  │  └─ strategies.py   # Stratégies IA (rush, economy, random, idle)
│  │
│  └─ web/
│     ├─ api.py          # Backend FastAPI
│     ├─ models.py       # Conversion GameState → GameStateView
│     ├─ replay_store.py # Lecture et écriture des replays
│     ├─ live_game_store.py # Gestion des parties live
│     ├─ serialize.py    # Sérialisation des états
│     └─ view_options.py # Options de visualisation
│
├─ rl/
│  ├─ gym_env.py         # Wrapper Gymnasium (single-agent)
│  ├─ pettingzoo_env.py  # Wrapper PettingZoo (multi-agent)
│  └─ session.py         # Session de simulation
│
├─ frontend/             # Interface web (React + TypeScript)
│  ├─ src/
│  │  ├─ App.tsx
│  │  ├─ api.ts
│  │  ├─ components/
│  │  │  ├─ Board.tsx
│  │  │  ├─ HUD.tsx
│  │  │  ├─ GameList.tsx
│  │  │  ├─ LiveGameView.tsx
│  │  │  └─ ...
│  │  └─ types.ts
│  └─ ...
│
├─ scripts/
│  ├─ generate_replay.py # Génère un replay bot vs bot
│  └─ run_web_demo.py    # Démo front + backend
│
├─ tests/
│  ├─ test_core.py       # Tests unitaires du moteur
│  ├─ test_web/          # Tests de l'API web
│  └─ test_ai/           # Tests des stratégies IA
│
├─ pyproject.toml
├─ SETUP.md              # Guide de configuration détaillé
└─ README.md
```

---

## 3. Cœur de simulation (module `polytopia_jax/core/`)

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

### Système de grille et mouvements

Le jeu utilise une grille simple avec des coordonnées alignées (x, y). Les unités peuvent se déplacer dans 8 directions avec des deltas {-1, 0, 1} en x et {-1, 0, 1} en y :
- UP: [0, -1]
- UP_RIGHT: [1, -1]
- RIGHT: [1, 0]
- DOWN_RIGHT: [1, 1]
- DOWN: [0, 1]
- DOWN_LEFT: [-1, 1]
- LEFT: [-1, 0]
- UP_LEFT: [-1, -1]

L'affichage utilise un rendu hexagonal visuel, mais les coordonnées logiques sont alignées (pas de décalage selon la parité de la ligne).

### Fonctions principales

- `init_random(key, config)` : génération d'un état initial.
- `step(state, action)` : transition d'état pure et JIT-compatible.
- `legal_actions_mask(state)` : masque des actions valides.

Aucune fonction dans `polytopia_jax/core/` ne doit interagir avec l'extérieur (pas d'IO, pas d'état mutable).

---

## 4. Environnements RL (`rl/`)

Deux wrappers sont fournis :

### Gymnasium (single-agent)

Implémente l'API standard :

```python
from rl.gym_env import PolytopiaEnv, SimulationConfig

env = PolytopiaEnv(
    SimulationConfig(
        opponents=2,
        difficulty="hard",
        ai_strategy="economy",
    )
)
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

`SimulationConfig.ai_strategy` accepte les mêmes valeurs que le backend live (`rush`, `economy`, `random`, `idle`) et contrôle la façon dont les adversaires IA sont simulés côté moteur.

### PettingZoo (multi-agent)

Permet le self-play et les agents indépendants (modes AEC/Parallel) en s'appuyant sur la même session sous-jacente.

---

## 5. Backend web (`polytopia_jax/web/`)

Backend FastAPI exposant plusieurs endpoints :

**Replays :**
- `GET /games` — liste des replays disponibles.
- `GET /games/{id}/state/{turn}` — état d'un tour donné.
- `GET /games/{id}/replay` — récupération du replay complet.

**Parties live :**
- `POST /live/perfection` — crée une partie live (paramètres `opponents`, `difficulty`, `seed`, `strategy`).
- `GET /live/{game_id}` — récupère l'état courant.
- `POST /live/{game_id}/action` — applique une action encodée.
- `POST /live/{game_id}/end_turn` — termine explicitement le tour du joueur humain.

`models.py` contient la version sérialisée de l'état (`GameStateView`), optimisée pour l'affichage.

---

## 6. Frontend (`frontend/`)

Frontend en React + TypeScript :

- `Board.tsx` : affichage du plateau via SVG (optimisé mobile).
- `HUD.tsx` : informations principales (tours, scores, joueur actif).
- `LiveGameView.tsx` : interface pour jouer des parties en direct.
- `GameList.tsx` : liste des replays disponibles.
- `api.ts` : communication avec FastAPI.
- `types.ts` : types TypeScript pour `GameStateView`.

Le frontend ne contient aucune logique de jeu : il se contente d'afficher.

---

## 7. Installation

> **Note** : Pour un guide d'installation détaillé, consultez [`SETUP.md`](SETUP.md).

### 7.1 Backend et simulation

Installer JAX selon la plateforme :

```bash
pip install "jax[cpu]"
# ou pour GPU : pip install "jax[cuda12]"  # selon votre configuration
```

Installer le projet :

```bash
pip install -e .
```

**Dépendances principales** :
- `jax`, `jaxlib` : moteur de calcul
- `fastapi`, `uvicorn`, `pydantic` : backend web
- `numpy` : calculs numériques

**Dépendances optionnelles** (pour les environnements RL) :
- `gymnasium` : wrapper single-agent (installer avec `pip install gymnasium`)
- `pettingzoo` : wrapper multi-agent (installer avec `pip install pettingzoo`)

Ces dépendances sont optionnelles car les wrappers RL utilisent des imports conditionnels. Si vous n'utilisez pas les environnements RL, vous pouvez ignorer ces dépendances.

### 7.2 Frontend

```bash
cd frontend
npm install
npm run dev
```

Le frontend sera accessible sur `http://localhost:5173` (ou un autre port si 5173 est occupé).

**Important** : Assurez-vous que le backend FastAPI est lancé sur `http://localhost:8000` (voir section 8).

---

## 8. Flux de travail recommandé

### Étape 1 : Développer un sous-ensemble minimal du jeu

- petite carte fixe ;
- unités simples ;
- villes basiques ;
- victoire par élimination.

### Étape 2 : Ajouter des tests unitaires

Tester les transitions élémentaires (step, mouvements, combats, fin de partie).

```bash
# Exécuter tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_rules.py -v
pytest tests/test_web/ -v
pytest tests/test_ai/ -v
```

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

- Extension du gameplay : monuments, temples, tribus spéciales (voir Phase 7).
- Optimisation du batching (simulation massive sur GPU/TPU).
- Mode spectateur live via WebSocket.
- Enregistrement compact des replays.
- Mode DOMINATION côté backend et frontend.
- Mode multijoueur en ligne.

---

## 11. Feuille de route gameplay

Cette section détaille les étapes proposées pour rapprocher progressivement la simulation d'une partie complète de Polytopia. Chaque phase peut être développée et testée indépendamment afin de conserver un moteur fonctionnel en permanence.

1. **Phase 0 – Stabiliser l'existant** ✅ **Terminée**
   
   **État initial (Phase 0)**  
   - Une seule unité jouable (`WARRIOR`) avec déplacement orthogonal et combat de mêlée.  
   - Aucune économie : pas d'étoiles, de bâtiments ni de technologies ; les villes sont réduites à un propriétaire et un niveau booléen.  
   - Captures instantanées : entrer sur la case d'une ville neutralise ou conquiert immédiatement la capitale adverse et réinitialise son niveau à 1.  
   - Condition de victoire unique : l'élimination. La partie se termine dès qu'un seul joueur possède encore au moins une ville.

2. **Phase 1 – Boucle économique minimale** ✅ **Terminée**
   
   **État actuel (Phase 1)**  
   - Chaque joueur possède une réserve `player_stars` initialisée à 5 et alimentée par les capitales (`2/4/6` ★ par niveau lors de `_apply_end_turn`).  
   - Les villes stockent `city_population` : capturer ou construire (ferme/mine/hutte) ajuste la population puis le `city_level` associé.  
   - `TRAIN_UNIT` et `BUILD` consomment automatiquement les ★ correspondantes et sont bloqués par le masque d'actions tant que le joueur n'a pas le budget requis.

3. **Phase 2 – Progression des villes et scoring** ✅ **Terminée**
   
   **État actuel (Phase 2)**  
   - `GameState` encode désormais le `game_mode` (Domination ou Perfection) et un `max_turns`, déclenchant la fin de partie au tour 30 pour Perfection.  
   - Un système de score agrège automatiquement territoire, population, armée et trésor (`player_score` + `score_breakdown`) et est exposé aux replays/API.  
   - Les conditions de victoire et les récompenses RL utilisent ces scores pour départager les joueurs lors d'une fin de partie en Perfection.

4. **Phase 3 – Arbre technologique** ✅ **Terminée**
   
   **État actuel (Phase 3)**  
   - `GameState` conserve désormais un tableau `player_techs`; l'action `RESEARCH_TECH` débloque Climbing, Sailing ou Mining selon le budget et les dépendances (Sailing requiert Climbing).  
   - Le moteur restreint les actions : montagnes et eaux peu profondes exigent la techno adaptée, et les mines ne peuvent être construites qu'après Mining.  
   - Le masque d'actions et les payloads API/replay exposent les technos restantes afin que les clients puissent piloter ou afficher l'arbre débloqué.

5. **Phase 4 – Diversité d'unités terrestres** ✅ **Terminée**
   
   **État actuel (Phase 4)**  
   - Trois nouvelles unités (`DEFENDER`, `ARCHER`, `RIDER`) sont disponibles avec des statistiques dédiées (PV, attaque, défense, coût, portée).  
   - Les archers tirent désormais à distance 2 sans subir de riposte lorsqu'ils restent hors de portée, et les mineurs doivent débloquer la techno Mining avant construction.  
   - Les tableaux `UNIT_*` et les tests de règles couvrent ces scénarios (capacité de tir longue portée, impossibilité d'attaquer hors portée pour les unités de mêlée, prérequis technologiques pour les bâtiments).

6. **Phase 5 – Navigation et terrains avancés** ✅ **Terminée**

   **État actuel (Phase 5)**  
   - Les villes peuvent construire des ports (tech Sailing requise) et permettre l'embarquement d'unités terrestres en `RAFT`, avec suivi du type transporté.  
   - Les déplacements/mouvements prennent en compte les ports et les restrictions d'eau : seuls les radeaux peuvent naviguer en eau peu profonde, l'accostage ne peut se faire que sur un port allié.  
   - Les replays/API exposent désormais la présence des ports et les métadonnées nécessaires (`city_has_port`, `player_techs`, `payload_type`) afin que le frontend puisse représenter la navigation fidèlement.

7. **Phase 6 – IA et difficultés** ✅ **Terminée**

   **État actuel (Phase 6)**  
   - Plusieurs stratégies IA sont disponibles (`rush`, `economy`, `random`, `idle`) via `polytopia_jax/ai/strategies.py`.  
   - Les wrappers RL (`rl/gym_env.py`, `rl/pettingzoo_env.py`) supportent plusieurs adversaires simultanés.  
   - Le backend live permet de choisir la stratégie IA lors de la création d'une partie.

8. **Phase 7 – Contenus avancés et tribus spéciales** 🚧 **À venir**
   - Implémenter monuments, temples et leur contribution au score.  
   - Ajouter des tribus à mécaniques uniques (Polaris, Cymanti, Aquarion, etc.) avec configuration activable/désactivable.  
   - Prévoir une API de configuration côté `web/api.py` et `frontend/` afin que les utilisateurs puissent choisir précisément quelles mécaniques activer lors d'une simulation ou d'un replay.

---

## 12. Mode Perfection live

Le backend expose désormais un mode Perfection jouable en temps réel :

- `POST /live/perfection` — crée une partie live (paramètres `opponents`, `difficulty`, `seed`).
- `GET /live/{game_id}` — récupère l’état courant.
- `POST /live/{game_id}/action` — applique une action encodée (mêmes bits que `core.actions.encode_action`).
- `POST /live/{game_id}/end_turn` — termine explicitement le tour du joueur humain.

L’interface React permet de lancer ce mode via le bouton PERFECTION → `START GAME`, puis de jouer (sélection des unités, déplacements, attaques, fin de tour). Les IA résolvent désormais leurs tours complètes côté serveur (selon la stratégie choisie) avant de rendre la main au joueur 0. L'endpoint `POST /live/perfection` accepte un champ supplémentaire `strategy` permettant de choisir le comportement IA (`rush`, `economy`, `random`, `idle`) sans avoir à modifier le code client.
