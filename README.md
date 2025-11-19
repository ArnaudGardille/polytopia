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

---

## 11. Feuille de route gameplay

Cette section détaille les étapes proposées pour rapprocher progressivement la simulation d'une partie complète de Polytopia. Chaque phase peut être développée et testée indépendamment afin de conserver un moteur fonctionnel en permanence.

1. **Phase 0 – Stabiliser l'existant**  
   - Documenter explicitement le périmètre MVP (guerrier unique, capture directe, victoire par élimination) dans `core/state.py` et `core/rules.py`.  
   - Étendre la suite de tests (`tests/test_rules.py`) pour couvrir mouvement, attaque, capture et fin de tour afin de verrouiller le comportement actuel avant d'ajouter de nouvelles mécaniques.  
   - Mettre à jour ce README avec les limitations connues afin que les équipes RL/web sachent exactement ce qui est supporté.
   
   **Limitations connues (Phase 0)**  
   - Une seule unité jouable (`WARRIOR`) avec déplacement orthogonal et combat de mêlée.  
   - Aucune économie : pas d'étoiles, de bâtiments ni de technologies ; les villes sont réduites à un propriétaire et un niveau booléen.  
   - Captures instantanées : entrer sur la case d'une ville neutralise ou conquiert immédiatement la capitale adverse et réinitialise son niveau à 1.  
   - Condition de victoire unique : l'élimination. La partie se termine dès qu'un seul joueur possède encore au moins une ville.  
   - Les masques d'actions restent permissifs (pas de validation complète côté moteur) tant que la partie n'est pas terminée.

2. **Phase 1 – Boucle économique minimale**  
   - Ajouter la notion d'étoiles et de population dans `GameState`, incrémenter le revenu des villes lors de `_apply_end_turn`, et déduire les coûts lorsque `TRAIN_UNIT`/`BUILD` sont déclenchées.  
   - Implémenter quelques bâtiments basiques (ferme, mine, hutte) depuis `core/rules.py` pour modifier la population et donc les niveaux de villes.  
   - Construire un masque d'actions légales qui bloque toute action non finançable.

3. **Phase 2 – Progression des villes et scoring**  
   - Faire évoluer `city_level` en fonction de la population, débloquer les bonus d'étoiles et introduire les améliorations de ville (mur, port, marché) avec leurs effets économiques.  
   - Implémenter les deux modes de victoire : Perfection (score au tour 30) et Domination (élimination), en branchant la logique dans `_check_victory` et la boucle principale.  
   - Créer une représentation de score compatible avec les replays/frontends.

4. **Phase 3 – Arbre technologique**  
   - Définir une structure de technologies avec coûts croissants et dépendances.  
   - Connecter chaque techno aux unités/bâtiments/terrains qu'elle déverrouille (Climbing, Sailing, Roads...).  
   - Étendre `core/actions.py` pour encoder la sélection d'une technologie et mettre à jour `legal_actions_mask`.

5. **Phase 4 – Diversité d'unités terrestres**  
   - Étendre `UnitType` + tables de stats pour inclure Défenseur, Archer, Cavalier, Mind Bender, Catapulte, etc., en respectant les compétences décrites dans `Polytopia.md`.  
   - Implémenter les capacités spéciales (Dash, portée >1, conversion, riposte asymétrique) et garantir leur traçabilité JAX.  
   - Ajouter des tests ciblés par type d'unité et exposer les nouveaux sprites/états au frontend.

6. **Phase 5 – Navigation et terrains avancés**  
   - Introduire les radeaux/bateaux et la transformation d'unités terrestres en navales via les ports.  
   - Gérer les terrains `WATER_SHALLOW`, `WATER_DEEP`, montagnes et la nécessité d'avoir la techno adaptée pour y entrer.  
   - Adapter les visualisations (backend + frontend) pour représenter les unités navales et les connexions maritimes.

7. **Phase 6 – IA et difficultés**  
   - Développer des heuristiques simples pour les IA (priorité expansion/combat) et appliquer des bonus d'étoiles par niveau de difficulté.  
   - Supporter plusieurs adversaires simultanés et synchroniser les wrappers RL (`rl/`) avec cette logique multi-agent.  
   - Enregistrer de nouveaux replays de référence pour tester les comportements.

8. **Phase 7 – Contenus avancés et tribus spéciales**  
   - Implémenter monuments, temples et leur contribution au score.  
   - Ajouter des tribus à mécaniques uniques (Polaris, Cymanti, Aquarion, etc.) avec configuration activable/désactivable.  
   - Prévoir une API de configuration côté `web/api.py` et `frontend/` afin que les utilisateurs puissent choisir précisément quelles mécaniques activer lors d'une simulation ou d'un replay.

---

## 11. Mode Perfection live

Le backend expose désormais un mode Perfection jouable en temps réel :

- `POST /live/perfection` — crée une partie live (paramètres `opponents`, `difficulty`, `seed`).
- `GET /live/{game_id}` — récupère l’état courant.
- `POST /live/{game_id}/action` — applique une action encodée (mêmes bits que `core.actions.encode_action`).
- `POST /live/{game_id}/end_turn` — termine explicitement le tour du joueur humain.

L’interface React permet de lancer ce mode via le bouton PERFECTION → `START GAME`, puis de jouer (sélection des unités, déplacements, attaques, fin de tour). Tant que les IA sont inactives, leurs tours sont automatiquement passés côté serveur pour revenir au joueur 0.
