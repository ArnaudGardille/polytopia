# Compte Rendu : Différences entre le Simulateur JAX et le Frontend

## Vue d'ensemble

Le projet Polytopia-JAX suit une architecture stricte en couches :
- **`polytopia_jax/core/`** : Moteur de jeu pur en JAX (simulateur)
- **`polytopia_jax/web/`** : Backend FastAPI (conversion et sérialisation)
- **`frontend/`** : Interface React + TypeScript (affichage uniquement)

## 1. Structure des Données

### 1.1 GameState (Simulateur JAX)

Le simulateur utilise une structure `GameState` optimisée pour JAX :

```python
@struct.dataclass
class GameState:
    # Arrays JAX de taille fixe
    terrain: jnp.ndarray  # [H, W]
    city_owner: jnp.ndarray  # [H, W] - grille dense
    city_level: jnp.ndarray  # [H, W] - grille dense
    units_type: jnp.ndarray  # [N_units_max] - tableau fixe
    units_pos: jnp.ndarray  # [N_units_max, 2]
    units_active: jnp.ndarray  # [N_units_max] - booléen
    # ... autres champs
```

**Caractéristiques :**
- Tous les champs sont des arrays JAX (`jnp.ndarray`)
- Structure dense avec tableaux de taille fixe
- Compatible avec `jit` et `vmap`
- Pas de listes dynamiques
- Les unités inactives sont marquées par `units_active[i] = False`

### 1.2 GameStateView (Frontend)

Le frontend reçoit une structure `GameStateView` optimisée pour l'affichage :

```typescript
interface GameStateView {
  terrain: number[][];
  cities: CityView[];  // Liste sparse
  units: UnitView[];   // Liste sparse
  // ... autres champs optionnels
}
```

**Caractéristiques :**
- Structures sérialisables en JSON
- Listes sparses (seulement les villes/unités existantes)
- Champs optionnels pour compatibilité
- Types TypeScript natifs (pas de JAX)

### 1.3 Conversion (Backend)

La conversion se fait dans `polytopia_jax/web/serialize.py` et `models.py` :

**Étapes de conversion :**
1. `GameState` (JAX) → `dict` (Python) via `state_to_dict()`
   - Conversion des arrays JAX en listes Python
   - Extraction des unités actives uniquement
   - Calcul des revenus et breakdowns de score

2. `dict` → `GameStateView` (Pydantic) via `GameStateView.from_raw_state()`
   - Extraction des villes depuis les grilles `city_owner` et `city_level`
   - Création de listes sparses
   - Validation avec Pydantic

3. `GameStateView` (Pydantic) → JSON → `GameStateView` (TypeScript)
   - Sérialisation JSON automatique
   - Types TypeScript synchronisés avec Pydantic

## 2. Représentation des Unités et Villes

### 2.1 Simulateur JAX

**Unités :**
- Stockées dans des tableaux de taille fixe `[max_units]`
- Les unités inactives ont `units_active[i] = False`
- Accès par index : `state.units_pos[unit_id]`
- Pas de structure d'objet, juste des arrays parallèles

**Villes :**
- Stockées dans des grilles denses `[H, W]`
- `city_owner[i, j] = -1` signifie pas de ville
- `city_level[i, j] = 0` signifie pas de ville ou niveau 0
- Accès par coordonnées : `state.city_owner[y, x]`

### 2.2 Frontend

**Unités :**
- Liste sparse d'objets `UnitView[]`
- Chaque unité a un `id`, `type`, `pos`, `hp`, `owner`
- Accès par itération : `units.find(u => u.id === id)`

**Villes :**
- Liste sparse d'objets `CityView[]`
- Chaque ville a `owner`, `level`, `pos`
- Accès par itération : `cities.find(c => c.pos[0] === x && c.pos[1] === y)`

**Impact :**
- Le frontend doit itérer sur les listes pour trouver une unité/ville
- Le simulateur accède directement par index/coordonnées
- Conversion coûteuse côté backend (extraction depuis grilles denses)

## 3. Système de Coordonnées et Rendu

### 3.1 Simulateur JAX

**Coordonnées logiques :**
- Système de grille simple aligné (x, y)
- Pas hexagonal logique, juste coordonnées cartésiennes
- 8 directions avec deltas `{-1, 0, 1}` en x et y
- Mouvements calculés avec `DIRECTION_DELTA[direction]`

**Pas de rendu :**
- Le simulateur ne connaît pas le rendu visuel
- Aucune conversion en coordonnées pixel
- Aucune notion de forme hexagonale

### 3.2 Frontend

**Coordonnées logiques :**
- Même système de grille aligné (x, y)
- Même système de 8 directions
- Conversion en coordonnées pixel pour affichage

**Rendu isométrique :**
- Transformation isométrique dans `Board.tsx` :
  ```typescript
  function hexToPixel(x: number, y: number, tileWidth: number): [number, number] {
    const pixelX = (x - y) * tileWidth / 2;
    const pixelY = (x + y) * tileHeight / 4;
    return [pixelX, pixelY];
  }
  ```
- Rendu hexagonal visuel mais coordonnées logiques alignées
- SVG pour le rendu avec images de terrain/unités

**Différence clé :**
- Le simulateur utilise uniquement les coordonnées logiques
- Le frontend ajoute une couche de transformation visuelle

## 4. Encodage des Actions

### 4.1 Format d'Encodage

Les deux systèmes utilisent le **même format** :

```
- ActionType: 3 bits (0-7)
- unit_id: 8 bits (0-255)
- direction: 3 bits (0-7)
- target_x: 6 bits (0-63)
- target_y: 6 bits (0-63)
- unit_type: 4 bits (0-15)
Total: 30 bits (fits in int32)
```

### 4.2 Simulateur JAX

**Fichier :** `polytopia_jax/core/actions.py`

```python
def encode_action(
    action_type: int,
    unit_id: Optional[int] = None,
    direction: Optional[int] = None,
    target_pos: Optional[Tuple[int, int]] = None,
    unit_type: Optional[int] = None,
) -> int:
    encoded = action_type
    if unit_id is not None:
        encoded |= (unit_id & 0xFF) << 3
    # ... autres champs
```

**Décodage :**
- Utilise `jnp.asarray` pour compatibilité JAX
- Gère les contextes tracés (`jit`)
- Retourne des arrays JAX ou valeurs Python selon le contexte

### 4.3 Frontend

**Fichier :** `frontend/src/utils/actionEncoder.ts`

```typescript
export function encodeAction(params: {
  actionType: ActionType;
  unitId?: number;
  direction?: Direction;
  targetPos?: [number, number];
  unitType?: number;
}): number {
  // Même logique que Python
}
```

**Différences :**
- TypeScript natif (pas de JAX)
- Fonctions helper : `encodeMove()`, `encodeAttack()`, etc.
- Pas de décodage côté frontend (pas nécessaire)

**Compatibilité :**
- Les actions encodées sont identiques entre les deux systèmes
- Le frontend peut envoyer directement l'entier encodé au backend

## 5. Calcul des Actions Légales

### 5.1 Simulateur JAX

**Fichier :** `polytopia_jax/core/rules.py`

- Fonction `legal_actions_mask()` calcule toutes les actions légales
- Retourne un masque booléen de taille fixe
- Utilise des opérations JAX vectorisées
- Vérifie les règles de jeu (mouvement, terrain, technologies, etc.)

**Caractéristiques :**
- Calcul côté serveur uniquement
- Optimisé pour batch processing
- Compatible avec `vmap` pour plusieurs états

### 5.2 Frontend

**Fichier :** `frontend/src/components/LiveGameView.tsx`

- Calcul approximatif des cibles de mouvement
- Utilise `useMemo` pour optimiser
- Génère les 8 voisins avec `NEIGHBOR_OFFSETS`
- Vérifie basiquement :
  - Bounds
  - Occupation par unité alliée
  - Terrain traversable (approximatif)

**Limitations :**
- Ne vérifie pas toutes les règles (technologies, ports, etc.)
- Calcul uniquement pour l'affichage visuel
- Le backend valide réellement les actions

**Différence critique :**
- Le frontend calcule des **cibles visuelles approximatives**
- Le simulateur calcule les **actions légales exactes**
- Le backend doit toujours valider les actions avant application

## 6. Architecture et Responsabilités

### 6.1 Simulateur JAX (`polytopia_jax/core/`)

**Responsabilités :**
- Logique de jeu pure
- Transitions d'état déterministes
- Calculs de règles (mouvement, combat, économie)
- Compatibilité JAX (`jit`, `vmap`)

**Contraintes :**
- Aucune dépendance externe (pas d'IO)
- Pas d'état mutable
- Toutes les fonctions doivent être traçables
- Utilise uniquement `jax.numpy`

**Ne fait PAS :**
- Sérialisation
- Rendu visuel
- Gestion de sessions
- API HTTP

### 6.2 Backend (`polytopia_jax/web/`)

**Responsabilités :**
- Conversion `GameState` → `GameStateView`
- Sérialisation JSON
- Gestion des replays et parties live
- API REST

**Ne fait PAS :**
- Logique de jeu (délègue au core)
- Rendu visuel
- Calculs de règles

### 6.3 Frontend (`frontend/`)

**Responsabilités :**
- Affichage visuel du jeu
- Interface utilisateur
- Interaction utilisateur (clics, sélection)
- Communication avec l'API

**Ne fait PAS :**
- Logique de jeu
- Validation des règles (approximative pour UX)
- Calculs de simulation

## 7. Différences Fonctionnelles

### 7.1 Gestion des Villes

**Simulateur :**
- Grille dense `city_owner[H, W]` et `city_level[H, W]`
- Accès O(1) par coordonnées
- Stockage de la population dans `city_population[H, W]`

**Frontend :**
- Liste sparse `cities: CityView[]`
- Accès O(n) par itération
- Population dans `city_population?: number[][]` (optionnel)

**Impact :**
- Le frontend doit itérer pour trouver une ville à une position
- Conversion coûteuse côté backend (parcours de toute la grille)

### 7.2 Gestion des Unités

**Simulateur :**
- Tableau fixe `units_pos[max_units]` avec flag `units_active`
- Accès O(1) par ID
- Unités inactives toujours présentes dans le tableau

**Frontend :**
- Liste sparse `units: UnitView[]`
- Accès O(n) par ID
- Seulement les unités actives sont présentes

**Impact :**
- Le frontend doit itérer pour trouver une unité par ID
- Conversion côté backend filtre les unités inactives

### 7.3 Calcul des Revenus

**Simulateur :**
- Stocke `player_income_bonus` (bonus de difficulté)
- Calcule les revenus à la volée si nécessaire
- Pas de champ `player_income` pré-calculé

**Frontend :**
- Reçoit `player_income` pré-calculé dans `GameStateView`
- Calculé dans `serialize.py` :
  ```python
  city_income_grid = tile_income_lookup[city_level_arr]
  owned_income = np.where(city_owner_arr == player_id, city_income_grid, 0)
  player_income = np.sum(owned_income) + bonus_value
  ```

**Impact :**
- Le frontend reçoit des données pré-calculées
- Économise des calculs côté client
- Le backend doit recalculer à chaque sérialisation

### 7.4 Breakdown des Scores

**Simulateur :**
- Stocke les composantes séparément :
  - `score_territory`
  - `score_population`
  - `score_military`
  - `score_resources`

**Frontend :**
- Reçoit un dictionnaire `score_breakdown` :
  ```typescript
  score_breakdown?: Record<string, number[]>;
  ```

**Impact :**
- Le frontend peut afficher le détail des scores
- Conversion simple dans `serialize.py`

## 8. Points d'Attention et Limitations

### 8.1 Synchronisation des Types

**Risque :**
- Les types TypeScript doivent correspondre aux modèles Pydantic
- Changements dans le core nécessitent mise à jour du backend ET frontend

**Solution actuelle :**
- Types définis dans `frontend/src/types.ts`
- Synchronisation manuelle avec `polytopia_jax/web/models.py`

### 8.2 Performance de Conversion

**Bottleneck :**
- Conversion `GameState` → `GameStateView` peut être coûteuse
- Parcours de grilles denses pour extraire villes/unités
- Recalcul des revenus et scores

**Optimisations actuelles :**
- Extraction uniquement des unités actives
- Utilisation de listes en compréhension
- Pré-calcul des revenus une seule fois

### 8.3 Calcul des Actions Légales

**Problème :**
- Le frontend calcule approximativement les cibles
- Peut suggérer des actions invalides
- Le backend doit toujours valider

**Solution :**
- Le backend pourrait exposer un endpoint pour les actions légales
- Actuellement, le frontend fait une approximation pour l'UX

### 8.4 Gestion des Erreurs

**Simulateur :**
- Retourne des états invalides si action invalide
- Pas de gestion d'erreur explicite

**Frontend :**
- Doit gérer les erreurs HTTP
- Affiche des messages d'erreur utilisateur
- Gère les états de chargement

## 9. Conclusion

### Points Clés

1. **Séparation stricte des responsabilités** : Le simulateur ne connaît pas le frontend et vice versa
2. **Structures différentes** : Dense (JAX) vs Sparse (Frontend)
3. **Même logique de coordonnées** : Les deux utilisent le système de grille aligné
4. **Encodage identique** : Les actions sont compatibles entre les deux systèmes
5. **Calculs approximatifs côté frontend** : Pour l'UX, mais validation backend requise

### Recommandations

1. **Maintenir la synchronisation** : Documenter les changements de types
2. **Optimiser la conversion** : Profiler et optimiser `serialize.py` si nécessaire
3. **Exposer les actions légales** : Endpoint API pour calcul exact côté frontend
4. **Tests de compatibilité** : Vérifier que les actions encodées fonctionnent des deux côtés






