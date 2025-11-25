# Compte Rendu : Aspects Implémentés dans le Simulateur JAX mais Absents du Frontend

**Date :** 2025-01-27  
**Objectif :** Identifier les fonctionnalités du moteur de jeu qui ne sont pas encore exposées dans l'interface utilisateur

---

## Résumé Exécutif

Le simulateur JAX implémente plusieurs mécaniques de jeu avancées qui ne sont pas encore accessibles ou visibles dans le frontend React. Ces différences concernent principalement les types d'unités, certaines actions, l'affichage des ressources et bâtiments, ainsi que des informations détaillées sur l'état du jeu.

---

## 1. Types d'Unités Non Affichés

### Simulateur JAX

Le moteur supporte **6 types d'unités** définis dans `polytopia_jax/core/state.py` :

```python
class UnitType(IntEnum):
    NONE = 0
    WARRIOR = 1
    DEFENDER = 2
    ARCHER = 3
    RIDER = 4
    RAFT = 5
    NUM_TYPES = 6
```

**Statistiques implémentées** (dans `rules.py`) :
- **WARRIOR** : HP=10, Attaque=2, Défense=2, Mouvement=1, Coût=2★
- **DEFENDER** : HP=15, Attaque=1, Défense=3, Mouvement=1, Coût=3★
- **ARCHER** : HP=8, Attaque=2, Défense=1, Mouvement=1, Portée=2, Coût=3★
- **RIDER** : HP=10, Attaque=3, Défense=1, Mouvement=2, Coût=4★
- **RAFT** : HP=8, Attaque=2, Défense=1, Mouvement=2, Naval, Coût=0★ (créé via embarquement)

### Frontend

**Problème :** Le fichier `frontend/src/types.ts` ne définit que `WARRIOR` :

```typescript
export const UnitType = {
  NONE: 0,
  WARRIOR: 1,
} as const;
```

**Conséquences :**
- ❌ Les unités DEFENDER, ARCHER, RIDER ne peuvent pas être affichées correctement
- ❌ Les unités RAFT (radeaux) ne sont pas distinguées visuellement
- ❌ L'entraînement d'unités autres que WARRIOR n'est pas possible via l'UI
- ❌ Les icônes et noms des autres types d'unités ne sont pas définis

**Impact :** Les joueurs ne peuvent actuellement entraîner et utiliser que des WARRIOR, alors que le moteur supporte 4 autres types d'unités.

---

## 2. Action RECOVER Non Exposée

### Simulateur JAX

L'action **RECOVER** est implémentée dans `polytopia_jax/core/rules.py` (lignes 469-539) :

- Permet à une unité de guérir **4 HP en territoire ami** (ville amie ou adjacente)
- Permet de guérir **2 HP en territoire neutre/ennemi**
- L'unité doit avoir déjà agi ce tour (`units_has_acted = False`)
- L'unité ne doit pas être à HP maximum

**Encodage :** L'action est encodée avec `ActionType.RECOVER = 8` et nécessite un `unit_id`.

### Frontend

**Problème :** Bien que l'action soit définie dans `frontend/src/utils/actionEncoder.ts` :

```typescript
export function encodeRecover(unitId: number): number {
  return encodeAction({
    actionType: ActionType.RECOVER,
    unitId,
  });
}
```

**Il n'existe aucune UI pour déclencher cette action :**
- ❌ Pas de bouton "Guérir" dans `LiveGameView.tsx`
- ❌ Pas d'indication visuelle qu'une unité peut guérir
- ❌ Pas de vérification si l'unité est blessée et peut guérir
- ❌ Pas d'affichage du territoire ami/ennemi pour déterminer le montant de guérison

**Impact :** Les joueurs ne peuvent pas guérir leurs unités blessées, ce qui limite significativement la stratégie militaire.

---

## 3. Système de Radeaux (Embark/Disembark) Non Visible

### Simulateur JAX

Le moteur implémente un système complet d'embarquement/débarquement :

- **Embark** : Une unité terrestre peut se transformer en RAFT lorsqu'elle entre dans l'eau peu profonde depuis un port ami (si SAILING est débloquée)
- **Disembark** : Un RAFT peut redevenir son type d'unité original lorsqu'il atteint la terre ferme
- Le type original est stocké dans `units_payload_type`

**Logique implémentée** (dans `rules.py`, lignes 278-300, 1098-1116) :
- Conversion automatique lors du mouvement
- Préservation du type d'unité original dans `payload_type`

### Frontend

**Problème :** Aucune indication visuelle ou fonctionnelle :

- ❌ Les unités RAFT ne sont pas distinguées visuellement des unités terrestres
- ❌ Le champ `units_payload_type` n'est pas exposé dans `UnitView`
- ❌ Pas d'indication qu'une unité est en radeau
- ❌ Pas d'affichage du type d'unité original lors de l'embarquement
- ❌ Pas d'icône spéciale pour les radeaux

**Impact :** Les joueurs ne peuvent pas distinguer les radeaux des unités terrestres, ce qui rend la navigation confuse.

---

## 4. Bâtiments Non Affichés

### Simulateur JAX

Le moteur supporte **5 types de bâtiments** (dans `rules.py`) :

```python
class BuildingType(IntEnum):
    NONE = 0
    FARM = 1      # Coût: 3★, +1 pop
    MINE = 2      # Coût: 4★, +2 pop, requiert MINING
    HUT = 3       # Coût: 2★, +1 pop
    PORT = 4      # Coût: 5★, requiert SAILING
    NUM_TYPES = 5
```

**Fonctionnalités :**
- Les bâtiments augmentent la population des villes
- Les ports permettent l'embarquement (`city_has_port`)
- Les coûts et prérequis technologiques sont gérés

### Frontend

**Problème :** Aucun affichage des bâtiments construits :

- ❌ Pas d'indicateur visuel des bâtiments sur les villes
- ❌ Le champ `city_ports` existe dans `GameStateView` mais n'est pas utilisé visuellement
- ❌ Pas de liste des bâtiments construits dans une ville
- ❌ Pas d'icônes pour les différents types de bâtiments
- ❌ L'action BUILD existe mais les bâtiments construits ne sont pas visibles

**Impact :** Les joueurs ne peuvent pas voir quels bâtiments ont été construits dans leurs villes, ce qui rend la gestion économique opaque.

---

## 5. Ressources Non Affichées sur le Terrain

### Simulateur JAX

Le moteur gère un système complet de ressources :

- **Types** : FRUIT (plaine), FISH (eau peu profonde), ORE (montagne)
- **Champs** : `resource_type` et `resource_available` dans `GameState`
- **Récolte** : Action `HARVEST_RESOURCE` avec coûts et prérequis technologiques
- **Effets** : Augmentation de population des villes adjacentes

**Logique implémentée** :
- Les ressources sont placées sur le terrain lors de la génération
- La récolte consomme la ressource (`resource_available = False`)
- Le terrain change après récolte (ex: `PLAIN_FRUIT` → `PLAIN`)

### Frontend

**Problème :** Affichage limité des ressources :

- ✅ Les ressources sont affichées dans la zone de récolte d'une ville sélectionnée
- ❌ **Les ressources ne sont pas affichées directement sur le terrain** (pas d'icônes sur les cases)
- ❌ Pas d'indication visuelle des ressources disponibles avant de sélectionner une ville
- ❌ Pas d'affichage des ressources déjà récoltées (cases vides)
- ❌ Le terrain ne reflète pas visuellement les ressources présentes

**Impact :** Les joueurs doivent sélectionner chaque ville pour voir les ressources disponibles, ce qui rend la planification économique difficile.

---

## 6. Informations d'Unité Manquantes

### Simulateur JAX

Le `GameState` contient plusieurs champs pour les unités :

- `units_has_acted` : Indique si l'unité a déjà agi ce tour
- `units_payload_type` : Type original pour les radeaux
- `units_active` : Indique si l'unité existe encore

### Frontend

**Problème :** Informations partielles dans `UnitView` :

```typescript
export interface UnitView {
  id?: number;
  type: number;
  pos: [number, number];
  hp: number;
  owner: number;
  has_acted?: boolean;  // ✅ Présent mais optionnel
}
```

**Manquants :**
- ❌ `has_acted` n'est pas toujours affiché visuellement
- ❌ Pas d'indication claire qu'une unité a déjà agi (pas de badge "✓" ou grisé)
- ❌ Pas d'affichage de `payload_type` pour les radeaux
- ❌ Pas de distinction visuelle entre unités actives et inactives

**Impact :** Les joueurs ne savent pas toujours quelles unités peuvent encore agir ce tour.

---

## 7. Score Breakdown Partiellement Affiché

### Simulateur JAX

Le moteur calcule un breakdown détaillé du score (dans `score.py`) :

- **Territoire** : 100 points par ville
- **Population** : 5 points par point de population
- **Militaire** : 20 points par unité active
- **Ressources** : 2 points par étoile restante

**Champs dans `GameState` :**
- `score_territory`
- `score_population`
- `score_military`
- `score_resources`

### Frontend

**Problème :** Le `Scoreboard` affiche le breakdown mais avec limitations :

- ✅ Le breakdown est affiché dans `Scoreboard.tsx`
- ❌ Les labels sont en anglais dans le code (`territory`, `population`, `military`, `economy`)
- ❌ Le mapping `economy` → `resources` peut être confus
- ❌ Pas d'explication de la formule de calcul
- ❌ Pas d'affichage des pondérations (100, 5, 20, 2)

**Impact :** Les joueurs peuvent voir le breakdown mais ne comprennent pas toujours comment il est calculé.

---

## 8. Ports Non Visibles

### Simulateur JAX

Le système de ports est implémenté :

- `city_has_port` : Booléen indiquant la présence d'un port
- Les ports permettent l'embarquement depuis cette ville
- Coût : 5★, requiert SAILING

### Frontend

**Problème :** Les ports ne sont pas visibles :

- ❌ Pas d'icône ou d'indicateur visuel pour les ports
- ❌ Le champ `city_ports` existe dans `GameStateView` mais n'est pas utilisé
- ❌ Pas d'indication qu'une ville a un port lors de la sélection
- ❌ Pas d'affichage dans la liste des villes

**Impact :** Les joueurs ne savent pas quelles villes ont des ports, ce qui rend la navigation impossible à planifier.

---

## 9. Types de Terrain avec Ressources Non Distingués

### Simulateur JAX

Le moteur définit des variantes de terrain avec ressources :

- `PLAIN_FRUIT` (5)
- `FOREST_WITH_WILD_ANIMAL` (6)
- `MOUNTAIN_WITH_MINE` (7)
- `WATER_SHALLOW_WITH_FISH` (8)

Ces types sont distincts des terrains de base.

### Frontend

**Problème :** Pas de distinction visuelle :

- ❌ Les icônes de terrain ne distinguent pas les variantes avec ressources
- ❌ `getTerrainIcon()` ne gère probablement pas tous les types
- ❌ Pas d'indication visuelle qu'une case contient une ressource avant récolte

**Impact :** Les joueurs ne peuvent pas voir les ressources sur le terrain sans sélectionner une ville.

---

## 10. Action BUILD Non Complètement Exposée

### Simulateur JAX

L'action BUILD permet de construire des bâtiments dans les villes :

- FARM, MINE, HUT, PORT
- Vérification des coûts, prérequis, et propriété de la ville

### Frontend

**Problème :** L'action BUILD n'est pas accessible via l'UI :

- ❌ Pas de bouton ou menu pour construire des bâtiments
- ❌ Pas de liste des bâtiments disponibles dans `LiveGameView`
- ❌ Pas d'affichage des coûts et prérequis des bâtiments
- ❌ L'action existe dans `actionEncoder.ts` mais n'est pas utilisée

**Impact :** Les joueurs ne peuvent pas construire de bâtiments, ce qui bloque complètement la progression économique.

---

## 11. Informations de Combat Non Affichées

### Simulateur JAX

Le moteur calcule des informations détaillées de combat :

- **Bonus de défense** selon terrain (forêt, montagne, eau, ville)
- **Formule complète** avec ratio HP actuel / HP max
- **Contre-attaque** conditionnelle selon portée et survie
- **Déplacement après combat** si l'attaquant tue la cible en mêlée

### Frontend

**Problème :** Aucune information de combat affichée :

- ❌ Pas de prévisualisation des dégâts avant attaque
- ❌ Pas d'affichage des bonus de défense
- ❌ Pas d'indication de la portée d'attaque
- ❌ Pas de calcul de dégâts estimés
- ❌ Pas d'affichage des stats d'attaque/défense des unités

**Impact :** Les joueurs attaquent à l'aveugle sans connaître les chances de succès ou les dégâts attendus.

---

## 12. Gestion des Technologies Partielle

### Simulateur JAX

Le système de technologies est complet :

- **6 technologies** : CLIMBING, SAILING, MINING, ARCHERY, AQUATISM
- **Dépendances** : SAILING requiert CLIMBING
- **Coûts** : 3-6★ selon la technologie
- **Effets** : Déblocage d'unités, bâtiments, bonus de défense

### Frontend

**Problème :** Affichage partiel :

- ✅ L'arbre technologique est affiché dans `LiveGameView`
- ✅ Les dépendances sont visibles
- ❌ **Les effets des technologies ne sont pas clairement expliqués**
- ❌ Pas d'indication que ARCHERY débloque l'unité ARCHER
- ❌ Pas d'indication que SAILING permet la construction de ports
- ❌ Pas d'affichage des bonus de défense accordés

**Impact :** Les joueurs ne comprennent pas toujours ce que chaque technologie débloque concrètement.

---

## Recommandations Prioritaires

### Priorité Haute

1. **Ajouter les types d'unités manquants** (DEFENDER, ARCHER, RIDER, RAFT)
   - Mettre à jour `types.ts`
   - Ajouter les icônes et noms
   - Permettre l'entraînement de tous les types

2. **Exposer l'action BUILD**
   - Ajouter un menu de construction dans `LiveGameView`
   - Afficher les bâtiments disponibles avec coûts et prérequis
   - Afficher visuellement les bâtiments construits sur les villes

3. **Exposer l'action RECOVER**
   - Ajouter un bouton "Guérir" pour les unités blessées
   - Afficher le montant de guérison selon le territoire
   - Indiquer visuellement les unités éligibles

### Priorité Moyenne

4. **Afficher les ressources sur le terrain**
   - Ajouter des icônes de ressources directement sur les cases
   - Distinguer visuellement les ressources disponibles/récoltées
   - Afficher les types de terrain avec ressources

5. **Améliorer l'affichage des radeaux**
   - Distinguer visuellement les RAFT des unités terrestres
   - Afficher le type d'unité original lors de l'embarquement
   - Ajouter des icônes spéciales pour les radeaux

6. **Afficher les ports**
   - Ajouter un indicateur visuel pour les villes avec ports
   - Afficher dans la liste des villes

### Priorité Basse

7. **Améliorer les informations de combat**
   - Prévisualisation des dégâts avant attaque
   - Affichage des bonus de défense
   - Calcul des dégâts estimés

8. **Améliorer l'affichage des unités**
   - Badge "✓" pour les unités ayant agi
   - Affichage des stats d'attaque/défense
   - Indication de la portée d'attaque

---

## Conclusion

Le simulateur JAX implémente un système de jeu riche avec de nombreuses mécaniques avancées. Cependant, le frontend n'expose qu'une partie limitée de ces fonctionnalités, principalement autour des WARRIOR, de la récolte de ressources, et de la recherche technologique.

Les différences les plus critiques concernent :
- Les types d'unités non supportés
- L'absence d'UI pour BUILD et RECOVER
- Le manque d'affichage visuel des ressources, bâtiments et ports

Ces limitations réduisent significativement la profondeur stratégique accessible aux joueurs via l'interface web.

