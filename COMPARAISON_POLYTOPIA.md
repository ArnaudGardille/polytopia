# Compte-rendu : Différences entre le Simulateur JAX et Polytopia

**Date :** 2025-01-27  
**Source documentation :** Wiki Polytopia (scripts/wiki_knowledge/)

---

## Résumé Exécutif

Le simulateur Polytopia-JAX implémente un sous-ensemble simplifié des mécaniques du jeu original Polytopia. Il se concentre sur les fonctionnalités essentielles pour l'apprentissage par renforcement, avec des simplifications importantes dans plusieurs domaines clés : combat, mouvement, technologies, génération de cartes, et modes de jeu.

---

## 1. Système de Combat

### Polytopia Original

Le système de combat utilise une formule complexe basée sur les forces d'attaque et de défense :

```
attackForce = attacker.attack * (attacker.health / attacker.maxHealth)
defenseForce = defender.defense * (defender.health / defender.maxHealth) * defenseBonus
totalDamage = attackForce + defenseForce
attackResult = round((attackForce / totalDamage) * attacker.attack * 4.5)
defenseResult = round((defenseForce / totalDamage) * defender.defense * 4.5)
```

**Caractéristiques :**
- Les dégâts dépendent du ratio HP actuel / HP max
- Bonus de défense selon terrain (forêt, montagne, eau, ville) : 1.5x standard, 4x pour murs de ville
- Pas de contre-attaque si l'attaquant tue la cible ou si la cible ne peut pas voir l'attaquant (brouillard)
- Les unités en mêlée prennent la place de l'unité tuée (sauf unités à distance)
- Guérison possible : 4 HP en territoire ami, 2 HP en territoire neutre/ennemi
- Poison (Cymanti) : réduit défense de 30%, empêche guérison
- Boost (Shaman Cymanti) : +0.5 attaque, +1 mouvement

### Simulateur JAX

**Implémentation actuelle :**
```python
# Formule complète de Polytopia implémentée
attack_force = attacker_attack * (attacker_hp / attacker_max_hp)
defense_force = target_defense * (target_hp / target_max_hp) * defense_bonus
total_damage = attack_force + defense_force
attack_result = round((attack_force / total_damage) * attacker_attack * 4.5)
defense_result = round((defense_force / total_damage) * target_defense * 4.5)
```

**Différences majeures :**
- ✅ **Formule complète** : Implémentée avec ratio HP actuel / HP max pour attaque et défense
- ✅ **Bonus de défense terrain** : 1.5x si ARCHERY (forêt), CLIMBING (montagne), AQUATISM (eau), ou en ville amie
- ✅ **Murs de ville** : Bonus 4x implémenté (City Wall)
- ✅ **Guérison** : Action RECOVER implémentée (4 HP territoire ami, 2 HP ailleurs)
- ✅ **Pas de contre-attaque si tué** : Si la cible est tuée par l'attaque, pas de riposte
- ✅ **Prise de place** : Implémentée pour unités mêlée tuant une cible
- ✅ **Promotions** : Système de vétérans implémenté (3 kills = +5 HP max, guérison complète)
- ❌ **Vérification vision/brouillard** : Contre-attaque ne vérifie pas si la cible peut voir l'attaquant
- ❌ **Poison/Boost** : Mécaniques spéciales tribus non implémentées

**Impact :** Le système de combat est maintenant très fidèle à Polytopia. Les unités blessées sont pénalisées, le terrain a un impact défensif significatif, et les murs de ville offrent une défense puissante. La guérison et les promotions ajoutent de la profondeur tactique. Il reste à implémenter la vérification de vision pour la contre-attaque.

---

## 2. Système de Mouvement

### Polytopia Original

**Caractéristiques :**
- Coût de mouvement : 1 par case adjacente
- **Routes** : Réduisent le coût à 0.5 entre deux cases avec routes
- **Terrain difficile** : Forêts et montagnes bloquent le mouvement (sauf si route)
- **Zone de contrôle** : Une unité ne peut pas passer à côté d'une unité ennemie (sauf compétence Creep)
- **Ports** : Les unités terrestres entrant dans un port deviennent des Rafts (ne peuvent plus agir ce tour)
- **Compétences spéciales** : Creep (ignore zone de contrôle), Dash, Escape, Fly, etc.
- **Brouillard de guerre** : Bloque le mouvement vers cases non explorées

### Simulateur JAX

**Implémentation actuelle :**
- ✅ Mouvement basique avec distance de Chebyshev
- ✅ Vérification terrain (montagnes nécessitent Climbing, eau nécessite Sailing)
- ✅ Transformation en Raft via ports
- ✅ **Routes et Ponts** : Système de routes (plaine/forêt) et ponts (eau peu profonde) implémentés
- ✅ **Brouillard de guerre** : Système d'exploration implémenté (mouvement bloqué vers cases non explorées ou non adjacentes à cases explorées)
- ✅ **Vision** : Système de vision par unité (3x3 standard, 5x5 sur montagnes) et par ville (3x3)
- ✅ **Exploration** : Cases explorées marquées de façon permanente, vision mise à jour chaque tour
- ❌ **Pas de zone de contrôle** : Les unités peuvent passer librement à côté d'ennemis
- ❌ **Pas de compétences spéciales** de mouvement (Dash, Creep, etc.)
- ❌ **Coût mouvement routes** : Routes ne réduisent pas encore le coût de mouvement (0.5 au lieu de 1)

**Impact :** Le mouvement est maintenant plus stratégique avec brouillard de guerre et exploration. Routes et ponts permettent la connexion entre villes. Il reste à implémenter la réduction de coût de mouvement via routes et la zone de contrôle.

---

## 3. Arbre Technologique

### Polytopia Original

**Structure :**
- **Tier 1** : ~3-5★ (Organization, Hunting, Fishing, Riding, Climbing, Archery, etc.)
- **Tier 2** : ~6-7★ (Construction, Mining, Roads, Navigation, etc.)
- **Tier 3+** : Technologies avancées (Diplomacy, Mathematics, Chivalry, etc.)
- **Coût dynamique** : Augmente avec le nombre de villes contrôlées
- **Philosophy** : Réduit coût des recherches suivantes de -33%
- **Dépendances complexes** : Chaque tech a des prérequis spécifiques
- **~30+ technologies** au total

**Technologies débloquent :**
- Unités (Archer, Rider, Defender, Knight, Catapult, etc.)
- Bâtiments (Farms, Mines, Ports, Temples, etc.)
- Compétences (Burn Forest, Clear Forest, Grow Forest, etc.)
- Bonus passifs (défense en forêt avec Archery, etc.)

### Simulateur JAX

**Implémentation actuelle :**
```python
class TechType(IntEnum):
    NONE = 0
    # Tier 1
    CLIMBING = 1
    FISHING = 2
    HUNTING = 3
    ORGANIZATION = 4
    RIDING = 5
    # Tier 2
    ARCHERY = 6
    RAMMING = 7
    FARMING = 8
    FORESTRY = 9
    FREE_SPIRIT = 10
    MEDITATION = 11
    MINING = 12
    ROADS = 13
    SAILING = 14
    STRATEGY = 15
    # Tier 3
    AQUATISM = 16
    PHILOSOPHY = 17
    SMITHERY = 18
    CHIVALRY = 19
    MATHEMATICS = 20
    NUM_TECHS = 21
```

**Différences majeures :**
- ✅ **21 technologies** implémentées (Tier 1-3) vs ~30+ dans Polytopia
- ✅ **Coût dynamique** : Augmente avec nombre de villes (formule: tier * num_cities + 4)
- ✅ **Philosophy** : Réduction de coût de -33% implémentée
- ✅ **Dépendances complexes** : Arbre technologique avec prérequis multiples (ex: ARCHERY → HUNTING, SAILING → CLIMBING, etc.)
- ✅ **Débloque unités/bâtiments** : Technologies débloquent unités (Archer, Rider, Defender, Knight, Swordsman, Catapult) et bâtiments (Ports, Mines, Routes, etc.)

**Impact :** L'arbre technologique est maintenant beaucoup plus complet. Progression technologique complexe avec coût dynamique et Philosophy, permettant des stratégies basées sur l'ordre de recherche des techs.

---

## 4. Types d'Unités

### Polytopia Original

**Unités terrestres :**
- Warrior (2 att, 2 def, 10 HP, 1 mov)
- Defender (1 att, 3 def, 15 HP, 1 mov)
- Archer (2 att, 1 def, 8 HP, 1 mov, portée 2)
- Rider (3 att, 1 def, 10 HP, 2 mov)
- Knight (4 att, 1 def, 15 HP, 3 mov)
- Swordsman (3 att, 2 def, 15 HP, 1 mov)
- Catapult (4 att, 1 def, 8 HP, 1 mov, portée 3)
- Giant (Super Unit, 5 att, 3 def, 40 HP)
- Et bien d'autres...

**Unités navales :**
- Raft (transport)
- Boat (navire basique)
- Ship (navire amélioré)
- Battleship (navire de guerre)

**Unités spéciales par tribu :**
- Cymanti : Hexapod, Doomux, Raychi, etc.
- Polaris : Mooni, Battle Sled, Ice Fortress
- Aquarion : Tridention, etc.

### Simulateur JAX

**Implémentation actuelle :**
```python
UNIT_HP_MAX = [0, 10, 15, 8, 10, 8, 15, 15, 8, 40]  # NONE, WARRIOR, DEFENDER, ARCHER, RIDER, RAFT, KNIGHT, SWORDSMAN, CATAPULT, GIANT
UNIT_ATTACK = [0, 2, 1, 2, 3, 2, 4, 3, 4, 5]
UNIT_DEFENSE = [0, 2, 3, 1, 1, 1, 1, 2, 1, 3]
UNIT_MOVEMENT = [0, 1, 1, 1, 2, 2, 3, 1, 1, 1]
UNIT_ATTACK_RANGE = [0, 1, 1, 2, 1, 1, 1, 1, 3, 1]  # Catapult portée 3
```

**Différences majeures :**
- ✅ **10 types d'unités** : Warrior, Defender, Archer, Rider, Raft, Knight, Swordsman, Catapult, Giant
- ✅ **Knight, Swordsman, Catapult, Giant** : Tous implémentés avec stats correctes
- ✅ **Portée d'attaque** : Implémentée pour Archer (portée 2) et Catapult (portée 3)
- ✅ **Promotions** : Système de vétérans implémenté (3 kills = +5 HP max)
- ❌ **Pas d'unités navales avancées** (Boat, Ship, Battleship)
- ❌ **Pas d'unités spéciales** par tribu
- ❌ **Pas de compétences spéciales** (Dash, Escape, Fortify, etc.)

**Impact :** Diversité tactique améliorée avec 10 types d'unités. Stratégies basées sur différentes compositions d'armées possibles. Le système de promotions ajoute de la profondeur.

---

## 5. Bâtiments et Économie

### Polytopia Original

**Bâtiments de base :**
- Farm (+1 pop, 3★)
- Mine (+2 pop, 4★, nécessite Mining)
- Hut (+1 pop, 2★)
- Port (permet embarquement, 5★, nécessite Sailing)
- Windmill (+pop selon fermes adjacentes)
- Forge (+2 pop par mine adjacente)
- Sawmill (+pop selon huttes adjacentes)
- Market (génère étoiles)
- Temples (100 pts + 50 pts par niveau)
- Monuments (400 pts)
- Park (250 pts)
- City Wall (bonus défense x4)

**Ressources récoltables :**
- Fruit (sur plaine, +1 pop, 2★)
- Fish (sur eau peu profonde, +1 pop, 2★, nécessite Sailing)
- Ore (sur montagne, +2 pop, 4★, nécessite Mining)
- Animaux sauvages (forêt, esthétique uniquement)

**Revenus :**
- Villes génèrent étoiles selon niveau : 2★ (niv1), 4★ (niv2), 6★ (niv3+)
- Bonus difficulté IA : +1 à +5★ par tour selon difficulté

### Simulateur JAX

**Implémentation actuelle :**
```python
BUILDING_COST = [0, 3, 4, 2, 5, 6, 7, 5, 8, 10, 20, 5, 15, 3, 5]  # NONE, FARM, MINE, HUT, PORT, WINDMILL, FORGE, SAWMILL, MARKET, TEMPLE, MONUMENT, CITY_WALL, PARK, ROAD, BRIDGE
BUILDING_POP_GAIN = [0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RESOURCE_COST = [0, 2, 2, 4]     # NONE, FRUIT, FISH, ORE
RESOURCE_POP_GAIN = [0, 1, 1, 2]
CITY_STAR_INCOME_PER_LEVEL = [0, 2, 4, 6, 6, 6]  # Niveau 3+ = 6★
```

**Différences majeures :**
- ✅ **Bâtiments de base** : Farm, Mine, Hut, Port implémentés
- ✅ **Bâtiments avancés** : Windmill (+pop selon fermes adjacentes), Forge (+2 pop par mine adjacente), Sawmill (+pop selon huttes adjacentes)
- ✅ **Marchés** : Market génère +1★ par tour
- ✅ **Temples** : Temple implémenté (niveau 1 initial, peut être amélioré)
- ✅ **Monuments** : Monument implémenté (400 pts score)
- ✅ **City Wall** : Murs de ville implémentés (bonus défense 4x)
- ✅ **Park** : Parc implémenté (250 pts score)
- ✅ **Routes et Ponts** : Routes (plaine/forêt) et ponts (eau peu profonde) implémentés
- ✅ **Ressources** : Fruit, Fish, Ore implémentées
- ✅ **Revenus villes** : 2/4/6★ selon niveau (niveau 3+ = 6★)
- ✅ **Bonus difficulté** : Implémenté via `player_income_bonus`
- ✅ **Connexions villes** : Système de connexions via routes/ponts/ports implémenté (+1 pop par connexion à capitale)

**Impact :** Économie complète avec bâtiments avancés. Stratégies basées sur optimisation économique avancée possibles (marchés, temples pour score, connexions villes).

---

## 6. Génération de Cartes

### Polytopia Original

**Tailles de carte :**
- Tiny (11x11, 121 tiles)
- Small (14x14, 196 tiles)
- Normal (16x16, 256 tiles)
- Large (18x18, 324 tiles)
- Huge (20x20, 400 tiles)
- Massive (30x30, 900 tiles)

**Types de cartes :**
- Drylands (0-10% eau)
- Lakes (25-30% eau)
- Continents (40-70% eau)
- Pangea (40-60% eau)
- Archipelago (60-80% eau)
- Waterworld (90-100% eau)

**Processus de génération :**
1. Placement capitales (quadrants pour équilibrer distances)
2. Villages (suburbs, pre-terrain, post-terrain selon type carte)
3. Terrain (avec taux de spawn par tribu)
4. Ressources (uniquement à 2 cases des villes)
5. Ruins (4-23 selon taille carte)
6. Starfish (1 pour 25 cases d'eau)

**Caractéristiques spéciales :**
- Répartition équitable du territoire entre joueurs
- Villages garantis à 2 cases minimum des capitales
- Ressources uniquement près des villes
- Ruins avec récompenses variées

### Simulateur JAX

**Implémentation actuelle :**
```python
class GameConfig:
    height: int = 10
    width: int = 10
    prob_plain: float = 0.45
    prob_forest: float = 0.2
    prob_mountain: float = 0.15
    prob_water: float = 0.15
    prob_water_deep: float = 0.05
```

**Différences majeures :**
- ✅ **Génération procédurale** : Terrain aléatoire avec probabilités
- ✅ **Placement capitales** : Aux coins opposés (stratégie simple)
- ✅ **Ressources** : Générées avec probabilités fixes (30% fruit, 35% fish, 40% ore)
- ❌ **Taille fixe** : 10x10 uniquement (pas de variantes)
- ❌ **Pas de types de cartes** : Pas de Drylands/Lakes/Continents/etc.
- ❌ **Pas de villages** : Seulement capitales
- ❌ **Pas de ruins** : Pas de récompenses aléatoires
- ❌ **Pas de starfish**
- ❌ **Pas de répartition équitable** : Capitales aux coins, pas de garantie d'équité

**Impact :** Cartes beaucoup plus simples et moins variées. Pas de mécaniques de découverte (ruins), pas de variété de tailles/types de cartes.

---

## 7. Modes de Jeu

### Polytopia Original

**Modes solo :**
- **Perfection** : 30 tours, score maximum, bonus difficulté selon nombre d'adversaires
- **Domination** : Élimination complète, pas de limite de tours
- **Creative** : Personnalisation complète (taille, type carte, nombre adversaires)

**Modes multijoueur :**
- **Glory** : Premier à 10,000 pts gagne
- **Might** : Capturer toutes les capitales ennemies

**Conditions de victoire Perfection :**
- Score basé sur : Territoire, Population, Militaire, Science, Monuments/Temples
- Bonus difficulté : 100% + 41% * ln(nb_adversaires) + bonus difficulté (20/40/80%)

**Conditions de victoire Domination :**
- Éliminer tous les adversaires (capturer dernière capitale)
- Rating basé sur : Speed skills, Battle skills, Tribes destroyed, Difficulty rating

### Simulateur JAX

**Implémentation actuelle :**
```python
class GameMode(IntEnum):
    DOMINATION = 0
    PERFECTION = 1
    CREATIVE = 2
    GLORY = 3
    MIGHT = 4
```

**Différences majeures :**
- ✅ **Cinq modes** : DOMINATION, PERFECTION, CREATIVE, GLORY, MIGHT
- ✅ **DOMINATION** : Élimination (vérifie si joueur a encore capitale)
- ✅ **PERFECTION** : Limite de tours (30 par défaut), score avec bonus difficulté
- ✅ **CREATIVE** : Mode personnalisable (pas de limite, partie continue jusqu'à élimination)
- ✅ **GLORY** : Premier à 10,000 pts gagne
- ✅ **MIGHT** : Capturer toutes les capitales ennemies
- ✅ **Score Perfection complet** : Calcul complet (territoire, population, militaire, exploration, temples, monuments, parcs, science)
- ✅ **Bonus difficulté** : Formule Polytopia implémentée (100% + 41% * ln(nb_adversaires) + bonus difficulté)

**Impact :** Modes de jeu complets avec variété de gameplay. Système de scoring avancé aligné avec Polytopia.

---

## 8. Système de Score

### Polytopia Original

**Composantes du score :**
- **Armée et Territoire** :
  - Unités : 5 pts par étoile de coût (ex: Warrior 2★ = 10 pts)
  - Super Unit : 50 pts
  - Territoire : 20 pts par case contrôlée
  - Exploration : 5 pts par case explorée
- **Villes** :
  - Niveau 1 : 100 pts + 5 pts par population
  - Niveau 2+ : +50 pts par niveau au-dessus de 1
  - Park : 250 pts
- **Monuments et Temples** :
  - Monuments : 400 pts
  - Temples : 100 pts + 50 pts par niveau
- **Science** :
  - Technologie : 100 pts par tier de la tech

**Calcul final Perfection :**
- Score brut × (100% + bonus difficulté)

### Simulateur JAX

**Implémentation actuelle :**
```python
TERRITORY_POINTS_PER_TILE = 20      # 20 pts par case contrôlée
EXPLORATION_POINTS_PER_TILE = 5     # 5 pts par case explorée
UNIT_POINTS_PER_STAR = 5            # 5 pts par étoile de coût d'unité
SUPER_UNIT_POINTS = 50              # 50 pts pour Super Unit (Giant)
CITY_BASE_POINTS = 100              # 100 pts de base pour niveau 1
CITY_POPULATION_POINTS = 5          # 5 pts par point de population
CITY_LEVEL_BONUS = 50               # +50 pts par niveau au-dessus de 1
TEMPLE_BASE_POINTS = 100            # 100 pts de base pour temple
TEMPLE_LEVEL_POINTS = 50            # 50 pts par niveau de temple
MONUMENT_POINTS = 400               # 400 pts par monument
PARK_POINTS = 250                   # 250 pts par parc
TECH_POINTS_PER_TIER = 100          # 100 pts par tier de technologie
```

**Différences majeures :**
- ✅ **Composantes complètes** : Territoire (20 pts/case), Population (100 + 5*pop + 50*niveau), Militaire (5 pts/étoile coût), Exploration (5 pts/case explorée)
- ✅ **Exploration** : Points pour cases explorées implémentés (5 pts/case)
- ✅ **Monuments/Temples** : Monuments (400 pts), Temples (100 + 50*niveau) implémentés
- ✅ **Parcs** : Parcs (250 pts) implémentés
- ✅ **Science** : Points pour technologies implémentés (100 pts/tier)
- ✅ **Bonus difficulté** : Formule Polytopia implémentée (100% + 41% * ln(nb_adversaires) + bonus difficulté)
- ✅ **Unités** : Points selon coût d'unité (5 pts/étoile), Super Unit = 50 pts

**Impact :** Score complet et fidèle à Polytopia. Stratégies basées sur optimisation du score possibles (temples, monuments, exploration, science).

---

## 9. Villes et Population

### Polytopia Original

**Niveaux de ville :**
- Niveau 1 : 1 population (seuil)
- Niveau 2 : 3 population (seuil)
- Niveau 3 : 5 population (seuil)
- Niveau 4+ : Seuils croissants

**Revenus par niveau :**
- Niveau 1 : 2★ par tour
- Niveau 2 : 4★ par tour
- Niveau 3+ : 6★ par tour

**Capture de ville :**
- Population réinitialisée à 1 (niveau 1)
- Bâtiments conservés (sauf certains cas)

**Connexions de villes :**
- Routes connectent villes pour bonus économiques
- Villes connectées partagent certains avantages

### Simulateur JAX

**Implémentation actuelle :**
```python
CITY_LEVEL_POP_THRESHOLDS = [0, 1, 3, 5]
CITY_STAR_INCOME_PER_LEVEL = [0, 2, 4, 6]
CITY_CAPTURE_POPULATION = 1
```

**Différences majeures :**
- ✅ **Niveaux de ville** : 1, 2, 3 implémentés (seuils 1, 3, 5 pop)
- ✅ **Revenus** : 2/4/6★ selon niveau
- ✅ **Capture** : Population réinitialisée à 1
- ❌ **Pas de connexions** : Pas de système de routes entre villes
- ❌ **Pas de niveau 4+** : Limité à niveau 3

**Impact :** Système de villes fonctionnel mais simplifié. Pas de mécaniques de connexion entre villes.

---

## 18. Autres Mécaniques Manquantes

### Polytopia Original (non implémentées)

**Diplomatie et Relations :**
- ❌ **Diplomatie** : Pas d'alliances, traités de paix, ambassades
- ❌ **Relations tribales** : Pas de système de relations entre joueurs
- ❌ **Espionnage** : Pas de Cloaks/Daggers pour infiltration

**Mécaniques de terrain avancées :**
- ❌ **Modifications terrain** : Pas de Grow/Clear/Burn Forest
- ❌ **Terrain spécial** : Pas d'Ice, Flooded, Algae
- ❌ **Brouillard de guerre** : Toute la carte visible dès le début

**Récompenses et exploration :**
- ❌ **Ruins** : Pas de récompenses aléatoires
- ❌ **Exploration** : Pas de système de découverte progressive
- ❌ **Explorer** : Pas d'unité spéciale d'exploration

**Tribes et unités spéciales :**
- ❌ **Tribes spéciales** : Pas de Cymanti, Polaris, Aquarion, ∑∫ỹriȱŋ
- ❌ **Super Units** : Pas de Giants ou unités spéciales par tribu
- ❌ **Promotions** : Pas de système de vétérans (3 kills = +5 HP)

**Compétences et actions :**
- ❌ **Compétences d'unités** : Pas de Dash, Escape, Fortify, Creep, Fly, etc.
- ❌ **Actions spéciales** : Pas de Recover, Disband, Heal Others, Boost, etc.
- ❌ **Abilities** : Pas de Burn Forest, Clear Forest, Grow Forest, etc.

**Bâtiments et améliorations :**
- ❌ **Routes/Ponts** : Pas de construction de routes
- ❌ **Temples/Monuments** : Pas de bâtiments spéciaux pour score
- ❌ **Marchés** : Pas de génération d'étoiles supplémentaires
- ❌ **City Walls** : Pas de bonus défense villes
- ❌ **Workshops/Parks** : Pas d'améliorations de villes niveau 2/5+

**Mécaniques de combat avancées :**
- ✅ **Guérison** : Action RECOVER implémentée (4 HP territoire ami, 2 HP territoire neutre/ennemi)
- ❌ **Poison/Boost** : Pas de mécaniques spéciales Cymanti
- ❌ **Splash damage** : Pas de dégâts de zone
- ❌ **Conversion** : Pas de Mind Bender convertissant ennemis

**Gestion d'unités :**
- ❌ **Disband** : Pas de retrait volontaire d'unités (rembourse 50%)
- ❌ **Capacité villes** : Pas de limite d'unités par ville (niveau+1)
- ❌ **Association ville** : Pas de système de ville d'origine

**Économie avancée :**
- ❌ **Coût tech dynamique** : Pas d'augmentation avec nombre de villes
- ❌ **Philosophy** : Pas de réduction de coût technologies (-33%)
- ❌ **Siège villes** : Villes assiégées produisent toujours des revenus
- ❌ **Connexions villes** : Pas de bonus population via routes

---

## 11. Système de Villes et Population

### Polytopia Original

**Capacité d'unités :**
- Une ville peut supporter **niveau + 1 unités** (ex: niveau 2 = 3 unités max)
- Chaque unité entraînée consomme un "slot" de population
- Les unités sont associées à une ville d'origine
- Si une unité capture une ville, elle migre vers cette nouvelle ville

**Améliorations de villes :**
- Niveau 2 : Choix entre **Workshop** (+1★/tour) ou **Explorer** (unités exploratrices)
- Niveau 3 : Choix entre **City Wall** (bonus défense x4) ou **Resources** (5★)
- Niveau 4 : Choix entre **Population Growth** (+3 pop) ou **Border Growth** (expansion territoire)
- Niveau 5+ : Choix entre **Park** (250 pts) ou **Super Unit** (Giant)

**Connexions de villes :**
- Routes connectent villes pour bonus de population
- Chaque connexion ajoute +1 population à la capitale ET à la ville connectée
- Villes connectées affichent un symbole route

**Villages :**
- Villages neutres à capturer (deviennent villes niveau 1)
- Distribution aléatoire selon règles de génération
- Capture nécessite qu'une unité commence son tour sur le village
- Capture termine le tour de l'unité

**Siège :**
- Villes assiégées (unité ennemie dessus) ne produisent **aucun revenu**

**Revenus capitales :**
- Humain/Normal bot : 2★/tour (base)
- Easy bot : 1★/tour
- Hard bot : 3★/tour
- Crazy bot : 5★/tour

### Simulateur JAX

**Implémentation actuelle :**
- ✅ Niveaux de ville (1, 2, 3, 4, 5+ avec seuils croissants)
- ✅ Population stockée pour calculer niveaux
- ✅ Revenus selon niveau (2/4/6★, niveau 3+ = 6★)
- ✅ **Améliorations** : City Wall (bonus défense 4x), Park (250 pts score) implémentés
- ✅ **Connexions** : Système de connexions via routes/ponts/ports implémenté (+1 pop par connexion à capitale)
- ❌ **Pas de limite d'unités** : Pas de système de capacité par ville (niveau+1)
- ❌ **Pas d'améliorations complètes** : Pas de Workshop, Explorer, Super Unit
- ❌ **Pas de villages** : Seulement capitales au départ
- ❌ **Pas de siège** : Villes produisent toujours des revenus
- ✅ **Bonus difficulté** : Implémenté via `player_income_bonus`

**Impact :** Système de villes amélioré avec connexions et améliorations. Les connexions permettent d'optimiser la population. Il reste à implémenter la limite de capacité d'unités et les autres améliorations.

---

## 12. Actions d'Unités et Compétences

### Polytopia Original

**Actions spéciales disponibles :**
- **Recover** : Guérit 4 HP (territoire ami) ou 2 HP (territoire neutre/ennemi)
- **Disband** : Retire l'unité et rembourse 50% du coût (arrondi bas)
- **Heal Others** : Soigne toutes les unités adjacentes de 4 HP (Mind Bender)
- **Break Ice** : Brise glace adjacente (sauf Mooni/Gaami)
- **Fill** : Transforme terrain inondé en terrain normal
- **Flood Tile** : Inonde une case (Aquarion uniquement)
- **Freeze Area** : Gèle zone adjacente (Mooni/Gaami)
- **Boost** : Boost unités adjacentes (+0.5 att, +1 mov) (Shaman)
- **Explode** : Explose et empoisonne ennemis adjacents (Doomux, etc.)

**Compétences de mouvement :**
- **Dash** : Attaquer après mouvement (Warrior, Archer, Rider, etc.)
- **Escape** : Se déplacer après attaque (Rider, etc.)
- **Persist** : Attaquer à nouveau après avoir tué (Knight, Tridention)
- **Double Attack** : Attaquer deux fois par tour (Phychi)
- **Fortify** : Bonus défense en ville (Warrior, Archer, Defender, etc.)
- **Creep** : Ignore barrières terrain et zone de contrôle (Cloak, Hexapod)
- **Sneak** : Ignore barrières imposées par unités ennemies (Hexapod)
- **Scout** : Vision 5x5 au lieu de 3x3 (Scout, Battleship, etc.)
- **Fly** : Ignore toutes barrières terrain (Phychi, Dragons)
- **Carry** : Transporte unités (Raft, Boat, Ship, etc.)
- **Water** : Restreint à eau uniquement (Raft, etc.)
- **Amphibious** : Mouvement terre/eau (Tridention, etc.)
- **Skate** : Double mouvement sur glace (Mooni, Battle Sled)
- **Surprise** : Pas de contre-attaque ennemie (Dagger, Phychi)
- **Stiff** : Pas de contre-attaque quand attaqué (Bomber, Catapult)
- **Splash** : Dégâts aux unités adjacentes (Bomber, Fire Dragon)
- **Poison** : Empoisonne ennemis attaqués (Kiton, Phychi)
- **Freeze** : Gèle ennemis attaqués (Ice Archer)
- **Convert/Parasite** : Convertit ennemis en alliés (Mind Bender, Shaman)

**Unités vétérans :**
- Promotion après **3 kills** (attaque ou contre-attaque)
- +5 HP maximum et guérison complète
- Promotion peut être retardée pour utiliser comme "guérison gratuite"
- Certaines unités ne peuvent pas être promues (navales, super units, etc.)

### Simulateur JAX

**Implémentation actuelle :**
- ✅ **Action Recover** : Guérison implémentée (4 HP territoire ami, 2 HP ailleurs)
- ✅ **Promotions** : Système de vétérans implémenté (3 kills = +5 HP max, guérison complète)
- ✅ **Attaque basique** : Mouvement + attaque séparés
- ✅ **Portée** : Implémentée pour Archer (portée 2) et Catapult (portée 3)
- ❌ **Autres actions spéciales** : Pas de Disband, Heal Others, Break Ice, etc.
- ❌ **Aucune compétence** : Pas de Dash, Escape, Fortify, Creep, etc.

**Impact :** Diversité tactique améliorée avec promotions. La guérison et les promotions permettent de gérer les unités blessées et de récompenser les unités expérimentées. Il reste à implémenter les compétences spéciales pour varier davantage les stratégies.

---

## 13. Vision et Exploration

### Polytopia Original

**Système de vision :**
- **Vision standard** : 3x3 (1 case de rayon) autour de l'unité
- **Montagnes** : Vision 5x5 (2 cases de rayon)
- **Compétence Scout** : Vision 5x5 pour certaines unités
- **Brouillard de guerre** : Cases non explorées masquées par des nuages
- **Exploration** : Découvrir une case révèle le terrain et les unités
- **Points d'exploration** : 5 pts par case explorée (score)

**Brouillard :**
- Cases non explorées = nuages (impossibles à voir)
- Unités ne peuvent pas se déplacer vers cases nuageuses
- Certaines tribus spéciales peuvent voir à travers nuages (∑∫ỹriȱŋ voit ruins)

**Explorer :**
- Unité spéciale obtenue via amélioration ville niveau 2
- Explore automatiquement zones autour d'elle
- Utile pour découvrir la carte rapidement

### Simulateur JAX

**Implémentation actuelle :**
- ✅ **Brouillard de guerre** : Cases non explorées masquées, mouvement bloqué vers cases non explorées ou non adjacentes à cases explorées
- ✅ **Système de vision** : Vision par unité (3x3 standard, 5x5 sur montagnes) et par ville (3x3)
- ✅ **Exploration** : Cases explorées marquées de façon permanente (`tiles_explored`), vision mise à jour chaque tour (`tiles_visible`)
- ✅ **Points d'exploration** : 5 pts par case explorée (score)
- ❌ **Pas d'Explorer** : Pas d'unité spéciale d'exploration
- ❌ **Pas de compétence Scout** : Pas de vision étendue pour certaines unités

**Impact :** Mécanique d'exploration complète. Les joueurs doivent explorer la carte progressivement, ce qui ajoute de la stratégie de découverte. Le brouillard de guerre bloque le mouvement vers zones inconnues.

---

## 14. Ruins et Récompenses

### Polytopia Original

**Système de ruins :**
- Générées aléatoirement sur la carte (4-23 selon taille)
- Ne peuvent pas être à côté d'une capitale ou d'une autre ruin
- Peuvent être sur montagnes, forêts, plaines, ou océan profond
- Ne peuvent pas être sur eau peu profonde

**Récompenses possibles :**
- **Resources** : 10★ (toujours disponible)
- **Scrolls of Wisdom** : Technologie gratuite aléatoire (si arbre incomplet)
- **Population** : +3 population à la capitale
- **Explorer** : Unité Explorer gratuite (si zone non explorée)
- **New Friends** : Swordsman vétéran (sur terre uniquement)
- **Rammer** : Rammer vétéran (sur eau uniquement)
- **Lost City** : Ville niveau 3 avec murs (Aquarion sur océan uniquement)

**Mécanique :**
- Une unité doit **commencer son tour** sur la ruin pour l'examiner
- Examiner termine le tour de l'unité (comme capturer un village)
- Si récompense = unité, l'unité examinant est poussée (forced spawn)

### Simulateur JAX

**Implémentation actuelle :**
- ❌ **Pas de ruins** : Aucun système de récompenses aléatoires
- ❌ **Pas de récompenses** : Pas de mécanique de découverte de bonus

**Impact :** Pas d'élément de chance/exploration. Pas de stratégies basées sur la recherche de ruins.

---

## 15. Terrain Spécial et Modifications

### Polytopia Original

**Types de terrain spéciaux :**
- **Clouds** : Brouillard de guerre (cases non explorées)
- **Ice** : Eau gelée (coût mouvement réduit pour certaines unités)
- **Flooded** : Terrain inondé (Aquarion, ralentit unités navales)
- **Algae** : Bâtiment spécial créé par poison/explosion sur eau

**Modifications de terrain :**
- **Grow Forest** : Faire pousser forêt sur plaine (Spiritualism)
- **Clear Forest** : Couper forêt (Forestry)
- **Burn Forest** : Brûler forêt (Chivalry)
- **Break Ice** : Briser glace (toutes unités sauf Mooni/Gaami)
- **Flood Tile** : Inonder terrain (Aquarion)
- **Freeze Area** : Geler zone (Polaris)

**Effets terrain :**
- Montagnes : Vision +2 cases, nécessite Climbing
- Forêts : Bloquent mouvement (sauf routes ou compétence Creep)
- Eau : Restreint mouvement (sauf unités navales/amphibies)
- Glace : Coût mouvement réduit (Polaris)

### Simulateur JAX

**Implémentation actuelle :**
- ✅ **Types de base** : Plain, Forest, Mountain, Water Shallow, Water Deep
- ✅ **Ressources visuelles** : Plain_Fruit, Forest_With_Wild_Animal, Mountain_With_Mine, Water_Shallow_With_Fish
- ❌ **Pas de Clouds** : Pas de brouillard
- ❌ **Pas d'Ice** : Pas de terrain gelé
- ❌ **Pas de Flooded** : Pas de terrain inondé
- ❌ **Pas d'Algae** : Pas de bâtiment spécial
- ❌ **Pas de modifications** : Pas de Grow/Clear/Burn Forest, etc.

**Impact :** Terrain statique. Pas de mécaniques de modification de terrain pour stratégie.

---

## 16. Coût des Technologies

### Polytopia Original

**Coût dynamique :**
- Coût de base selon tier (T1: ~3-5★, T2: ~6-7★, T3: ~10+★)
- **+1★ par ville** pour T1 technologies
- **+2★ par ville** pour T2 technologies  
- **+3★ par ville** pour T3 technologies
- Exemple : T2 tech avec 5 villes = 6★ (base) + 10★ (5×2) = 16★

**Philosophy :**
- Réduit coût des recherches suivantes de **-33%**
- Utile pour accélérer progression technologique

**Dépendances :**
- Chaque tech a des prérequis spécifiques
- Arbre technologique complexe avec chemins multiples

### Simulateur JAX

**Implémentation actuelle :**
```python
# Coût dynamique : tier * num_cities + 4
# Avec Philosophy : ceil(cost * 0.67) (réduction de 33%)
def _compute_tech_cost(state, tech_id, player_id):
    tier = TECH_TIER[tech_id]
    num_cities = jnp.sum((state.city_owner == player_id) & (state.city_level > 0))
    base_cost = tier * num_cities + 4
    has_philosophy = state.player_techs[player_id, TechType.PHILOSOPHY]
    discounted_cost = jnp.ceil(base_cost * 0.67).astype(jnp.int32)
    return jnp.where(has_philosophy, discounted_cost, base_cost)
```

**Différences majeures :**
- ✅ **Coût dynamique** : Augmente avec nombre de villes (formule: tier * num_cities + 4)
- ✅ **Philosophy** : Réduction de coût de -33% implémentée
- ✅ **Dépendances** : Arbre technologique complet avec prérequis multiples

**Impact :** Technologies coûtent plus cher avec l'expansion, créant un dilemme stratégique entre expansion et recherche. Philosophy permet d'optimiser la progression technologique.

---

## 17. Disband et Gestion d'Unités

### Polytopia Original

**Disband :**
- Action disponible après recherche de **Free Spirit**
- Retire une unité et rembourse **50% du coût** (arrondi bas)
- Utile pour libérer capacité de ville ou récupérer ressources
- L'unité disparaît complètement

**Gestion d'unités :**
- Unités associées à une ville d'origine
- Migration automatique si unité capture ville/village
- Unités de ruins héritent ville d'origine de l'unité exploratrice
- Unités sans ville (compétence Independent) : Dagger, Polytaur, etc.

### Simulateur JAX

**Implémentation actuelle :**
- ❌ **Pas de Disband** : Impossible de retirer une unité volontairement
- ❌ **Pas d'association ville** : Pas de système de ville d'origine
- ✅ **Unités actives/inactives** : Système de slots avec `units_active`

**Impact :** Impossible de gérer la capacité d'unités. Pas de stratégie de libération d'espace.

---

## Conclusion

Le simulateur Polytopia-JAX est une **implémentation simplifiée mais fonctionnelle** des mécaniques de base de Polytopia. Il se concentre sur :

✅ **Points forts :**
- Moteur de jeu pur en JAX (compatible jit/vmap)
- Système de combat complet (formule complète ✅, bonus terrain ✅, guérison ✅, murs de ville 4x ✅, promotions ✅)
- Économie complète avec bâtiments avancés (Windmill, Forge, Sawmill, Market, Temples, Monuments, Parks)
- Arbre technologique étendu (21 techs Tier 1-3, coût dynamique ✅, Philosophy ✅)
- Diversité d'unités améliorée (10 types : Warrior, Defender, Archer, Rider, Raft, Knight, Swordsman, Catapult, Giant)
- Modes DOMINATION, PERFECTION, CREATIVE, GLORY, MIGHT
- Système de vision/exploration complet (brouillard de guerre ✅, exploration progressive ✅)
- Routes et ponts implémentés
- Score complet (exploration ✅, monuments ✅, temples ✅, parcs ✅, science ✅, bonus difficulté ✅)
- Connexions villes implémentées
- Architecture modulaire propre

❌ **Limitations principales :**
- Combat : Vérification vision pour contre-attaque non implémentée
- Mouvement : Zone de contrôle non implémentée, routes ne réduisent pas encore le coût de mouvement (0.5)
- Compétences spéciales : Pas de Dash, Escape, Fortify, Creep, etc.
- Actions spéciales : Recover ✅, promotions ✅, mais pas de Disband, Heal Others, etc.
- Villes : Pas de limite de capacité d'unités (niveau+1), pas de Workshop/Explorer/Super Unit
- Génération de cartes : Pas de variétés/types (Drylands/Lakes/etc.), pas de villages, pas de ruins
- Unités navales avancées : Pas de Boat, Ship, Battleship
- Unités spéciales : Pas de tribus spéciales (Cymanti, Polaris, etc.)

**Recommandations pour alignement avec Polytopia :**
1. **Combat** : ✅ Formule complète implémentée, bonus terrain implémentés, guérison implémentée, murs de ville (4x bonus) ✅, promotions ✅. Reste : vérification vision pour contre-attaque
2. **Mouvement** : ✅ Routes et ponts implémentés, ✅ brouillard de guerre implémenté. Reste : zone de contrôle, réduction coût mouvement routes (0.5)
3. **Technologies** : ✅ Arbre étendu (21 techs Tier 1-3), ✅ coût dynamique selon villes, ✅ Philosophy. Aligné avec Polytopia
4. **Unités** : ✅ Knight, Catapult, Swordsman, Giant implémentés. Reste : compétences (Dash, Escape, etc.), unités navales avancées
5. **Villes** : ✅ Connexions implémentées, ✅ City Wall et Park implémentés. Reste : capacité limite (niveau+1), améliorations (Workshop, Explorer, Super Unit)
6. **Vision** : ✅ Système d'exploration progressif implémenté, ✅ vision différente selon terrain (montagnes 5x5). Aligné avec Polytopia
7. **Ruins** : Système de récompenses aléatoires (★, tech, population, unités)
8. **Actions** : ✅ Recover implémenté, ✅ promotions implémentées. Reste : Disband, Heal Others, modifications terrain
9. **Cartes** : Variétés (Drylands/Lakes/Continents), villages, équité de répartition
10. **Score** : ✅ Système complet avec exploration, monuments, temples, parcs, science, bonus difficulté. Aligné avec Polytopia

Le simulateur est **adapté pour l'apprentissage par renforcement** avec ses mécaniques essentielles, mais reste **loin de la complexité** du jeu original Polytopia.

