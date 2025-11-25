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
- ✅ **Guérison** : Action RECOVER implémentée (4 HP territoire ami, 2 HP ailleurs)
- ✅ **Pas de contre-attaque si tué** : Si la cible est tuée par l'attaque, pas de riposte
- ✅ **Prise de place** : Implémentée pour unités mêlée tuant une cible
- ❌ **Murs de ville** : Bonus 4x non implémenté (retourne 1.5x comme ville normale)
- ❌ **Vérification vision/brouillard** : Contre-attaque ne vérifie pas si la cible peut voir l'attaquant
- ❌ **Poison/Boost** : Mécaniques spéciales tribus non implémentées

**Impact :** Le système de combat est maintenant beaucoup plus fidèle à Polytopia. Les unités blessées sont pénalisées (moins de dégâts infligés, plus de dégâts reçus), et le terrain a un impact défensif significatif. La guérison permet de gérer les unités blessées. Il reste à implémenter les murs de ville et la vérification de vision pour la contre-attaque.

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
- ❌ **Pas de routes** : Aucun système de routes/ponts
- ❌ **Pas de zone de contrôle** : Les unités peuvent passer librement à côté d'ennemis
- ❌ **Pas de brouillard de guerre** : Toute la carte est visible
- ❌ **Pas de compétences spéciales** de mouvement (Dash, Creep, etc.)

**Impact :** Le mouvement est beaucoup plus simple et moins stratégique. Pas de gestion de routes pour optimiser la mobilité, et pas de mécanisme de zone de contrôle pour bloquer les mouvements ennemis.

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
    CLIMBING = 1      # Coût 3★
    SAILING = 2       # Coût 4★ (dépend de CLIMBING)
    MINING = 3        # Coût 3★
    NUM_TECHS = 4
```

**Différences majeures :**
- ❌ **Seulement 3 technologies** vs ~30+ dans Polytopia
- ❌ **Coût fixe** : Pas d'augmentation avec nombre de villes
- ❌ **Pas de Philosophy** : Pas de réduction de coût
- ❌ **Dépendances minimales** : Seulement SAILING → CLIMBING
- ✅ **Débloque unités/bâtiments** : CLIMBING (montagnes), SAILING (ports/radeaux), MINING (mines)

**Impact :** L'arbre technologique est extrêmement simplifié. Pas de progression technologique complexe, pas de stratégies basées sur l'ordre de recherche des techs.

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
UNIT_HP_MAX = [0, 10, 15, 8, 10, 8]      # NONE, WARRIOR, DEFENDER, ARCHER, RIDER, RAFT
UNIT_ATTACK = [0, 2, 1, 2, 3, 2]
UNIT_DEFENSE = [0, 2, 3, 1, 1, 1]
UNIT_MOVEMENT = [0, 1, 1, 1, 2, 2]
UNIT_ATTACK_RANGE = [0, 1, 1, 2, 1, 1]
```

**Différences majeures :**
- ✅ **5 types d'unités** : Warrior, Defender, Archer, Rider, Raft
- ❌ **Pas de Knight, Swordsman, Catapult, Giant**
- ❌ **Pas d'unités navales avancées** (Boat, Ship, Battleship)
- ❌ **Pas d'unités spéciales** par tribu
- ❌ **Pas de compétences spéciales** (Dash, Escape, Fortify, etc.)
- ✅ **Portée d'attaque** : Implémentée pour Archer (portée 2)

**Impact :** Diversité tactique très limitée. Pas de stratégies basées sur différentes compositions d'armées.

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
BUILDING_COST = [0, 3, 4, 2, 5]  # NONE, FARM, MINE, HUT, PORT
BUILDING_POP_GAIN = [0, 1, 2, 1, 0]
RESOURCE_COST = [0, 2, 2, 4]     # NONE, FRUIT, FISH, ORE
RESOURCE_POP_GAIN = [0, 1, 1, 2]
CITY_STAR_INCOME_PER_LEVEL = [0, 2, 4, 6]
```

**Différences majeures :**
- ✅ **Bâtiments de base** : Farm, Mine, Hut, Port implémentés
- ✅ **Ressources** : Fruit, Fish, Ore implémentées
- ✅ **Revenus villes** : 2/4/6★ selon niveau
- ❌ **Pas de bâtiments avancés** : Windmill, Forge, Sawmill, Market, Temples, Monuments
- ❌ **Pas de City Wall**
- ❌ **Pas de Park**
- ✅ **Bonus difficulté** : Implémenté via `player_income_bonus`

**Impact :** Économie simplifiée mais fonctionnelle. Pas de stratégies basées sur optimisation économique avancée (marchés, temples pour score, etc.).

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
```

**Différences majeures :**
- ✅ **Deux modes** : DOMINATION et PERFECTION
- ✅ **DOMINATION** : Élimination (vérifie si joueur a encore capitale)
- ✅ **PERFECTION** : Limite de tours (30 par défaut)
- ❌ **Pas de Creative** : Pas de personnalisation
- ❌ **Pas de multijoueur** : Pas de modes Glory/Might
- ❌ **Score Perfection simplifié** : Calcul basique (territoire, population, militaire, ressources)
- ❌ **Pas de bonus difficulté** : Pas de calcul de bonus selon adversaires/difficulté

**Impact :** Modes de jeu fonctionnels mais simplifiés. Pas de variété de gameplay, pas de système de scoring avancé.

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
TERRITORY_POINTS = 100      # Par case avec ville
POPULATION_POINTS = 5      # Par point de population
MILITARY_POINTS = 20       # Par unité active
RESOURCE_POINTS = 2        # Par étoile possédée
```

**Différences majeures :**
- ✅ **Composantes de base** : Territoire, Population, Militaire, Ressources
- ❌ **Pas d'exploration** : Pas de points pour cases explorées
- ❌ **Pas de monuments/temples** : Pas de bâtiments spéciaux
- ❌ **Pas de science** : Pas de points pour technologies
- ❌ **Calcul simplifié** : Pas de bonus difficulté, pas de multiplicateurs complexes
- ❌ **Unités** : Points fixes par unité (20), pas selon coût

**Impact :** Score fonctionnel mais très simplifié. Pas de stratégies basées sur optimisation du score (temples, monuments, exploration).

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
- ✅ Niveaux de ville (1, 2, 3)
- ✅ Population stockée pour calculer niveaux
- ✅ Revenus selon niveau (2/4/6★)
- ❌ **Pas de limite d'unités** : Pas de système de capacité par ville
- ❌ **Pas d'améliorations** : Pas de Workshop, Explorer, City Wall, Park, Super Unit
- ❌ **Pas de connexions** : Pas de système de routes entre villes
- ❌ **Pas de villages** : Seulement capitales au départ
- ❌ **Pas de siège** : Villes produisent toujours des revenus
- ✅ **Bonus difficulté** : Implémenté via `player_income_bonus`

**Impact :** Système de villes très simplifié. Pas de gestion de capacité d'unités, pas de choix stratégiques d'améliorations, pas de mécaniques de connexion.

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
- ❌ **Autres actions spéciales** : Pas de Disband, Heal Others, Break Ice, etc.
- ❌ **Aucune compétence** : Pas de Dash, Escape, Fortify, Creep, etc.
- ❌ **Pas de promotions** : Pas de système de vétérans
- ✅ **Attaque basique** : Mouvement + attaque séparés
- ✅ **Portée** : Implémentée pour Archer (portée 2)

**Impact :** Diversité tactique limitée. La guérison permet de gérer les unités blessées, mais pas de compétences spéciales pour varier les stratégies.

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
- ❌ **Pas de brouillard** : Toute la carte est visible dès le début
- ❌ **Pas de système de vision** : Pas de limitation de portée de vision
- ❌ **Pas d'exploration** : Pas de mécanique de découverte progressive
- ❌ **Pas d'Explorer** : Pas d'unité spéciale d'exploration
- ✅ **Visibilité complète** : Tous les joueurs voient toute la carte

**Impact :** Pas de mécanique d'exploration. Les joueurs connaissent immédiatement toute la carte, ce qui élimine la stratégie de découverte.

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
TECH_COST = [0, 3, 4, 3]  # NONE, CLIMBING, SAILING, MINING
```

**Différences majeures :**
- ❌ **Coût fixe** : Pas d'augmentation avec nombre de villes
- ❌ **Pas de Philosophy** : Pas de réduction de coût
- ✅ **Dépendances** : SAILING nécessite CLIMBING

**Impact :** Technologies toujours au même prix, pas de pénalité d'expansion. Pas de stratégie basée sur timing de recherche vs expansion.

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
- Système de combat basique fonctionnel
- Économie simplifiée mais cohérente
- Modes DOMINATION et PERFECTION
- Architecture modulaire propre

❌ **Limitations principales :**
- Combat amélioré mais incomplet (formule complète ✅, bonus terrain ✅, guérison ✅, mais pas de murs de ville 4x, pas de vérification vision pour contre-attaque)
- Mouvement simplifié (pas de routes, zone de contrôle, brouillard, compétences spéciales)
- Arbre technologique minimaliste (3 techs vs 30+, coût fixe vs dynamique)
- Diversité d'unités limitée (5 types vs 20+, pas de compétences, pas de promotions)
- Génération de cartes basique (pas de variétés/types, pas de villages, pas de ruins)
- Score simplifié (pas de monuments, temples, exploration)
- Villes simplifiées (pas de capacité limite, pas d'améliorations, pas de connexions)
- Pas de vision/exploration (carte entièrement visible dès le début)
- Actions spéciales limitées (Recover ✅, mais pas de Disband, Heal Others, etc.)

**Recommandations pour alignement avec Polytopia :**
1. **Combat** : ✅ Formule complète implémentée, bonus terrain implémentés, guérison implémentée. Reste : murs de ville (4x bonus), vérification vision pour contre-attaque
2. **Mouvement** : Ajouter routes, zone de contrôle, brouillard de guerre
3. **Technologies** : Étendre arbre (tier 1-2 complets), coût dynamique selon villes
4. **Unités** : Ajouter Knight, Catapult, Swordsman, compétences (Dash, Escape, etc.)
5. **Villes** : Capacité limite (niveau+1), améliorations (Workshop, City Wall, Park), connexions
6. **Vision** : Système d'exploration progressif, vision différente selon terrain
7. **Ruins** : Système de récompenses aléatoires (★, tech, population, unités)
8. **Actions** : Recover ✅, reste Disband, Heal Others, modifications terrain
9. **Cartes** : Variétés (Drylands/Lakes/Continents), villages, équité de répartition
10. **Promotions** : Système de vétérans (3 kills = +5 HP)

Le simulateur est **adapté pour l'apprentissage par renforcement** avec ses mécaniques essentielles, mais reste **loin de la complexité** du jeu original Polytopia.

