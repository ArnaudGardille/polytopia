# Combat

**Source:** https://polytopia.fandom.com/wiki/Combat  
**Licence:** CC-BY-SA  
**Date de scraping:** 2025-11-24

---

**Combat** occurs when two enemy [units](</wiki/Units> "Units") attack each other.   
  
Each unit has two damage stats: attack, which determines how much damage it deals when it attacks, and defence, which determines how much retaliation damage it deals. (The actual amount of damage dealt is usually much greater than the attack or defence stat; for example, a full-health [Warrior](</wiki/Warrior> "Warrior") can deal 5 or more damage even though its attack and defence stats are both 2.) The defence stat also affects how much damage the defending unit takes; for example, a [Rider](</wiki/Rider> "Rider"), with a defence stat of 1, would take much more damage from the same attack than a [Defender](</wiki/Defender> "Defender"), with a defence stat of 3. Note that the attacker’s defence stat and the defending unit’s attack stat are irrelevant. 

Damaged units are weaker in all regards. They do less offensive and defensive damage, and they receive more damage when attacked. 

If a melee unit (like a [Warrior](</wiki/Warrior> "Warrior")) kills an adjacent enemy unit, it will take the place of the killed unit, (barring movement restrictions imposed by terrain). Ranged units (like an [Archer](</wiki/Archer> "Archer")) will not do so, even when killing at melee range. This movement can move units into a friendly port and transform them. 

## Contents

  * 1 Battle Preview
  * 2 Defence Bonus
  * 3 Healing
  * 4 Damage Formula
  * 5 Vision
  * 6 Boost
  * 7 Poison
  * 8 Resources
  * 9 See also

## Battle Preview[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D1&uselang=en> "Sign in to edit")]

[![Battle Preview Example](images/Battle_Preview_Example.webp)](<https://static.wikia.nocookie.net/supertribes/images/a/a3/Battle_Preview_Example.png/revision/latest?cb=20210610004017>) [](</wiki/File:Battle_Preview_Example.png>)

An example of the battle preview

The battle preview shows the result of an attack. To see how much damage will be dealt to either unit, hold on (mobile) or hover over (Steam) an enemy unit within range. Sweating indicates that the enemy unit will be killed by the attack, while a black and red ring indicates that the attacking unit will die from retaliation. 

## Defence Bonus[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D2&uselang=en> "Sign in to edit")]

Certain tiles provide a defence bonus to units on them. Tiles of certain [terrain](</wiki/Terrain> "Terrain") types provide a defence bonus once the corresponding [technology](</wiki/Technology> "Technology") is researched, and [cities](</wiki/Cities> "Cities") without a city wall (not including uncaptured villages) provide a defence bonus to friendly units (invading units besieging a city do not receive any defence bonus). Units with a defence bonus receive less damage and deal more retaliation damage. A single shield around the unit's HP indicates that the unit has a defence bonus. 

The [city wall](</wiki/City_wall> "City wall") provides a defence bonus much stronger than the standard defence bonus. A double shield around a unit's HP indicates that the unit is standing in a city with a city wall. 

Cities will only provide a defence bonus (wall or no wall) to units with the fortify [skill](</wiki/Unit_Skills> "Unit Skills"). For example, a [Giant](</wiki/Giant> "Giant") will never receive a defence bonus in a city because it does not have the fortify skill, although it may receive a defence bonus on other tiles. 

Icon  | Terrain  | Unlocked By   
---|---|---  
[![Forest defense](images/Forest_defense.webp)](<https://static.wikia.nocookie.net/supertribes/images/a/a1/Forest_defense.png/revision/latest?cb=20180607125544>) | Forest  | [Archery](</wiki/Archery> "Archery")  
[![Mountain defense](images/Mountain_defense.webp)](<https://static.wikia.nocookie.net/supertribes/images/0/02/Mountain_defense.png/revision/latest?cb=20180605114023>) | Mountain  | [Climbing](</wiki/Climbing> "Climbing")  
[![Shallow water defense](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](<https://static.wikia.nocookie.net/supertribes/images/7/76/Shallow_water_defense.png/revision/latest?cb=20180613230353>) [![Shallow water defense](images/Shallow_water_defense.webp)](<https://static.wikia.nocookie.net/supertribes/images/7/76/Shallow_water_defense.png/revision/latest?cb=20180613230353>) | Shallow water  | [Aquatism](</wiki/Aquatism> "Aquatism")  
[![Deep water defense](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](<https://static.wikia.nocookie.net/supertribes/images/c/c7/Deep_water_defense.png/revision/latest?cb=20180613230403>) [![Deep water defense](images/Deep_water_defense.webp)](<https://static.wikia.nocookie.net/supertribes/images/c/c7/Deep_water_defense.png/revision/latest?cb=20180613230403>) | Ocean  | Aquatism   
  
## Healing[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D3&uselang=en> "Sign in to edit")]

A damaged unit can recover (heal) instead of moving and/or attacking on any given turn. Healing restores up to 4 HP in friendly territory and 2 HP in neutral or enemy territory. (A unit cannot have more than its maximum HP.) 

The [Mind Bender](</wiki/Mind_Bender> "Mind Bender") can heal all adjacent friendly units by 4 HP instead of moving, attacking, and/or healing itself on any given turn (if there is a damaged unit nearby). 

## Damage Formula[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D4&uselang=en> "Sign in to edit")]

Damage is calculated as follows. _attackResult_ is the amount of damage dealt by the attacker, while _defenceResult_ is the amount of retaliation damage dealt by the defender. _defenceBonus_ is 1 when there is no defence bonus, 1.5 for the standard forest/mountain/ocean/city bonus, 4 for the city wall bonus, and overridden to 0.7 whenever a unit is poisoned. Note that _defenceBonus_ is applied to _defenceForce_ , not _defenceResult._ Rounding is done to the nearest whole number. 
    
    
    attackForce = attacker.attack * (attacker.health / attacker.maxHealth)
    defenseForce = defender.defense * (defender.health / defender.maxHealth) * defenseBonus 
    totalDamage = attackForce + defenseForce 
    attackResult = round((attackForce / totalDamage) * attacker.attack * 4.5) 
    defenseResult = round((defenseForce / totalDamage) * defender.defense * 4.5)
    

  
Splash damage and explosion damage are calculated (individually) by taking the _attackResult_ , and then dividing it by 2 _without rounding_. 
    
    
    attackForce = attacker.attack * (attacker.health / attacker.maxHealth)
    defenseForce = defender.defense * (defender.health / defender.maxHealth) * defenseBonus 
    totalDamage = attackForce + defenseForce 
    attackResult = round((attackForce / totalDamage) * attacker.attack * 4.5) 
    attackSplash = attackResult / 2
    

  
When _attackResult_ is an odd number, this will cause the received splash damage to be a .5 decimal, which in turn will make the receiving unit's hp a .5 decimal. This is presumed to be a bug, in particular because it invokes another bug: the _n_.5 hp is rounded up on the map display and rounded down in the unit info box, enabling units to seem to have 0 hp on the info box. 

## Vision[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D5&uselang=en> "Sign in to edit")]

No retaliation occurs if the attacker kills the unit being attacked or the unit being attacked cannot see the attacker (the attacker is hidden by [fog](</wiki/Terrain> "Terrain")). 

Mountains provide extra sight (a two tile radius instead of the normal one). 

Units with the scout skill can see two tiles in every direction. 

## Boost[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D6&uselang=en> "Sign in to edit")]

[![Boost](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](<https://static.wikia.nocookie.net/supertribes/images/c/cd/Boost.png/revision/latest?cb=20210301203127>) [![Boost](images/Boost.webp)](<https://static.wikia.nocookie.net/supertribes/images/c/cd/Boost.png/revision/latest?cb=20210301203127>) [](</wiki/File:Boost.png>)

The Boost icon

The [Shaman](</wiki/Shaman> "Shaman"), a unit unique to the [Cymanti](</wiki/Cymanti> "Cymanti") [tribe](</wiki/Tribe> "Tribe"), can boost friendly units. Boosted units get +0.5 attack and +1 movement. This effect lasts until the boosted unit attacks another unit, is attacked, uses most abilities, examines a ruin, captures a village/city, or is poisoned. 

## Poison[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D7&uselang=en> "Sign in to edit")]

[![Xin-Xi Warrior - Poisoned](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](<https://static.wikia.nocookie.net/supertribes/images/d/da/Xin-Xi_Warrior_-_Poisoned.png/revision/latest?cb=20210305235545>) [![Xin-Xi Warrior - Poisoned](images/Xin-Xi_Warrior_-_Poisoned.webp)](<https://static.wikia.nocookie.net/supertribes/images/d/da/Xin-Xi_Warrior_-_Poisoned.png/revision/latest?cb=20210305235545>) [](</wiki/File:Xin-Xi_Warrior_-_Poisoned.png>)

A poisoned Xin-xi Warrior

Poison is applied by Cymanti [units](</wiki/Units> "Units") and [buildings](</wiki/Building> "Building"). It reduces defence by 30%, prevents the unit from being healed, prevents the unit from receiving any defence bonus, and causes the unit to drop spores (on land) or [Algae](</wiki/Algae> "Algae") (in water) upon death. Poison can be removed by healing once. This can be through self-healing, a [Mind Bender](</wiki/Mind_Bender> "Mind Bender"), or a [Mycelium](</wiki/Mycelium> "Mycelium"). When healed in this way, the poison is removed but the unit does not get any health back. 

## Resources[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D8&uselang=en> "Sign in to edit")]

  * [Polytopia Battle Calculator](<https://polytopia-damage-calculator.firebaseapp.com/beta>)
  * Drag and drop [Polytopia Calculator](<https://polytopiacalculator.com>)

## See also[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FCombat%3Fveaction%3Dedit%26section%3D9&uselang=en> "Sign in to edit")]

  * [Unit skills](</wiki/Unit_Skills> "Unit Skills"), which influence the behavior of units in battle

  
---  
Game Mechanics  
Game|  [The Battle of Polytopia: Moonrise](</wiki/The_Battle_of_Polytopia:_Moonrise> "The Battle of Polytopia: Moonrise"), [Update History](</wiki/Update_History> "Update History")  
General|  **Combat** , [Easter Eggs](</wiki/Easter_Eggs> "Easter Eggs"), [Game Modes](</wiki/Game_Modes> "Game Modes"), [Map Generation](</wiki/Map_Generation> "Map Generation"), [Movement](</wiki/Movement> "Movement"), [Ruin](</wiki/Ruin> "Ruin"), [Score](</wiki/Score> "Score"), [Star](</wiki/Star> "Star"), [Technology](</wiki/Technology> "Technology"), [Terrain](</wiki/Terrain> "Terrain"), [Tribes](</wiki/Tribes> "Tribes")  
[Abilities](</wiki/Abilities> "Abilities")|  [Burn Forest](</wiki/Burn_Forest> "Burn Forest"), [Clear Forest](</wiki/Clear_Forest> "Clear Forest"), [Destroy](</wiki/Destroy> "Destroy"), [Grow Forest](</wiki/Grow_Forest> "Grow Forest"), [Starfish Harvesting](</wiki/Starfish_Harvesting> "Starfish Harvesting"), [Whale Hunting](</wiki/Whale_Hunting> "Whale Hunting")  
[Buildings](</wiki/Buildings> "Buildings")|  [Buildings](</wiki/Buildings> "Buildings"), [Bridge](</wiki/Bridge> "Bridge"), [Embassy](</wiki/Embassy> "Embassy"), [Roads](</wiki/Roads_\(Building\)> "Roads \(Building\)"), [Temples](</wiki/Temples> "Temples")  
[City](</wiki/City> "City")|  [City](</wiki/City> "City"), [City Connection](</wiki/City_Connection> "City Connection"), [City Upgrades](</wiki/Category:City_Upgrades> "Category:City Upgrades"), [Population](</wiki/Population> "Population")  
[Diplomacy](</wiki/Category:Diplomacy> "Category:Diplomacy")|  [Diplomacy](</wiki/Diplomacy> "Diplomacy"), [Embassy](</wiki/Embassy> "Embassy"), [Peace Treaty](</wiki/Peace_Treaty> "Peace Treaty"), [Tribe Relations](</wiki/Tribe_Relations> "Tribe Relations")  
[Units](</wiki/Units> "Units")|  [Units](</wiki/Units> "Units"), [List of Units](</wiki/List_of_Units> "List of Units"), [Unit Skills](</wiki/Unit_Skills> "Unit Skills"), [Super Unit](</wiki/Super_Unit> "Super Unit"), [Disband](</wiki/Disband> "Disband")


---

*Ce contenu est extrait du wiki Polytopia et est sous licence CC-BY-SA.*
