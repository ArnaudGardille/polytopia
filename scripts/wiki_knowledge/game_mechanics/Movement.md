# Movement

**Source:** https://polytopia.fandom.com/wiki/Movement  
**Licence:** CC-BY-SA  
**Date de scraping:** 2025-11-24

---

The **movement** of a [unit](</wiki/Unit> "Unit") is determined by its movement [stat](</wiki/Unit_Stats> "Unit Stats"), its [skills](</wiki/Unit_Skills> "Unit Skills"), and the [terrain](<Terrain.md> "Terrain") it moves across. If not impacted by any other factors, a unit can move to tiles whose distance to the unit's starting tile is not more than its movement stat, as measured using [Chebyshev distance](<https://en.wikipedia.org/wiki/Chebyshev_distance> "wikipedia:Chebyshev distance"). However, this is affected by the terrain the unit move across, the presence of [roads](</wiki/Roads_\(Building\)> "Roads \(Building\)"), as well as some other factors. A unit may only move once per turn (besides the use of the dash and escape skills), and each tile can only be occupied by one unit at a time. A forced unit spawn occurs when an unit is spawned on a tile that already has an unit on it. 

A unit's movement stat determines how far it can move every turn. Moving from one tile to an adjacent tile incurs a cost of 1, but moving between tiles with roads has a cost of 0.5. 

## Contents

  * 1 Terrain
  * 2 Zone of Control
  * 3 Roads
  * 4 Glide
  * 5 Bubbled
  * 6 Forced Unit Spawns
  * 7 Unit Skills
  * 8 Examples
    * 8.1 Terrain and Zone of Control
    * 8.2 Roads
    * 8.3 Polarism (outdated)
  * 9 References

## Terrain[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D1&uselang=en> "Sign in to edit")]

Certain types of [terrain](<Terrain.md> "Terrain") impact movement. Rough terrain, including mountains and forests (without roads) block movement. Units cannot move through them (they can move onto them, but no further in the same turn). Also, units cannot move into clouds (the fog of war). 

Water and amphibious units are capable of moving through water and (if the [appropriate technology](</wiki/Sailing> "Sailing") has been researched) ocean tiles. Water units can also move through flooded tiles and amphibians can also move through land tiles, but these will respectively slow them, as if moving through rough terrain. 

Most land units moving into a [Port](</wiki/Port> "Port") will be turned into [Rafts](</wiki/Raft> "Raft") and will be unable to perform actions for the rest of that turn, even if they had excess movement points or had not attacked. However, this does not apply to water, amphibious or flying units. 

## Zone of Control[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D2&uselang=en> "Sign in to edit")]

All units exert a passive Zone of Control onto enemy units. This takes priority over movement bonuses conferred by [Roads](</wiki/Roads> "Roads") or [Polarism](</wiki/Polarism> "Polarism") (see #Glide). A unit that moves to a tile adjacent to an enemy unit cannot move any further, even if the unit had excess movement points. 

The exception to this mechanic is units that have the [Creep skill](</wiki/Unit_Skills#Creep> "Unit Skills"). Units with the creep skill completely bypass zone of control restrictions for unit movement. 

This also does not prevent units with the [Dash skill](</wiki/Unit_Skills#Dash> "Unit Skills") from attacking after moving, and the Zone of Control goes away when the enemy unit is destroyed. 

A unit that starts its turn within Zone of Control may be able to move out and then back into it, occasionally leading to seemingly impossible but valid movement destinations. 

## Roads[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D3&uselang=en> "Sign in to edit")]

Movement between two tiles with [Roads](</wiki/Roads> "Roads") (or a Road and a [city](</wiki/Cities> "Cities")) requires only 0.5 moves, ignoring the movement penalty from Forests, but not Zone of Control (it is impossible to place Roads on Mountains). This only applies to Roads in friendly territory or in neutral territory, but not when moving onto Roads in enemy territory. [Bridges](</wiki/Bridge> "Bridge") follow these same rules, but with the added benefit of allowing land units to cross one tile of water. 

In order to utilize bonus movement from roads, you need to place a road where your troop is standing and then on the following tile. City tile works as a road tile. Roads can be placed on villages, but they disappear when the village is claimed. 

When calculating movement cost of a tile, the game rounds up - meaning even if you have 0.5 movement points remaining, you can still move onto a tile costing 1 or more movement points. The place the unit ends up is irrelevant - you don't need a road at the final tile and movement is not blocked if final tile is an obstacle. 

## Glide[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D4&uselang=en> "Sign in to edit")]

As [Polaris](<../tribes/Polaris.md> "Polaris"), researching the unique [Polarism](</wiki/Polarism> "Polarism") technology unlocks the Glide skill, which halves the movement cost of moving into ice tiles, functioning more or less as if every ice tile had a [Road](</wiki/Roads_\(Building\)> "Roads \(Building\)") on it. However, moving from an ice tile to a land tile with a Road still costs a full movement point. 

## Bubbled[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D5&uselang=en> "Sign in to edit")]

As [Aquarion](<../tribes/Aquarion.md> "Aquarion"), researching the unique [Waterways](</wiki/Roads_\(Technology\)> "Roads \(Technology\)") technology unlocks the Bubbled skill. This gives any Aquarion unit a speed bubble whenever they movement onto a flooded tile, which increases their movement by 1 until they are attacked or move onto non-flooded land. 

## Forced Unit Spawns[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D6&uselang=en> "Sign in to edit")]

A forced unit spawn occurs when an [unit](</wiki/Unit> "Unit") is spawned in a tile that already has a unit on it. The unit that is already on the tile is pushed in a specific direction. Forced unit spawns can happen when a [super unit](</wiki/Super_unit> "Super unit") spawns in a city, a [Ruin](</wiki/Ruins> "Ruins") spawns a unit, or [∑∫ỹriȱŋ](<../general/∑∫ỹriȱŋ.md> "∑∫ỹriȱŋ") spawns a [Polytaur](</wiki/Polytaur> "Polytaur"). The forced spawn of super units are commonly used in besieged cities to push out the invader, and the forced spawn of Polytaurs are commonly used to "boost" the movement of friendly units. 

The push order for forced unit spawns determines the direction in which the unit already on the tile where the new unit is spawned will be pushed. The push order is as follows: 

  * Friendly units that previously moved will be pushed in the same direction of their movement. 
    * Example: A friendly unit that previously moved north will be pushed north.
  * Enemy units that previously moved will be pushed in the opposite direction of their movement. 
    * Example: An enemy unit that previously moved north will be pushed south.
  * Ranged units get pushed in the direction of their last move or last attack.
  * Units that were not previously moved will be pushed toward the center of the map.
  * If the city where the unit spawns is on the exact center of the map, the unit will be pushed south.
  * If the tile where the unit is supposed to go is occupied or impassable, it will try counterclockwise and then clockwise one tile at a time until it finds a free tile, or if there is no free tile, it will be removed from the game entirely. This does not count as a kill for the [Gate of Power](</wiki/Gate_of_Power> "Gate of Power"), nor does it count as an attack for the [Altar of Peace](</wiki/Altar_of_Peace> "Altar of Peace").

## Unit Skills[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D7&uselang=en> "Sign in to edit")]

    _This section needs to be updated._

The unit skills listed below impact movement in some way. (Units without any of these skills are called “land units”.) 

  * Carry ([Raft](</wiki/Raft> "Raft"), [Scout](</wiki/Scout> "Scout"), [Bomber](</wiki/Bomber> "Bomber"), [Rammer](</wiki/Rammer> "Rammer"), [Juggernaut](</wiki/Juggernaut> "Juggernaut"), [Dinghy](</wiki/Dinghy> "Dinghy") and [Pirate](</wiki/Pirate> "Pirate")): The carry skill allows a unit to carry another unit inside and travel on water. (Navigation or Free Diving must be researched to travel on ocean tiles.) A unit with the carry skill can move to land tiles adjacent to water, but doing so turns it into the unit it was carrying and ends the unit's turn.
  * Fly ([Phychi](</wiki/Phychi> "Phychi"), [Baby](</wiki/Baby_Dragon> "Baby Dragon") and [Fire Dragon](</wiki/Fire_Dragon> "Fire Dragon")): The fly skill allows a unit to ignore all movement barriers imposed by terrain. However, units with the fly skill cannot utilize roads.
  * Navigate ([Raychi](</wiki/Raychi> "Raychi")): The navigate skill allows a unit to move across shallow water and ocean even if the prerequisite technologies are not researched. Units with the navigate skill cannot move to land tiles, except if they can capture cities
  * Skate ([Mooni](</wiki/Mooni> "Mooni"), [Battle Sled](</wiki/Battle_Sled> "Battle Sled"), [Ice Fortress](</wiki/Ice_Fortress> "Ice Fortress")): The skate skill doubles movement on ice but on land limits movement to one and bars units with it from using the dash and escape skills.
  * Creep([Cloak](</wiki/Cloak> "Cloak"), [Hexapod](</wiki/Hexapod> "Hexapod"), [Doomux](</wiki/Doomux> "Doomux"), [Raychi](</wiki/Raychi> "Raychi"), and [Centipede](</wiki/Centipede> "Centipede")): The creep skill allows a unit to ignore movement penalties due to mountains and forests , but disabling bonuses from roads. Movement restrictions from zone of control are also eliminated.
  * Sneak ([Hexapod](</wiki/Hexapod> "Hexapod"), [Cloak](</wiki/Cloak> "Cloak") ): The sneak skill allows a unit to bypass barriers presented by tiles adjacent to enemies at no cost.
  * Boost ([Shaman](</wiki/Shaman> "Shaman")): The boost skill grants a unit the Boost [unit ability](</wiki/Unit_ability> "Unit ability"), which increases the movement of all adjacent nearby units by 1 until their next action (excluding moving).

## Examples[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D8&uselang=en> "Sign in to edit")]

### Terrain and Zone of Control[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D9&uselang=en> "Sign in to edit")]

[![The Rider in above picture can't reach marked tiles because of an enemy archer.](images/The_Rider_in_above_picture_can't_reach_marked_tiles_because_of_an_enemy_archer..webp)![The Rider in above picture can't reach marked tiles because of an enemy archer.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement01.JPG> "Movement01.JPG \(38 KB\)")

The [Rider](</wiki/Rider> "Rider") in above picture can't reach marked tiles because of an enemy [archer](</wiki/Archer> "Archer").

[![Here, the rider can’t reach the marked tiles because of forests.](images/Here,_the_rider_can’t_reach_the_marked_tiles_because_of_forests..webp)![Here, the rider can’t reach the marked tiles because of forests.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement02.JPG> "Movement02.JPG \(36 KB\)")

Here, the rider can’t reach the marked tiles because of forests.

[![An enemy hidden by the fog is blocking movement.](images/An_enemy_hidden_by_the_fog_is_blocking_movement..webp)![An enemy hidden by the fog is blocking movement.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Unit_in_Fog.png> "Unit in Fog.png \(148 KB\)")

An enemy hidden by the fog is blocking movement.

[![This Cloak can travel freely past the defending enemy Warrior because of its Creep ability, ignoring the warrior's zone of control.](images/This_Cloak_can_travel_freely_past_the_defending_enemy_Warrior_because_of_its_Creep_ability,_ignoring.webp)![This Cloak can travel freely past the defending enemy Warrior because of its Creep ability, ignoring the warrior's zone of control.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Cloak-Bypassing-Enemy-ZOC2.png> "Cloak-Bypassing-Enemy-ZOC2.png \(2.19 MB\)")

This [Cloak](</wiki/Cloak> "Cloak") can travel freely past the defending enemy [Warrior](</wiki/Warrior> "Warrior") because of its [Creep ability](</wiki/Units> "Units"), ignoring the warrior's zone of control.

### Roads[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D10&uselang=en> "Sign in to edit")]

[![Rider can reach mountain here.](images/Rider_can_reach_mountain_here..webp)![Rider can reach mountain here.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement03.JPG> "Movement03.JPG \(47 KB\)")

Rider can reach mountain here.

[![Same goes for starting point - if you're standing on Forest or Mountain tiles, you’ll still be able to move 2 tiles with your rider and 3 with a knight. However, you won’t be able to reach maximum movement of 4/6 since you don’t start on a road.](images/Same_goes_for_starting_point_-_if_you're_standing_on_Forest_or_Mountain_tiles,_you’ll_still_be_able_.webp)![Same goes for starting point - if you're standing on Forest or Mountain tiles, you’ll still be able to move 2 tiles with your rider and 3 with a knight. However, you won’t be able to reach maximum movement of 4/6 since you don’t start on a road.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement04.JPG> "Movement04.JPG \(42 KB\)")

Same goes for starting point - if you're standing on Forest or Mountain tiles, you’ll still be able to move 2 tiles with your rider and 3 with a [knight](</wiki/Knight> "Knight"). However, you won’t be able to reach maximum movement of 4/6 since you don’t start on a road.

[![Road on Forest tiles will neutralize movement restriction. However, Roads can't be placed on mountains.](images/Road_on_Forest_tiles_will_neutralize_movement_restriction._However,_Roads_can't_be_placed_on_mountai.webp)![Road on Forest tiles will neutralize movement restriction. However, Roads can't be placed on mountains.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement05.JPG> "Movement05.JPG \(67 KB\)")

Road on Forest tiles will neutralize movement restriction. However, Roads can't be placed on mountains.

### Polarism (outdated)[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D11&uselang=en> "Sign in to edit")]

[![This Catapult only has 1 movement point, but due to Polarism gains an extra move when moving NE, resulting in it being able to cross 2 tiles. Alternatively if moving SW, the Road allows the catapult to move 2 tiles onto the Ice. However, since Polarism is unlocked, the Catapult then gains another movement point and is allowed to move an additional tile, resulting in a total displacement of 3 tiles - which is impossible to accomplish with any other tribe using direct movement.](images/This_Catapult_only_has_1_movement_point,_but_due_to_Polarism_gains_an_extra_move_when_moving_NE,_res.webp)![This Catapult only has 1 movement point, but due to Polarism gains an extra move when moving NE, resulting in it being able to cross 2 tiles. Alternatively if moving SW, the Road allows the catapult to move 2 tiles onto the Ice. However, since Polarism is unlocked, the Catapult then gains another movement point and is allowed to move an additional tile, resulting in a total displacement of 3 tiles - which is impossible to accomplish with any other tribe using direct movement.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement14.JPG> "Movement14.JPG \(57 KB\)")

This [Catapult](</wiki/Catapult> "Catapult") only has 1 movement point, but due to Polarism gains an extra move when moving NE, resulting in it being able to cross 2 tiles. Alternatively if moving SW, the Road allows the catapult to move 2 tiles onto the Ice. However, since Polarism is unlocked, the Catapult then gains another movement point and is allowed to move an additional tile, resulting in a total displacement of 3 tiles - which is impossible to accomplish with any other tribe using direct movement.

[![Here, the knight \(3 movement\) moves up to 7 tiles thanks to Polarism.](images/Here,_the_knight_\(3_movement\)_moves_up_to_7_tiles_thanks_to_Polarism..webp)![Here, the knight \(3 movement\) moves up to 7 tiles thanks to Polarism.](data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D)](</wiki/File:Movement16.JPG> "Movement16.JPG \(71 KB\)")

Here, the knight (3 movement) moves up to 7 tiles thanks to Polarism.

## References[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMovement%3Fveaction%3Dedit%26section%3D12&uselang=en> "Sign in to edit")]

  * ["Movement in Polytopia"](<https://docs.google.com/document/d/1D5jMQVxpmyFtDlw-S_21uH1IsSWPY-4A7u9miuWeals/edit>) by QuasiStellar
  * ["Movement Guide"](<https://docs.google.com/document/d/1I82qRpIx1ynklrE8J8rHGYiKBc-8jdbX71hyqhtOGAs/edit>) by Glouc3stershire
  * ["Push Rules"](<https://docs.google.com/document/d/1C3nQm6SnRFc5pkWMy_WOj8LRosuj2XlINCmCoDfswn4/edit>) by Legorooj

  
---  
Game Mechanics  
Game|  [The Battle of Polytopia: Moonrise](</wiki/The_Battle_of_Polytopia:_Moonrise> "The Battle of Polytopia: Moonrise"), [Update History](<../general/Update_History.md> "Update History")  
General|  [Combat](<Combat.md> "Combat"), [Easter Eggs](<../general/Easter_Eggs.md> "Easter Eggs"), [Game Modes](<Game_Modes.md> "Game Modes"), [Map Generation](<Map_Generation.md> "Map Generation"), **Movement** , [Ruin](<../general/Ruin.md> "Ruin"), [Score](</wiki/Score> "Score"), [Star](</wiki/Star> "Star"), [Technology](</wiki/Technology> "Technology"), [Terrain](<Terrain.md> "Terrain"), [Tribes](</wiki/Tribes> "Tribes")  
[Abilities](</wiki/Abilities> "Abilities")|  [Burn Forest](</wiki/Burn_Forest> "Burn Forest"), [Clear Forest](</wiki/Clear_Forest> "Clear Forest"), [Destroy](</wiki/Destroy> "Destroy"), [Grow Forest](</wiki/Grow_Forest> "Grow Forest"), [Starfish Harvesting](</wiki/Starfish_Harvesting> "Starfish Harvesting"), [Whale Hunting](</wiki/Whale_Hunting> "Whale Hunting")  
[Buildings](</wiki/Buildings> "Buildings")|  [Buildings](</wiki/Buildings> "Buildings"), [Bridge](</wiki/Bridge> "Bridge"), [Embassy](</wiki/Embassy> "Embassy"), [Roads](</wiki/Roads_\(Building\)> "Roads \(Building\)"), [Temples](</wiki/Temples> "Temples")  
[City](</wiki/City> "City")|  [City](</wiki/City> "City"), [City Connection](</wiki/City_Connection> "City Connection"), [City Upgrades](</wiki/Category:City_Upgrades> "Category:City Upgrades"), [Population](</wiki/Population> "Population")  
[Diplomacy](</wiki/Category:Diplomacy> "Category:Diplomacy")|  [Diplomacy](</wiki/Diplomacy> "Diplomacy"), [Embassy](</wiki/Embassy> "Embassy"), [Peace Treaty](</wiki/Peace_Treaty> "Peace Treaty"), [Tribe Relations](</wiki/Tribe_Relations> "Tribe Relations")  
[Units](</wiki/Units> "Units")|  [Units](</wiki/Units> "Units"), [List of Units](</wiki/List_of_Units> "List of Units"), [Unit Skills](</wiki/Unit_Skills> "Unit Skills"), [Super Unit](</wiki/Super_Unit> "Super Unit"), [Disband](</wiki/Disband> "Disband")


---

*Ce contenu est extrait du wiki Polytopia et est sous licence CC-BY-SA.*
