# Map Generation

**Source:** https://polytopia.fandom.com/wiki/File:We_Analyzed_2,306_Lines_of_Code_to_Discover_THIS_about_Polytopia_Maps  
**Licence:** CC-BY-SA  
**Date de scraping:** 2025-11-24

---

Every game occurs on a unique, randomly generated square map. Maps come in six sizes and six water amounts. [Terrain](<../game_mechanics/Terrain.md> "Terrain") is spawned in proportions that vary by [tribe](<Tribes.md> "Tribe"). Each tribe has a unique set of spawn rates and unique aesthetics for terrain and resources. During the initial map generation, the land is distributed roughly equally between all tribes in the game.   
  
## Contents

  * 1 Map Size
  * 2 Water Amount
  * 3 Spawn Rates
    * 3.1 Spawn Rate Modifiers by Tribe
  * 4 Map Generation Process
    * 4.1 Capital spawns
    * 4.2 Village Spawning Rules
    * 4.3 Suburbs
    * 4.4 Pre-terrain villages
    * 4.5 Post-terrain villages
      * 4.5.1 Continents and Pangea maps
        * 4.5.1.1 Pangea Map Generation process
        * 4.5.1.2 Continents Map Generation process
      * 4.5.2 Tiny Island Villages
    * 4.6 Ruins
      * 4.6.1 Starfish
  * 5 Other
  * 6 References

## Map Size[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D1&uselang=en> "Sign in to edit")]

There are six map sizes in Polytopia: 

  * Tiny: 121 tiles, 11x11
  * Small: 196 tiles, 14x14
  * Normal: 256 tiles, 16x16
  * Large: 324 tiles, 18x18
  * Huge: 400 tiles, 20x20
  * Massive: 900 tiles, 30x30

Map size changes based on the [game mode](<../game_mechanics/Game_Modes.md> "Game Modes"). Perfection games use a normal map. In Domination, the map size changes depending on the number of opponents: Games with one opponent are on a tiny map, two opponents on a 196-tile (14x14) map, three opponents on a normal map, and four or more on a large map. 

In multiplayer and creative, map size is set by the host when creating a new game. Tiny maps support a maximum of nine players, while other maps support up to 16. 

## Water Amount[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D2&uselang=en> "Sign in to edit")]

Each map type has a different level of wetness that guides the algorithm in how much water to place. 

  * Drylands: Entirely land. ([Kickoo](<../tribes/Kickoo.md> "Kickoo") and [Aquarion](<../tribes/Aquarion.md> "Aquarion") capitals will include two water tiles with a fish in each.)
  * Lakes: A border of land bridges will always generate on the edges of the map. Every player is guaranteed to spawn with land connections to at least two villages.
  * Continents: Large but discrete landmasses. Most similar to pre-[Moonrise](<The_Battle_of_Polytopia__Moonrise.md> "Moonrise") maps.
  * Pangea: A large landmass in the middle of the map surrounded by ocean on the outskirts.
  * Archipelago: Misshapen and discontinuous landmasses. Almost every city is coastal.
  * Water World: Small landmasses of one or two villages.

**Map Type** | **Wetness** | **How It Affects Water/Ocean**  
---|---|---  
**Drylands** | 0-10%  | Almost no water   
**Lakes** | 25-30%  | Moderate wetness results in scattered inland lakes.   
**Continents** | 40-70%  | High wetness shrinks continents, expands surrounding ocean.   
**Pangea** | 40-60%  | Large land mass surrounded by ocean and the occasional island   
**Archipelago** | 60-80%  | High wetness leads to small island chains and more ocean.   
**Waterworld** | 90-100%  | Almost all tiles become water or ocean; land is manually forced for cities.   
  
## Spawn Rates[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D3&uselang=en> "Sign in to edit")]

The standard spawn rates for land tiles are listed in the table below. "Inner city" refers to tiles that are adjacent to a city or village, while "outer city" refers to those that are not. 

Since [update 2.0.58 (Balance Pass 2)](<Update_History.md> "Update History"), terrain will always be generated in accordance with the percentage rates - for example, [Luxidoor](<../tribes/Luxidoor.md> "Luxidoor") will always have 48% of their biome (or as close as possible) as field tiles. 

| Inner City  | Outer City   
---|---|---  
Field (Total)  | 48%  | 48%   
Fruit  | 18%  | 6%   
Crop  | 18%  | 6%   
Empty Field  | 12%  | 36%   
Forest (Total)  | 38%  | 38%   
Animal  | 19%  | 6%   
Empty Forest  | 19%  | 32%   
Mountain (Total)  | 14%  | 14%   
Metal  | 11%  | 3%   
Empty Mountain  | 3%  | 11%   
  
The standard spawn rate for fish is 50% among shallow water tiles. Fish can also spawn in ocean tiles within a tribe's borders and some outside borders, which can be obtained with Border Growth. Starfish can spawn on any water tiles, and are scattered across the map. 

All resources spawn only within two tiles of cities or villages. This fact can be used to locate villages and enemy cities. 

### Spawn Rate Modifiers by Tribe[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D4&uselang=en> "Sign in to edit")]

Modifiers by tribe are direct multipliers on the base rates. For example, Xin-xi has 2 times, or 100% more mountains compared to the base rate (Luxidoor). This will indirectly change the ration of the other resources. Modifiers are a little bit tricky to work with as not every modifier impacts all other terrain. 

  1. First the mountain modifier is checked. If this is not the base rate, it will affect both the forest and field rate. An increase of mountains will thus proportionally change the forest rate and field rate.
  2. After this was changed, the forest multiplier is checked. Since this is checked after the mountain modifier, it will no longer impact the mountain rate, but only impact the fields.
  3. Finally the field rate. There is no tribe that has a direct modifier for this and fields are thus just calculated via the simple 100% - Mountain% - Forest%.

Note that if forced in a 121 tile map, surrounding tribes can affect the spawn rates in the capital city. (ex. A naturally spawning Bardur crop.) 

  * [Xin-xi](<../tribes/Xin-xi.md> "Xin-xi"): 1.5x mountain, 1.5x metal
  * [Imperius](<../tribes/Imperius.md> "Imperius"): 0.5x wild animal, 2.0x fruit
  * [Bardur](<../tribes/Bardur.md> "Bardur"): 0.8x forest, 0x crop
  * [Oumaji](<../tribes/Oumaji.md> "Oumaji"): 0.2x forest, 0.2x wild animal, 0.5x mountain, 0.5x water[note 1]
  * [Kickoo](<../tribes/Kickoo.md> "Kickoo"): 0.5x mountain, 1.5x fish, 2.0x water[note 2]
  * [Hoodrick](<../tribes/Hoodrick.md> "Hoodrick"): 0.5x mountain, 1.5x forest 
  * [Luxidoor](<../tribes/Luxidoor.md> "Luxidoor"): same as base rate
  * [Vengir](<../tribes/Vengir.md> "Vengir"): 2.0x metal, 0.1x wild animal, 0.1x fruit, 0.1x fish 
  * [Zebasi](<../tribes/Zebasi.md> "Zebasi"): 0.5x mountain, 0.5x forest, 0.5x fruit
  * [Ai-Mo](<../tribes/Ai-Mo.md> "Ai-Mo"): 1.5x mountain, 0.1x crop
  * [Quetzali](<../tribes/Quetzali.md> "Quetzali"): 2.0x fruit, 0.1x crop
  * [Yădakk](<Yădakk.md> "Yădakk"): 0.5x mountain, 0.5x forest, 1.5x fruit
  * [Aquarion](<../tribes/Aquarion.md> "Aquarion"): 0.5x forest, 1.5x water[note 2]
  * [∑∫ỹriȱŋ](<∑∫ỹriȱŋ.md> "∑∫ỹriȱŋ"): 0.5x mountain, 1.5x crop
  * [Polaris](<../tribes/Polaris.md> "Polaris"): same as non-polaris opponent(s), else base rate
  * [Cymanti](<../tribes/Cymanti.md> "Cymanti"): 1.2x mountain, crop rate replaced with spore rate, cannot spawn crop. [![Resource rates by tribe](images/Resource_rates_by_tribe.webp)](<https://static.wikia.nocookie.net/supertribes/images/5/58/Resource_rates_by_tribe.png/revision/latest?cb=20250608214300>) [](<Map_Generation.md>)

Resource reates by tribe

  1. ↑ Reductions to the amount of water are found in the source code but are not translated into the world generation and do not impact water rates.
  2. ↑ 2.0 2.1 On continents, Kickoo and Aquarion should have 40% and 30%, respectively, of their tiles replaced with water after the initial map generation, but this effect is bugged and does not currently apply to map generation.

## Map Generation Process[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D5&uselang=en> "Sign in to edit")]

[ ![We_Analyzed_2,306_Lines_of_Code_to_Discover_THIS_about_Polytopia_Maps](images/We_Analyzed_2,306_Lines_of_Code_to_Discover_THIS_about_Polytopia_Maps.webp) ](<https://polytopia.fandom.com/wiki/File:We_Analyzed_2,306_Lines_of_Code_to_Discover_THIS_about_Polytopia_Maps>) [](</wiki/File:We_Analyzed_2,306_Lines_of_Code_to_Discover_THIS_about_Polytopia_Maps>)

We Analyzed 2,306 Lines of Code to Discover THIS about Polytopia Maps

A video explaining the map generation process. By [Espark](</wiki/User:Espark> "User:Espark").

The typical order for map creation is listed below, however each map has variations in this process:[1]

  1. Capital Spawns
  2. Villages
  3. Terrain
  4. Resources
  5. Ruins/Starfish

### Capital spawns[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D6&uselang=en> "Sign in to edit")]

Most maps use quadrants to place capitals so players don’t start too close to one another. Quadrants are evenly sized zones on the map. The number of quadrands/domains depends on number of players: 1-4 players creates 4 domains, 5-9 players makes 9 domains, and 10-16 players makes 16 domains. For the map types that use quandrants, Drylands, Lakes, Archipelago, and Waterworld, capital cities are distributed in an unoccupied quadrant at the begining of map generation based on weighted randomness. 

### Village Spawning Rules[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D7&uselang=en> "Sign in to edit")]

Spawning rules can be used to predict the locations of ruins, villages, and cities. 

The three kinds of villages are: 

  1. Suburbs (Lakes and Archipelago only)
  2. Pre-terrain villages, also called primary villages
  3. Post-terrain, also called secondary villages

Depending on the map type, the algorithm uses a different combination of rules to place villages. 

**Feature** | **Drylands** | **Archipelago** | **Lakes** | **Waterworld** | **Continents / Pangea**  
---|---|---|---|---|---  
Quadrants for capitals  | ✅ Yes  | ✅ Yes  | ✅ Yes  | ✅ Yes  | ❌ No   
Suburbs  | ❌ No  | ✅ Yes  | ✅ Yes  | ❌ No  | ❌ No   
Pre-Terrain Cities  | ❌ No  | ✅ Yes  | ✅ Yes  | ✅ Yes  | ❌ No (not applicable)   
Post-Terrain Cities  | ✅ Yes  | ✅ Yes  | ✅ Yes  | ✅ Yes  | ✅ Yes   
Tiny Island Cities  | ❌ No  | ❌ No  | ❌ No  | ✅ Yes  | ✅ Yes   
Pre-terrain city coefficient  | None  | 0.3  | 0.3  | 0.1  | None   
  
### Suburbs[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D8&uselang=en> "Sign in to edit")]

In Polytopia, a **suburb** is a special village that is associated with a capital. For Lakes and Archipelago, after the capital city is placed, the algorithm puts up to two villages nearby. It is possible to get zero or only one suburb per capital on any map, but a single suburb is less likely to occur on larger maps; you usually get two. Drylands, Continents, Pangea, and Waterworld do not use the suburb algorithm. While there is no guarantee those suburbs will be close to the capital, because suburbs are placed before terrain, they are usually a couple tiles away. 

### Pre-terrain villages[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D9&uselang=en> "Sign in to edit")]

Archipelago, Lakes and Waterworld use forumula to determine how many villages placed after capitals and sububs. 

Formula:

> Pre-terrain villages placed = (([map width]/3)^2-[amount of capitals and suburbs])* (map density coefficent) 

Lakes and Archipelago use a denisty coefficent of 0.3; waterworld uses a density coefficent of 0.1. The map width/3 number is rounded down before being squared. 

Pre-terrain villages must two tiles from other villages and at least one tile from the edge of the map. 

### Post-terrain villages[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D10&uselang=en> "Sign in to edit")]

After terrain is added to the map, including mountains, forests, and water, additional villages are added. At this point, the game does not care about the number of villages on the map. It just randomly throws in villages following post-terrain rules until there is no tile left where it can place one. 

Post terrain villages must: 

  * not be placed within two tiles of the edge of the map
  * not be within two tiles of other villages
  * not be three tiles from the edge of the map

All Drylands map villages are post-terrain villages. 

##### Continents and Pangea maps[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D11&uselang=en> "Sign in to edit")]

Contients and Pangea are made differently than other maps. There are no suburbs or pre-terrain villages. Villages are added after all the land is made, that's why Continents and Pangea don’t use quadrants for capital placement. 

###### Pangea Map Generation process[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D12&uselang=en> "Sign in to edit")]

  1. A single starting point in the center of the map is chosen to seed the landmass. The ratio of land and water generated is based on wetness. For Pangea, about half of the tiles are water.
  2. Pangea does not use quadrants. Instead, villages are placed on land one at a time until there is no space left as long as the villages are two tiles from each other.
  3. Next, some of those villages that are already generated are converted to capitals based on two factors. 
     * First, the game wants the capitals to be as far away from each other as possible
     * Second, the algorithm prefers to have capitals next to water tiles. The means you almost always get capitals spawns on the coast
     * After the villages and capitals are set on the main land mass, island villages are generated based on the map size.

###### Continents Map Generation process[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D13&uselang=en> "Sign in to edit")]

Continents land is added through a procedure called noise, which gives natural looking variation to the map. The number of land masses is based on the number of players, wetness level, and map size. For continents maps, about half the tiles are water. Continents have between 30 and 200 tiles of land and must be at least one tile apart. For a 196 tiles map and two players, half the map being water, then each continent would have about 50 tiles. You often get one tile strips of shallow water, also called rivers, which can be great for bridges. 

After continents are created, the algorithm places one village at a time on each continent. Villages must be two tiles away from each other. The game also makes sure to put at least one village on each continent. Because this is a small size map, we get one tiny island village. 

Then, like on Pangea, one village per player is converted to a capital. Capitals are placed on different land masses, space permitting. 

#### Tiny Island Villages[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D14&uselang=en> "Sign in to edit")]

Continents and Pangea have a fixed amount of villages that spawn in the water without any land connections. These island villages are created after the mainland villages. 

**Map size** | **Tiles** | **Tiny Island villages**  
---|---|---  
Tiny  | 121  | 0   
Small  | 196  | 1   
Normal  | 256  | 2   
Large  | 324  | 3   
Huge  | 400  | 4   
Massive  | 900  | 9   
  
  
One exception to these rules is the [Aquarion](<../tribes/Aquarion.md> "Aquarion") tribe’s ability to discover Lost cities from deep water ruins. If aquarion claims a ruin in deep ocean, it will always turn into a level 3 city plus the water around the city will spawn new resources, like fish and aquacrops 

### Ruins[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D15&uselang=en> "Sign in to edit")]

[Ruins](</wiki/Ruin> "Ruin") are placed randomly after villages, resources, and lighthouses. Ruins can spawn on mountains, forests, fields, or deep ocean. They cannot spawn right next to another ruin or a village. The number of ruins is based on the size of the map. On Lakes, a maximum of one third of these ruins are allowed to spawn on water. 

Number of ruins based on map size  **Map size** | **Tiles** | **Ruins**  
---|---|---  
Tiny  | 121  | 4   
Small  | 196  | 5   
Normal  | 256  | 7   
Large  | 324  | 9   
Huge  | 400  | 11   
Massive  | 900  | 23   
  
#### Starfish[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D16&uselang=en> "Sign in to edit")]

The map makes approximately one starfish for every 25 water tiles. Starfish may spawn on shallow or deep ocean tiles. Starfish cannot be next to another starfish, lighthouse, or city. 

## Other[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D17&uselang=en> "Sign in to edit")]

<https://youtu.be/fJ562xzAIVs> Map Gen explained on PolyChampions YouTube 

PolyChampions YouTube

## References[[](<https://auth.fandom.com/signin?redirect=https%3A%2F%2Fpolytopia.fandom.com%2Fwiki%2FMap_Generation%3Fveaction%3Dedit%26section%3D18&uselang=en> "Sign in to edit")]

  1. ↑ Post by Espark on [YouTube](<https://youtu.be/fJ562xzAIVs>). 

  
  
---  
Game Mechanics  
Game|  [The Battle of Polytopia: Moonrise](</wiki/The_Battle_of_Polytopia:_Moonrise> "The Battle of Polytopia: Moonrise"), [Update History](<Update_History.md> "Update History")  
General|  [Combat](</wiki/Combat> "Combat"), [Easter Eggs](</wiki/Easter_Eggs> "Easter Eggs"), [Game Modes](<../game_mechanics/Game_Modes.md> "Game Modes"), **Map Generation** , [Movement](</wiki/Movement> "Movement"), [Ruin](</wiki/Ruin> "Ruin"), [Score](</wiki/Score> "Score"), [Star](</wiki/Star> "Star"), [Technology](</wiki/Technology> "Technology"), [Terrain](<../game_mechanics/Terrain.md> "Terrain"), [Tribes](</wiki/Tribes> "Tribes")  
[Abilities](</wiki/Abilities> "Abilities")|  [Burn Forest](</wiki/Burn_Forest> "Burn Forest"), [Clear Forest](</wiki/Clear_Forest> "Clear Forest"), [Destroy](</wiki/Destroy> "Destroy"), [Grow Forest](</wiki/Grow_Forest> "Grow Forest"), [Starfish Harvesting](</wiki/Starfish_Harvesting> "Starfish Harvesting"), [Whale Hunting](</wiki/Whale_Hunting> "Whale Hunting")  
[Buildings](</wiki/Buildings> "Buildings")|  [Buildings](</wiki/Buildings> "Buildings"), [Bridge](</wiki/Bridge> "Bridge"), [Embassy](</wiki/Embassy> "Embassy"), [Roads](</wiki/Roads_\(Building\)> "Roads \(Building\)"), [Temples](</wiki/Temples> "Temples")  
[City](</wiki/City> "City")|  [City](</wiki/City> "City"), [City Connection](</wiki/City_Connection> "City Connection"), [City Upgrades](</wiki/Category:City_Upgrades> "Category:City Upgrades"), [Population](</wiki/Population> "Population")  
[Diplomacy](</wiki/Category:Diplomacy> "Category:Diplomacy")|  [Diplomacy](</wiki/Diplomacy> "Diplomacy"), [Embassy](</wiki/Embassy> "Embassy"), [Peace Treaty](</wiki/Peace_Treaty> "Peace Treaty"), [Tribe Relations](</wiki/Tribe_Relations> "Tribe Relations")  
[Units](</wiki/Units> "Units")|  [Units](</wiki/Units> "Units"), [List of Units](</wiki/List_of_Units> "List of Units"), [Unit Skills](</wiki/Unit_Skills> "Unit Skills"), [Super Unit](</wiki/Super_Unit> "Super Unit"), [Disband](</wiki/Disband> "Disband")


---

*Ce contenu est extrait du wiki Polytopia et est sous licence CC-BY-SA.*
