import { TerrainType, UnitType } from '../types';

/**
 * Obtient la lettre de la tribu à partir du playerId
 * Pour l'instant : 0 = Imperius (I), 1 = Xin-xi (X), défaut = Imperius (I)
 */
function getTribeLetter(playerId?: number): string {
  if (playerId === undefined || playerId === null) {
    return 'I'; // Imperius par défaut
  }
  // Joueur 0 = Imperius (I), Joueur 1 = Xin-xi (X)
  return playerId === 1 ? 'X' : 'I';
}

/**
 * Mapping des types de terrain vers les chemins d'icônes Polytopia
 * Utilise les vraies images PNG téléchargées
 */
export function getTerrainIcon(terrainType: number): string | null {
  switch (terrainType) {
    case TerrainType.PLAIN:
      return '/icons/terrain/Grass.png';
    case TerrainType.PLAIN_FRUIT:
      return '/icons/terrain/Field_with_Fruit.png';
    case TerrainType.FOREST:
      return '/icons/terrain/Imperius_ground_with_forest.png';
    case TerrainType.FOREST_WITH_WILD_ANIMAL:
      return '/icons/terrain/Imperius_ground_with_forest_with_wild_animal.png';
    case TerrainType.MOUNTAIN:
      return '/icons/terrain/Imperius_ground_with_mountain.png';
    case TerrainType.MOUNTAIN_WITH_MINE:
      return '/icons/terrain/Imperius_ground_with_mountain_and_metal.png';
    case TerrainType.WATER_SHALLOW:
      return '/icons/terrain/Shallow_water-1.png';
    case TerrainType.WATER_SHALLOW_WITH_FISH:
      return '/icons/terrain/Shallow_water_with_fish-0.png';
    case TerrainType.WATER_DEEP:
      return '/icons/terrain/Ocean.png';
    default:
      return '/icons/terrain/Grass.png';
  }
}

/**
 * Mapping des types d'unités vers les chemins d'icônes Polytopia
 * Utilise les vraies images PNG spécifiques à chaque tribu
 */
export function getUnitIcon(unitType: number, playerId?: number): string | null {
  const tribe = getTribeLetter(playerId);
  
  switch (unitType) {
    case UnitType.WARRIOR:
      return `/icons/units/Warrior${tribe}.png`;
    default:
      // Par défaut, utiliser un guerrier
      return `/icons/units/Warrior${tribe}.png`;
  }
}

/**
 * Mapping des villes vers les chemins d'icônes Polytopia
 * Utilise les icônes spécifiques à chaque tribu et niveau
 */
export function getCityIcon(level: number, playerId?: number): string | null {
  const tribe = getTribeLetter(playerId);
  
  // Pour les villes de niveau élevé (châteaux)
  if (level >= 3) {
    if (tribe === 'I') {
      return '/icons/cities/castles/Imperius_city_castle.png';
    } else if (tribe === 'X') {
      return '/icons/cities/castles/Xin-xi_city_castle.png';
    }
  }
  
  // Pour les villes de niveau basique
  if (tribe === 'I') {
    return '/icons/cities/IMPERIUS_CITY.png';
  }
  
  // Par défaut, utiliser le village générique
  return '/icons/terrain/Village.png';
}

/**
 * Couleurs pour les joueurs (inspirées de Polytopia)
 */
export function getPlayerColor(playerId: number): string {
  const colors = [
    '#3B82F6', // Bleu (joueur 0)
    '#EF4444', // Rouge (joueur 1)
    '#10B981', // Vert (joueur 2)
    '#F59E0B', // Orange (joueur 3)
    '#8B5CF6', // Violet (joueur 4)
    '#EC4899', // Rose (joueur 5)
  ];
  return colors[playerId % colors.length];
}

/**
 * Couleurs de fond pour les types de terrain
 */
export function getTerrainColor(terrainType: number): string {
  switch (terrainType) {
    case TerrainType.PLAIN:
    case TerrainType.PLAIN_FRUIT:
      return '#D4A574'; // Beige/sable
    case TerrainType.FOREST:
    case TerrainType.FOREST_WITH_WILD_ANIMAL:
      return '#2D5016'; // Vert foncé
    case TerrainType.MOUNTAIN:
    case TerrainType.MOUNTAIN_WITH_MINE:
      return '#6B7280'; // Gris
    case TerrainType.WATER_SHALLOW:
    case TerrainType.WATER_SHALLOW_WITH_FISH:
      return '#60A5FA'; // Bleu clair
    case TerrainType.WATER_DEEP:
      return '#3B82F6'; // Bleu foncé
    default:
      return '#D4A574';
  }
}
