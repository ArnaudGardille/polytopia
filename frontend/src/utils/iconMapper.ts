import { TerrainType, UnitType } from '../types';

/**
 * Obtient le nom de la tribu à partir du playerId
 * Mapping des playerId vers les noms de tribus Polytopia
 */
function getTribeName(playerId?: number): string {
  if (playerId === undefined || playerId === null || playerId < 0) {
    return 'Imperius'; // Par défaut
  }
  
  // Mapping des playerId vers les tribus
  const tribeMapping: Record<number, string> = {
    0: 'Imperius',
    1: 'Xin-xi',
    2: 'Bardur',
    3: 'Oumaji',
    4: 'Kickoo',
    5: 'Hoodrick',
    6: 'Luxidoor',
    7: 'Vengir',
    8: 'Zebasi',
    9: 'Quetzali',
    10: 'Yadakk',
    11: 'Ai-mo',
    12: 'Aquarion',
    13: 'Polaris',
    14: 'Elyrion',
    15: 'Cymanti',
  };
  
  return tribeMapping[playerId] || 'Imperius';
}

/**
 * Obtient la lettre de la tribu à partir du playerId (pour compatibilité avec les unités)
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
 * Pour les villes colonisées, affiche le château de la tribu
 */
export function getCityIcon(level: number, playerId?: number): string | null {
  // Si pas de propriétaire ou propriétaire invalide (village neutre)
  if (playerId === undefined || playerId === null || playerId < 0) {
    return '/icons/cities/Village.png';
  }
  
  // Pour les villes colonisées, utiliser le château de la tribu
  const tribeName = getTribeName(playerId);
  return `/icons/cities/castles/${tribeName}_city_castle.png`;
}

/**
 * Couleurs pour les joueurs (inspirées de Polytopia)
 */
export function getPlayerColor(playerId: number): string {
  // Si playerId invalide (village neutre), utiliser une couleur neutre
  if (playerId < 0) {
    return '#9CA3AF'; // Gris neutre
  }
  
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
