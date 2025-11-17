import { TerrainType, UnitType } from '../types';

/**
 * Mapping des types de terrain vers les chemins d'icônes Polytopia
 * Utilise les vraies images téléchargées depuis le wiki
 */
export function getTerrainIcon(terrainType: number): string | null {
  switch (terrainType) {
    case TerrainType.PLAIN:
      return '/icons/terrain/plain.svg';
    case TerrainType.FOREST:
      return '/icons/terrain/forest.svg';
    case TerrainType.MOUNTAIN:
      return '/icons/terrain/mountain.svg';
    case TerrainType.WATER_SHALLOW:
      return '/icons/terrain/water_shallow.svg';
    case TerrainType.WATER_DEEP:
      return '/icons/terrain/water_deep.svg';
    default:
      return '/icons/terrain/plain.svg';
  }
}

/**
 * Mapping des types d'unités vers les chemins d'icônes Polytopia
 * Utilise les vraies images téléchargées depuis le wiki
 */
export function getUnitIcon(unitType: number, playerId?: number): string | null {
  switch (unitType) {
    case UnitType.WARRIOR:
      // Utiliser une image de guerrier générique
      return '/icons/units/WarriorA.png';
    default:
      return '/icons/units/WarriorA.png';
  }
}

/**
 * Mapping des villes vers les chemins d'icônes Polytopia
 */
export function getCityIcon(level: number, playerId?: number): string | null {
  // Utiliser une image de ville générique
  // On pourrait utiliser des images spécifiques selon le niveau
  return '/icons/cities/Village.png';
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
      return '#D4A574'; // Beige/sable
    case TerrainType.FOREST:
      return '#2D5016'; // Vert foncé
    case TerrainType.MOUNTAIN:
      return '#6B7280'; // Gris
    case TerrainType.WATER_SHALLOW:
      return '#60A5FA'; // Bleu clair
    case TerrainType.WATER_DEEP:
      return '#3B82F6'; // Bleu foncé
    default:
      return '#D4A574';
  }
}

