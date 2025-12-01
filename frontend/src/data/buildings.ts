// Définitions complètes des bâtiments (correspondant à rules.py dans le backend)

import { BuildingType, TechType } from '../types';

export interface BuildingDefinition {
  id: number;
  name: string;
  cost: number;
  popGain: number;
  requiredTech: number;
  description: string;
  requiresCity: boolean; // true = doit être construit dans une ville, false = sur le terrain
  icon?: string; // Chemin vers l'icône
}

// Définitions des bâtiments selon rules.py
export const BUILDING_DEFINITIONS: Record<number, BuildingDefinition> = {
  [BuildingType.FARM]: {
    id: BuildingType.FARM,
    name: 'Ferme',
    cost: 3,
    popGain: 1,
    requiredTech: TechType.NONE,
    description: 'Augmente la population de la ville',
    requiresCity: true,
  },
  [BuildingType.MINE]: {
    id: BuildingType.MINE,
    name: 'Mine',
    cost: 4,
    popGain: 2,
    requiredTech: TechType.MINING,
    description: 'Récolte du minerai, +2 population',
    requiresCity: true,
  },
  [BuildingType.HUT]: {
    id: BuildingType.HUT,
    name: 'Hutte',
    cost: 2,
    popGain: 1,
    requiredTech: TechType.NONE,
    description: 'Augmente la population de la ville',
    requiresCity: true,
  },
  [BuildingType.PORT]: {
    id: BuildingType.PORT,
    name: 'Port',
    cost: 5,
    popGain: 0,
    requiredTech: TechType.SAILING,
    description: 'Permet aux unités d\'embarquer',
    requiresCity: true,
  },
  [BuildingType.WINDMILL]: {
    id: BuildingType.WINDMILL,
    name: 'Moulin',
    cost: 6,
    popGain: 0, // Dépend du nombre de fermes adjacentes
    requiredTech: TechType.NONE, // Tier 2
    description: '+1 pop par ferme adjacente',
    requiresCity: true,
  },
  [BuildingType.FORGE]: {
    id: BuildingType.FORGE,
    name: 'Forge',
    cost: 7,
    popGain: 0, // Dépend du nombre de mines adjacentes
    requiredTech: TechType.MINING,
    description: '+2 pop par mine adjacente',
    requiresCity: true,
  },
  [BuildingType.SAWMILL]: {
    id: BuildingType.SAWMILL,
    name: 'Scierie',
    cost: 5,
    popGain: 0, // Dépend du nombre de huttes adjacentes
    requiredTech: TechType.NONE, // Tier 2
    description: '+1 pop par hutte adjacente',
    requiresCity: true,
  },
  [BuildingType.MARKET]: {
    id: BuildingType.MARKET,
    name: 'Marché',
    cost: 8,
    popGain: 0,
    requiredTech: TechType.NONE, // Tier 2
    description: '+1 étoile par tour par port connecté',
    requiresCity: true,
  },
  [BuildingType.TEMPLE]: {
    id: BuildingType.TEMPLE,
    name: 'Temple',
    cost: 10,
    popGain: 0,
    requiredTech: TechType.NONE, // Tier 3
    description: 'Génère des points de score chaque tour',
    requiresCity: true,
  },
  [BuildingType.MONUMENT]: {
    id: BuildingType.MONUMENT,
    name: 'Monument',
    cost: 20,
    popGain: 0,
    requiredTech: TechType.NONE, // Tier 3
    description: 'Bonus de score important',
    requiresCity: true,
  },
  [BuildingType.CITY_WALL]: {
    id: BuildingType.CITY_WALL,
    name: 'Murs',
    cost: 5,
    popGain: 0,
    requiredTech: TechType.NONE, // Amélioration ville niveau 3
    description: 'Défense améliorée pour la ville',
    requiresCity: true,
  },
  [BuildingType.PARK]: {
    id: BuildingType.PARK,
    name: 'Parc',
    cost: 15,
    popGain: 0,
    requiredTech: TechType.NONE, // Amélioration ville niveau 5
    description: 'Bonus de score',
    requiresCity: true,
  },
  [BuildingType.ROAD]: {
    id: BuildingType.ROAD,
    name: 'Route',
    cost: 3,
    popGain: 0,
    requiredTech: TechType.ROADS,
    description: 'Permet un déplacement rapide',
    requiresCity: false, // Peut être construit sur le terrain
  },
  [BuildingType.BRIDGE]: {
    id: BuildingType.BRIDGE,
    name: 'Pont',
    cost: 5,
    popGain: 0,
    requiredTech: TechType.ROADS,
    description: 'Traverse l\'eau peu profonde',
    requiresCity: false, // Sur eau peu profonde
  },
};

// Bâtiments constructibles dans une ville (exclut ROAD et BRIDGE)
export const CITY_BUILDINGS = [
  BuildingType.PORT,
  BuildingType.WINDMILL,
  BuildingType.FORGE,
  BuildingType.SAWMILL,
  BuildingType.MARKET,
  BuildingType.TEMPLE,
  BuildingType.MONUMENT,
  BuildingType.CITY_WALL,
  BuildingType.PARK,
];

// Bâtiments constructibles sur le terrain
export const TERRAIN_BUILDINGS = [
  BuildingType.ROAD,
  BuildingType.BRIDGE,
];

/**
 * Retourne le nom d'un bâtiment à partir de son type
 */
export function getBuildingName(buildingType: number): string {
  return BUILDING_DEFINITIONS[buildingType]?.name ?? `Bâtiment #${buildingType}`;
}

/**
 * Retourne la définition complète d'un bâtiment
 */
export function getBuildingDefinition(buildingType: number): BuildingDefinition | undefined {
  return BUILDING_DEFINITIONS[buildingType];
}

/**
 * Vérifie si un bâtiment nécessite d'être dans une ville
 */
export function requiresCity(buildingType: number): boolean {
  return BUILDING_DEFINITIONS[buildingType]?.requiresCity ?? true;
}


